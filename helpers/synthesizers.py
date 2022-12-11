import copy
from abc import ABC, abstractclassmethod
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
from kornia import augmentation
from torchvision import transforms
from tqdm import tqdm
import torchvision.utils as vutils
from helpers.utils import ImagePool, DeepInversionHook, average_weights, kldiv

upsample = torch.nn.Upsample(mode='nearest', scale_factor=7)


class MultiTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [t(x) for t in self.transform]

    def __repr__(self):
        return str(self.transform)

class Ensemble_A(torch.nn.Module):
    def __init__(self, model_list):
        super(Ensemble_A, self).__init__()
        self.models = model_list

    def forward(self, x):
        logits_total = 0
        for i in range(len(self.models)):
            logits = self.models[i](x)
            logits_total += logits
        logits_e = logits_total / len(self.models)

        return logits_e

class Ensemble_M(torch.nn.Module):
    def __init__(self, model_list):
        super(Ensemble_M, self).__init__()
        self.models = model_list

    def forward(self, x):
        logits_list = []
        for i in range(len(self.models)):
            logits = self.models[i](x)
            logits_list.append(logits)
        # 把list送入到mlp中
        logits_e = torch.stack((logits_list[0], logits_list[1],
                                logits_list[2], logits_list[3], logits_list[4]))
        data = logits_e.permute(1, 2, 0)  # [bs,num_cls,5]
        return data


def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


class MLP(nn.Module):
    """
    输入为logits,输出为weight矩阵
    [[bs,num_cls]*5]=[bs,num_cls,5]  ----> [[bs,1]*5], 搭配上[bs,num_cls]
    给每个样本配上一个权重，应该为[bs,1]*[bs,num_cls]
    """

    def __init__(self, dim_in=500, dim_hidden=100, dim_out=5):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        # --------------------
        bs = x.shape[0]  # x:[bs,num_cls,5]
        ori_data = x.permute(2, 0, 1)  # [5,bs,num_cls]
        logits_total = 0
        # --------------------
        x = x.reshape(-1, x.shape[2] * x.shape[1])  # [bs,num_cls*5]
        x = self.layer_input(x)  # [bs,dim_hidden]
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)  # [bs, dim_out]
        # ----------
        y_prob = F.softmax(x, dim=1)  # [bs, 5]
        for i in range(5):
            tmp = y_prob[:, i].reshape(bs, -1).cuda()
            logits = ori_data[i].mul(tmp)  # [bs,10] [bs,5]取第i列，对应点乘
            logits_total += logits
        logits_final = logits_total / 5.0
        return logits_final


class AdvSynthesizer():
    def __init__(self, teacher, model_list, student, generator, nz, num_classes, img_size,
                 iterations, lr_g,
                 synthesis_batch_size, sample_batch_size,
                 adv, bn, oh, save_dir, dataset):
        super(AdvSynthesizer, self).__init__()
        self.student = student
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.nz = nz
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.save_dir = save_dir
        self.data_pool = ImagePool(root=self.save_dir)
        self.data_iter = None
        self.teacher = teacher
        self.dataset = dataset

        self.generator = generator.cuda().train()
        self.model_list = model_list

        self.aug = MultiTransform([
            # global view
            transforms.Compose([
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
            ]),
            # local view
            transforms.Compose([
                augmentation.RandomResizedCrop(size=[self.img_size[-2], self.img_size[-1]], scale=[0.25, 1.0]),
                augmentation.RandomHorizontalFlip(),
            ]),
        ])
        # =======================
        if not ("cifar" in dataset):
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])

        # datasets = self.data_pool.get_dataset(transform=self.transform)  # 获取程序运行到现在所有的图片
        # if len(datasets) != 0:
        #     self.data_loader = torch.utils.data.DataLoader(
        #         datasets, batch_size=self.sample_batch_size, shuffle=True,
        #         num_workers=4, pin_memory=True, )

    def gen_data(self, cur_ep):
        self.synthesize(self.teacher, cur_ep)

    def get_data(self):
        datasets = self.data_pool.get_dataset(transform=self.transform)  # 获取程序运行到现在所有的图片
        self.data_loader = torch.utils.data.DataLoader(
            datasets, batch_size=self.sample_batch_size, shuffle=True,
            num_workers=4, pin_memory=True, )
        return self.data_loader

    def synthesize(self, net, cur_ep):
        net.eval()
        best_cost = 1e6
        best_inputs = None
        z = torch.randn(size=(self.synthesis_batch_size, self.nz)).cuda()  #
        z.requires_grad = True
        targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        targets = targets.sort()[0]
        targets = targets.cuda()
        reset_model(self.generator)
        optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, {'params': [z]}], self.lr_g,
                                     betas=[0.5, 0.999])
        hooks = []
        #############################################
        dim_in = 500 if "cifar100" == self.dataset else 50
        net = Ensemble_A(self.model_list)
        net.eval()
        # net_mlp = MLP(dim_in).cuda()
        # net_mlp.train()
        # optimizer_mlp = torch.optim.SGD(net_mlp.parameters(), lr=0.01,
        #                                 momentum=0.9)
        #############################################
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                hooks.append(DeepInversionHook(m))

        with tqdm(total=self.iterations) as t:
            for it in range(self.iterations):
                optimizer.zero_grad()
                # optimizer_mlp.zero_grad()
                inputs = self.generator(z)  # bs,nz
                global_view, _ = self.aug(inputs)  # crop and normalize
                #############################################
                # Gate
                t_out = net(global_view)
                # data_ensm = net(global_view)
                # t_out = net_mlp(data_ensm)
                #############################################
                # t_out = net(global_view)
                loss_bn = sum([h.r_feature for h in hooks])  # bn层loss
                loss_oh = F.cross_entropy(t_out, targets)  # ce_loss
                # if cur_ep <= 20:
                #     adv = 1
                # elif cur_ep <= 50:
                #     adv = 10
                # elif cur_ep <= 100:
                #     adv = 20
                # elif cur_ep <= 150:
                #     adv = 30
                # else:
                #     adv = 50
                # self.adv = adv
                s_out = self.student(global_view)
                mask = (s_out.max(1)[1] != t_out.max(1)[1]).float()
                loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(
                    1) * mask).mean()  # decision adversarial distillation

                loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv
                # loss = loss_inv
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data

                loss.backward()
                optimizer.step()
                # optimizer_mlp.step()
                t.set_description('iters:{}, loss:{}'.format(it, loss.item()))
            vutils.save_image(best_inputs.clone(), '1.png', normalize=True, scale_each=True, nrow=10)

        # save best inputs and reset data iter
        self.data_pool.add(best_inputs)  # 生成了一个batch的数据
