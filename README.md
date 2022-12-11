# DENSE: Data-Free One-Shot Federated Learning

This repository is the official PyTorch implementation of:

[DENSE: Data-Free One-Shot Federated Learning (NeurIPS 2022)](https://arxiv.org/abs/2112.12371).

<img width="1530" alt="image" src="https://user-images.githubusercontent.com/33173674/206908843-00928ac2-0d1f-4636-8023-5ed37c893140.png">


## Requirements
- This codebase is written for python3 (used python 3.6 while implementing).
- We use Pytorch version of 1.8.2, 11.4 CUDA version.
- To install necessary python packages,

`pip install -r requirements.txt`

- The logs are uploaded to the wandb server. If you do not have a wandb account, just install and use as offline mode.
```shell
pip install wandb
wandb off
```

## How to Run Codes?

### Local Training 
At first, we divide the data into $m$ clients, train the model on each client until it converges and send it to the central server. The sample code is as follows:
```shell
python3 loop_df_fl.py --type=pretrain  --iid=0 --lr=0.01 --model=cnn \
--dataset=cifar10 --beta=0.5 --seed=1 --num_users=5 --local_ep=400
```
Here is an explanation of some parameters,
- `--dataset`: name of the datasets (e.g., `mnist`, `cifar10`, `cifar100` or `cinic10`).
- `--num_users`: the number of total clients (e.g., 5, 20, 100).
- `--batch_size`: the size of batch to be used for local training. (default: 256)
- `--iid`: IID or non-IID partition strategy (0 for non-IID).
- `--beta`: concentration parameter beta for latent Dirichlet Allocation (default: beta=0.5).
- `--model`: model architecture to be used (e.g., `cnn`, `resnet`, or `vit`).
- `--epochs`: the number of total communication rounds. (default: `200`)
- `--frac`: fraction of clients to be ramdonly sampled at each round (default: `1`)
- `--local_ep`: the number of local epochs (default: `400`).
- `--lr`: the initial learning rate for local training (default: `0.01`)
- `--momentum`: the momentum for SGD (default: `0.9`).
- `--seed`: random seed
- `--adv`: scaling factor for adv loss
- `--bn`: scaling factor for BN regularization
- `--lr_g`: learning rate for the generator

**Note that the same random seed must be fixed for fair comparison. Because different random seeds mean that the data distribution on each client is different.** Therefore, we should use several random seeds in the experiments. For `args.seed=1`. Here is an example for `--seed=1` (cifar10, 5 clients, $beta$=0.5),
```
Data statistics: {client 0: {0: 156, 1: 709, 2: 301, 3: 2629, 4: 20, 5: 651, 6: 915, 7: 113, 8: 180, 9: 2133}, \
client 1: {0: 1771, 1: 2695, 2: 1251, 3: 1407, 4: 665, 5: 314, 6: 1419, 7: 3469}, \
client 2: {0: 236, 1: 15, 2: 1715, 3: 76, 4: 1304, 5: 34, 6: 1773, 7: 75, 8: 3289, 9: 2360}, \
client 3: {0: 2809, 1: 575, 2: 157, 3: 853, 4: 2555, 5: 2557, 6: 203, 7: 1213}, \
client 4: {0: 28, 1: 1006, 2: 1576, 3: 35, 4: 456, 5: 1444, 6: 690, 7: 130, 8: 1531, 9: 507}}
```

The sample learning curves for local training:

<img width="549" alt="image" src="https://user-images.githubusercontent.com/33173674/206910177-6f7bdd5e-0f4d-4ade-be99-91c0c8865f15.png">


The accuracy for model ensemble (teacher) and the accuracy after FedAvg:
```
For each client, Accuracy: 55 / 55 / 59 / 60 / 62
FedAvg Accuracy: 30.9300
Ensemble Accuracy: 71.8100
```


### Global Distillation

Then we use the ensemble of local models for KD. Here is the sample code,
```
python loop_df_fl.py --type=kd_train --iid=0 --epochs=200 --lr=0.005 --batch_size 256 --synthesis_batch_size=256 \
--g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/cifar10 --other=cifar10 --model=cnn --dataset=cifar10 \
--adv=1 --beta=0.5 --seed=1

```
<img width="552" alt="image" src="https://user-images.githubusercontent.com/33173674/206911016-e79429d0-e206-4876-b2b8-970d62a0ad4b.png">

Note that the parameters are not well-designed in the experiments, intuitively the accuracy of KD should be close to the performance of the model ensemble.


The synthetic data for one batch (CIFAR10):

<img width="264" alt="image" src="https://user-images.githubusercontent.com/33173674/206911598-156a59ad-7cd9-4910-a655-05dbcdcdb4c6.png">


## Citing this work

```
@inproceedings{zhangdense,
  title={DENSE: Data-Free One-Shot Federated Learning},
  author={Zhang, Jie and Chen, Chen and Li, Bo and Lyu, Lingjuan and Wu, Shuang and Ding, Shouhong and Shen, Chunhua and Wu, Chao},
  booktitle={Advances in Neural Information Processing Systems}
}
```

