"""
训练
"""
import torch
from torch import nn, optim
from model import Model
import argparse
import os

from torch.utils.data import DataLoader

from DataSet import MyDataSet
from torch.utils.data.distributed import DistributedSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 多核训练
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    print('local_rank', local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)


def train(opt):
    batch = opt.batch
    epochs = opt.epoch
    lr = opt.lr

    if not os.path.exists(opt.savePath):
        os.makedirs(opt.savePath)

    if not os.path.exists(opt.saveEpochPath):
        os.mkdir(opt.saveEpochPath)

    # todo 放入数据
    dataset = MyDataSet()
    # 3）使用DistributedSampler
    rand_loader = DataLoader(dataset=dataset,
                             batch_size=batch,
                             shuffle=True,
                             sampler=DistributedSampler(dataset))

    model = Model()
    # 封装之前要把模型移到对应的gpu
    model.to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    pass


def cfg():
    parse = argparse.ArgumentParser()
    parse.add_argument('--batch', type=int, default=64)
    parse.add_argument('--epoch', type=int, default=3001)
    # parse.add_argument('--weight', type=str, default='./checkpoint/model/lqf.pth', help='load pre train weight')
    parse.add_argument('--weight', type=str, default='', help='load pre train weight')
    parse.add_argument('--savePath', type=str, default='./checkpoint/model', help='weight save path')
    parse.add_argument('--saveEpochPath', type=str, default='./checkpoint/model/epoch', help='weight epoch save path')
    parse.add_argument('--perEpoch', type=int, default=50, help='per epoch to save')
    parse.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parse.add_argument('--seed', type=int, default=11, help='seed for initializing training. ')
    parse.add_argument('--gpu', type=int, default=None, help='GPU id to use.')

    opt = parse.parse_args()
    return opt


if __name__ == '__main__':
    opt = cfg()
    print(opt)
    train(opt)
