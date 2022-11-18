"""
输出预测
"""
import argparse
import os
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(opt):
    if not os.path.exists(opt.savePath):
        os.makedirs(opt.savePath)

    pass


def cfg():
    parse = argparse.ArgumentParser()
    parse.add_argument('--batch', type=int, default=1000)
    parse.add_argument('--weight', type=str, default='./checkpoint/model/lqf.pth', help='load pre train weight')
    parse.add_argument('--savePath', type=str, default='', help='test_out save path')

    opt = parse.parse_args()
    return opt


if __name__ == '__main__':
    opt = cfg()
    print(opt)
    test(opt)
