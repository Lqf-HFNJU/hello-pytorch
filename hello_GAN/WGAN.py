"""
WGAN
解决了刚开始训练的时候难以训练的问题
希望梯度越接近于 1
做了一个线性插值

WGAN 模拟 GMM 分布
交替训练
for epoch:
    for D: ...
    for G: ...

"""
import random

import numpy as np
import torch
from torch import nn, optim, autograd
from matplotlib import pyplot as plt

# 隐藏层数量
h_dim = 400
batch_size = 512
cuda = torch.cuda.is_available()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            # z = [b, 2] => [b, 2]
            nn.Linear(2, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, 2),
        )

    def forward(self, z):
        output = self.net(z)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()  # 输出来自真实分布的概率
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1)


def data_generator():
    """
    生成数据集 8-GMM
    返回一个无限数据生成器
    """
    scale = 2
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), - 1. / np.sqrt(2))
    ]

    centers = [(scale * x, scale * y) for x, y in centers]

    while True:
        dataset = []

        for i in range(batch_size):
            point = np.random.randn(2) * 0.02  # 分布为N(0, 1)
            center = random.choice(centers)

            point[0] += center[0]
            point[1] += center[1]

            dataset.append(point)

        dataset = np.array(dataset).astype(np.float32)
        dataset /= 1.414
        yield dataset  # 无限循环数据生成器


def generate_img(D, G, xr, epoch):
    N_POINTS = 128
    RANGE = 3
    plt.clf()

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))

    with torch.no_grad():
        points = torch.Tensor(points)
        if cuda:
            points = points.cuda()
        disc_map = D(points).cpu().numpy()
    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    cs = plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    plt.clabel(cs, inline=1, fontsize=10)

    with torch.no_grad():
        z = torch.randn(batch_size, 2)
        if cuda:
            z = z.cuda()
        samples = G(z).cpu().numpy()
    plt.scatter(xr[:, 0], xr[:, 1], c='orange', marker='.')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')
    plt.title(("epoch = " + str(epoch)))
    plt.show()


def gradient_penalty(D, xr, xf):
    """
    约束梯度在 0 ~ 1 之间
    :param D:
    :param xr: [b, 2]
    :param xf: [b, 2]
    :return: 结构风险项
    """
    # 随机产生一个均值分布
    t = torch.rand(batch_size, 1)
    if cuda:
        t = t.cuda()
    # [b, 1] => [b, 2] 保证 t 的两个维度的值一样
    t = t.expand_as(xr)

    # 随机插值 x_bar = tx + (1-t)x with t in (0, 1)
    mid = t * xr + (1 - t) * xf
    # 需要求导
    mid.requires_grad_()

    pred = D(mid)
    grads = autograd.grad(outputs=pred, inputs=mid, grad_outputs=torch.ones_like(pred),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()
    return gp


def main():
    torch.manual_seed(11)
    np.random.seed(11)

    data_iter = data_generator()
    xr = next(data_iter)

    G = Generator()
    D = Discriminator()

    optim_G = optim.Adam(G.parameters(), lr=5e-4, betas=(0.5, 0.9))
    optim_D = optim.Adam(D.parameters(), lr=5e-4, betas=(0.5, 0.9))

    if cuda:
        print(cuda)
        G = G.cuda()
        D = D.cuda()
    # print(G)
    # print(D)

    # 先训练 D 再训练 G 交替训练
    for epoch in range(50000):
        # 1. 训练 D
        for _ in range(5):
            # 1.1 train on real data

            xr = torch.from_numpy(next(data_iter))
            if cuda:
                xr = xr.cuda()
            predr = D(xr)
            lossr = -predr.mean()  # maximize  lossr

            # 1.2 train on fake data
            z = torch.randn(batch_size, 2)
            if cuda:
                z = z.cuda()
            xf = G(z).detach()  # 梯度不能传到 G 里
            predf = D(xf)
            lossf = predf.mean()

            """wgan重要步骤 计算 gradient penalty"""
            # xf 要 detach 从 G 来的
            gp = gradient_penalty(D, xr, xf.detach())

            # 1.3 aggravate all
            """加上结构风险 控制梯度在 0~1 之间"""
            loss_D = lossf + lossr + 0.2 * gp

            # 1.4 optimize
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        # 2. 训练 G
        # G 不能训练的太快
        z = torch.randn(batch_size, 2)
        if cuda:
            z = z.cuda()

        xf = G(z)
        predf = D(xf)
        loss_G = -predf.mean()  # maximize pred

        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch % 100 == 0:
            print("epoch: {} , loss_D = {} , loss_G = {} ".format(epoch, loss_D.item(), loss_G.item()))

            generate_img(D, G, xr.cpu(), epoch)


if __name__ == '__main__':
    main()
