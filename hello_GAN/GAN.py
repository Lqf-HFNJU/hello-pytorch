"""
WGAN
解决了刚开始训练的时候难以训练的问题

Generator 生成器
Discriminator 判别器

WGAN 模拟 GMM 分布
交替训练
for epoch:
    for D: ...
    for G: ...

"""
import random

import numpy as np
import visdom
import torch
from torch import nn, optim, autograd
from matplotlib import pyplot as plt

# 隐藏层数量
h_dim = 400
batch_size = 512
cuda = torch.cuda.is_available()
# viz = visdom.Visdom()


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
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, 1]
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

    plt.show()
    # viz.matplot(plt, win='contour', opts=dict(title='p(x):%d' % epoch))


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

    # viz.line([[0, 0]], [0], win='loss', opts=dict(title='loss', legend=['D', 'G']))

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

            # 1.3 aggravate all
            loss_D = lossf + lossr

            # 1.4 optimizer
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        # 2. 训练 G
        for _ in range(5):
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
            # viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')
            print("epoch: {} , loss_D = {} , loss_G = {} ".format(epoch, loss_D.item(), loss_G.item()))

            generate_img(D, G, xr, epoch)


if __name__ == '__main__':
    main()
