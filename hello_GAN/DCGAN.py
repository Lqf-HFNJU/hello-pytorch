import torch
from torch import nn, optim
from torch.autograd import Variable
import os
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import transforms

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

os.makedirs("images", exist_ok=True)


n_epochs = 200
batch_size = 64
lr = 2e-4
# 动量梯度下降的两个参数
b1 = 0.5
b2 = 0.999
latent_dim = 100  # 噪声数据生成的维度
img_size = 32  # 输入照片的维度
channels = 1  # 信道个数
sample_interval = 400  # 保存图像的迭代数

torch.manual_seed(11)
np.random.seed(11)


# 自定义初始化参数
def weights_init_normal(m):
    classname = m.__class__.__name__  # 获得类名
    if classname.find("Conv") != -1:  # 卷积层
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:  # BN 层
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 刚开始的维度是 8*8
        self.init_size = img_size // 4
        # 线性变换
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size ** 2)
        )

        # 卷积层
        self.conv_blocks = nn.Sequential(
            # BatchNorm2d的目的是使我们的一批（batch）feature map 规范化, 满足均值0方差1
            nn.BatchNorm2d(128),
            # 上采样, 将图片放大两倍 8*8*128 => 16*16*128
            nn.Upsample(scale_factor=2),
            #  16*16*128 => 16*16*128
            nn.Conv2d(128, 128, 3, stride=1, padding=1),  # 输入数据channel，输出的channel，卷积核大小，步长，padding的大小
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # 16*16*128 => 32*32*128
            nn.Upsample(scale_factor=2),
            #  32*32*128 => 32*32*64
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()  # 激活函数
        )

    def forward(self, z):
        # batch_size*100*1 => batch_size*(8*8*128)*1
        out = self.l1(z)
        # batch_size*(8*8*128)*1 => batch_size*128*8*8
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        # batch_size*128*8*8 => batch_size*1*128*128
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            # 卷积 非线性激活 失活
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(p=0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))

            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)  # 打平 送入线性层
        validity = self.adv_layer(out)
        return validity


adversarial_loss = torch.nn.BCELoss()

G = Generator()
D = Discriminator()

if cuda:
    G.cuda()
    D.cuda()
    adversarial_loss.cuda()

G.apply(weights_init_normal)
D.apply(weights_init_normal)

optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(b1, b2))

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/",  # 进行训练集下载
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)

for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        batch = imgs.shape[0]
        # 生成对抗标签
        valid = Variable(Tensor(batch, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(batch, 1).fill_(0.0), requires_grad=False)

        real_imgs = Variable(imgs.type(Tensor))

        # 训练 G
        optimizer_G.zero_grad()

        z = Variable(Tensor(np.random.normal(0, 1, (batch, latent_dim))))

        gen_imgs = G(z)

        g_loss = adversarial_loss(D(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # 训练 D
        optimizer_D.zero_grad()

        real_loss = adversarial_loss(D(real_imgs), valid)
        fake_loss = adversarial_loss(D(gen_imgs.detach()), fake)  # 记得训练 D 的时候要把 G detach 掉

        d_loss = real_loss + fake_loss

        d_loss.backward()
        optimizer_D.step()

        batches_done = epoch * len(dataloader) + i
        if batches_done % (sample_interval / 4) == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

        if batches_done % sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
