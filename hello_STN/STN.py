import torch
from torch import nn
import torch.nn.functional as F


class STN(nn.Module):
    def __init__(self, channels):
        super(STN, self).__init__()

        self.channel = channels
        # 定位网络 卷积层
        self.localization_convs = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        # 生成参数矩阵 2*3
        self.localization_linear = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 2 * 3)
        )

    def forward(self, x):
        x2 = self.localization_convs(x)
        x2 = x2.view(x2.size()[0], -1)  # 拉平
        x2 = self.localization_linear(x2)
        theta = x2.view(x2.size()[0], 2, 3)

        # 网格生成器，根据 θ 建立原图片的坐标仿射矩阵
        grid = F.affine_grid(theta, x.size(), align_corners=True)

        # 采样器，根据网格对原图片进行转换
        x = F.grid_sample(x, grid, align_corners=True)
        return x


if __name__ == '__main__':
    net = STN(3)
    x = torch.randn(64, 3, 28, 28)
    res = net.forward(x)
