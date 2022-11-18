from torch import nn

"""定义卷积层和BN层的初始化参数"""


def weight_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


"""残差块"""


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.in_features = in_features
        self.block = nn.Sequential(
            nn.ReflectionPad2d(padding=1),  # 利用输入的边界反射来填充输入张量
            nn.Conv2d(self.in_features, self.in_features, 3),
            nn.InstanceNorm2d(in_features),  # 在图像像素上做归一化，用于图像风格迁移
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.in_features, self.in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)  # 输出维图像+残差输出


"""生成器 G """


class G(nn.Module):
    # input_shape = (3,256,256) num_residual_blocks = 9
    def __init__(self, input_shape, num_residual_blocks):
        super(G, self).__init__()
        self.num_residual_blocks = num_residual_blocks
        self.channels = input_shape[0]  # 信道
        out_features = 64  # 输出信道数 64
        model = [
            nn.ReflectionPad2d(self.channels),  # [3,256,256] -> [3,262,262]
            nn.Conv2d(self.channels, out_features, 7),  # [3,262,262] -> [64,256,256]
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        in_features = out_features

        # 下采样两次
        for _ in range(2):
            out_features *= 2
            model += [  # [64,256,256] -> [128,128,128] -> [256,64,64]
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # 9个残差块
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # 上采样两次
        for _ in range(2):
            out_features //= 2
            model += [  # [256,64,64] -> [128,128,128]->[64,256,256]
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # 网络输出层
        model += [  # [64,256,256] -> [3,256,256]
            nn.ReflectionPad2d(self.channels),
            nn.Conv2d(out_features, self.channels, 7),
            nn.Tanh()  # 映射回 [-1, 1]之间
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


"""判别器 D """


class D(nn.Module):
    def __init__(self, input_shape):
        super(D, self).__init__()

        channels, height, width = input_shape

        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):  # [3,256,256] -> [1,30,30]
        return self.model(img)
