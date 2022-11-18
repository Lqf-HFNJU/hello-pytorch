import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

"""
生成样本数据
"""
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

"""
自定义网络，继承nn.Module类
"""


class Net(nn.Module):
    # 构造函数
    def __init__(self):
        super(Net, self).__init__()

        self.stn = STN(1)

        #      二维卷积层     输入信号通道，输出信号通道，卷积核大小
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 第一个卷积层
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # 第二个卷积层
        self.conv2_drop = nn.Dropout2d()  # 丢失层
        self.fc1 = nn.Linear(320, 50)  # 线性全连接层 默认有偏置 bias = True
        self.fc2 = nn.Linear(50, 10)  # 线性全连接层

    def forward(self, x):
        x = self.stn(x)
        # 对第一层卷积层出来的参数做池化和reLU
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 池化核大小为2
        # 继续做卷积、池化，然后做一层舍弃
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)  # 张量拉伸为向量
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


"""
初始化网络和优化器
"""
network = Net()
# 随机梯度下降                                                  动量因子
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * batch_size_train) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), 'model.pth')
            torch.save(optimizer.state_dict(), 'optimizer.pth')


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    train(1)
    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()
