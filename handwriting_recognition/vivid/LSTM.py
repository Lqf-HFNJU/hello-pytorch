"""
LSTM 做手写数据分类 (没想到吧)
就是说把 [100,1,28,28] 维的数据转换成 [100,28,28] 维(第一个 28 是指有 28 个序列, 第二个 28 只每个序列的维度是 28)
这里要注意的是要拿 hx[-1] 来进行全连接层操作 ,也就是最后一个时刻的最后一层的 h 值
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

n_epochs = 3
batch_size = 100
lr = 0.01
momentum = 0.5
input_size = 28
hidden_size = 128
num_classes = 10
num_layers = 2
random_seed = 1
torch.manual_seed(random_seed)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=torchvision.datasets.MNIST(root='./data/', train=True,
                                                                              transform=transforms.ToTensor(),
                                                                              download=True),
                                           batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=torchvision.datasets.MNIST(root='./data/', train=False,
                                                                             transform=transforms.ToTensor()),
                                          batch_size=batch_size, shuffle=False)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out, (hx, c) = self.lstm(x)
        output = hx[-1].view(-1, hidden_size)
        output = self.fc(output)
        return output


model = RNN()
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


def train(epoch):
    log_interval = 10
    train_losses = []
    train_counter = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.reshape(-1, input_size, input_size)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * batch_size) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(model.state_dict(), 'model.pth')
            torch.save(optimizer.state_dict(), 'optimizer.pth')


def test():
    model.eval()
    test_loss = 0
    correct = 0
    test_losses = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.reshape(-1, input_size, input_size)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()
