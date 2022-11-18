import torch
import torch.nn as nn
import torch.nn.functional as F
import prodata


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #      两层卷积层
        self.conv = nn.Sequential(
            #        输入信号通道，输出信号通道，卷积核大小，步长，边界补充
            nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 两层全连接层
        self.dense = nn.Sequential(
            nn.Linear(8 * 8 * 12, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 8 * 8 * 12)
        x = self.dense(x)
        return x


lr = 1e-4
epoch_n = 5000


def format(x):
    x = x.data.reshape(x.shape[0], 32, 32)
    return x.unsqueeze(1)


def predict(test_x, test_y, model):
    correct = 0
    total = test_y.shape[0]

    with torch.no_grad():
        test_y_pred = model(test_x)
        # print(test_y_pred)
        test_loss = F.cross_entropy(test_y_pred, test_y)
        print("test_loss: {}".format(test_loss))

        correct = torch.where(torch.argmax(test_y, dim=1) == torch.argmax(test_y_pred, dim=1),
                              True, False).sum()

    print("准确率为{}/{}，即{:.4f}%".format(correct, total, correct * 100 / total))


if __name__ == '__main__':
    training_x, training_y, test_x, test_y = prodata.load_data()
    training_x = format(training_x)
    test_x = format(test_x)
    print(training_x.shape)
    print(test_x.shape)

    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epoch_n + 1):
        y_pred = model.forward(training_x)
        loss = F.cross_entropy(y_pred, training_y)
        if epoch % 100 == 0:
            print("epoch:{},loss:{:.4f}".format(epoch, loss.data))
            predict(test_x, test_y, model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
