"""
预测 sin 曲线的变化值
为了简化，只预测下一个相位的值

因为这是用多个数据来训练的，所以每次都要保存训练下来的参数(h0),并于下一次迭代时传递进去,否则模型的记忆性差
"""

import torch
import numpy as np
from torch import nn
from torch import optim
import matplotlib.pyplot as plt

num_time_steps = 50  # 一段时间序列有多少个点 ,也就是每层循环50次

# start = np.random.randint(3, size=1)[0]  # 相位：x值
# time_steps = np.linspace(start, start + 10, num_time_steps)
# data = np.sin(time_steps).reshape(num_time_steps, 1)
# x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
# y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

input_size = 1  # 每个点只用一个维度表示
hidden_size = 10  # 隐藏层维度
output_size = 1  # 输出大小为1
lr = 0.01


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True  # 所有参数都是batch在前，循环次数在后
        )
        self.linear = nn.Linear(hidden_size, output_size)  # 从h到y的全连接

    def forward(self, x, hx):
        out, hx = self.rnn(x, hx)
        out = out.view(-1, hidden_size)  # 这里要打平才能送到线性层
        out = self.linear(out)
        # out = out.unsqueeze(dim=0)  # 补上第一维和y(batch,seq,1)比较
        return out, hx


model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

hx = torch.zeros(1, 1, hidden_size)
# 6000次迭代，这里每次迭代都用的带初始值干扰（也就是不同起点的数据）
for i in range(1, 1001):
    start = np.random.randint(3, size=1)[0]  # 相位：x值
    time_steps = np.linspace(start, start + 10, num_time_steps)
    data = np.sin(time_steps).reshape(num_time_steps, 1)
    x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)  # 输入
    y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)  # 输出

    output, hx = model(x, hx)
    hx = hx.detach()

    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print("Iteration: {} loss {}".format(i, loss.item()))

# 测试数据
start = np.random.randint(3, size=1)[0]  # 相位：x值
time_steps = np.linspace(start, start + 10, num_time_steps)
data = np.sin(time_steps).reshape(num_time_steps, 1)
x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)

predictions = []
input = x[:, 0, :]
with torch.no_grad():
    for _ in range(x.shape[1]):
        input = input.view(1, 1, 1)
        (pred, hx) = model(input, hx)
        input = pred
        predictions.append(pred.numpy().ravel()[0])

    x = x.data.numpy().ravel()
    plt.scatter(time_steps[:-1], x, s=90)
    plt.plot(time_steps[:-1], x)

    plt.scatter(time_steps[1:], predictions)
    plt.show()
