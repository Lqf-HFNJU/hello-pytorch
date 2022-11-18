"""
手写数字识别
pytorch版
"""

import torch
import torch.nn.functional as F
from torch import optim
import prodata

"""
预测
"""


def predict(test_x, test_y, model):
    correct = 0
    total = test_y.shape[0]

    with torch.no_grad():
        test_y_pred = model(test_x)
        # print(test_y_pred)
        test_loss = F.mse_loss(test_y_pred, test_y)
        print("test_loss: {}".format(test_loss))

        correct = torch.where(torch.argmax(test_y, dim=1) == torch.argmax(test_y_pred, dim=1),
                              True, False).sum()

    print("准确率为{}/{}，即{:.4f}%".format(correct, total, correct * 100 / total))


# 学习率和迭代次数,隐藏节点数
epoch_n = 2000
lr = 1e-4
hidden = 100

if __name__ == '__main__':

    training_x, training_y, test_x, test_y = prodata.load_data()

    model = torch.nn.Sequential(
        torch.nn.Linear(1024, hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(),
        torch.nn.Linear(hidden, 10),
        torch.nn.ReLU()
    )

    # 也可以使用优化器来更新参数 自适应动量的随机优化方法
    #                           被优化的参数  学习率的初始值
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoch_n):
        y_pred = model(training_x)

        # loss_fn = torch.nn.MSELoss()
        # loss = loss_fn(y_pred, training_y)
        # 两种方法等价
        loss = F.mse_loss(y_pred, training_y)
        if epoch % 100 == 0:
            print("epoch:{},loss:{:.4f}".format(epoch, loss.data))

        # model.zero_grad()  # 清零梯度
        # 使用优化器方式
        optimizer.zero_grad()
        loss.backward()  # 反向传播

        # for param in model.parameters():
        #    param.data -= lr * param.grad
        # 使用优化器方式
        optimizer.step()

    # 预测测试集数据
    predict(test_x, test_y, model)
