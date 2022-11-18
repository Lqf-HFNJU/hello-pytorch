import torch
from torch import nn

"""
一步到位
RNN
"""

rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=4)
print(rnn)  # RNN(100, 20)

# x = (多少个单词（多少次输入），batch，每一个单词用多少维度)
x = torch.randn(10, 3, 100)

# out , h = RNN.forward(x,hx)
# 一次性传入全部数据，hx是0时刻的h值(layer,batch,hidden_size)
out, h = rnn(x, hx=torch.zeros(4, 3, 20))
# out 是最后一层所有时刻的h值 h是最后一个时刻的所有层的h值
# out(多少个单词（多少次输入）,batch,hidden_size)
print(out.shape, h.shape)  # torch.Size([10, 3, 20]) torch.Size([4, 3, 20])
print("=============================")
"""
手动输入多个
RNNCell
"""

# 多少层就要多少个cell
cell1 = nn.RNNCell(input_size=100, hidden_size=30)
# 第二层的输入维==第一层的输出维
cell2 = nn.RNNCell(input_size=30, hidden_size=20)
# h(batch,hidden_size)
h1 = torch.zeros(3, 30)
h2 = torch.zeros(3, 20)
for xt in x:
    h1 = cell1(input=xt, hx=h1)
    h2 = cell2(input=h1, hx=h2)
print(h1.shape)  # torch.Size([3, 30])
print(h2.shape)  # torch.Size([3, 20])
