"""
pixelRNN 生成新的宝可梦
所有宝可梦都已经被简化为 20*20 = 400 的列向量 每一个向量表示一种颜色 (pixel_color.txt)
颜色和 RGB 值对应表是 colormap.txt 第 n 行（从0开始） 表示这个颜色的 RGB 值

训练的时候遮住一半让他预测下一个，不断迭代至全部预测完
太弔慢了
学得不好，原因有很多：
    batch_size 太大了
    训练集没有打散
    学习率可能太大
    这不能弄成回归任务！！
    生成式模型本身就很难学
"""
import torch
from torch import nn, optim
import math
from torchvision.utils import save_image
import os


# torch.manual_seed(7)


def load_data(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(list(map(int, line.split())))
    return torch.tensor(data)


def load_colormap(path):
    dic = {}
    with open(path, 'r') as f:
        i = 0
        for line in f:
            dic[i] = tuple(map(int, line.split()))
            i += 1
    return dic


# 输入的是一个 n*1600 的列向量
def tensor2img(data, path, dic):
    l = int(math.sqrt(data.shape[1]))
    data_rgb = torch.tensor(list(map(lambda x: dic[int(x)], data.view(-1).tolist())))  # (n*1600,3)
    data_rgb = data_rgb.view(data.shape[0], l, l, 3).permute(0, 3, 1, 2).float()
    for i in range(data_rgb.shape[0]):
        save_image(data_rgb[i], path + str(i) + ".jpg", normalize=True)


class pixel_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(pixel_RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, s, h = x.shape
        out, _ = self.lstm(x)
        out = out.reshape(b * s, self.hidden_size)
        out = self.linear(out)
        out = self.relu(out)
        out = out.view(b, s, -1)  # 补上第一维和y(batch,seq,dim)比较
        return out


# x (seq,dim)
# predictions (seq,dim)
def test(model, x):
    predictions = torch.zeros(x.shape[0], x.shape[1]).to(device)
    mid = x.shape[0] // 2
    input = x[:mid].float()
    predictions[:mid] = input
    input = input.unsqueeze(0)
    with torch.no_grad():
        for i in range(mid):
            pred = model(input)

            predictions[mid + i] = pred[0][-1].view(-1, 1)
            input = input.view(-1).tolist()
            input.append(pred[0][-1][0].to('cpu').tolist())
            input = torch.tensor(input).to(device).view(1, -1, 1)
    return predictions


input_size = 1
hidden_size = 512
output_size = 1
num_layers = 2
lr = 0.01
n_epoch = 600
cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    pokemon_path = './pokemon_image/pixel_color.txt'
    data = load_data(pokemon_path).to(device)
    data = data.unsqueeze(dim=2)  # (batch,seq,1)
    start_epoch = 0
    model = pixel_RNN(input_size, hidden_size, output_size, num_layers).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if os.path.exists('./weight/pixelRNN.pth'):
        ckpt = torch.load('./weight/pixelRNN.pth', map_location=device)
        model.load_state_dict(ckpt['pixelRNN'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1

    # 每次都是遮一半 让他生成另一半好了
    mid = data.shape[1] // 2

    if not os.path.exists('./weight'):
        os.mkdir('./weight')

    if not os.path.exists('./weight/epoch'):
        os.mkdir('./weight/epoch')

    for epoch in range(start_epoch, n_epoch):
        for i in range(mid):
            x = data[:, i:mid + i].view(data.shape[0], -1, 1).float()
            y = data[:, 1 + i:mid + i + 1].view(data.shape[0], -1, 1).float()
            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i == mid - 1:
                with open('log', mode='a') as log:
                    print("epoch: {} loss {}".format(str(epoch), loss.item()))
                    log.write("epoch: {} loss {}\n".format(str(epoch), loss.item()))
        torch.save({
            'pixelRNN': model.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict()
        }, './weight/pixelRNN.pth')

        if epoch % 50 == 0:
            torch.save({
                'pixelRNN': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, './weight/epoch/' + str(epoch) + '.pth')

    res_data = torch.zeros(data.shape[0], 400).to(device)
    for i in range(data.shape[0]):
        output = test(model, data[i])
        res_data[i] = output.view(-1)
    colormap_path = './pokemon_image/colormap.txt'
    dic = load_colormap(colormap_path)
    tensor2img(res_data.int(), "./pokemon_image/out/", dic)
