import torch
from torch.autograd import Variable
from os import listdir


def num2vector(filename):
    vector = torch.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            vector[0, i * 32 + j] = int(line[j])
    return vector


# 封装数据集为Variable
def markData(dataPath):
    filelist = listdir(dataPath)
    m = len(filelist)  # 多少个数据
    x = torch.zeros(m, 1024)
    y = torch.zeros(m, 10)
    for i in range(m):
        file = filelist[i]
        fileName = file.split('.')[0]
        y[i, int(fileName.split('_')[0])] = 1  # 把这一维设为1
        x[i, :] = num2vector(dataPath + '/' + file)

    y = Variable(y, requires_grad=False)
    x = Variable(x, requires_grad=False)
    print("已初始化 {} 的数据".format(dataPath))
    return x, y


def load_data():
    training_x = torch.load('training_x.pt')
    training_y = torch.load('training_y.pt')
    test_x = torch.load('test_x.pt')
    test_y = torch.load('test_y.pt')

    return training_x, training_y, test_x, test_y


if __name__ == '__main__':
    path = 'resources/digits/'
    training_x, training_y = markData(path + 'trainingDigits')
    test_x, test_y = markData(path + 'testDigits')

    torch.save(training_x, "training_x.pt")
    torch.save(training_y, "training_y.pt")

    torch.save(test_x, "test_x.pt")
    torch.save(test_y, "test_y.pt")
