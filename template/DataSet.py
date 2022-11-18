"""自定义数据集"""
from torch.utils.data.dataset import Dataset


class MyDataSet(Dataset):
    def __init__(self, data1, label):
        self.data1 = data1
        self.label = label
        self.length = label.shape[0]

    def __getitem__(self, mask):
        return self.data1[mask], self.label[mask]

    def __len__(self):
        return self.length
