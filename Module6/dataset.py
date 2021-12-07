#  ****************************************************************  #
import torch 
# CREATE RANDOM DATA POINTS
from sklearn.datasets import make_blobs
from torch.utils.data import Dataset, DataLoader, random_split
from torchsummary import summary
import numpy as np
import math
import matplotlib.pyplot as plt

# 0. 超参数
LR = 0.1
num_epoch = 10
batch_size = 1   # 这个先不用讲解

# Helper plot function (将训练后结果做可视化)
def plot(data, prediction, mode = "Train"):
    x = data[0][:, 0]
    y = data[0][:, 1]
    plt.subplot(1, 2, 1)
    plt.scatter(x[data[1] <0.5], y[data[1] <0.5], label = "0", c = "red")
    plt.scatter(x[data[1] >=0.5], y[data[1] >=0.5], label = "1", c = "blue")
    plt.legend(loc='lower left')
    plt.title(mode)

    plt.subplot(1, 2, 2)
    plt.scatter(x[prediction < 0.5], y[prediction < 0.5], label = "0", c = "red")
    plt.scatter(x[prediction >= 0.5], y[prediction >= 0.5], label = "1", c = "blue")
    plt.legend(loc='lower left')
    plt.title("After training")
    plt.show()
    return
#  ****************************************************************  #
# 1. 训练，测试数据
# 使用 torch.utils.data 模块的 Dataset
class PointsDataset(Dataset):
    def __init__(self):
        x, y = make_blobs(n_samples=1000, centers=2, n_features=2, cluster_std=1.5, shuffle=True)
        self.x = torch.FloatTensor(x)    # 格式转换（numpy to tensor）
        self.y = torch.FloatTensor(y)
        self.n_samples = y.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

# 我们的数据集
dataset = PointsDataset()