import torch 
# CREATE RANDOM DATA POINTS
from sklearn.datasets import make_blobs
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import math


class PointsDataset(Dataset):
    def __init__(self):
        x, y = make_blobs(n_samples=1000, centers=2, n_features=2, cluster_std=1.5, shuffle=True)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.n_samples = y.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

dataset = PointsDataset()


# Testing
firstdata = dataset[0]
features, labels = firstdata
print(features, labels)
print(len(firstdata))

# Use DataLoader
train_data, test_data = random_split(dataset, [800, 200])
batch_size = 5
dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)

num_epoch = 2
total_sample = len(dataset)
num_iteration = math.ceil(total_sample/batch_size)
for epoch in range(num_epoch):
    for i, (inputs, labels) in enumerate(dataloader):
        if (i + 1) % 5 == 0:
            print(f"epoch {epoch + 1}/{num_epoch}, step {i + 1}/{num_iteration}, inputs {inputs.shape}")