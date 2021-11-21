import torch 
# CREATE RANDOM DATA POINTS
from sklearn.datasets import make_blobs
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import math

# x_train = torch.FloatTensor(x_train)
# y_train = torch.FloatTensor(blob_label(y_train, 0, [0]))
# y_train = torch.FloatTensor(blob_label(y_train, 1, [1,2,3]))
# x_test, y_test = make_blobs(n_samples=100, n_features=2, cluster_std=1.5, shuffle=True)
# x_test = torch.FloatTensor(x_test)
# y_test = torch.FloatTensor(blob_label(y_test, 0, [0]))
# y_test = torch.FloatTensor(blob_label(y_test, 1, [1,2,3]))

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

# class Perceptron(torch.nn.Module):
#     def __init__(self):
#         super(Perceptron, self).__init__()
#         self.fc = torch.nn.Linear(1,1)
#         self.relu = torch.nn.ReLU() # instead of Heaviside step fn
#     def forward(self, x):
#         output = self.fc(x)
#         output = self.relu(x) # instead of Heaviside step fn
#         return output
# class Feedforward(torch.nn.Module):
#         def __init__(self, input_size, hidden_size):
#             super(Feedforward, self).__init__()
#             self.input_size = input_size
#             self.hidden_size  = hidden_size
#             self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
#             self.relu = torch.nn.ReLU()
#             self.fc2 = torch.nn.Linear(self.hidden_size, 1)
#             self.sigmoid = torch.nn.Sigmoid()
#         def forward(self, x):
#             hidden = self.fc1(x)
#             relu = self.relu(hidden)
#             output = self.fc2(relu)
#             output = self.sigmoid(output)
#             return output

# # def blob_label(y, label, loc): # assign labels
# #     target = np.copy(y)
# #     for l in loc:
# #         target[y == l] = label
# #     return target


# model = Feedforward(2, 10)
# criterion = torch.nn.BCELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# model.train()
# epoch = 200000
# for epoch in range(epoch):
#     optimizer.zero_grad()
#     # Forward pass
#     y_pred = model(x_train)
#     # Compute Loss
#     loss = criterion(y_pred.squeeze(), y_train)
   
#     print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
#     # Backward pass
#     loss.backward()
#     optimizer.step()

# model.eval()
# y_pred = model(x_test)
# after_train = criterion(y_pred.squeeze(), y_test) 
# print('Test loss after Training' , after_train.item())