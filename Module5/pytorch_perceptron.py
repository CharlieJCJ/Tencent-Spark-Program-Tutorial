#  ****************************************************************  #
import torch 
# CREATE RANDOM DATA POINTS
from sklearn.datasets import make_blobs
from torch.utils.data import Dataset, DataLoader, random_split
from torchsummary import summary
import numpy as np
import math
import matplotlib.pyplot as plt

# Helper plot function
def plot(data, prediction):
    x = data.x[:, 0]
    y = data.x[:, 1]
    plt.subplot(1, 2, 1)
    plt.scatter(x[data.y <0.5], y[data.y <0.5], label = "0", c = "red")
    plt.scatter(x[data.y >=0.5], y[data.y >=0.5], label = "1", c = "blue")
    plt.legend(loc='lower left')
    plt.title("True Dataset")

    plt.subplot(1, 2, 2)
    plt.scatter(x[prediction < 0.5], y[prediction < 0.5], label = "0", c = "red")
    plt.scatter(x[prediction >= 0.5], y[prediction >= 0.5], label = "1", c = "blue")
    plt.legend(loc='lower left')
    plt.title("After training")
    plt.show()
    return
#  ****************************************************************  #
# 训练，测试数据
# 使用 torch.utils.data 模块的 Dataset
class PointsDataset(Dataset):
    def __init__(self):
        x, y = make_blobs(n_samples=1000, centers=2, n_features=2, cluster_std=2, shuffle=True)
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
        self.n_samples = y.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

# 我们的数据集
dataset = PointsDataset()



firstdata = dataset[0]
features, labels = firstdata
print(features, labels)
print(len(firstdata))

# 训练集和测试集
train_data, test_data = random_split(dataset, [800, 200])
batch_size = 5
dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)
total_sample = len(dataset)
num_iteration = math.ceil(total_sample/batch_size)

#  ****************************************************************  #
# 定义神经网络
class Perceptron(torch.nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc = torch.nn.Linear(2,1)
        self.sigmoid = torch.nn.Sigmoid() # instead of Heaviside step fn
    def forward(self, x):
        output = self.fc(x)
        output = self.sigmoid(output) # instead of Heaviside step fn
        return output

class MLP(torch.nn.Module): # 下节课 Module 6 的内容
        def __init__(self, input_size, hidden_size):
            super(MLP, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.sigmoid(output)
            return output
#  ****************************************************************  #
# 设置训练模型，参数
# optimizer 就是优化器，包含了需要优化的参数有哪些，
# loss_func 就是我们设置的损失函数
# epoch 是指所有数据被训练的总轮数

model = Perceptron()
# model = MLP(2, 20) # 下节课 Module 6 的内容
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
num_epoch = 10

#  ****************************************************************  #
# 训练模型
model.train()
for epoch in range(num_epoch):
    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(inputs)
        # `criterion()` 计算【损失函数】结果， (output, target)作为输入(output为网络的输出,target为实际值)
        loss = criterion(y_pred.squeeze(), labels)
        # `loss.backward` 反向传播 - 利用损失函数反向传播计算梯度
        loss.backward()
        # `optimizer.step` 梯度下降，更新模型参数 - 用我们定义的优化器将每个需要优化的参数进行更新
        optimizer.step()
        # 在训练过程中print出来训练中的损失函数结果（观察
        if (i + 1) % 5 == 0:
            print(f"epoch {epoch + 1}/{num_epoch}, step {i + 1}/{num_iteration}, train loss {loss.item()}")


#  ****************************************************************  #
# 测试模型
model.eval()
y_pred = model(test_data.dataset.x)
after_train = criterion(y_pred.squeeze(), test_data.dataset.y)
print('Test loss after Training' , after_train.item())

#  ****************************************************************  #
# (optional) print 模型训练后的模型参数 - weights, bias
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)


#  ****************************************************************  #
# 模型在测试集上的表现可视化
plot(test_data.dataset, y_pred.squeeze())
