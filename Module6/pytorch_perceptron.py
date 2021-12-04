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
LR = 0.01
num_epoch = 10
batch_size = 5   # 这个先不用讲解

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

'''
first_row = dataset[0] # first row
features, labels = firstdata
print(features, labels)
print(len(firstdata))
'''

#  ****************************************************************  #
# 2. 将整个数据集分成训练集和测试集
train_data, test_data = random_split(dataset, [800, 200])
dataloader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
total_sample = len(train_data)
num_iteration = math.ceil(total_sample/batch_size)

#  ****************************************************************  #
# 3. 定义神经网络结构
class Perceptron(torch.nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc = torch.nn.Linear(2,1)
        self.sigmoid = torch.nn.Sigmoid() # instead of Heaviside step fn
    def forward(self, x):
        output = self.fc(x)
        output = self.sigmoid(output) # instead of Heaviside step fn
        return output

class MLP(torch.nn.Module):
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
# 4. 设置训练模型，参数
# optimizer 就是优化器，包含了需要优化的参数有哪些，
# loss_func 就是我们设置的损失函数
# epoch 是指所有数据被训练的总轮数

model = Perceptron()
# model = MLP(2, 20) # 下节课 Module 6 的内容
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = LR)

#  ****************************************************************  #
# 5. 训练模型
model.train()
for epoch in range(num_epoch):
    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()

        # 使用当前模型 <训练的参数> 去预测数据相对应的标签 (label)，即 `前向传播`
        y_pred = model.forward(inputs)

        # `criterion()` 计算【损失函数】结果， (output, target) 作为输入 (output为网络的输出,target为实际值)
        loss = criterion(y_pred.squeeze(), labels)

        # `loss.backward` 反向传播 - 利用损失函数反向传播计算梯度
        loss.backward()

        # `optimizer.step` 梯度下降，更新模型参数 - 用我们定义的优化器将每个需要优化的参数进行更新
        optimizer.step()

        # 在训练过程中print出来训练中的损失函数结果（观察损失函数的变化）
        if (i + 1) % 5 == 0:
            print(f"epoch {epoch + 1}/{num_epoch}, step {i + 1}/{num_iteration}, train loss {loss.item()}")


#  ****************************************************************  #
# 6. 测试模型
train_set, test_set = dataset[train_data.indices], dataset[test_data.indices]

model.eval()
y_pred = model(test_set[0])
after_train = criterion(y_pred.squeeze(), test_set[1])
print('Test loss after Training', after_train.item())

#  ****************************************************************  #
# (optional) print 模型训练后的模型参数 - weights, bias
print("模型训练后的模型参数:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

#  ****************************************************************  #
# 7. 模型在 <训练集> 和 <测试集> 上的表现可视化
plot(train_set, model(train_set[0]).squeeze(), "Train set")
plot(test_set, y_pred.squeeze(), "Test set")

