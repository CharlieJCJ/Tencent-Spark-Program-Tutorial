import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import datasets

X, y = datasets.make_blobs(n_samples=5,n_features=2,
                           centers=2,cluster_std=1.5,
                           random_state=2)
# X, y = torch.FloatTensor(X), torch.FloatTensor(y)
data = [[1,3], [2,6], [3,9], [4,12], [5,15], [6,18]]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1,1)
    def forward(self, x):
        x = self.fc1(x)
        return x

def criterion(out, label):
    return (label - out)**2

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

for epoch in range(100):
    for i, item in enumerate(data):
        X, Y = iter(item)
        # print(X, Y, "before")
        X, Y = Variable(torch.FloatTensor([X]), requires_grad=True), Variable(torch.FloatTensor([Y]), requires_grad=False)
        # print(X, Y)
        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        if (i % 10 == 0):
            print("Epoch {} - loss: {}".format(epoch, loss.data[0]))

print(list(net.parameters()))
