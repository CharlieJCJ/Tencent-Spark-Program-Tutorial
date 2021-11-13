import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
class dataset(Dataset):
  def __init__(self,x,y):
    self.x = torch.tensor(x,dtype=torch.float32)
    self.y = torch.tensor(y,dtype=torch.float32)
    self.length = self.x.shape[0]
 
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]
  def __len__(self):
    return self.length


X, y = datasets.make_blobs(n_samples=5000,n_features=2,
                           centers=2,cluster_std=1.5,
                           random_state=2)

trainset = dataset(X,y)
trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
# X, y = torch.FloatTensor(X), torch.FloatTensor(y)
data = [[1,3], [2,6], [3,9], [4,12], [5,15], [6,18]]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 1)
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        return x

def criterion(out, label):
    return (label - out) ** 2

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.005)

for epoch in range(10):
    for i, (x_train, y_train) in enumerate(trainloader):
        # X, Y = iter(item)
        # print(X, Y, "before")
        # X, Y = Variable(torch.FloatTensor([X]), requires_grad=True), Variable(torch.FloatTensor([Y]), requires_grad=False)
        # print(X, Y)
        optimizer.zero_grad()
        outputs = net(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
    print("Epoch {} - loss: {}".format(epoch, loss.data[0]))

print(list(net.parameters()))
