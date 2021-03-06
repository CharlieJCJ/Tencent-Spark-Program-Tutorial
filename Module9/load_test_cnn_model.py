# import libraries
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from PIL import *
import os

# how many samples per batch to load
batch_size = 20

# number of epochs to train the model
n_epochs = 20  # suggest training between 20-50 epochs

directory = './transformed_hand_written_digits'

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # `nn.Conv2d`
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        # stride 默认是 1， padding 默认是 0
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        # torch.nn.functional.MaxPool2d(input, kernel_size, stride=None)
        # `kernel_size` – the size of the window to take a max over
        # `stride` – the stride of the window. Default value is `kernel_size`
        y = F.max_pool2d(y, 2)
        y = F.relu(self.conv2(y))
        y = F.max_pool2d(y, 2)
        y = y.view(y.shape[0], -1)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        return y

model = LeNet()
print(model)
model.load_state_dict(torch.load('./model/cnn.pth'))

test_transforms = transforms.Compose([transforms.Resize(28),
                                      transforms.ToTensor(),
                                     ])
imsize = 28
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.autograd.Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image 

for image in sorted(os.listdir(directory)):
    if image == '.DS_Store':
        continue
    img = image_loader(os.path.join(directory, image))
    print(image, np.argmax(model(img).detach().numpy()))