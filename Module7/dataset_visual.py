# import libraries
import torch
import numpy as np

from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Hyperparameters

# how many samples per batch to load
batch_size = 20

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


# **** obtain one batch of training images **** # 
dataiter = iter(train_loader)
images, labels = dataiter.next()
print(type(images))
images = images.numpy()
print(type(images))

# **** plot the images in the batch, along with the corresponding labels **** # 
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    # print out the correct label for each image
    # .item() gets the value contained in a Tensor
    ax.set_title(str(labels[idx].item()))
plt.show()


# **** View an Image in More Detail **** #
img = np.squeeze(images[1])

fig = plt.figure(figsize = (8,8)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y],2) if img[x][y] !=0 else 0
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<thresh else 'black',
                    fontsize = 7)
plt.show()