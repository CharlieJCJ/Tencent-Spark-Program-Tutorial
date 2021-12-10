# Module 7: 人工智能神经网络基础-4 深度神经网络，卷积神经网络引入

[TOC]

*教学前需要装的python library*

```python
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
```



## 1. 训练多层深度神经网络

*建议教学时长：*20分钟

### [`dataset_visuals`](/Module7/dataset_visual.py)
- 简单介绍MNIST手写数字数据集的背景（在做任何数据分析前需要对于数据集的背景做一定的了解，有助于我们对于模型最终的结果有效性做出合理解释）
- 理解电脑中是如何存储图片数据 （用`dataset_visuals.py`来讲解）

### [`mlp.py`](/Module7/mlp.py)

- 比较浅层Perceptron和多层深度神经网络在MNIST手写数据集上的正确率表现（在我的本地电脑上：Perception overall accuracy 92%， MLP overall accuracy 97 ~ 98%）
  - Perceptron基本上在2分钟内可以跑完，MLP大概在5～6分钟左右

```
Perceptron(
  (fc1): Linear(in_features=784, out_features=10, bias=True)
)
Epoch: 1        Training Loss: 0.674300
Epoch: 2        Training Loss: 0.408527
Epoch: 3        Training Loss: 0.367693
Epoch: 4        Training Loss: 0.346877
Epoch: 5        Training Loss: 0.333568
Epoch: 6        Training Loss: 0.324056
Epoch: 7        Training Loss: 0.316800
Epoch: 8        Training Loss: 0.311018
Epoch: 9        Training Loss: 0.306258
Epoch: 10       Training Loss: 0.302245
Epoch: 11       Training Loss: 0.298788
Epoch: 12       Training Loss: 0.295773
Epoch: 13       Training Loss: 0.293103
Epoch: 14       Training Loss: 0.290719
Epoch: 15       Training Loss: 0.288564
Epoch: 16       Training Loss: 0.286605
Epoch: 17       Training Loss: 0.284806
Epoch: 18       Training Loss: 0.283145
Epoch: 19       Training Loss: 0.281611
Epoch: 20       Training Loss: 0.280177
Test Loss: 0.277265

Test Accuracy of     0: 98% (963/980)
Test Accuracy of     1: 97% (1108/1135)
Test Accuracy of     2: 88% (913/1032)
Test Accuracy of     3: 90% (917/1010)
Test Accuracy of     4: 92% (911/982)
Test Accuracy of     5: 87% (777/892)
Test Accuracy of     6: 95% (911/958)
Test Accuracy of     7: 91% (942/1028)
Test Accuracy of     8: 88% (864/974)
Test Accuracy of     9: 90% (915/1009)

Test Accuracy (Overall): 92% (9221/10000)
```



```
MLP(
  (fc1): Linear(in_features=784, out_features=512, bias=True)
  (fc2): Linear(in_features=512, out_features=512, bias=True)
  (fc3): Linear(in_features=512, out_features=10, bias=True)
)
Epoch: 1        Training Loss: 0.786223
Epoch: 2        Training Loss: 0.299194
Epoch: 3        Training Loss: 0.237961
Epoch: 4        Training Loss: 0.196748
Epoch: 5        Training Loss: 0.165963
Epoch: 6        Training Loss: 0.142040
Epoch: 7        Training Loss: 0.123203
Epoch: 8        Training Loss: 0.107992
Epoch: 9        Training Loss: 0.095503
Epoch: 10       Training Loss: 0.085099
Epoch: 11       Training Loss: 0.076297
Epoch: 12       Training Loss: 0.068748
Epoch: 13       Training Loss: 0.062256
Epoch: 14       Training Loss: 0.056566
Epoch: 15       Training Loss: 0.051538
Epoch: 16       Training Loss: 0.047088
Epoch: 17       Training Loss: 0.043092
Epoch: 18       Training Loss: 0.039477
Epoch: 19       Training Loss: 0.036197
Epoch: 20       Training Loss: 0.033190
Test Loss: 0.068800

Test Accuracy of     0: 98% (969/980)
Test Accuracy of     1: 99% (1128/1135)
Test Accuracy of     2: 97% (1006/1032)
Test Accuracy of     3: 98% (990/1010)
Test Accuracy of     4: 98% (963/982)
Test Accuracy of     5: 98% (879/892)
Test Accuracy of     6: 96% (927/958)
Test Accuracy of     7: 96% (996/1028)
Test Accuracy of     8: 96% (943/974)
Test Accuracy of     9: 97% (984/1009)

Test Accuracy (Overall): 97% (9785/10000)
```





## 2. 卷积神经网络引入

- 理解为什么需要卷积网络

Demo: https://generic-github-user.github.io/Image-Convolution-Playground/src/

