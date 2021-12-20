# Module 8: 卷积神经网络核心，训练卷积神经网络
*教学前需要装的python library*

## 0. 课前Demo
学习知识前先放一个最终手势识别的网页版Demo，向同学展示这门课的最终成果是如何的：https://www.cs.ryerson.ca/~aharley/vis/conv/
![CNN handwritten digit demo]()

## 1. 卷积神经网络核心
*建议教学时长：25分钟*

### a. 继续Module7对于卷积的讨论，定义一些卷积层的超参数`hyperparameters`：
1. Padding
2. Stride
3. Kernel size (常用 3x3, 5x5等)

#### 建议使用下方可视化讲解：https://ezyang.github.io/convolution-visualizer/

### b. 介绍一下简单的卷积网络结构 (从LeNet-5开始举例，最基础的卷积神经网络结构)
![LeNet-5]()
Q: 观察一下LeNet-5网络的特征，规律?


### EXTRA: 在RGB图片上的卷积层

一篇关于卷积层中`卷积核大小`，`padding`，`stride`对于output feature image 形状的关系和影响: https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/

## 2. 训练卷积神经网络
> Bringing it all together

*建议教学时长：25分钟*

### [train_CNN_network.py](train_CNN_network.py)
```python
Epoch: 1        Training Loss: 2.299405
Epoch: 2        Training Loss: 1.538855
Epoch: 3        Training Loss: 0.403314
Epoch: 4        Training Loss: 0.329937
Epoch: 5        Training Loss: 0.306231
...
Epoch: 99       Training Loss: 0.000238
Epoch: 100      Training Loss: 0.000238
Test Loss: 0.059046

Test Accuracy of     0: 99% (977/980)
Test Accuracy of     1: 99% (1127/1135)
Test Accuracy of     2: 99% (1024/1032)
Test Accuracy of     3: 99% (1006/1010)
Test Accuracy of     4: 99% (974/982)
Test Accuracy of     5: 98% (883/892)
Test Accuracy of     6: 98% (946/958)
Test Accuracy of     7: 98% (1015/1028)
Test Accuracy of     8: 98% (962/974)
Test Accuracy of     9: 97% (987/1009)

Test Accuracy (Overall): 99% (9901/10000)
```
**预告下节课内容**：图像增广`image augmentation`，transfer learning

