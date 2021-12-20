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

**预告下节课内容**：图像增广`image augmentation`，transfer learning

