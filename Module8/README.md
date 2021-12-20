# Module 8: 卷积神经网络核心，训练卷积神经网络
*教学前需要装的python library*
- [Module 8: 卷积神经网络核心，训练卷积神经网络](#module-8-卷积神经网络核心训练卷积神经网络)
  - [0. 课前Demo](#0-课前demo)
  - [1. 卷积神经网络核心](#1-卷积神经网络核心)
    - [a. 继续Module7对于卷积的讨论，定义一些卷积层的超参数`hyperparameters`：](#a-继续module7对于卷积的讨论定义一些卷积层的超参数hyperparameters)
      - [建议使用下方可视化讲解：https://ezyang.github.io/convolution-visualizer/](#建议使用下方可视化讲解httpsezyanggithubioconvolution-visualizer)
      - [浅谈为什么需要Padding, Stride, Kernel size 这些超参数](#浅谈为什么需要padding-stride-kernel-size-这些超参数)
    - [b. 介绍一下简单的卷积网络结构 (从LeNet-5开始举例，最基础的卷积神经网络结构)](#b-介绍一下简单的卷积网络结构-从lenet-5开始举例最基础的卷积神经网络结构)
    - [***EXTRA***: output_size formula:](#extra-output_size-formula)
      - [浅谈卷积网络每一层](#浅谈卷积网络每一层)
  - [2. 训练卷积神经网络](#2-训练卷积神经网络)
    - [`LeNet_model.py`](#lenet_modelpy)
    - [`train_CNN_network.py`](#train_cnn_networkpy)
    - [`LeNet_log.txt`](#lenet_logtxt)
  - [**预告下节课内容**：](#预告下节课内容)

## 0. 课前Demo
学习知识前先放一个最终手写数字识别的网页版Demo，向同学展示这门课的最终成果是如何的：https://www.cs.ryerson.ca/~aharley/vis/conv/
![CNN handwritten digit demo](/Module8/img/CNN%20demo.png)

## 1. 卷积神经网络核心
*建议教学时长：25分钟*

### a. 继续Module7对于卷积的讨论，定义一些卷积层的超参数`hyperparameters`：
1. Padding
2. Stride
3. Kernel size (常用 3x3, 5x5等)

#### 建议使用下方可视化讲解：https://ezyang.github.io/convolution-visualizer/
![kernel demo](/Module8/img/CNN%20interactive.png)

#### 浅谈为什么需要Padding, Stride, Kernel size 这些超参数

### b. 介绍一下简单的卷积网络结构 (从LeNet-5开始举例，最基础的卷积神经网络结构)
![LeNet-5](/Module8/img/Lenet-5%20architecture.jpeg)
Q: 观察一下LeNet-5网络的特征，规律?

### ***EXTRA***: output_size formula:
![output_size_formula](/Module8/img/n_out%20formula.png)
- 比如拿最简单的例子，一个6x6的图片(n_in = 6)，卷积核大小3x3 (kernel size = 3)，padding = 0, stride = 1
  - (6 + 2 * 0 - 3)/1 + 1 = 4 --> n_out (output size)

#### 浅谈卷积网络每一层
Optional: 一篇关于卷积层中`卷积核大小`，`padding`，`stride`对于output feature image 形状的关系和影响: https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/

## 2. 训练卷积神经网络
> Bringing it all together

*建议教学时长：25分钟*

### [`LeNet_model.py`](LeNet_model_structure.py)
### [`train_CNN_network.py`](train_CNN_network.py)
***NOTE***: 这是跑完100个epoch的结果，上课的时候只需要跑20个epoch即可（正确率能到98%），20个epoch大概要跑10分钟左右，可以先让同学把代码跑起来，然后开始讲网络结构如何用pytorch写。讲完了基本上网络也训练完了，正好同学可以看一些训练结果。
![CNN result](/Module8/img/CNN%20result.png)
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


完整model training（用了100 epoch）的训练过程log记录在下面的文件中：
### [`LeNet_log.txt`](/Module8/LeNet_log.txt)


## **预告下节课内容**：
图像增广 `image augmentation`，卷积网络在其他数据集上的应用，迁移学习

