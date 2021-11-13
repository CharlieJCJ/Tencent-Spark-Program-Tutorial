# Module 4 Introduction to opencv library and kernal matrix calculation

使用到的库 (课前需要提前安装好)：
```python
from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn import datasets
from scipy.optimize import *
import pandas as pd
```
## 1. 矩阵乘法 (Matrix multiplication)

[`matrix_multiplication.py`](/Module4/matrix_multiplication.py)
1. 向量的点乘 Vector dot product
   1. 从公式出发不用自带函数
   2. 用`np.dot()`函数
   3. Exercise: 以下向量的点乘结果是多少呢？
2. 矩阵乘法 Matrix multiplication
   1. 从公式出发不用自带函数
   2. 用`np.dot()`函数
   3. Exercise: 请试着计算以下两个矩阵的矩阵乘法结果
   4. 只有哪些矩阵可以做矩阵乘法呢？
**Conclusion: 矩阵乘法对于理解你马上要搭建的第一个神经网络有密不可分的关系**

建议配合使用网页版 interactive demo 演示矩阵乘法过程：http://matrixmultiplication.xyz/
![matmul1](/Module4/img/Matmul1.png)

## 2. 感知机运算 (Perceptron calculation)

[`perceptron_calc.py`](/Module4/perceptron_calc.py)

感知机（Perceptron）: 感知机是神经网络（深度学习）的起源算法，学习感知机的构造是通向神经网络和深度学习的一种重要思想。
感知机是 *二类分类* 的线性分类模型，由输入特征 x 得到输出类别 1 或 0 的函数。它只有一层输出层，一个神经元。从这个文件中我们将会学会如何将输入特征进行单个神经元的计算。
学习完了计算，我们会浅谈损失函数，梯度下降在神经网络中的作用
![perceptron](/Module4/img/perceptron.png)
1. Perceptron 一个神经元的计算
   1. activation function(weights * x + bias)
2. 激活层
   1. 思考一下激活层对于输出结果有着什么样的影响？
3. 多个神经元计算 （如果输出层有多个(> 1), 比如，两神经元呢？如何计算结果？(需要矩阵乘法的知识)）
   
## 3. 损失函数 (cost function)

[`cost_func.py`](/Module4/cost_func.py)

损失函数 cost function
损失函数用来评价模型的预测值和真实值不一样的程度。
损失函数结果越小，通常模型的性能越好。
所以在机器学习问题中通常都会遇到 - 【寻找损失函数最小值问题】

简单介绍一种常用的损失函数
RMSE - root mean squared error

1. 使用 numpy 函数从公式出发，计算数据集的RMSE
2. 使用sklearn library 的 `mean_squared_error` 函数

## 5. 神经网络训练 demo

人工智能会帮我们最小化损失函数，让预测值变得越来越准确
1. 使用网页版 interactive demo 演示神经网络训练过程：https://playground.tensorflow.org/#activation=relu&batchSize=1&dataset=gauss&regDataset=reg-plane&learningRate=0.00001&regularizationRate=0&noise=35&networkShape=&seed=0.17057&showTestData=false&discretize=true&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false
![Tensorflow playground](/Module4/img/Tensorflow.png)
2. 代码 demo，观察一下如何用python代码实现感知机的训练
[`perceptron_1layer.py`](perceptron_1layer.py)
   1. 不需要讲解此代码，直接让同学run，会有visualization的demo，演示用代码实现的训练神经网络过程

预告：下节课我们会【浅谈 梯度下降】了解人工智能如何最小化损失函数，以及 介绍【多层神经网络】