# Module 5: 感知机 + Pytorch 介绍 （用代码实现第一个神经网络）


使用到的库 (课前需要提前安装好)：
```python
from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn import datasets
from scipy.optimize import *
import pandas as pd
import plotly.graph_objects as go
```

## 1. 感知机运算 (Perceptron calculation)
***推荐教学时长：15分钟***

[`perceptron_calc.py`](/Module5/perceptron_calc.py)

感知机（Perceptron）: 感知机是神经网络（深度学习）的起源算法，学习感知机的构造是通向神经网络和深度学习的一种重要思想。
感知机是 *二类分类* 的线性分类模型，由输入特征 x 得到输出类别 1 或 0 的函数。它只有一层输出层，一个神经元。从这个文件中我们将会学会如何将输入特征进行单个神经元的计算。
学习完了计算，我们会浅谈损失函数，梯度下降在神经网络中的作用
![perceptron](/Module5/img/perceptron.png)
1. Perceptron 一个神经元的计算
   1. activation function(weights * x + bias)
2. 激活层
   1. 思考一下激活层对于输出结果有着什么样的影响？

*Extra*: 多个神经元计算 （如果输出层有多个(> 1), 比如，两神经元呢？如何计算结果？(需要矩阵乘法的知识)）

这个网页里有一个*辅助动画*帮助理解：https://appliedgo.net/perceptron/
![perceptron_calc](/Module5/img/perceptron_calc.png)
## 2. 浅谈 损失函数 (cost function)
***推荐教学时长：15分钟***
### [`cost_func.py`](/Module4/cost_func.py)

* 损失函数 cost function
* 损失函数用来评价模型的预测值和真实值不一样的程度。
* 损失函数结果越小，通常模型的性能越好。
* 所以在机器学习问题中通常都会遇到 - 【寻找损失函数最小值问题】

简单介绍一种常用的损失函数
MSE - mean squared error

  1. 使用 numpy 函数从公式出发，计算数据集的MSE
  2. 使用sklearn library 的 `mean_squared_error` 函数

### [`cost_func_init.py`](/Module5/cost_func/cost_func_init.py)
- 用线性回归的问题（用一条直线去表达现有数据）来引入损失函数的概念，可以变化斜率，观察一下损失的值是如何变化的

![cost_func_init](/Module5/img/cost1.png)
### [`cost_func_visualize.py`](/Module5/cost_func/cost_func_visualize.py)
- 同学经过一些`cost_func_init.py`尝试，`cost_func_visualize.py`会把所有斜率的可能性枚举，并用`line plot`做可视化，程序只需要跑一遍即可

![cost_func_visualize](/Module5/img/cost2.png)
### [`3d_cost_function_visualization.py`](/Module5/cost_func/3d_cost_function_visualization.py)
- 不需要理解代码，只作为可视化功能
- 对于 >1 变量的cost function（这里有两个变量，所以cost function的graph是三维的）的可视化
![3d_cost_function_visualization](/Module5/img/cost3.png)



## 3. 浅谈 梯度下降 (gradient descent)
***推荐教学时长：10分钟***

[`gradient_descent_demo.py`](/Module5/gradient_descent/gradient_descent_demo.py)


1. `gradient_descent_demo.py`
   1. 不需要讲解此代码，直接让同学run，会有visualization的demo，演示gradient descent的过程
    ![gradient demo](/Module5/img/Snipaste_2021-11-12_22-57-24.png)
2. 建议配合使用网页版 interactive demo 演示梯度下降过程：https://uclaacm.github.io/gradient-descent-visualiser/#playground
![gradient descent](/Module5/img/gradient_desc.png)



### 神经网络基本流程小结：
1. 搭建神经网络结构
2. 前向传播（感知机运算）
3. 计算损失函数 (cost function)
4. 梯度下降（如果对这个内容感兴趣，同学可以搜索关键词**反向传播，偏导数计算**）

## 4. 用 python `Pytorch` 实现的感知机演示 demo
[`pytorch_perceptron.py`](pytorch_perceptron.py)

NOTE: 每次运行程序生成的数据都是随机的，所以每次分类结果都会不一样
![pytorch](/Module5/img/pytorch1.png)

***之后的内容***：训练集/测试集，模型准确率，多层神经网络，卷积神经网络


***EXTRA*** 下面这段因为时间原因，暂时不放进课程内
1. [`gradient_descent_student.py`](/Module5/gradient_descent/gradient_descent_student.py)
   1. 告诉学生们通常在机器学习，人工智能场景下，会出现非常复杂的损失函数，但是在计算他的最小值，用传统数学方法是很难求解的。所以我们可以利用numerical optimization方法中的一种 - 梯度下降的方法找到函数最小值（有些时候找到的是函数局部最小值）
   2. 这个python文件我定义了一个比较复杂的方程，可以让同学通过更改x值，来观察cost函数的值是如何变化的
      1. 可以尝试 x = 10 -> 12 -> 8 -> 3 -> 0 -> -4 -> -2 -> -1.8 （试不同参数，理解gradient descent的intuition，朝着负梯度的方向慢慢移动，移动到最小值）
      2. 这里不用引入复杂的数学概念，只需要大概了解gradient descent的思想就行
    1. 我在python文件里有两段代码，一个是标题写着`Numerical optimization. What's the absolute minimum of this function? `可以找到任何函数的最小值。另一个写着`What does the graph look like? Uncomment below and check it out!`可以将`cost`函数 plot 出来