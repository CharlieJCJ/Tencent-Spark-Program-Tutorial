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
推荐教学时长：15分钟

[`perceptron_calc.py`](/Module5/perceptron_calc.py)

感知机（Perceptron）: 感知机是神经网络（深度学习）的起源算法，学习感知机的构造是通向神经网络和深度学习的一种重要思想。
感知机是 *二类分类* 的线性分类模型，由输入特征 x 得到输出类别 1 或 0 的函数。它只有一层输出层，一个神经元。从这个文件中我们将会学会如何将输入特征进行单个神经元的计算。
学习完了计算，我们会浅谈损失函数，梯度下降在神经网络中的作用
![perceptron](/Module5/img/perceptron.png)
1. Perceptron 一个神经元的计算
   1. activation function(weights * x + bias)
2. 激活层
   1. 思考一下激活层对于输出结果有着什么样的影响？
3. 多个神经元计算 （如果输出层有多个(> 1), 比如，两神经元呢？如何计算结果？(需要矩阵乘法的知识)）

## 2. 浅谈 损失函数 (cost function)
推荐教学时长：10分钟

[`cost_func_init.py`](/Module5/cost_func/cost_func_init.py)

![cost_func_init](/Module5/img/cost1.png)
[`cost_func_visualize.py`](/Module5/cost_func/cost_func_visualize.py)

![cost_func_visualize](/Module5/img/cost2.png)

[`3d_cost_function_visualization.py`](/Module5/cost_func/3d_cost_function_visualization.py)

![3d_cost_function_visualization](/Module5/img/cost3.png)

## 3. 使用 `pytorch` 库搭建你的第一个神经网络
推荐教学时长：15分钟
1. 训练，测试数据
   1. 使用 torch.utils.data 模块的 Dataset
2. 将整个数据集分成训练集和测试集
3. 定义神经网络结构
4. 设置训练模型，参数
   1. optimizer 就是优化器，包含了需要优化的参数有哪些，
   2. loss_func 就是我们设置的损失函数
   3. och 是指所有数据被训练的总轮数
5. 训练模型
   1. # 使用当前模型 <训练的参数> 去预测数据相对应的标签 (label)，即 `前向传播`
   2. `criterion()` 计算【损失函数】结果， (output, target) 作为输入 (output为网络的输出,target为实际值)
   3. `loss.backward` 反向传播 - 利用损失函数反向传播计算梯度
   4. `optimizer.step` 梯度下降，更新模型参数 - 用我们定义的优化器将每个需要优化的参数进行更新
   5. 在训练过程中print出来训练中的损失函数结果（观察损失函数的变化）
6. 测试模型
7. 模型在 <训练集> 和 <测试集> 上的表现可视化
[`pytorch_perceptron.py`](pytorch_perceptron.py)

![pytorch](/Module5/img/pytorch1.png)
## EXTRA 浅谈 梯度下降 (gradient descent) - 【学生自行阅读】

[`gradient_descent_student.py`](/Module5/gradient_descent/gradient_descent_student.py)
[`gradient_descent_demo.py`](/Module5/gradient_descent/gradient_descent_demo.py)

1. `gradient_descent_student.py`
   1. 告诉学生们通常在机器学习，人工智能场景下，会出现非常复杂的损失函数，但是在计算他的最小值，用传统数学方法是很难求解的。所以我们可以利用numerical optimization方法中的一种 - 梯度下降的方法找到函数最小值（有些时候找到的是函数局部最小值）
   2. 这个python文件我定义了一个比较复杂的方程，可以让同学通过更改x值，来观察cost函数的值是如何变化的
      1. 可以尝试 x = 10 -> 12 -> 8 -> 3 -> 0 -> -4 -> -2 -> -1.8 （试不同参数，理解gradient descent的intuition，朝着负梯度的方向慢慢移动，移动到最小值）
      2. 这里不用引入复杂的数学概念，只需要大概了解gradient descent的思想就行
    3. 我在python文件里有两段代码，一个是标题写着`Numerical optimization. What's the absolute minimum of this function? `可以找到任何函数的最小值。另一个写着`What does the graph look like? Uncomment below and check it out!`可以将`cost`函数 plot 出来
2. `gradient_descent_demo.py`
   1. 不需要讲解此代码，直接让同学run，会有visualization的demo，演示gradient descent的过程
    ![gradient demo](/Module5/img/Snipaste_2021-11-12_22-57-24.png)
建议配合使用网页版 interactive demo 演示梯度下降过程：https://uclaacm.github.io/gradient-descent-visualiser/#playground
![gradient descent](/Module5/img/gradient_desc.png)

