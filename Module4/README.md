# Module 4 Introduction to opencv library and kernal matrix calculation

使用到的库：
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

使用网页版 interactive demo 演示矩阵乘法过程：http://matrixmultiplication.xyz/

## 2. 感知机运算 (Perceptron calculation)

[`perceptron_calc.py`](/Module4/perceptron_calc.py)

## 3. 损失函数 (cost function)

[`cost_func.py`](/Module4/cost_func.py)

## 4.浅谈 梯度下降 (gradient descent)

[`gradient_descent_student.py`](/Module4/gradient_descent_student.py)
[`gradient_descent_demo.py`](/Module4/gradient_descent_demo.py)

使用网页版 interactive demo 演示梯度下降过程：https://uclaacm.github.io/gradient-descent-visualiser/#playground
## 5. 神经网络训练 demo

1. 使用网页版 interactive demo 演示神经网络训练过程：https://playground.tensorflow.org/#activation=relu&batchSize=1&dataset=gauss&regDataset=reg-plane&learningRate=0.00001&regularizationRate=0&noise=35&networkShape=&seed=0.17057&showTestData=false&discretize=true&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false

2. 代码 demo，观察一下如何用python代码实现感知机的训练