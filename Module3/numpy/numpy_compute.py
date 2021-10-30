# NumPy(Numerical Python) 是 Python 语言的一个扩展程序库，支持
# 大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
# `NumPy array` - 新的数据结构 ndarray (n-dimensional array)，N 维数组

# NumPy 数组，矩阵计算

import numpy as np

a = np.array([1,2,3,4])
b = np.array([1,0,1,0])

#*************************************#
# 1. Element-wise addition (这里 a 和 b 必须是一样长度的) 加减乘除运算，这里用np函数或者 python 自带 + - * / 运算是一样的
c = a + b
c = np.add(a, b)
print(c)

c = a - b
c = np.subtract(a, b)
print(c)

c = np.multiply(a, b)
c = a * b
print(c)

c = a / b
c = np.divide(a, b)
print(c)
#*************************************#
# 2. (重要！): NumPy 广播 Broadcast 
a = np.array([1,2,3,4]) 
b = np.array([10,20,30,40]) 
c = a * b 
print (c)


a = np.array([[ 0, 0, 0],
           [10,10,10],
           [20,20,20],
           [30,30,30]])
b = np.array([1,2,3])
print(a + b)


# 更多关于 NumPy broadcast 请阅读官方文件： https://numpy.org/doc/stable/user/basics.broadcasting.html

x = a ** 2
print(x)

