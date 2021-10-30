# NumPy(Numerical Python) 是 Python 语言的一个扩展程序库，支持
# 大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
# `Numpy array` - 新的数据结构 ndarray (n-dimensional array)，N 维数组

import numpy as np

#*************************************#
# 1. 创建数组 Initialize numpy arrays
# a: 1D array
a = np.array([1,2,3])
print(a)
# b: a 2D array
b = np.array([[9.0,8.0,7.0],[6.0,5.0,4.0]])
print(b)

# Get Dimension 维度
print(a.ndim)
print(b.ndim)

# Get Shape 表示数组形状（shape）的元组，表示各维度大小的元组（tuple）
print(a.shape)
print(b.shape)

# Get Data Type 数组数据类型
print(a.dtype)
print(b.dtype)

# 从列表转换成numpy数组
x = [1, 2, 3, 4]
x = np.asarray(x)     # 现在 x 是一个numpy数组了
#*************************************#
# 2. NumPy 切片和索引 - Accessing numpy arrays through indexing

a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
print(a)
# Get a specific element [r, c]
a[1, 5]

# Get a specific row 
a[0, :]

# Get a specific column
a[:, 2]


# Changing specific elements, rows, columns
a[1,5] = 20
print(a)

#*************************************#
#  3. Initializing Different Types of Arrays

# `np.zeros` 创建指定大小的数组，数组元素以 0 来填充
x = np.zeros((2,3))
print(x)
# `np.ones` 创建指定形状的数组，数组元素以 1 来填充
y = np.ones((4,2,2))
print(y)
# `np.full` 创建指定形状的数组，数组元素以 参数值 来填充
z = np.full((2,2), 99)
print(z)


# 随机数数组
# Random decimal numbers
np.random.rand(4,2)

# Random Integer values
np.random.randint(-4,8, size=(3,3))

# The identity matrix
np.identity(5)

# Repeat an array
arr = np.array([[1,2,3]])
r1 = np.repeat(arr, 3, axis=0)       # axis = 0 
print(r1)