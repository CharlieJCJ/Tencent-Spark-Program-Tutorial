# NumPy(Numerical Python) 是 Python 语言的一个扩展程序库，支持
# 大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
# `NumPy array` - 新的数据结构 ndarray (n-dimensional array)，N 维数组

# NumPy 数组，矩阵计算

import numpy as np

a = np.array([1,2,3,4])
b = np.array([1,0,1,0])

#*************************************#
# 1. Element-wise computation (这里 a 和 b 必须是一样长度的，或符合数组广播规则（见2）) 加减乘除运算，这里用np函数或者 python 自带 + - * / 运算是一样的
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


# 更多关于 NumPy broadcast 请阅读 EXTRA_numpy_broadcasting_rules.md 文件
# 请阅读官方文件： https://numpy.org/doc/stable/user/basics.broadcasting.html

x = a ** 2
print(x)

#*************************************#
# 3. Mathematics

a = np.array([1,2,3,4])
print(a)

x = a + 2
print(x)

x = a * 2
print(x)

# ** operator to calculate powers
x = a ** 2
print(x)

# Take the cos
print(np.cos(a))

# `numpy.around(a,decimals)` 返回指定数字的四舍五入值
# decimals: 舍入的小数位数。 默认值为0。 如果为负，整数将四舍五入到小数点左侧的位置
a = np.array([1.0,5.55,  123,  0.567,  25.532])  
print  ('原数组：')
print (a)
print ('\n')
print ('舍入后：')
print (np.around(a))
print (np.around(a, decimals =  1))
print (np.around(a, decimals =  -1))

#*************************************#
# 4. 统计

# 平均值
# `numpy.mean()` 函数返回数组中元素的算术平均值。 如果提供了轴，则沿其计算。
# 1D array
a = np.array([1,2,3,4])
print (np.mean(a))

# nD array (n-dimensional)
# 如果不指定axis，默认对所有数字求平均
# 指定axis，axis=0 是 `列`；axis=1 是 `行`
a = np.array([[1,2,3],[3,4,5],[4,5,6]])  
print ('我们的数组是：')
print (a)
print ('\n')
print ('调用 mean() 函数：')
print (np.mean(a))
print ('\n')
print ('沿轴 0 调用 mean() 函数：')
print (np.mean(a, axis =  0))
print ('\n')
print ('沿轴 1 调用 mean() 函数：')
print (np.mean(a, axis =  1))

# 中位数
a = np.array([1,2,3,4])
print (np.median(a))

# 标准差
print (np.std(a))

#*************************************#
# 5. 线性代数

# a. 转置矩阵
a = np.arange(12).reshape(3,4)
 
print ('原数组：')
print (a)
print ('\n')
 
print ('转置数组：')
print (a.T)

# b. `numpy.dot()`对于两个一维的数组，计算的是这两个数组
# 对应下标元素的乘积和(数学上称之为内积)

# 计算式为：[[1*11+2*13, 1*12+2*14],[3*11+4*13, 3*12+4*14]]
a = np.array([[1,2],[3,4]])
b = np.array([[11,12],[13,14]])
print(np.dot(a,b))


# c. numpy.inner()
# numpy.inner() 函数返回一维数组的向量内积
print (np.inner(np.array([1,2,3]),np.array([0,1,0])))
# 等价于 1*0+2*1+3*0