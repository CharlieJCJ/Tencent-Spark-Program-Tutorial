# Matrix multiplication
import numpy as np

#**************************************************#
# 1. 向量的点乘 Vector dot product
a = np.array([1, 2, 3])
b = np.array([0, 1, 1])

dot_product = sum(a * b)  # 1 * 0 + 2 * 1 + 3 * 1 = 5
print(f"Dot Product of {repr(a)} * {repr(b)} = {dot_product}")

# Using np.dot() - numpy 自带函数
dot_product = np.dot(a, b)
print("Using np.dot(a, b) gets", dot_product)

# Exercise: 以下向量的点乘结果是多少呢？
c = np.array([6, 7, 8])
d = np.array([1, 2, 3])

# dot_product = ...(...)      # 填空
# print(dot_product)
 

#**************************************************#
# 2. 矩阵乘法 Matrix multiplication
a = np.array([[1, 2], [3, 4]])
b = np.array([[0, 1], [1, 0]])
print(f"\na = {repr(a)} \nb = {repr(b)} \n\t\tare 2x2 matrices\n")

mat_mul = np.array([[np.dot(a[0, :], b[:, 0]), np.dot(a[0, :], b[:, 1])], 
                    [np.dot(a[1, :], b[:, 0]), np.dot(a[1, :], b[:, 1])]])
                    # 这边一长串 code 解读: 
                    # [[a.row1 * b.col1, a.row1 * b.col2],
                    #  [a.row2 * b.col1, a.row2 * b.col2]]
                    
                    # General formula for 2x2 matrix multiplication:
                    # [[dot(a_R1, b_C1), dot(a_R1, b_C2)],
                    #  [dot(a_R2, b_C1), dot(a_R2, b_C2)]]
                    
                    # a dot b -> a 是左边的matrix, b 是右边的matrix
                    # R1, R2, C1, C2 are vectors 向量 (equivalent to 1D np.array())
                    
print(f'a * b = \n{repr(mat_mul)}\n')

# np.dot() calculates matrix multiplications
a = np.array([[1, 2], [3, 4]])
b = np.array([[0, 1], [1, 0]])
mat_mul = np.dot(a, b)
print(f'Using np.dot(a, b) = \n{repr(mat_mul)}')


# Exercise: 请试着计算以下两个矩阵的矩阵乘法结果
c = np.array([[5, 3], [2, 1]])
d = np.array([[1, 1], [0, 1]])


# Conclusion: 矩阵乘法对于理解你马上要搭建的第一个神经网络有密不可分的关系，
# 马上来看看吧！