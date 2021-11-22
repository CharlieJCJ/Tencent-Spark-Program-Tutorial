# Perceptron calculation
# 你从零搭建的第一个神经网络

# 感知机（Perceptron）: 感知机是神经网络（深度学习）的起源算法，学习感知机的构造是通向神经网络和深度学习的一种重要思想。
# 感知机是 *二类分类* 的线性分类模型，由输入特征 x 得到输出类别 1 或 0 的函数
# 它只有一层输出层，一个神经元。从这个文件中我们将会学会如何将输入特征进行单个神经元的计算
# 学习完了计算，我们会浅谈损失函数，梯度下降在神经网络中的作用

import numpy as np
from matplotlib import pyplot as plt
import math

#*******************************************************************#
# 1. Perceptron 一个神经元的计算
# x = feature vector
x = np.array([1, 3]); x = x.reshape(-1, 1)       # 可以想像 x 这是一个点，有 (x1, x2) 坐标
weights = np.array([2, 4])                       # 假设这是训练好的权重 weights [w1, w2] 对应 feature vector 的两个 feature
bias = 1                                         # bias是一个常量，它和weights都是一个可训练的参数
                                                 # 神经元的计算公式: w1 * x1 + w2 * x2 + bias

# Compute output (without activation layer) -> `activation layer` 之后会讲到
result = np.dot(weights, x) + bias
print(f'{repr(weights)} * {repr(x)} + {bias} = {result}')

#*******************************************************************#
# 2. 激活层
# 什么是激活层？
# ** Activation functions: ** #
def sign(x):
    if x >= 0:
        return 1
    else:
        return 0

def relu(x):
    return max(0, x)

def sigmoid(num):
	return  1/(1+math.e**(-num))


# 可以更换activation function，把sin换成别的函数
activated_result = sign(result)
print(f'Result {result} after passing into activation layer: {activated_result}') # 将神经元计算的结果放入激活层

# 思考一下激活层对于输出结果有着什么样的影响？
input = range(-100, 100)
plt.plot(input, list(map(sign, input)))
plt.title('sign activation function')
plt.show()
plt.plot(input, list(map(relu, input)))
plt.title('relu activation function')
plt.show()
plt.plot(input, list(map(sigmoid, input)))
plt.title('sigmoid activation function')
plt.show()

#*******************************************************************#
# 3. 多个神经元计算
# 如果输出层有多个(> 1), 比如，两神经元呢？如何计算结果？(需要矩阵乘法的知识) 
# 这里一层有两个神经元 weights 有两组 [[2, 4],    <- w1, w2 (第一组)
#                                  [3, 5]]    <- w1, w2 (第二组)
#                   bias 有两组    [1, 2]
x = np.array([1, 3]); x = x.reshape(-1, 1)
weights = np.array([[2, 4], 
                    [3, 5]]) 
bias = np.array([1, 2]); bias = bias.reshape(-1, 1)

result = np.dot(weights, x) + bias


print(f'\nweights = {repr(weights)} has shape {weights.shape}\nx = {repr(x)} has shape {x.shape}\nbias = {repr(bias)} has shape {bias.shape}\n{result} has shape {result.shape}\n')
print(f'{repr(weights)} * {repr(x)} + {repr(bias)} = {result}')

activated_result = list(map(sign, result))
print(f'Result {result} after passing into activation layer: {activated_result}') # 将神经元计算的结果放入激活层
