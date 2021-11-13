# Perceptron calculation
# 你第一个从零搭建的神经网络

# 感知机（Perceptron）: 感知机是神经网络（深度学习）的起源算法，学习感知机的构造是通向神经网络和深度学习的一种重要思想。
# 感知机是 *二类分类* 的线性分类模型，由输入特征 x 得到输出类别 1 或 0 的函数
# 它只有一层输出层，一个神经元。从这个文件中我们将会学会如何将输入特征进行单个神经元的计算
# 学习完了计算，我们会浅谈损失函数，梯度下降在神经网络中的作用

import numpy as np
from matplotlib import pyplot as plt

# feature vector
x = np.array([1, 3])       # 可以想像这是一个点，有 (x, y) 坐标
weights = np.array([2, 4]) # 假设这是训练好的权重 weights [w1, w2] 对应 feature vector 的两个 feature
bias = 1                   # bias是一个常量，它和weights都是一个可训练的参数

# Compute output (without activation layer) -> `activation layer` 之后会讲到
result = np.dot(x, weights) + bias
print(f'{repr(x)} * {repr(weights)} + {bias} = {result}')




# 什么是激活层 
def sign(x):
    if x >= 0:
        return 1
    else:
        return 0

def relu(x):
    return max(0, x)

input = range(-100, 100)
plt.plot(input, list(map(sign, input)))
plt.show()
plt.plot(input, list(map(relu, input)))
plt.show()

# 拓展：如果输出层有多个(> 1)神经元呢？如何计算结果？(需要矩阵乘法的知识)
x = np.array([1, 3]) 
weights = np.array([[2, 4], 
                    [3, 5]]) 
bias = np.array([1, 2])

result = np.dot(x, weights) + bias
print(f'{repr(x)} * {repr(weights)} + {bias} = {result}')