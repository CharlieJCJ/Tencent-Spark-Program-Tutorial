# 梯度下降 intuition student version

from math import * 
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import *

# 假设以下函数是我们想去 minimize 的损失函数 cost function (损失函数越小越好)，这个函数可以是任何的函数
def cost(x):
    return 1/30 * (x + 3) * (x - 2)**2 * (x - 5)


my_guess = 10  # 我觉得 cost function 在 x = ... 的时候是最少的  (可以尝试10 -> 12 -> 8 -> 3 -> 0 -> -4 -> -2 -> -1.8)
print(cost(my_guess))


# Numerical optimization. What's the absolute minimum of this function? 
# Uncomment below and check it out!
'''
result = minimize(cost, 0)
solution = result['x']
evaluation = cost(solution)
print(f"Solution: f({solution}) = {evaluation}")
'''

# What does the graph look like? Uncomment below and check it out!
'''
input = np.arange(-5, 5, 0.01)
plt.plot(input, list(map(cost, input)))
plt.axis([-5,5,-5,5])
plt.show()
'''