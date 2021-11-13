# 损失函数 cost function
# 损失函数用来评价模型的预测值和真实值不一样的程度。
# 损失函数结果越小，通常模型的性能越好。
# 所以在机器学习问题中通常都会遇到 - 【寻找损失函数最小值问题】

import math
import numpy as np
import pandas as pd

y_actual = np.array([1,2,3,4,5])
y_predicted = np.array([1.6,2.5,2.9,3,4.1])

d = {'y_actual':y_actual, 
    'y_predicted':y_predicted, 
    'error': y_actual-y_predicted, 
    'squared error': (y_actual-y_predicted)**2}

df = pd.DataFrame(data = d)
print(repr(df))


MSE = np.square(np.subtract(y_actual,y_predicted)).mean() 
RMSE = math.sqrt(MSE)
print("Root Mean Square Error:\n")
print(RMSE)


#**************************************************#
# Using sklearn library
from sklearn.metrics import mean_squared_error
import math
y_actual = [1,2,3,4,5]
y_predicted = [1.6,2.5,2.9,3,4.1]
 
MSE = mean_squared_error(y_actual, y_predicted)
 
RMSE = math.sqrt(MSE)
print("Root Mean Square Error:\n")
print(RMSE)

