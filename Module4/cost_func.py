# 损失函数 cost_func 
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

