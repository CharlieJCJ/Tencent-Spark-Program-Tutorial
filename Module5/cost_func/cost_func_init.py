import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error as mse

# 这是一个线性回归的问题，用一条直线去表达现有数据
# 可以变化slope的值，观察一下cost function的值是如何变化的
# making lines for different Values of Beta 0.1, 0.8, 1.5
slope = 0.1

# keeping intercept constant
b = 1.1

# 数据集
experience = [1.2,1.5,1.9,2.2,2.4,2.5,2.8,3.1,3.3,3.7,4.2,4.4]
salary = [1.7,2.4,2.3,3.1,3.7,4.2,4.4,6.1,5.4,5.7,6.4,6.2]

data = pd.DataFrame({
"salary" : salary,
"experience" : experience
})

plt.subplot(1, 2, 1)
plt.scatter(data.experience, data.salary, color = 'red', label = 'data points')
plt.xlim(1,4.5)
plt.ylim(1,7)
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.legend()


# to store predicted points
line1 = []

# generating predictions for every data point
for i in range(len(data)):
    line1.append(data.experience[i] * slope + b)

plt.subplot(1, 2, 2)

# Plotting the line
plt.scatter(data.experience, data.salary, color = 'red')
plt.plot(data.experience, line1, color = 'black', label = 'line')
plt.xlim(1,4.5)
plt.ylim(1,7)
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.legend()
MSE = mse(data.salary,line1)
plt.title("Slope "+ str(slope)+" with MSE "+ str(MSE))
MSE = mse(data.salary, line1)
plt.show()