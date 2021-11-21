import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from IPython.display import display

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


def Error(slope, data):
    # b is constant
    b = 1.1

    salary = []
    experience = data.experience

    # Loop to calculate predict salary variables
    for i in range(len(data.experience)):
        tmp = data.experience[i] * slope + b
        salary.append(tmp)

    MSE = mse(data.salary, salary)
    return MSE

# 枚举 slope 的值
slope = [i/100 for i in range(0, 300)]
Cost = []
for i in slope:
    cost = Error(slope = i, data = data)
    Cost.append(cost)

Cost_table = pd.DataFrame({
'slope' : slope,
'Cost' : Cost
})

print("Slope from 0.00 ~ 0.19 相对应的 MSE")
display(Cost_table.head(20))

# plotting the cost values corresponding to every value of slope

plt.plot(Cost_table.slope, Cost_table.Cost, color = 'blue', label = 'Cost Function Curve')
plt.xlabel('Value of Slope')
plt.ylabel('Cost')
plt.legend()
plt.show()