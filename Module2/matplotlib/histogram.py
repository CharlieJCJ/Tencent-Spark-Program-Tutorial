from matplotlib import pyplot as plt
import pandas as pd




data = pd.read_csv('data.csv')
ids = data['Responder_id']
ages = data['Age']
bins = range(10, 101, 10)   #  range(10, 101, 10) -> [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
plt.hist(ages, bins=bins, edgecolor='black')



#*************************************#
# plt 自定义属性
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Respondent age distribution')

plt.show()