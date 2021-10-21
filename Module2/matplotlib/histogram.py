from matplotlib import pyplot as plt
import pandas as pd

# pandas 可以从各种文件格式比如 CSV、JSON、SQL、Microsoft Excel 导入数据。
# 可以对各种数据进行运算操作，比如归并、再成形、选择，还有数据清洗和数据加工特征。

# `pd.read_csv`读取 CSV 文件数据
data = pd.read_csv('data.csv')


# data[...] 选择单列，这里选择'Age'列
ages = data['Age']
bins = range(10, 101, 10)   #  range(10, 101, 10) -> [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
plt.hist(ages, bins=bins, edgecolor='black')


#*************************************#
# plt 自定义属性
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Respondent age distribution')

plt.show()