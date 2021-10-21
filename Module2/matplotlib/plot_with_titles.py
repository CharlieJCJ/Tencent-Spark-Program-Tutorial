from matplotlib import pyplot as plt

# 数据（以列表形式定义）
x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

y = [38496, 42000, 46752, 49320, 53200,
         56000, 62316, 64928, 67317, 68748, 73752]

plt.plot(x, y)

#*************************************#
# plt 基础自定义属性
# `xlabel` - x坐标轴标注
# `ylabel` - y坐标轴标注
# `title` - 图像标题

plt.xlabel('Ages')
plt.ylabel('Median Salary (USD) in 2019')
plt.title('Median Salary (USD) by Age')
plt.show()