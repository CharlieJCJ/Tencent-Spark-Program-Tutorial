from matplotlib import pyplot as plt

# 数据（以列表形式定义）
x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

y1 = [38496, 42000, 46752, 49320, 53200,
         56000, 62316, 64928, 67317, 68748, 73752]
y2 = [45372, 48876, 53850, 57287, 63016,
            65998, 70003, 70000, 71496, 75370, 83640]


# 你可以用子图来将图样（plot）放在均匀的坐标网格中。用 `plt.subplot` 函数的时候，
# 你需要指明网格的行列数量，以及你希望将图样放在哪一个网格区域中。
# py.subplot(行序号，列序号，图序号)

# 创建一个两行一列的子图集，在第一行绘制图表
plt.subplot(2,1,1)
plt.plot(x, y1, label = "All developer", color="blue", linewidth=2.5, linestyle="-", marker = ".")
plt.ylabel('Median Salary (USD) 2019')
plt.title('Median Salary (USD) by Age')
plt.xlim(24,36)
plt.ylim(20000,90000)
plt.legend(loc='upper left')

# 创建一个两行一列的子图集，在第二行绘制图表
plt.subplot(2,1,2)
plt.plot(x, y2, label = "Python developer", color="red", linewidth=2.5, linestyle="--", marker = "p")
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD) 2019')
plt.xlim(24,36)
plt.ylim(20000,90000)
plt.legend(loc='upper left')


plt.show()