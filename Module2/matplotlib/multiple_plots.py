from matplotlib import pyplot as plt

# 数据（以列表形式定义）
x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

y1 = [38496, 42000, 46752, 49320, 53200,
         56000, 62316, 64928, 67317, 68748, 73752]
y2 = [45372, 48876, 53850, 57287, 63016,
            65998, 70003, 70000, 71496, 75370, 83640]

# plot 函数里有许多参数可以自定义图表
# 使用蓝色的、连续的、宽度为 2.5 （像素）的线条
# `color` - 线条颜色
# `linewidth` - 线条宽度
# `linestyle` - 线条风格 可以尝试 {'-', '--', '-.', ':'} 等
# `marker` - 样本点标记样式 可以尝试 {'.', ',', 'o', 'p', 'x'} 等

plt.plot(x, y1, label = "All developer", color="blue", linewidth=2.5, linestyle="-", marker = ".")
plt.plot(x, y2, label = "Python developer", color="red", linewidth=2.5, linestyle="--", marker = "p")

#*************************************#
# plt 自定义属性
# `xlabel` - x坐标轴标注
# `ylabel` - y坐标轴标注
# `title` - 图像标题
# `xlim` - 设置横轴的上下限
# `ylim` - 设置纵轴的上下限
# `legend` - 图例位置

plt.xlabel('Ages')
plt.ylabel('Median Salary (USD) 2019')
plt.title('Median Salary (USD) by Age')
plt.xlim(24,36)
plt.ylim(20000,90000)
plt.legend(loc='upper left')
plt.show()


# 这些自定义参数同学们可以自行调整，看看绘制的图表的变化