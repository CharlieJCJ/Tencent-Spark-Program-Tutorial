# Matplotlib 是一个绘图工具，类似excel或者matlab里绘制图表的功能
# 可能是 Python 2D-绘图领域使用最广泛的套件。它能让使用者很轻松地将数据图形化，并且提供多样化的输出格式
from matplotlib import pyplot as plt

# 数据（以列表形式定义）
x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

y = [38496, 42000, 46752, 49320, 53200,
         56000, 62316, 64928, 67317, 68748, 73752]

# plt.plot() 线型图 plot y versus x as lines and/or markers
plt.plot(x, y)

# 在屏幕上显示
plt.show()


