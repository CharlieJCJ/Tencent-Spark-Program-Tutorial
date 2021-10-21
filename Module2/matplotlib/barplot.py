from matplotlib import pyplot as plt


# 柱形图
# plt.bar(类型, 频率) 
langs = ['C', 'C++', 'Java', 'Python', 'PHP']
students = [23,17,35,29,12]
plt.bar(langs, students)



#*************************************#
# plt 自定义属性
plt.xlabel('Programming language')
plt.ylabel('Students')
plt.title('Students using different programming languages')


plt.show()