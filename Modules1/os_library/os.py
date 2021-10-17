import os

# os 是python自带的非常实用的标准库
# 顾名思义，就是与操作系统相关的库。如：文件，目录，执行系统命令等。

#*************************************#
# 常用函数
# 1. `os.getcwd` 查看当前目录 Check the current working directory of the file
print(os.getcwd())

# 2. `os.listdir` 列举当前目录里所有文件名
print(os.listdir())

#*************************************#
# 3. `os.makedirs` 新建文件夹
os.makedirs('new_directory')

# 4. `os.rmdir` 删除文件夹
os.rmdir('new_directory')

#*************************************#
# os的好处：批量化处理文件数据
# 举例：Create many folders at the same time

"""
for i in range(1, 11):
    os.makedirs(f'My folder {i}')
"""

# 同样，你也可以一次性删除很多文件夹 Remove many folders at the same time
"""
for i in range(1, 11):
    os.rmdir(f'My folder {i}')
"""

# 如果想尝试，可以用`os.rename`重命名文件

# Rename filename
os.rename('one.txt', 'two.txt')


#*************************************#
# 5. `os.path.join(path, *paths)` 用于路径拼接文件路径

# Path
path = "/home"
# Join various path components
print(os.path.join(path, "file.txt"))
#/home/file.txt

print(os.path.join(path, "Documents", "myfolder", "file.txt")) # 可以加任何数量的参数
# /home/Documents/myfolder/file.txt


# 其他os.path 函数
# os.path.basename()
# os.path.dirname()
# os.path.split()
# os.path.exists()


#*************************************#
# 6. 课堂练习：如何print出当前目录所有文件的完整路径
# Hint: Use `os.path.join`, `os.getcwd`, `os.listdir`
print("\n\nComplete path of every file in current directory:")

for file in os.listdir():
    print(os.path.join(os.getcwd(), file))