import os

# 查看当前目录 Check the current working directory of the file
print(os.getcwd())

# os.chdir('...')


# 列举当前目录里所有文件名
print(os.listdir())


##############################################################
##############################################################

"""
os.makedirs('new_directory')
os.rmdir('new_directory')
"""



# Create many folders at the same time
"""
for i in range(1, 11):
    os.makedirs(f'My folder {i}')
"""



# Remove many folders at the same time
"""
for i in range(1, 11):
    os.rmdir(f'My folder {i}')
"""

"""
# Rename filename
os.rename('test.txt', 'demo.txt')
"""

"""
# Get information of a file
os.stat('demo.txt')
"""


##############################################################
##############################################################


# os.environ.get()
# os.path.join()
# os.path.basename()
# os.path.dirname()
# os.path.split()
# os.path.exists()