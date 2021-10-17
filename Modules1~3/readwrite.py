# Old practice (不推荐用第一个方法)
f = open('test.txt', 'r')
print(f.name)   # 文件名
print(f.mode)   # 文件权限: 'r' = only read, 'w' = only write, 'r+' = read and write
f.close()


# Context manager, 'with' statement 推荐在使用open()时候，用以下方法
# The advantage is that the file is properly closed after its suite 
# finishes, even if an exception is raised at some point.


# Practice: read in a text file from the current directory.
'''
with open('test.txt', 'r') as f:    
    f_contents = f.read()
    print(f_contents)
'''


'''
with open('test.txt', 'r') as f:    
    f_contents = f.readlines()
    print(f_contents)
'''



