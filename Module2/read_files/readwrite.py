# 读写数据 open() 函数用于创建或打开指定文件

#*************************************#
# 1. Old practice (不推荐用第一个方法) 
f = open('the_adventures_of_sherlock_holmes.txt', 'r')
print("文件名:", f.name)   # 文件名
print("文件权限:", f.mode)   # 文件权限: 'r' = only read, 'w' = only write, 'r+' = read and write
f.close()

# 2. Better practice: context manager --> `with` statement 推荐在使用open()时候，用以下方法
# 文件读取完毕后自动关闭
# The advantage is that the file is properly closed after its suite 
# finishes, even if an exception is raised at some point.
with open('the_adventures_of_sherlock_holmes.txt', 'r') as f:  
    print("文件名:", f.name) 
    print("文件权限:", f.mode) 
    
#*************************************#
# 3. `<file>.read()` 读取整个文件，返回的是一个字符串，字符串包括文件中的所有内容
with open('the_adventures_of_sherlock_holmes.txt', 'r') as f:    
    f_contents = f.read() # 这会把整个txt文件全部放在一个string里

print("书籍的前200个字符:\n", f_contents[:200])  # 看前200个字符

#*************************************#
# 4. `<file>.readlines()` readlines() 用于读取文件中的所有行，
# 该函数返回是一个字符串列表，其中每个元素为文件中的一行内容。
with open('test.txt', 'r') as f:    
    f_contents_lst = f.readlines()
print(f_contents_lst)

#*************************************#
# 5. 简单的数据清理
# 
f_contents_lst = []
with open('test.txt', 'r') as f:    
    for line in f:
        line = line.replace("\n", "") 
        line = line.split(" ")        
        f_contents_lst.append(line)
print(f_contents_lst)
