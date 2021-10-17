# Warmup python exercises
# 对于基础python语句复习回顾

#*************************************#
# 1. Primitive data types
# Define a variable containing a `Boolean` 
my_boolean = True
# Define a variable containing an integer 整数
my_int = 12
# Define a float
my_float = 233453.1231
# Define a string
my_name = "Charlie"
# String concatenation
many_str = "Hello " + "World!"  # 合并两个小string成一个大的string "Hello " + "World!" -> "Hello World!"

#*************************************#
# 2. for循环
for i in range(10):   # range(10) 代表着0～9的所有整数， 10不包括; 如果是range(1,10) ，代表着初始值1，1～9的所有整数
    print("Hello World")

#*************************************#
# 3. function 函数
# 自定义函数
def square(x):
    return x * x

# function call expression
# 使用函数
result = square(3)  # result should be 9
print(result)
#*************************************#

# 4. list python列表：一种序列，序列中的每个值都有对应的位置值，称之为索引，第一个索引是0，第二个索引是1，依此类推。
my_list = [1, 3, 5, 7]
print(my_list[0]) # 第一个元素 --> 1
print(my_list[1]) # 第二个元素 --> 3
print(my_list[-1]) # 倒数第一个元素 --> 7

# list slicing
print("my_list[1:3]", my_list[1:3])
print(my_list[0:2])

# list manipulation
print("Original list", my_list)
my_list.append(9)    # 把元素(9)附加在python列表的最后 [1, 3, 5, 7] --> [1, 3, 5, 7, 9]
print("After appending 9 to the list", my_list)

my_list.remove(7)    # 把元素(7)从python列表中去除 [1, 3, 5, 7, 9] --> [1, 3, 5, 9]
print("After removing 7 from the list", my_list)

# 除了append, remove; list还有很多其他list manipulation的函数, 比如insert, extend, pop 等
#*************************************#
my_list = [1, 3, 5, 7]
# 5. for能遍历整个python列表

for item in my_list:
    print(f"Item {item}")

# 6. 非常实用的`enumerate`函数，如果我们同时需要列表中的索引和元素值
for index, item in enumerate(my_list):
    print(f"Index {index} of list is {item}")

# Side note: f string -> string formating 非常简洁实用
a = 2
b = 3
print(f"{a} + {b} = {a + b}")
print(str(a) + " + " + str(b) + " = " + str(a + b))   # 比较一下用f string和string concatenation代码的不同
#*************************************#