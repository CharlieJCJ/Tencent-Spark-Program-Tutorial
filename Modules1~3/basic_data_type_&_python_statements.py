# Warmup python exercises
# 对于基础python语句复习回顾


# Primitive data types
# Define a variable containing a `Boolean` 
my_boolean = True
# Define a variable containing an integer 整数
my_int = 12
# Define a float
my_float = 233453.1231
# Define a string
my_name = "Charlie"


# for循环
for i in range(10):   # range(10) 代表着0～9的所有整数， 10不包括; 如果是range(1,10) ，代表着初始值1，1～9的所有整数
    print("Hello World")

# function 函数
# 自定义函数
def square(x):
    return x * x

# function call expression
# 使用函数
result = square(3)  # result should be 9
print(result)



# list python列表：一种序列，序列中的每个值都有对应的位置值，称之为索引，第一个索引是0，第二个索引是1，依此类推。
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

