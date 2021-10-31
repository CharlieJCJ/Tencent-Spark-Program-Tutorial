# 非常重要: What's the difference between `列表` (list) and `numpy array 数组`(Numerical Python)

NOTE: 在我一开始学NumPy的时候，并没有真正理解为什么要使用NumPy，只知道很多做ML的都在用这个库，所以就用了。但我觉得在学习NumPy，先理解为什么使用NumPy会对于之后的实践有巨大的帮助。
![Numpy](/Module3/numpy/img/6381635634163_.pic_hd.jpg)

1. 速度快
   - Numpy的并行化运算
```python
import time
import numpy as np

size_of_vec = 100000

def pure_python_version():
    t1 = time.time()
    X = range(size_of_vec)
    Y = range(size_of_vec)
    Z = [X[i] + Y[i] for i in range(len(X)) ]
    return time.time() - t1

def numpy_version():
    t1 = time.time()
    X = np.arange(size_of_vec)
    Y = np.arange(size_of_vec)
    Z = X + Y
    return time.time() - t1


t1 = pure_python_version()
t2 = numpy_version()
print(t1, t2)
print("Numpy is in this example " + str(t1/t2) + " faster!")

```

```python
0.021614789962768555 0.0003986358642578125
Numpy is in this example 54.22188995215311 faster!
```

2. NumPy数据结构内存占用小
   - NumPy数组是一个一个统一值的数组中（只能存储固定的一样的数据类型，比如int，float），单精度数字每4个字节，双精度数字每8个字节
   - python列表是指向python对象的指针数组，每个指针至少4个字节加上16个字节，即使是最小的python对象(4个用于类型指针，4个用于引用计数，4个用于值——内存分配器四舍五入为16)
![Memory](/Module3/numpy/img/6411635634299_.pic_hd.jpg)

3. 更多功能
   - 通过numpy做卷积、快速搜索、基本统计、线性代数、柱状图等，您可以获得比列表很更内置功能。
![Function](/Module3/numpy/img/6481635645902_.pic_hd.jpg)

## `列表` (list) and `numpy array 数组` 计算时的不同的特征：

```python
>>> import numpy as np

>>> np.array([1,2]) + np.array([3,4])
array([4, 6])
>>> [1, 2] + [3, 4]
[1, 2, 3, 4]

>>> np.array([1,2]) * np.array([3,4])
array([3, 8])
>>> [1, 2] * [3, 4]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: can't multiply sequence by non-int of type 'list'

>>> np.array([1,2]) * 2
array([2, 4])
>>> [1, 2] * 2
[1, 2, 1, 2]
```