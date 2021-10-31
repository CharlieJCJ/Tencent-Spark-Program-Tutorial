# Module 3 Intro to Numpy
 NumPy(Numerical Python) 是 Python 语言的一个扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
`Numpy array` - 新的数据结构 ndarray (n-dimensional array)，N 维数组

## 1. 理解在处理大数据问题上为什么推荐使用NumPy，而不是列表



`list_vs_numpy.md`
- 速度快 (Numpy的并行化运算)
- NumPy数据结构内存占用小
- 更多功能 (卷积、快速搜索、基本统计、线性代数、柱状图等，您可以获得比列表很更内置功能。)
- 允许逐个元素的基础运算

## 2. NumPy 基础
`numpy_intro.py`
- 创建数组 Initialize numpy arrays 
  - 1D array
  - 2D array
  - Get Dimension 维度
  - Get Shape 表示数组形状（shape）的元组，表示各维度大小的元组（tuple）
  - Get Data Type 数组数据类型
  - 从列表转换成numpy数组
- NumPy 切片和索引 - Accessing numpy arrays through indexing
  - Get a specific element [r, c]
  - Get a specific row 
  - Get a specific column
  - Changing specific elements, rows, columns
- Initializing Different Types of Arrays
  - `np.zeros` 创建指定大小的数组，数组元素以 0 来填充
  - `np.ones` 创建指定形状的数组，数组元素以 1 来填充
  - `np.full` 创建指定形状的数组，数组元素以 参数值 来填充
  - 随机数数组 `np.random.rand` `np.random.randint`
  - `np.identity` identity matrix
  - Repeat an array `np.repeat`
- Numpy 数组操作， 包含了一些函数用于处理数组
  - `numpy.reshape` 函数可以在不改变数据的条件下修改形状
  - `numpy.ndarray.flatten` 返回一份数组拷贝，对拷贝所做的修改不会影响原始数组
## 3. NumPy 数组基础运算，简单矩阵乘法
`numpy_compute.py`
- Element-wise computation
- NumPy 广播 Broadcast 机制
  - 更多关于 NumPy broadcast 请阅读 `EXTRA_numpy_broadcasting_rules.md` 文件请阅读官方文件： https://numpy.org/doc/stable/user/basics.broadcasting.html

- Mathematics
  - 三角函数
  - `numpy.around(a,decimals)` 返回指定数字的四舍五入值decimals: 舍入的小数位数。 默认值为0。 如果为负，整数将四舍五入到小数点左侧的位置
- 统计
  - 平均值 `numpy.mean()` 函数返回数组中元素的算术平均值。 如果提供了轴，则沿其计算。
    - 1D vs 2D 矩阵在使用函数时的区别
  - 中位数 `np.median`
  - 标准差 `np.std`
- 线性代数
  - 转置矩阵 `<NumPy array>.T`
  - `numpy.dot()`对于两个一维的数组，计算的是这两个数组对应下标元素的乘积和(数学上称之为内积)
  - `numpy.inner()` 函数返回一维数组的向量内积
## 4. 常见错误 (当需要复制np array的时候的注意事项)
  `copy_notice.py`

NOTE: 当两个数组指向同一个数组，当其中一个受到了更改，另一个的值也受到了更改
## 5. 【应用】cv2 import的图片是使用 numpy array 来表示的
`numpy_in_cv2.py`