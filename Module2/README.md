# Module 2
## 1. 读写数据 `open()`
推荐教学时长：10分钟
```python
open() 函数
```
open函数可用于读取训练测试集数据标签和图片名称的txt file。读取后的数据需要做一些简单的数据清理放入一个列表中。
1. Old practice
2. Better practice: context manager
3. `<file>.read()` 读取整个文件，返回的是一个字符串，字符串包括文件中的所有内容
4. `<file>.readlines()` readlines() 用于读取文件中的所有行
5. 简单的数据清理

## 2. Python Library `os`
推荐教学时长：10分钟
```python
import os
```
os是python自带的非常实用的标准库之一，是与操作系统相关的库。如：文件，目录，执行系统命令等。用于批量化处理文件，可运用于路径拼接图片训练测试数据集文件路径，批量化重命名图片名称等。

1. `os.getcwd` 查看当前目录
2. `os.listdir` 列举当前目录里所有文件名
3. `os.makedirs` 新建文件夹
4. `os.rmdir` 删除文件夹
5. `os.path.join(path, *paths)` 用于路径拼接文件路径
6. 课堂练习：如何print出当前目录所有文件的完整路径

## 3. Python Library `matplotlib.pyplot`
推荐教学时长：15分钟
```python
from matplotlib import pyplot as plt
```
`matplotlib` 是一个绘图工具，类似excel或者matlab里绘制图表的功能，可能是 Python 2D-绘图领域使用最广泛的套件。它能让使用者很轻松地将数据图形化，并且提供多样化的输出格式
1. `basic_plots.py`
   - 一个最基础的line plot例子
2. `plot_with_titles.py`
   - plot里自定义参数
3. `multiple_plots.py`
   - 一张图表里可以graph多个line plots
4. `barplot.py`
   - plt 柱状图
5. `histogram.py`
   - 使用 **pandas 库** 来读取csv文件
   - 选取其中的 `Age` 列，进行直方图的绘制
     - `pandas` 可以从各种文件格式比如 CSV、JSON、SQL、Microsoft Excel 导入数据。可以对各种数据进行运算操作，比如归并、再成形、选择，还有数据清洗和数据加工特征。
6. `im_show.py`
   - 使用 `opencv` 结合 `os.path.join` 读取图片数据
   - 可以更改图片路径和读取图片模式
   - 用`plt.imshow()`来显示图片
   - 为cv库开一个头，之后会对于cv库的其他功能做展开
     - `CV2`指的是`OpenCV2`，`OpenCV`是一个跨平台计算机视觉库,主要用的模块大概分为以下几类：
       1. 图片读写
       2. 图像滤波
       3. 图像增强
       4. 阈值分割
       5. 形态学操作
