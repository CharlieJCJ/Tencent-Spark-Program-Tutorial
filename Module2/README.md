# Module 2
## 1. 读写数据 `open()`
```python
open() 函数
```
## 2. Python Library `os`
```python
import os
```
## 3. Python Library `matplotlib.pyplot`
```python
from matplotlib import pyplot as plt
```
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
6. `im_show.py`
   - 使用 `opencv` 结合 `os.path.join` 读取图片数据
   - 可以更改图片路径和读取图片模式
   - 用`plt.imshow()`来显示图片
   - 为cv库开一个头，之后会对于cv库的其他功能做展开
