# Broadcast（广播）的规则
[Source website](https://zhuanlan.zhihu.com/p/35010592)
[NumPy 官网讲解](https://numpy.org/doc/stable/user/basics.broadcasting.html)
![Markdown logo](/Module3/numpy/img/WechatIMG647.png)
* All input arrays with ndim smaller than the input array of largest ndim, have 1’s prepended to their shapes.
* The size in each dimension of the output shape is the maximum of all the input sizes in that dimension.
* An input can be used in the calculation if its size in a particular dimension either matches the output size in that dimension, or has value exactly 1.
* If an input has a dimension size of 1 in its shape, the first data entry in that dimension will be used for all calculations along that dimension. In other words, the stepping machinery of the ufunc will simply not step along that dimension (the stride will be 0 for that dimension).

翻译如下：

1. 让所有输入数组都向其中shape最长的数组看齐，shape中不足的部分都通过在前面加1补齐
2. 输出数组的shape是输入数组shape的各个轴上的最大值
3. 如果输入数组的某个轴和输出数组的对应轴的长度相同或者其长度为1时，这个数组能够用来计算，否则出错
4. 当输入数组的某个轴的长度为1时，沿着此轴运算时都用此轴上的第一组值
来看更为一般的broadcasting rules：

***当操作两个array时，numpy会逐个比较它们的shape（构成的元组tuple），只有在下述情况下，两arrays才算兼容:***

- 相等
- 其中一个为1，（进而可进行拷贝拓展已至，shape匹配）

```
Image (3d array):  256 x 256 x 3
Scale (1d array):              3
Result (3d array): 256 x 256 x 3

A      (4d array):  8 x 1 x 6 x 1
B      (3d array):      7 x 1 x 5
Result (4d array):  8 x 7 x 6 x 5

A      (2d array):  5 x 4
B      (1d array):      1
Result (2d array):  5 x 4

A      (2d array):  15 x 3 x 5
B      (1d array):  15 x 1 x 5
Result (2d array):  15 x 3 x 5
``` 

```python
>>> x = np.arange(4)
>> xx = x.reshape(4, 1)
>> y = np.ones(5)
>> z = np.ones((3, 4))

>>> x.shape
(4,)
>>> y.shape
(5,)
>>> x+y
ValueError: operands could not be broadcast together with shapes (4,) (5,) 

>>> xx.shape
(4, 1)
>>> y.shape
(5,)
>>> (xx+y).shape
(4, 5)
>>> xx + y
array([[ 1.,  1.,  1.,  1.,  1.],
       [ 2.,  2.,  2.,  2.,  2.],
       [ 3.,  3.,  3.,  3.,  3.],
       [ 4.,  4.,  4.,  4.,  4.]])
```