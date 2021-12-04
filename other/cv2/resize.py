import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
cv2.resize() 放大和缩小图像
    参数：
        src: 输入图像对象
        dsize：输出矩阵/图像的大小，为0时计算方式如下：dsize = Size(round(fx*src.cols),round(fy*src.rows))
        fx: 水平轴的缩放因子，为0时计算方式：  (double)dsize.width/src.cols
        fy: 垂直轴的缩放因子，为0时计算方式：  (double)dsize.heigh/src.rows
        interpolation：插值算法
            cv2.INTER_NEAREST : 最近邻插值法
            cv2.INTER_LINEAR   默认值，双线性插值法
            cv2.INTER_AREA        基于局部像素的重采样（resampling using pixel area relation）。对于图像抽取（image decimation）来说，这可能是一个更好的方法。但如果是放大图像时，它和最近邻法的效果类似。
            cv2.INTER_CUBIC        基于4x4像素邻域的3次插值法
            cv2.INTER_LANCZOS4     基于8x8像素邻域的Lanczos插值
                     
    cv2.INTER_AREA 适合于图像缩小， cv2.INTER_CUBIC (slow) & cv2.INTER_LINEAR 适合于图像放大
'''

path = './images/img1.jpg'

img = cv2.imread(path)

plt.subplot(1,3,1)
plt.imshow(img)
plt.title('Original')
print(f'Original Dimensions : {img.shape}')


plt.subplot(1,3,2)
double = cv2.resize(img, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
plt.imshow(double)
plt.title('Double size')
print(f'Double sized image Dimensions : {double.shape}')

plt.subplot(1,3,3)
half = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
plt.imshow(half)
plt.title('Half size')
print(f'Half sized image Dimensions : {half.shape}')
plt.show()