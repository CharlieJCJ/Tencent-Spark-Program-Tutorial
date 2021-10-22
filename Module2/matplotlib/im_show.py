# CV2指的是OpenCV2，OpenCV是一个跨平台计算机视觉库
# 主要用的模块大概分为以下几类：
# 1. 图片读写，2. 图像滤波，3.图像增强，4.阈值分割，5.形态学操作
# 今天主要会运用图片读写功能
# 并且用matplotlib的`imshow()`显示图片

import cv2 as cv
from matplotlib import pyplot as plt
import os

data_dir = 'images'
img_pth = ['img' + str(i) + '.jpg' for i in range(1, 6)]
img = [os.path.join(data_dir, i) for i in img_pth]

# 修改这里的数字（0～4）都可以用
img_num = 1
# mode 可以改为 cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
mode = cv.IMREAD_UNCHANGED

# `imread()` decodes the image into a matrix with the color channels 
# stored in the order of Blue, Green, Red


img = cv.imread(img[img_num], mode)
print('Image Dimensions :', img.shape)
plt.imshow(img)
plt.show()

