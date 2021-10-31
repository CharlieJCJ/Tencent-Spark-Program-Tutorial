import cv2 as cv
from matplotlib import pyplot as plt
import os


img = './img/img2.jpg'

# mode 可以改为 cv.IMREAD_COLOR, cv.IMREAD_GRAYSCALE, cv.IMREAD_UNCHANGED
mode = cv.IMREAD_COLOR

# `imread()` decodes the image into a matrix with the color channels 
# stored in the order of Blue, Green, Red

img = cv.imread(img, mode)
print(img)
print(type(img)) # <class 'numpy.ndarray'> 图片是以numpy array的形式存储的，如果是彩色图片，则有三通道BGR；如果是黑白图片，则只有一通道：灰度值
print(img.shape) # (638, 640, 3)

# 可以尝试 mode = cv.IMREAD_GRAYSCALE, 看看结果有什么变化~