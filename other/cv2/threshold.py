import cv2 as cv
import matplotlib.pyplot as plt
'''
cv2.threshold(): 
参数：
    img:图像对象，必须是灰度图
    thresh:阈值
    maxval：最大值
    type:
        cv2.THRESH_BINARY:     小于阈值的像素置为0，大于阈值的置为maxval
        cv2.THRESH_BINARY_INV： 小于阈值的像素置为maxval，大于阈值的置为0
        cv2.THRESH_TRUNC：      小于阈值的像素不变，大于阈值的置为thresh
        cv2.THRESH_TOZERO       小于阈值的像素置0，大于阈值的不变
        cv2.THRESH_TOZERO_INV   小于阈值的不变，大于阈值的像素置0
返回两个值
    ret:阈值
    img：阈值化处理后的图像
'''

path = './images/img1.jpg'

img = cv.imread(path, 0)

ret,thre1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
adaptive_thre1 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 2)
adaptive_thre2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7, 2)

titles = ["img","thre1","adaptive_thre1","adaptive_thre2"]
imgs = [img,thre1, adaptive_thre1 ,adaptive_thre2]

for i in range(4):
    plt.subplot(2,2,i+1), plt.imshow(imgs[i],"gray")
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()