import cv2  # OpenCV library
import numpy as np
import matplotlib.pyplot as plt
import os 

image = "img2.jpeg"
directory= "./img"
print("Joined image path is :", os.path.join(directory, image))
path = os.path.join(directory, image)

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
cv2.imshow('1. Original Image',img)

ret,thre1 = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
cv2.imshow('2. Black-and-white Threshold image', thre1)

img_invert = cv2.bitwise_not(thre1)
cv2.imshow('3. Inverted black-and-white image', img_invert)

resized_invert = cv2.resize(img_invert, (28, 28))
cv2.imshow('4. Resized image into 28 x 28 (final image)', resized_invert)

cv2.imwrite(os.path.join('./', "Transformed_image.jpg"), resized_invert)
cv2.waitKey(0)
cv2.destroyAllWindows()