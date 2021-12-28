import cv2  # OpenCV library
import numpy as np
import matplotlib.pyplot as plt
import os 


directory= "./img"

print(os.listdir(directory))

images_names = os.listdir(directory); images_names.remove('.DS_Store')
for ind, image in enumerate(images_names):
    if image == '.DS_Store':
        continue
    path = os.path.join(directory, image)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ret,thre1 = cv2.threshold(img, 165, 255, cv2.THRESH_BINARY)
    img_invert = cv2.bitwise_not(thre1)
    resized_invert = cv2.resize(img_invert, (28, 28))
    cv2.imwrite(os.path.join("./transformed_hand_written_digits", f"Transformed_image{ind}.jpg"), resized_invert)

# Plaese check ./transformed_hand_written_digits folder! 