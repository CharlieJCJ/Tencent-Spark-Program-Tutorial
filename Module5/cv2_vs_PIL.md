# Difference between opencv2 and PIL
[Answer reference](https://www.quora.com/Whats-the-difference-between-following-python-packages-CV2-PIL-and-OPENCV-When-can-I-use-each-of-them "Answer reference")
CV2 is `OpenCV`.

`OpenCV` and `PIL` both have image processing tools such as:

Image filters (blur, sharpen, etc.)
Image transformation: flip, rotate, warp,…
Conversion between image types.
Other basic things you can do with image.
To put it simply, using either OpenCV or PIL, you can create your own photoshop. However, OpenCV also provides:

Tools to work with videos.
Feature extraction methods for computer vision: SIFT, HOG, HAAR,...
Machine learning: things like neural network, SVM, K-NN, and so on.
***OpenCV covers most of what PIL offers and more.*** If you only need image processing functions and you want something light, use PIL. Otherwise, go for OpenCV.