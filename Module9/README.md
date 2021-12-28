# 运用卷积神经网络，`bad case`分析，图像增广引入

- [运用卷积神经网络，`bad case`分析，图像增广引入](#运用卷积神经网络bad-case分析图像增广引入)
  - [1. 运用卷积神经网络，分类同学自己的手写数字图片](#1-运用卷积神经网络分类同学自己的手写数字图片)
    - [`one-image-preprocess`](#one-image-preprocess)
    - [`all-images-preprocess`](#all-images-preprocess)
    - [`train+save_model`](#trainsave_model)
    - [`load_test_cnn`](#load_test_cnn)
  - [2. 使用手写数字可视化来看看网络对于哪些数字的分类效果不佳](#2-使用手写数字可视化来看看网络对于哪些数字的分类效果不佳)
  - [3. 图像增广引入](#3-图像增广引入)

## 1. 运用卷积神经网络，分类同学自己的手写数字图片

*建议教学时长：25分钟*

### [`one-image-preprocess`](/Module9/img_preprocess.py)

NOTE: 拍照的时候需要用比较粗一点的记号黑笔，白纸，开闪光灯的拍，尽量提高黑字和白底的光线和颜色差
- 用opencv对于一张手写数字图片进行transformation
![transformations]()
### [`all-images-preprocess`](/Module9/all_img_preprocess.py)

- 批量化处理，将`img_preprocess.py`对于整一个文件夹的img（原始图片）做处理
### [`train+save_model`](/Module9/train_CNN_network_save.py)

### [`load_test_cnn`](/Module9/load_test_cnn_model.py)

```python
# 分类结果：
Transformed_image0.jpg 4
Transformed_image1.jpg 8
Transformed_image2.jpg 9
Transformed_image3.jpg 5
Transformed_image4.jpg 2
Transformed_image5.jpg 3
Transformed_image6.jpg 0
Transformed_image7.jpg 1
Transformed_image8.jpg 6
Transformed_image9.jpg 7
```
## 2. 使用手写数字可视化来看看网络对于哪些数字的分类效果不佳

*建议教学时长：10分钟*

学习知识前先放一个最终手写数字识别的网页版Demo，向同学展示这堂课的最终成果是如何的：https://www.cs.ryerson.ca/~aharley/vis/conv/

![CNN handwritten digit demo](/Module8/img/CNN%20demo.png)

## 3. 图像增广引入

*建议教学时长：10分钟*
