# Module 6: 人工智能神经网络基础-3 训练神经网络，深度神经网络
- [Module 6: 人工智能神经网络基础-3 训练神经网络，深度神经网络](#module-6-人工智能神经网络基础-3-训练神经网络深度神经网络)
  - [1. 训练神经网络核心知识点](#1-训练神经网络核心知识点)
    - [* ***超参数*** (hyperparameter)](#-超参数-hyperparameter)
    - [* ***训练集、验证集、测试集 (Train, Validation, Test Sets)***](#-训练集验证集测试集-train-validation-test-sets)
      - [`train_test_split.py`](#train_test_splitpy)
    - [* ***过拟合（over-fitting ）欠拟合（under-fitting）***](#-过拟合over-fitting-欠拟合under-fitting)
    - [* ***Epoch*** (中文翻译：时期。一般机器学习工程师直接会用`epoch`而不是它的中文翻译)](#-epoch-中文翻译时期一般机器学习工程师直接会用epoch而不是它的中文翻译)
    - [* *EXTRA* ***Batch_size*** (这个课上不用讲解)](#-extra-batch_size-这个课上不用讲解)
    - [* ***模型准确率*** (Accuracy)](#-模型准确率-accuracy)
  - [2. 使用 `pytorch` 库搭建并训练你的第一个神经网络](#2-使用-pytorch-库搭建并训练你的第一个神经网络)
    - [`pytorch_perceptron.py`](#pytorch_perceptronpy)
    - [`dataset.py`](#datasetpy)
  - [3. 使用 Tensorflow playground 可视化 比较浅层神经网络和深度神经网络](#3-使用-tensorflow-playground-可视化-比较浅层神经网络和深度神经网络)
  - [***预告下周 Module7***：](#预告下周-module7)

## 1. 训练神经网络核心知识点
***推荐教学时长：15分钟***
### * ***超参数*** (hyperparameter)
  * 区分于`可训练参数`，超参数是在开始学习过程之前设置值的参数，而不是通过训练得到的参数数据。通常情况下，需要对超参数进行优化，给学习机选择一组最优超参数，以提高学习的性能和效果。
  * 比如：模型的学习率 learning rate，深层神经网络隐藏层数 hidden layers（在本章后面会介绍）
![hyper](/Module6/img/hyperparameter_vs_param.png)
### * ***训练集、验证集、测试集 (Train, Validation, Test Sets)***
![TVT](/Module6/img/TVT.png)
![TVT](/Module6/img/TVT_workflow.png)
#### [`train_test_split.py`](train_test_split.py)
   * 训练集：用来训练模型内参数的数据集
   * 验证集：验证集通常用于调整超参数，根据几组模型验证集上的表现决定哪组超参数拥有最好的性能。
   * 测试集：评估模型的泛化能力（看模型有没有出现过拟合）
### * ***过拟合（over-fitting ）欠拟合（under-fitting）***
![fit](/Module6/img/overfit_underfit.png)
  * 欠拟合： 欠拟合是指模型不能在训练集上获得足够低的误差。换句换说，就是模型复杂度低，模型在训练集上就表现很差，没法学习到数据背后的规律。
  * 过拟合： 过拟合是指训练误差和测试误差之间的差距太大。换句换说，就是模型复杂度高于实际问题，模型在训练集上表现很好，但在测试集上却表现很差。模型对训练集"死记硬背"（记住了不适用于测试集的训练集性质或特点），没有理解数据背后的规律，泛化能力差。
### * ***Epoch*** (中文翻译：时期。一般机器学习工程师直接会用`epoch`而不是它的中文翻译)
  * 一个Epoch就是将所有训练样本训练一次的过程
  * 目的：在神经网络中传递完整的数据集一次是不够的，而且我们需要将完整的数据集在同样的神经网络中传递多次。但请记住，我们使用的是有限的数据集，并且我们使用一个迭代过程即梯度下降来优化学习过程。如下图所示。因此仅仅更新一次或者说使用一个epoch是不够的。
  * 随着epoch数量增加，神经网络中的权重的更新次数也在增加，曲线从欠拟合变得过拟合。
![model](/Module6/img/Model_perf.png)

### * *EXTRA* ***Batch_size*** (这个课上不用讲解)
  * 所谓Batch就是每次送入网络中训练的一部分数据，而Batch Size就是每个batch中训练样本的数量，这是另一个`超参数`
  * 优点：通过并行化提高内存的利用率，提高训练速度。适当Batch Size使得梯度下降方向更加准确。
### * ***模型准确率*** (Accuracy)
![acc](/Module6/img/Accuracy.png)
  * Extra: 其他分类模型评估指标请同学自行阅读 https://easyai.tech/ai-definition/accuracy-precision-recall-f1-roc-auc/


## 2. 使用 `pytorch` 库搭建并训练你的第一个神经网络

### [`pytorch_perceptron.py`](pytorch_perceptron.py)
### [`dataset.py`](/Module6/dataset.py)

***推荐教学时长：15分钟***
1. 训练，测试数据
   1. 使用 torch.utils.data 模块的 Dataset
2. 将整个数据集分成训练集和测试集
3. 定义神经网络结构
4. 设置训练模型，参数
   1. optimizer 就是优化器，包含了需要优化的参数有哪些，
   2. loss_func 就是我们设置的损失函数
   3. epoch 是指所有数据被训练的总轮数
5. 训练模型
   1. 使用当前模型 <训练的参数> 去预测数据相对应的标签 (label)，即 `前向传播`
   2. `criterion()` 计算【损失函数】结果， (output, target) 作为输入 (output为网络的输出,target为实际值)
   3. `loss.backward` 反向传播 - 利用损失函数反向传播计算梯度
   4. `optimizer.step` 梯度下降，更新模型参数 - 用我们定义的优化器将每个需要优化的参数进行更新
   5. 在训练过程中print出来训练中的损失函数结果（观察损失函数的变化）
6. 测试模型
7. 模型在 <训练集> 和 <测试集> 上的表现可视化


![pytorch](/Module5/img/pytorch1.png)

## 3. 使用 Tensorflow playground 可视化 比较浅层神经网络和深度神经网络
***推荐教学时长：10分钟***
* 什么是深度神经网络？
深度神经网络是一种具备至少一个隐层的神经网络。
![SNN](/Module6/img/SNN.jpg)
![DNN](/Module6/img/neural-net.png)

**图一是perceptron（浅层神经网络）；图二是深度神经网络（有>=1个隐层）**

https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.41236&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false

![playground](/Module6/img/compare.png)

**引导学生回答下列问题：**
1. 多层神经网络有什么作用？
2. 观察overfit, underfit (打开test data的distribution）
      - 过度fit原始数据集

## ***预告下周 Module7***：
用代码搭建，训练深度神经网络（对于图片数据进行分类）。卷积神经网络introduction

Extra：对于python如何写面向对象编程，这里是一个参考网站：https://www.runoob.com/python/python-object.html
