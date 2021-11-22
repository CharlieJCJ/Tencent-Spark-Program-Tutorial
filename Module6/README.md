## 3. 使用 `pytorch` 库搭建你的第一个神经网络
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
[`pytorch_perceptron.py`](pytorch_perceptron.py)

![pytorch](/Module5/img/pytorch1.png)