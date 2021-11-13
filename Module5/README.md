# Module 5: First classification task

## 1.浅谈 梯度下降 (gradient descent)

[`gradient_descent_student.py`](/Module5/gradient_descent_student.py)
[`gradient_descent_demo.py`](/Module5/gradient_descent_demo.py)

1. `gradient_descent_student.py`
   1. 告诉学生们通常在机器学习，人工智能场景下，会出现非常复杂的损失函数，但是在计算他的最小值，用传统数学方法是很难求解的。所以我们可以利用numerical optimization方法中的一种 - 梯度下降的方法找到函数最小值（有些时候找到的是函数局部最小值）
   2. 这个python文件我定义了一个比较复杂的方程，可以让同学通过更改x值，来观察cost函数的值是如何变化的
      1. 可以尝试 x = 10 -> 12 -> 8 -> 3 -> 0 -> -4 -> -2 -> -1.8 （试不同参数，理解gradient descent的intuition，朝着负梯度的方向慢慢移动，移动到最小值）
      2. 这里不用引入复杂的数学概念，只需要大概了解gradient descent的思想就行
    3. 我在python文件里有两段代码，一个是标题写着`Numerical optimization. What's the absolute minimum of this function? `可以找到任何函数的最小值。另一个写着`What does the graph look like? Uncomment below and check it out!`可以将`cost`函数 plot 出来
2. `gradient_descent_demo.py`
   1. 不需要讲解此代码，直接让同学run，会有visualization的demo，演示gradient descent的过程
    ![gradient demo](/Module5/img/Snipaste_2021-11-12_22-57-24.png)
建议配合使用网页版 interactive demo 演示梯度下降过程：https://uclaacm.github.io/gradient-descent-visualiser/#playground
![gradient descent](/Module5/img/gradient_desc.png)
