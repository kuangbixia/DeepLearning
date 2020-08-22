# 神经网络loss函数

references：[[1]Loss and Loss Functions for Training Deep Learning Neural Networks](https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/)

​					   [[2]How to Choose Loss Functions When Training Deep Learning Neural Networks](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)



## 0 loss只是用来评估模型的优化

- 损失函数的具体形式依赖于神经网络**输出层**的选择
- **最大似然法**（Maximum Likelihood）提供了一个选择训练模型时使用的损失函数的框架
- **均方差**和**交叉熵**是主要的两种loss函数



## 1 MSE(Mean Squared Error)

​	通常用于回归（近似）的问题
$$
MSE(W)=\frac{1}{m}\sum^m_{i=1}(W^T·X^{(i)}-y^{(i)})^2
$$


### (1) （线性）回归问题

​	**输出层结构**：有线性（Linear）激活单元的节点



## 2 Cross Entropy Error

​	通常用于分类问题
$$
\begin{align}
&J(W)=-\frac{1}{m}\sum^m_{i=1}\sum^K_{j=1}y_k^{(i)}log(\hat{p}_k^{(i)})\\\\
&其中，当第i个实例的预测类别是k时，y_k^{(i)}=1，否则为0\\
&当只有两个类别时，K=2，这个损失函数等价于逻辑回归的log损失函数
\end{align}
$$


### (1) 二分类问题

​	**输出层结构**：有sigmoid激活单元的节点

### (2) 多分类问题

​	**输出层结构**：对每个类别使用softmax激活函数的节点

