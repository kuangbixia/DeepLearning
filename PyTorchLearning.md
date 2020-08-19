# PyTorch是什么？

PyTorch是一个开源的Python机器学习（Torch）库。

PyTorch有两大特征：

1. 类似于NumPy的张量计算，可使用GPU加速
2. 基于带自动微分系统的**深度神经网络**

# 入门

## 1 张量Tensor

PyTorch的张量相当于NumPy的多维数组

### (1) 声明和初始化

```python
import torch

# 创建一个张量，未初始化
x = torch.empty(5,3)
print(x)
'''
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
'''

# 张量生成函数
# 随机数——初始值为随机数[0,1]f
x = torch.rand(5,3)
# 标准正态分布
x = torch.randn(5,3)
# 离散正态分布
x = torch.normal(5,3)
# 线性间距向量
x = torch.linespace(5,3)
# 范围函数
x = torch.arange(1,6) # 包含start，不包含end
'''
tensor([1.,2.,3.,4.,5.])
'''
x = torch.range(1,6) #包含start和end
'''
tensor([1.,2.,3.,4.,5.,6.])
'''

# 自定义数据类型
# 初始值设为0
x = torch.zeros(5,3,dtype=torch.long)
# 初始值设为1
x = torch.new_ones(5,3,dtype=torch.double)

# 生成张量的其他方式
# 用已有张量创建新的张量
y = torch.randn_like(x,dtype=torch.float)
# 将矩阵转化为张量
x = torch.tensor([5.5, 3])
```

### (2) 算术运算

```python
import torch

x = torch.rand(5,3)
y = torch.rand(5,3)

# 相加
# 加法1
z = x+y
# 加法2
z = torch.add(x,y)
# 加法3，添加参数
result = torch.empty(5,3)
torch.add(x,y,out=result)
# 加法4，'_'在所有的自身操作符的末尾都有
y.add_(x) # 将x加到y
```

### (3) 改变形状和大小

```python
import torch

x = torch.rand(5,3)

# 获取张量的形状，返回的是一个元组——Tensor.size()
print(x.size())
'''
torch.Size([5,3])
'''

# 重新设置张量的形状——Tensor.view(x,y)
y = x.view(16)
z = x.view(2,8)
# 改变维度——Tensor.transpose(d_x,d_y)
q = x.transpose(0,1) # 交换第一维和第二维
'''
torch.Size([3,5])
'''

# 查看张量的大小/值——Tensor.item()
x = torch.randn(1)
print(x.item())
```

- view() vs transpose()：

  - 都可以改变Tensor的形状
  - view()并不改变Tensor的内存布局

- view()：

  - 某一维度可以填入-1，实际的值可以通过其他维度计算出来（必须是int）

  - 示例

    ```python
    import torch
    
    x = torch.arange(6)
    '''
    Tensor([0.,1.,2.,3.,4.,5.])
    '''
    y = x.view(3,-1) # 6 / 3 = 2
    '''
    Tensor([[0.,1.],
    		[2.,3.],
    		[4.,5.]])
    '''
    z = x.view(-1,6) # 6 / 6 = 1
    '''
    Tensor([[0.,1.,2.,3.,4.,5.]])
    '''
    q = x.view(1,-1,2) # 6 / (1 * 2) = 3
    '''
    Tensor([[[0.,1.],
    		 [2.,3.],
    		 [4.,5.]]])
    '''
    ```

## 2 Tensor和NumPy的转换

CPU下，他们共享物理地址，改变其中一个，另一个也会随之改变

### (1) Tensor到NumPy

```python
import torch

t = torch.ones(5)
'''
tensor([1.,1.,1.,1.,1.])
'''
n = t.numpy()
'''
[1. 1. 1. 1. 1.]
'''
```

### (2) NumPy到Tensor

```python
import numpy as np
import torch

n = np.ones(5)
t = torch.from_numpy(n)
```

## 3 Cuda Tensors

.to(device)，在“Cuda”和“cpu”之间转移，还可以设置参数改变数据类型

```python
import torch

x = torch.rand(5,3)

if torch.cuda.is_available():
    device = torch.device("cuda")
    # ① 在GPU上创建Tensor
    y = torch.ones_like(x, device=device)
    # ② 直接转换到cuda——.to(device)
    x = x.to(device)
    z = x+y
    print(z)
    '''
    tensor([-0.4743], device='cuda:0')
    '''
    # 直接转换到cpu——.to(device)，设置参数改变了数据类型
    print(z.to("cpu",torch.double))
    '''
    tensor([-0.4743], dtype=torch.float64)
    '''
```

## 4 Pytorch相关操作

### (1) Tensor.size()

​	获取Tensor的形状

- 无参数：返回整个Tensor的形状

- 有参数（int）：返回Tensor的某个维度的size，示例：

  ```python
  import torch
  
  x = torch.rand(5,3)
  y = x.size(0)
  '''
  y: 5
  '''
  ```

### (2) Tensor == Tensor

​	返回包含每个位置的元素取等运算后的bool值的Tensor

# 自动微分——autograd

​	在pyTorch中，神经网络的核心是autograd包，它会为Tensor上所有的操作提供自动微分，是一个由运行定义的框架。以代码运行的方式定义**后向传播**，所以每次迭代都可以不同。

## 1 Tensor

### (1) 属性Tensor.requires_grad

​	将属性.requires_grad（默认False）设置为True，会开始跟踪针对tensor的所有操作

```python
import torch

# 在创建tensor时，设置requires_grad
x = torch.ones(2,2,requires_grad=True)

# 直接设置已创建tensor的requires_grad属性的值
x = torch.randn(2,2)
x.requires_grad_(True)
```

### (2) 属性Tensor.grad_fn（引用autograd.Function类）

​	**跟踪**过程，会对操作后**得到的Tensor**添加相应的**.grad_fn**属性值——如<AddBackward0>，<MulBackward0>，<MeanBackward0>等，记录进行操作的Tensor的操作。

### (3) 跟踪过程示例

```python
import torch

x = torch.ones(2,2,requires_grad=True)
# 对x操作——Add
y = x + 2
'''
y:
tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)
'''
# 对y操作——Multiply
z = y * y * 3
'''
z:
tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>)
'''
# 对z操作——Mean
out = z.mean()
'''
out:
tensor(27., grad_fn=<MeanBackward0>)
'''
```

## 2 梯度Gradients

### (1) 函数Tensor.backward()

- 结束操作后，调用**最后一个Tensor**的**Tensor.backward()**函数自动计算**所有的梯度**（微分）

### (2) 属性Tensor.grad

- 梯度计算后保存到**被操作Tensor**的**Tensor.grad**属性中

### (3) 反向传播示例

#### ① 简单示例

​	Tensor是标量（只有一个元素），.backward()参数可以为空

```python
# 接上一个代码块

# 计算梯度（微分）
# PS：out只包含一个标量，所以可以直接简单的调用.backward()，不设置参数
out.backward() # out.backward(torch.tensor(1.))

# 获取被操作Tensor的梯度
print(x.grad)
'''
tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
'''
```

#### ② 另一个示例

​	Tensor不是标量（有多个元素），.backward()要设置参数

```python
import torch

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)
'''
tensor([-479.4348,   33.9321,  913.9350], grad_fn=<MulBackward0>)
'''

# 计算梯度
# PS：y不再是只包含一个标量（有多个元素），调用.backward()时要设置向量作为参数
v = torch.tensor([0.1,1.0,0.0001],dtype=torch.float)
y.backward(v)

# 获取被操作Tensor的梯度
print(x.grad)
'''
tensor([1.0240e+02, 1.0240e+03, 1.0240e-01])
'''
```

## 3 停止跟踪

### 1 封装代码块——with torch.no_grad(): xxxxxxx

```python
# 接上一个代码块

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
    
'''
True
True
False
'''
```

### 2 函数.detach()

```python
# 接上一个代码块

print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
'''
True
False
'''
```

# 线性模型

## 1 线性回归

### (1) 模型预测

$$
\begin{align}
\hat{y}&=w_1x_1+w_2x_2+···+w_nx_n+b\\
&=W^T·X
\end{align}
$$

$$
其中，W^T=\begin{bmatrix}
w_1&w_2&\cdots&w_n&b
\end{bmatrix}
,
X=\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_n\\
1
\end{bmatrix}
$$



### (2) MSE损失函数——凸函数

$$
MSE(W)=\frac{1}{m}\sum^m_{i=1}(W^T·X^{(i)}-y^{(i)})^2
$$



## 2 逻辑回归——二分类

### (1) 逻辑函数——sigmoid函数（S型）

$$
σ(t)=\frac{1}{1+e^{-t}}=\frac{e^t}{1+e^t}
$$

### (2) 概率估算

$$
\hat{p}=σ(W^T·X)
$$

### (3) 模型预测

$$
\hat{y}=
\begin{cases}
0&(\hat{p}<0.5)\cr
1&(\hat{p}>=0.5)
\end{cases}
$$

### (4) log损失函数

$$
\begin{align}
&单个实例：j(W)=\begin{cases}
-log(\hat{p})&(y=1)\cr
-log(1-\hat{p})&(y=0)
\end{cases}\\\\
&整个训练集：J(W)=-\frac{1}{m}\sum^m_{i=1}[y^{(i)}log(\hat{p}^{(i)})+(1-y^{(i)})log(1-\hat{p}^{(i)})]\\\\
&其中，第i个实例的\hat{y}=1时，y^{(i)}=1；\hat{y}=0时，y^{(i)}=0
\end{align}
$$

## 3 softmax回归——多类别

​	对每个实例x，先计算出每个类别k的分数，然后对K个分数应用softmax函数（归一化指数），估算出每个类别的概率。

### (1) 计算类别k的分数

​	每个类别都有自己特定的参数W_k
$$
s_k(X)=W_k^T·X
$$

### (2) 估算概率

- 通过softmax函数计算每个分数的指数
- 对指数进行归一化处理（除以所有指数的总和）
- 得到概率：在给定类别k的分数下，实例X属于类别k的概率

$$
\hat{p_k}=σ(s(X))_k=\frac{e^{s_k(X)}}{\sum^K_{j=1}e^{s_j(X)}}
$$

### (3) 模型预测

$$
\hat{y}=argmax_k\ \hat{p}=argmax_k\ s_k(X)=argmax_k\ (W^T_k·X)
$$

- argmax函数：返回使得函数最大化所对应的变量的值
- 这里，返回的是，使得估算概率最大的类别k的值

### (4) CrossEntropy交叉熵损失函数

$$
\begin{align}
&J(W)=-\frac{1}{m}\sum^m_{i=1}\sum^K_{j=1}y_k^{(i)}log(\hat{p}_k^{(i)})\\\\
&其中，当第i个实例的预测类别是k时，y_k^{(i)}=1，否则为0\\
&当只有两个类别时，K=2，这个损失函数等价于逻辑回归的log损失函数
\end{align}
$$

# 优化算法

## 1 梯度下降

- 目的是迭代地调整参数，使得损失函数最小化。
- 首先随机初始化（取一个随机的参数值），然后逐步改进，每一步都尝试降低损失函数，直到收敛到一个最小值（比如：MSE就是一个凸函数，有全局最小值）
- 梯度下降有一个重要的参数——每一步的步长
  - 取决于超参数**学习率（learning rate）**
    - 学习率太低：算法需要大量迭代才能收敛
    - 学习率太高：导致算法发散，越过最小值

### (1) 随机梯度下降——SGD

​	每一步在训练集中随机选择一个实例，并基于这个实例来计算梯度。



# 神经网络

- 使用**torch.nn包**来构建神经网络
- 神经网络取决于**自动微分autograd**来定义模型，区分这些模型
- **nn.Module**包含不同的**层layers**，使用函数**forward(input)**返回output结果

## 0 神经网络相关名词

### (1) epoch

- 当一个完整（有限）的数据集经过神经网络一次并返回一次，即为一个epoch
- 一般会设置多个epoch，在神经网络中传递数据集一次是不够的，需要在同一个网络中多次传递
- 随着epoch的数量增加，网络中的参数的更新次数也增加，曲线会由欠拟合变得过拟合，很难确定设置几个epoch是最合适的

### (2) batch与batch_size

- batch size是指一个batch中的样本数量
- 当数据集不能一次性通过神经网络时，需要将数据集划分为几个batch

## 1 一个典型的神经网络的训练过程

1. 定义一个包含可训练参数的神经网络
2. 迭代整个输入的数据集
3. 通过神经网络处理输入
4. 计算**损失(loss)**——output的结果与正确值之间的差距
5. 反向传播梯度到神经网络的参数
6. 更新网络的参数，典型的用一个简单的更新方法：weight = weight - learning_rate *gradient

## 2 定义神经网络

- 构建神经网络是要定义nn.Module的一个派生类
- 需要重写初始化\__init__()函数（相当于构造函数）和forward()函数
  - \__init__()函数中，需要调用父类（nn.module）的初始化函数
  - forward()函数，用来构建网络从输入到输出的过程

```python
import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self):
        # 调用父类的初始化函数
        super(MyNet,self).__init__()
        # Linear:线性变换函数
        self.fc1 = nn.Linear(100,200)
        self.fc2 = nn.Linear(200,10)
    def forward(self,x):
        x = self.fc1(x)
        # relu:线性整流函数
        x = F.relu(x)
        x = self.fc2(x)
        # softmax:柔性最大值传输函数
        x = F.softmax(x)
        return x
```

## 3 损失函数

- (output, target)作为输入
- 计算output和target之间的差距的估算值
- nn有不同的损失函数
  - 简单的：nn.MSELoss()——mean squared均方
  - nn.CrossEntropyLoss()——交叉熵

## 4 ~~反向传播~~

## 5 ~~更新参数~~

# 神经网络结构

## 1 卷积神经网络（Convolutional Neural Network）

### (1) 卷积层——CNN最重要的构建块

- 第一卷积层的神经元不会连接到输入图像中的每个像素，只与其接受视野内的像素相连接。（->**矩形局部接受野**，f_h 和 f_w分别表示接受野的高和宽）

- 连接方式

  - 相关名词
    - 零填充：在输入周围填充零
    - 步幅：两个相邻接受野之间的距离

  - 通过零填充，（步幅为1），使得连接的两个层有**相同的高和宽**

  <img src=".\Figures\zero_padding.JPG" style="zoom:67%;" />

  - 使用步幅**降低维度**：使得一个大的输入层连接到一个更小的层（s_h和s_w分别表示垂直和水平方向的步幅）

  <img src=".\Figures\reduce_dimension.JPG" style="zoom: 67%;" />

- **过滤器/卷积内核**
  
  - 神经元的权重用接受野的大小表示
  - 对输入图像进行过滤（卷积）得到**一个特征图**
  
- **多个特征图叠加**
  
  - 使用**三维**来表示多个特征图
  - 在同一个特征图中，所有的神经元共享参数，具有相同的参数（weight和bias）；不同的特征图可能有不同的参数
  - 卷积层对输入图像**使用多个过滤器**，使得它可以检测到输入的多个特征
  - 即：一个位于给定卷积层 ***l*** 的特征图 ***k*** 上的 ***i*** 行 ***j*** 列的神经元，与上一层 ***l - 1*** 的接受野内的神经元相连接，**并且穿过 *l - 1* 层中所有的特征图**。PS：***l*** 层的不同特征图上的 ***i*** 行 ***j*** 列的神经元都连接到上一层 ***l - 1*** 输出中完全相同的神经元。
  
- **卷积层中神经元的输出**——即位于该卷积层特征图***k***上的第***i***行***j***列的神经元的输出

  - 累加是：对 位于上一层***l-1***中<u>每一个特征图</u>上【该层 ***l*** 中（某一个特征图***k***上）第***i***行***j***列对应的**接受野（u行v列）**】的神经元的输出 进行权重卷积运算（权重指的是：***l***层中某一特征图 ***k*** 第i行j列的神经元和对应位于上一层特征图 ***k'*** 的u行v列个神经元的输入之间的连接权重）的<u>结果累加</u>

    ->相当于：对上一层 ***l-1*** 的每一个特征图上的连接到（接受野内）的神经元都使用该层 ***l*** 的某一个过滤器后叠加得到特征图 ***k***上该神经元的输出（后面加上了bias，相当于将特征图 ***k*** 调亮）

  - ***b_k***是：该卷积层 ***l*** 中特征图***k***的bias参数

$$
z_{i,j,k} = \sum^{f_h}_{u=1}\sum^{f_w}_{v=1}\sum^{f_{n'}}_{k'=1}x_{i',j',k'}·w_{u,v,k',k}+b_k
$$



<img src=".\Figures\multiple_feature_maps.png" style="zoom:67%;" />

### (2) 线性整流层

- 线性整流函数relu——激活函数（输出非负数）

$$
ReLU(x) = (x)^+ = max(0,x)
$$

### (3) 池化层/采样层

- 池化层是对输入图像进行**二次采样**，来减小计算负载、内存利用率和参数数量。
- 跟卷积层一样，每个神经元都会连接到上一层的接受野内的神经元，但**池化层神经元没有权重**，要做的是使用**聚合**函数（max，mean等）聚合输入

- 池化层类型
  
  - 最大池化max pooling（**最常见**）
    
    池化内核为2 x 2，步幅为2——每隔2个元素从图像中划分出2 x 2的区块，对每个区块中的4个元素取最大值
  
  <img src=".\Figures\max_pooling.JPG" alt="https://github.com/kuangbixia/DeepLearning/blob/master/max_pooling.JPG" style="zoom: 67%;" />
  
  - 平均池化mean pooling

### (4) 全连接层 Fully Connected Layer

- 相当于卷积层的特例，对输入进行一次卷积，把特征整合到一起，输出为一维，最后交给分类器或者进行回归
- 全连接：即每个神经元都与上一层所有神经元一一连接
- 线性变换函数 torch.nn.Linear()

$$
y = xA^T+b
$$



<img src=".\Figures\fully_connected_layer.JPG" style="zoom:67%;" />

### (5) 损失函数层



## 2 递归神经网络（Recurrent Neural Network）

# 深度学习是什么？——很深层的神经网络

- 提高模型容量的方法
  - 增加隐层数目
    - 神经元连接权 *w_i*  和阈值 *θ*  等参数会增多
  - 增加隐层神经元的数目
- 综述
  - 从增加模型复杂度的角度看，**增加隐层数目**比增加隐层神经元数目更有效
  - 不仅增加了拥有激活函数的神经元数目，还增加了激活函数嵌套的层数



# torch相关函数

- 类神经网络层——torch.nn
- 函数神经网络层——torch.nn.functional

## 1 二维卷积torch.nn.Conv2d()

<img src=".\Figures\convolution.JPG" style="zoom: 67%;" />

- 卷积结果的高和宽：

  - $$
    height\_3 = height\_1 - height\_2 + 1
    $$

  - $$
    width\_3 = width\_1 - width\_2 +1
    $$


```python
import torch

x = torch.randn(2,1,7,3)
'''
batch_size：一个batch中样例的个数	2
channels：通道数/当前层的深度		1
height_1：图片的高度		7
width_1：图片的宽度		3
'''
conv = torch.nn.Conv2d(1,8,(2,3))
'''
in_channels：通道数/当前层的深度		1
out_channels：输出的深度	8
kernel_size(int/tuple):
	height_2：过滤器的高度		2
	width_2：过滤器的宽度		3
'''
res = conv(x)
'''
batch_size：2
output：8
height_3：6
width_3：1
'''
```



## 2 线性变换函数torch.nn.Linear(in_features : int, out_features : int, bias : bool = True)

$$
y = xA^T+b
$$



## 3 线性整流函数torch.nn.functional.relu(input,inplace=False)/torch.nn.ReLU(inplace:bool=False)

$$
ReLU(x) = (x)^+ = max(0,x)
$$



## 4 柔性最大值传输函数torch.nn.functional.softmax(input,dim=None,_stacklevel=3,dtype=None)

$$
Softmax(x_i) = \frac{exp(x_i)}{\sum_jexp(x_j)}
$$



## 5 保存/加载模型

### (1) 整个网络

- 保存整个网络：

  ```python
  import torch
  
  net = MyNet()
  torch.save(net,PATH)
  ```

- 加载：

  ```
  torch.load(PATH)
  ```

### (2) 网络中的参数

- 保存网络中的参数：

  ```python
  import torch
  
  net = MyNet()
  torch.save(net.state_dict(),PATH)
  ```

- 加载：

  ```python
  net.load_state_dict(torch.load(PATH))
  ```



## 6 torch.max(input,dim)

### (1) 参数

- input：输入的Tensor
- dim：max函数索引的维度（0/1）
  - 0：每列的最大值
  - 1：每行的最大值

### (2) 输出

- Tensor：包含每行/列最大值的Tensor
- LongTensor：包含每行/列最大值对应的下标的Tensor

## 7 [torch.argmax(input,dim)](https://blog.csdn.net/weixin_42494287/article/details/92797061)

### (1) 参数

- input：输入的Tensor
- dim：减掉的维度

### (2) 输出

- LongTensor：跟max函数类似，输出压缩dim维度后的Tensor

## 8 torch.histc(input, bins, min,max)

### (1) 参数

- input：输入的Tensor
- bins：int，直方图箱的数量（长度，max-min+1）
- min：int，最小范围
- max：int，最大范围

### (2) 输出

- Tensor：用Tensor的形式返回直方图



# TensorBoard可视化

​	pytorch版本（1.0.1）太低，不能直接使用TensorBoard，使用了[TensorBoardX](https://github.com/lanpa/tensorboardX)，用法和[api文档](https://tensorboardx.readthedocs.io/en/latest/tensorboard.html#tensorboardX.SummaryWriter.add_images )跟TensorBoard差不多

## 1 建立TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

# log_dir是文件夹目录，默认为'runs'
writer = SummaryWriter(log_dir) # 'runs/fashion_mnist_experiment_1'
```

## 2 写入TensorBoard

```python
# 图片训练集
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 构建图像网格grid
img_grid = torchvision.utils.make_grid(images)

# 写入TensorBoard
writer.add_images('four_fashion_mnist_images', img_grid) # 可以不构建网格
```

## 3 观察model

```python
writer.add_graph(net, images)
writer.close()
```

## 4 添加投影Projector

将高维的数据用低维表示

```python
features = images.view(-1, 28*28)
writer.add_embedding(features,
                     metadata=class_labels,
                     label_img=images.unsqueeze(1)) # 解缩
writer.close()
```

## 5 跟踪模型

- 显示损失值变化

  ```python
  writer.add_scalar('training_loss',
                   running_loss/1000, # y 
                   epoch*len(trainloader)+i) # x
  ```

- 预测与实际的对比

  ```python
  writer.add_figure('predictions vs. actuals',
                   plot_classes_preds(net, inputs, labels), # 自定义函数：画图显示预测的结果和实际标签的对比
                   global_step=epoch*len(trainloader)+i)
  ```

## 6 评估模型

```python
writer.add_pr_curve(classes[class_index],
                   tensorboard_preds,
                   tensorboard_prods,
                   global_step=global_step)
writer.close()
```

# 应用

## 1 训练一个分类器

### (1) 步骤

1. 用**torchvision**<u>加载和规范化</u>数据集（训练集和测试集）
2. 定义一个**卷积神经网络**（CNN）
3. 定义**loss函数**
4. 用训练集训练神经网络
5. 用测试集测试神经网络

### (2) torchvision

- torchvision.datasets
- torchvision.transforms

### (3) DataLoader

- torch.utils.data.DataLoader

