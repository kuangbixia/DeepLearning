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

# 查看张量的大小——Tensor.item()
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

# 神经网络

- 使用**torch.nn包**来构建神经网络
- 神经网络取决于**自动微分autograd**来定义模型，区分这些模型
- **nn.Module**包含不同的**层layers**，使用函数**forward(input)**返回output结果

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

## 4 ~~反向传播~~

## 5 ~~更新参数~~

# 神经网络相关函数

- 类神经网络层——torch.nn
- 函数神经网络层——torch.nn.functional

## 1 二维卷积torch.nn.Conv2d()

- 卷积结果的计算：
  - 
    $$
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

## 2 线性变换torch.nn.Linear(in_features : int, out_features : int, bias : bool = True)

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

  ```
  net.load_state_dict(torch.load(PATH))
  ```

  

# 神经网络结构

## 1 卷积神经网络（Convolutional Neural Network）

- 卷积层
- 线性整流层
  
  - 线性整流函数relu
- 池化层
  - 最大池化max pooling（**最常见**）
    
    - 最常用的池化层：池化窗口为2 x 2，步幅为2——每隔2个元素从图像中划分出2 x 2的区块，对每个区块中的4个元素取最大值
    
    ![../DeepLearning/max_pooling.JPG](..\DeepLearning\max_pooling.JPG)
  - 平均池化
- 完全连接层
- 损失函数层

## 2 相关名词

### (1) epoch

- 当一个完整（有限）的数据集经过神经网络一次并返回一次，即为一个epoch
- 一般会设置多个epoch，在神经网络中传递数据集一次是不够的，需要在同一个网络中多次传递
- 随着epoch的数量增加，网络中的参数的更新次数也增加，曲线会由欠拟合变得过拟合，很难确定设置几个epoch是最合适的

### (2) batch与batch_size

- batch size是指一个batch中的样本数量
- 当数据集不能一次性通过神经网络时，需要将数据集划分为几个batch

# TensorBoard可视化

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
writer.add_images('four_fashion_mnist_images', img_grid)
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