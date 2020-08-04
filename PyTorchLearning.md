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

### (3) 其他功能

```python
import torch

x = torch.rand(4,4)

# 获取张量的形状，返回的是一个元组——Tensor.size()
print(x.size())
'''
torch.Size([5,3])
'''

# 重新设置张量的形状——Tensor.view(x,y)
y = x.view(16)
z = x.view(2,8)

# 查看张量的大小——Tensor.item()
x = torch.randn(1)
print(x.item())
```

## 2 Tensor和NumPy的转换

### (1) CPU下，他们共享物理地址，改变其中一个，另一个也会随之改变

#### ① Tensor到NumPy

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

#### ② NumPy到Tensor

```python
import numpy as np
import torch

n = np.ones(5)
t = torch.from_numpy(n)
```

### (2) GPU下——.to(device)，还可以设置参数改变数据类型

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

### (2) 属性Tensor.grad_fn

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
# PS：y不再是只包含一个标量（像是一个向量），调用.backward()时要设置向量作为参数
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

## 1 一个典型的神经网络的训练过程

1.定义一个包含可训练参数的神经网络

2.迭代整个输入

3.通过神经网络处理输入

4.计算**损失(loss)**

5.反向传播梯度到神经网络的参数

6.更新网络的参数，典型的用一个简单的更新方法：weight = weight - learning_rate *gradient

## 2 定义神经网络

