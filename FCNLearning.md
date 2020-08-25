# 学习论文 [Fully Convolutional Networks for Semantic Segmentation 基于全卷积网络的语义分割](https://arxiv.org/abs/1411.4038 "Title")

## 摘要

### 1 卷积网络

- 是强大的视觉模型
- 可以产生多层次的特征

### 2 全卷积网络FCN

#### (1) 重点

- 端到端，像素到像素地训练
  - 在语义分割领域上超越了最先进的技术
- 接受任意大小的输入，经过推理和学习，产生相应大小的输出

<img src=".\Figures\FCN\FCN_figure1.JPG" style="zoom:67%;" />

#### (2) 内容

- 定义并描述**全卷积网络**，将分类转移到**密集预测（像素到像素）**
- 将现有的**分类网络**调整为**FCN**
  - 现有分类模型
    - AlexNet
    - VGG net
    - GoogLeNet
  - 采用数据集
    - PASCAL VOC
    - NYUDv2
    - SIFT Flow
- 通过**微调（fine-tuning）**将学习到的参数转移到分割任务上（迁移学习？）
- **定义一个新颖的“跳跃”结构**
  - 解决局部信息随着网络加深而丢失的问题（将浅层信息备份）
  - 将来自深的、粗糙（图像小，物体的空间信息比较丰富）的层的语义信息 和 来自浅的、细致（图像大，物体的几何信息比较丰富）的层的表征信息结合



##  介绍

### 1 卷积网络

#### (1) 已经实现的语义分割

​	每个像素都被标记为属于 包含它的目标或区域 的类别

#### (2) 从粗糙到细致的预测的改进

​	对每个像素进行预测

### 2 端到端、像素到像素的全卷积网络FCN

- 有监督的预训练（supervised pre-training）
- 像素预测（pixelwise prediction）



## 相关工作

### 1 深度分类网络

- AlexNet
- VGG net
- GoogLeNet

### 2 迁移学习

- 视觉识别
- 检测
- 实例分割和语义分割

### 3 FCNs历史

#### (1) 具有全卷积推理和学习的检测

- Matan首次将卷积网络LeNet扩展为任意大小的输入（数字串的识别）
- Wolf&Platt将卷积网络扩展为2维图像的输出

#### (2) 具有全卷积推理

- Ning粗糙的多类别分割
- Sermanet滑动窗口检测
- Pinheiro&Collobert语义分割
- Eigen图像恢复

#### (3) 具有全卷积训练

- Tompson姿势估计

### 4 使用卷积网络进行密集预测

#### (1) 历史工作

- Ning语义分割
- Ciresan边界预测
- Eigen图像恢复和深度估计

#### (2) 历史例子的特点

- 限制容量和接受野
- patchwise拼凑式训练
- 输入移位-输出交错等

### 5 在深度分类架构上扩展

#### (1) 类似的历史方法——不是端到端学习

- 在一个混合模型中，采用了深度分类网络来做语义分割
- 用采样边界盒/区域来微调一个R-CNN

#### (2) 本文——端到端

- 使用图像分类作为**有监督的预训练**
- 通过 微调全卷积 简单高效地学习输入的**整个图像**



## FCNs

### 1 使用分类器（图像级别）进行密集预测（像素级别）

#### (1) 经典识别网络（全连接层）特点

​	全连接层 要求**固定大小的输入**，且产生非空间**（一维）的输出**

#### (2) FCN：全连接层 -> 卷积层

- 可以输入任意大小的图像并产生相应大小的输出
- 产生问题：
  - 输出的维度会因为二次采样不断降低，图像越来越小
  - 需要将图像进行不断放大到原图像的大小

### 2 粗糙的输出->精确的密集预测

#### (1) 移动和缝合shift-and-stitch（没有采用）

- 不插值
- 输入移位+输出交错

#### (2) 网络内向上采样upsampling（即反卷积deconvolution）

- 插值
  - 简单的双线性插值仅依赖于输入和输出神经元的相对位置
- 反卷积操作简单
  - 只要将卷积的向前和向后传递做反向处理
- 向上采样应用在网络内端到端的学习（从像素到像素的损失开始反向传播）
- 反卷积的过滤器不需要固定，可以通过学习得到

### 3 patchwise拼凑式（小块）训练 vs 整个图像的全卷积训练

<img src=".\Figures\FCN\FCN_figure5.JPG" style="zoom:67%;" />

#### (1)（采样）patchwise训练（没有采用）

- 可以减少图像的冗余信息
- 常用来解决图像的空间相关性
- 对过多的图像需要花费更多时间收敛

#### (2)（整个图像）全卷积训练（采用）

- 对loss加权 -> 解决类不平衡性
- 对loss采样 -> 解决输入图像的空间相关性



## 分割架构

### 1 从分类器到密集FCN

- 丢弃每个网络的分类器层
- 将全连接层替换为卷积层

<img src=".\Figures\FCN\FCN_figure2.JPG" style="zoom:67%;" />

### 2 改进预测输出的精度

#### (1) 层融合——建立连接将最后的预测层与较低层，以合适的步长结合

![](.\Figures\FCN\FCN_figure3.JPG)

- 对较浅层（比如pool4，1/16）添加1x1的卷积层 -> 产生类别预测（1/16）
- 对较深层（比如conv7，1/32）添加2x的向上采样层 -> 产生类别预测（1/16）
- 将两个类别预测（1/16）相加（融合）后，向上采样/反卷积（步长16）-> 产生跟原图像一样大小的预测图（FCN-16s）
- 实验表示，继续到FCN-8s（与pool3融合）会使结果有所改进，但不需要继续与更浅层（pool2、pool1）融合，改进不大

#### (2) 其他方法

- 减小池化层的步长 
  - 需要增大卷积核
- shift-and-stitch 
  - 改进不如层融合

### 3 实验框架

#### (1) 优化

​	SGD

#### (2) fine-tuning微调

​	通过整个网络的反向传播对所有层进行微调

#### (3) patch sampling小块采样（不必要）

​	对过多的图像需要花费更多时间收敛

#### (4) 类别平衡（不必要）

- 类别不平衡——有3/4的部分是背景
- 全卷积训练可以通过**对loss进行加权和采样**来平衡

#### (5) 密集预测（像素到像素）——通过向上采样upsampling

- 最后一层的反卷积过滤器固定为 双线性插值
- 中间的向上采样层，初始化为双线性向上采样，之后学习得到

#### (6) 在每个方向预测为32像素（没明显改进）

#### (7) 使用更多的训练数据（有改进）



## 结果

​	使用PASCAL数据集训练效果最好，FCN特点：

- 可以恢复精细的结构（第一行）
- 可以分离紧密相交的物体（第二行）
- 受遮挡物的影响小（第三行）

<img src=".\Figures\FCN\FCN_figure6.JPG" style="zoom:67%;" />



# 细化[FCN](https://arxiv.org/abs/1411.4038)学习

## 1 FCN结构

### (1) VGG16结构

<img src="./Figures\FCN/VGG16.jpg" style="zoom:67%;" />

- 前部分的卷积层（包含ReLU）和池化层

  - 最大池化层：使用2x2的池化核，步长为2，使得图像缩小一半
  - 卷积层：使用3x3的卷积核，步长为1，填充设为1，使得经过卷积层时输出图像和输入图像保持相同的高和宽
    - 使用多个卷积层，小卷积核，减少参数
    - 如下面的代码段中cfg，分别是两个输出通道为64 ->（经过最大池化层）-> 两个输出通道为128 ->（经过最大池化层）-> 三个输出通道为256 ->（经过最大池化层）-> 三个输出通道为512 ->（经过最大池化层）-> 再接着三个输出通道为512的卷积层

  ```python
  cfg = {
      # ...
      'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
      # ...
  }
  
  def make_layers(cfg, batch_norm=False):
      layers = []
      in_channels = 3
      for v in cfg:
          if v == 'M':
              layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
          else:
              conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
              if batch_norm:
                  layers += (conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True))
              else:
                  layers += [conv2d, nn.ReLU(inplace=True)]
              in_channels = v
      return nn.Sequential(*layers)
  ```

- 后部分的全连接层（分类器）

  - 先经过一个平均池化层，将图像输出为7x7的大小
    - VGG16要求输出图像大小固定为224x224的RGB图像（3通道）
    - 通过上面所述的卷积层和池化层后，在最后一个池化层输出图像刚好是7x7的大小
  - 最后是三个全连接层

  ```python
  self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
  self.classifier = nn.Sequential(
      nn.Linear(512 * 7 * 7, 4096),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(4096, num_classes)
  )
  ```

### (2) 从VGG16到FCN

![](./Figures\FCN/FCN_figure3_notes.JPG)

- 沿用了VGG16的前部分——卷积层和池化层

  - 卷积层：如上图conv1-conv5
  - 池化层：如上图pool1-pool5

- 将全连接层替换为卷积层

  - pool5后的卷积层（512通道 -> nclass通道，不改变图像大小）——_FCNHead()

    - 第一个卷积层：卷积核为3x3，步长为1，填充设为1，不改变输入图像的宽和高，且不改变通道数量
    - 第二个卷积层：卷积核为1x1，步长为1，无填充，不改变输入图像的宽和高，输出通道数量为类别数量

    ```python
    class _FCNHead(nn.Module):
        def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, **kwargs):
            super(_FCNHead, self).__init__()
            inter_channels = in_channels // 512
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                norm_layer(inter_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, channels, 1)
            )
    ```

  - pool4后的卷积层（512通道 -> nclass通道，不改变图像大小）——score_pool4()

    - 卷积核为1x1，步长为1，无填充不改变输入图像的宽和高，输出通道数量为类别数量

    ```python
    self.score_pool4 = nn.Conv2d(512, nclass, 1)
    ```

  - pool3后的卷积层（256通道 -> nclass，不改变图像大小）——score_pool3()

    - 卷积核为1x1，步长为1，无填充不改变输入图像的宽和高，输出通道数量为类别数量

    ```python
    self.score_pool3 = nn.Conv2d(256, nclass, 1)
    ```

- 层融合

  - 通过反卷积（向上取样）放大图像

    - 使用双线性插值
    - 使得来自不同层的两个图像大小一致，进行融合 / 最后输出原图像大小的图像

  - 从后往前逐层融合

    - fcn32s

      <img src="./Figures\FCN/FCN32s_train.JPG" style="zoom:67%;" />

    - fcn16s

      <img src="./Figures\FCN/FCN16s_train.JPG" style="zoom:67%;" />

    - fcn8s

      <img src=".\Figures\FCN\FCN8s_train.JPG" style="zoom:67%;" />

- FCN vs VGG16

  - VGG16在经过一系列的卷积层（和池化层）后，输出512通道7x7大小的图像，需要经过全连接层（分类器），产生一维（nclass大小）的输出
    - 分类器需要考虑到输入图像的每一个像素，产生分类输出
    - 因此，VGG16要求固定大小图像的输入，来确定到达全连接层的图像大小（像素个数），确定全连接的输入参数
    - VGG16要求输入224x224的RGB图像，经过五个池化层后，缩小到7x7（512通道，通道数量由卷积层决定），所以分类器的输入是512x7x7（固定的值）
  - FCN将全连接层替换为卷积层后，可以接受任意大小的图像的输入，通过反卷积可以产生相应大小的输出
    - FCN由各个卷积层和池化层构成
      - 卷积层通过设置3x3的卷积核，步长为1，填充为1，保证输出图像不改变大小，只关心输入输出通道（和图像大小无关）
      - 每个池化层只对输入图像缩小一半，对任意大小的输入图像都可用
    - 只要在反卷积时对图像依次放大到前一层输出图像的大小，最后直接放大到原图像的大小，即可输出相应大小的图像，所以不需要考虑输入图像的大小

## 2 loss函数和优化

### (1) loss函数——MixSoftmaxCrossEntropyLoss

- 实际上采用了交叉熵损失函数F.cross_entropy()

- 对output上每个像素预测的类别和target上的像素类别比较，计算loss值

  - $$
    \begin{align}
    &loss(i)=-\sum^K_{k=1}y_k^{(i)}log(\hat{p}_k^{(i)})\\
    &设有K个类别；\\
    &当target上第i个像素的类别是k时，y^{(i)}_k=1，否则为0;\\
    &\hat{p}^{(i)}_k表示第i个像素属于类别k的概率
    \end{align}
    $$

- 整个output的loss值即为它上面所有像素的loss值的平均值

  - $$
    loss(output)=\frac{1}{m}\sum_{i=1}^mloss(i)
    $$

### (2) 优化

- 采用随机梯度下降SGD
- 实验中学习率设为0.0001



## 3 量化评估指标

### (1) IoU/IU 交并比(Intersection over Union)

$$
IoU=\frac{target \cap prediction}{target \cup prediction}
$$

- 基于类进行计算，将每一类的IoU计算后，累加计算平均值 -> mean IoU均交并比

  - $$
    MIoU=\frac{1}{k+1}\sum_{i=0}^k\frac{n_{ii}}{\sum_{j=0}^kn_{ij}+\sum_{j=0}^kn_{ji}-n_{ii}}
    $$

  - $$
    其中，n_{ii}表示target中类别为i的像素预测为类别i的像素的个数\\
    \sum_{j=0}^kn_{ij}相当于target中类别为i的面积（单个类别的area\_lab），\\
    \sum_{j=0}^kn_{ji}相当于prediction中类别为i的面积（单个类别的area\_pred），\\
    n_{ii}相当于它们相交的面积（单个类别的area\_inter）
    $$

  - ```python
    def batch_intersection_union(output, target, nclass):
        """mIoU"""
        # inputs are numpy array, output 4D, target 3D
        mini = 1
        maxi = nclass
        nbins = nclass
        predict = torch.argmax(output, 1) + 1
        target = target.float() + 1
    
        predict = predict.float() * (target > 0).float()
        intersection = predict * (predict == target).float()
        # areas of intersection and union
        # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
        area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
        area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
        area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
        area_union = area_pred + area_lab - area_inter
        assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
        return area_inter.float(), area_union.float()
    
    # 返回了统计了每个类别的像素的个数的Tensor
    inter, union = batch_intersection_union(pred, label, self.nclass)
    self.total_inter += inter
    self.total_union += union
    
    IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
    mIoU = IoU.mean().item()
    ```

    

### (2) pixcal accuracy 像素精度(PA)

$$
PA=\frac{\sum_{i=0}^{k}n_{ii}}{\sum_{i=0}^k \sum_{j=0}^kn_{ij}}\\
其中，n_{ii}表示target中类别为i的像素预测为类别i的像素的个数（correct）\\
\sum_{j=0}^kn_{ij}相当于target中类别为i的像素个数（labeled）
$$

```python
def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    predict = torch.argmax(output.long(), 1) + 1
    target = target.long() + 1

    pixel_labeled = torch.sum(target > 0).item()
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled

# 返回像素个数
correct, labeled = batch_pix_accuracy(pred, label)

self.total_correct += correct
self.total_label += labeled

pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)  # remove np.spacing(1)
```



  