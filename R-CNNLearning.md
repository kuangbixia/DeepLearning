# 语义分割，实例分割，全景分割

- 语义分割：对输入图像的每一个像素进行分类
  - 同一个类别的目标打上相同的标签
  - 模型：
    - FCN
    - DeepLabv1
    - DeepLabv2
    - DeepLabv3
    - DeepLabv3+
- 实例分割：实际上是目标检测和语义分割的结合
  - 检测目标的包围盒/边框
  - 同一类别的不同目标（相当于类的不同对象）打上不同的标签
  - 模型：
    - R-CNN
    - Fast R-CNN
    - Faster R-CNN
    - Mask R-CNN

- 全景分割：实例分割的升级
  - 不仅对图像中的目标进行检测和分割，也对背景进行检测和分割



# R-CNN(Regions with CNN features)：区域卷积神经网络

- 论文 [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524) 提出将**区域建议**和**CNN**结合，构建新的网络，叫R-CNN
  - 作者
    - Ross Girshick
    - Jeff Donahue
    - Trevor Darrell
    - Jitendra Malik

## 1 摘要&介绍

- 当时最好的目标检测方法是比较复杂的，将不同的低级别图像特征和高级别上下文结合。
- 本文提出了一种简单的可扩展的(scalable)检测算法

### (1) 两个关键点

1. 可以将**高容量的CNN**应用到**自底向上的区域建议**(region proposals) -> 用于对目标进行定位和分割
2. 当**标记的训练数据不足**时，对辅助任务进行有监督的预训练，随后对特定的区域进行微调 -> 提升性能

### (2) 目标检测的难题1——需要定位图像中的不同目标

- 经典：
  - 使用**SIFT**和**HOG**进行视觉识别
- 本文：
  - “ 和基于简单的类似HOG的系统相比，CNN在**目标检测**任务中可以带来更高的性能 ”

#### ① 把定位设计成回归问题

​	在实践中进展不好

#### ② 构建滑动窗口检测器——OverFeat

- 在网络的高层有五个卷积层，在输入图像上有较大的接受野和较大的步长，定位的精确性受到挑战
- OverFeat使用滑动窗口的CNN（sliding-window）来检测，是在ILSVRC2013数据集上有最好的性能的方法

#### ③ 使用区域来识别（√）——R-CNN（如下图）

​	对目标检测和语义分割都很成功，和OverFeat相比，性能更高

- 对输入图像生成**约2000个**与类别无关的**区域建议**（自底向上）
  - 区域建议：定义了用于检测器的候选检测集
- 对每一个区域，使用**大型的CNN**，提取**固定长度**的特征向量
- 使用与类别相关的**线性SVM**（支持向量机），对每一个区域进行分类
  - 点乘计算转换成矩阵乘积
    - 特征矩阵2000x4096
    - SVM权重矩阵4096xN（类别数量）
- 输出2000xN的分数后，应用**贪婪的非最大抑制（NMS, non-maximum suppression）**算法 -> 去掉重复的区域
  - 对**每一个类别**，进行NMS
  - 如果当前区域和一个**分数更高的区域**的IoU的值大于阈值，则去掉当前区域（因为它更可能是另一个类别）

<img src="./Figures/R-CNN/R-CNN_overview.jpg" style="zoom:67%;" />

### (3) 目标检测的难题2——标记的数据不多，不足以训练一个大型的CNN

#### ① 传统：无监督的预训练+有监督的微调

#### ② 本文：有监督的预训练+特定区域的微调

- 在大的辅助数据集ILSVRC上进行有监督的预训练
- 在小的数据集PASCAL上进行特定区域的微调



## 2 R-CNN目标检测

### (1) 模块的设计

#### ① 生成区域建议

- 近年有多种方法生成与类别无关的区域建议：
  - objectness
  - selective search 选择性搜索
  - category-independent object proposals 与类别无关的目标建议
  - constrained parametric min-cuts 最小切割
  - multi-scale combinatorial grouping 不同尺度组合分组
- 本文R-CNN采用**selective search**

#### ② 提取特征

- 将**区域内的图像数据**转换成与CNN兼容的格式 -> 227x227的大小（注：区域是**任意大小**的）

   - 在变换前，**扩展包围盒**，使得在变换后的尺寸中有p个像素是在原包围盒内的（采用p=16）

   - **三种变换方法**（如下图）：

     - **warp(D)**：直接将包围盒内的像素缩放变换到指定的大小（不考虑方向比例）
     - **tightest square with context(B)**：包含区域周围的原图像数据（考虑方向比例）
     - **tightest square without context(C)**：不包含区域周围的原图像数据（考虑方向比例）

     <img src=".\Figures\R-CNN\object_proposal_transformations.JPG" style="zoom:67%;" />

- 特征的计算：227x227的图像 经过5个卷积层和2个全连接层

  - 使用Caffe框架（深度卷积网络实现）提取**4096维的特征向量**

### (2) 运行时间的分析

​	使得检测高效的两个属性：

- CNN的所有**参数**在各个类别中**共享** -> 可以让各个类别分摊（计算区域建议和特征）花费的时间
- 和其他常用方法相比，CNN计算的**特征向量维度更低** -> 占用内存小

### (3) 训练

#### ① 有监督的预训练

- 使用大型的辅助数据集ILSVRC2012（图像级别的分类）
- 采用开源的Caffe CNN库
- 有区别地预训练CNN

#### ② 特定区域的微调

- 不改变ImageNet的架构，仅将最后1000路的分类器替换成N+1路的分类器（N=类别数量，**1=背景**）
  - PASCAL VOC中，N=20
  - ILSVRC2013中，N=200

#### ③ 正类和负类

- **正类**：区域建议和ground truth上的框的IoU>=0.5，判为属于ground truth上该框的类别
- **负类**：区域建议和ground truth上的框的IoU<0.5，判为不属于ground truth上该框的类别

#### ④ 目标类别分类器

- 举例：当考虑某一类别（如，car车）时，如果当前区域紧密围绕着一个car（有很大的IoU)，则判为正类；反之，（如，背景）几乎和car没有什么关系（有极小的IoU），则判为负类
- 设置**阈值**，当IoU>=阈值，判为正类；否则，判为负类

#### ⑤ hard negative mining 困难负例挖掘

### (4) 结果

​	比较指标**mAP**

#### ① 数据集PASCAL VOC 2010-12

​	**R-CNN BB(bounding-box regression)** > **R-CNN** > SegDPM > UVA

#### ② 数据集ILSVRC 2013

​	**R-CNN BB** > OverFeat(2) > UvA-Euvision > NEC-MU >OverFeat(1)



## 3 网络架构，可视化，消融，误差

### (1) 网络的架构

​	实验表明，O-Net的mAP比T-Net更高，但耗时更长

#### ① T-Net: TorontoNet

​	以上的实验基本采用T-Net

#### ② O-Net: OxfordNet——应用到R-CNN中

- 13个3x3卷积层 + （5个最大池化层） + 3个全连接层

- 从Caffe Model Zoo中下载了**VGG_ILSVRC_16_layers**模型的预训练网络权重
- 微调网络：使用小的批量（24个样本）-> 适应GPU内存

### (2) 将学习的特征可视化

- 将（最后一个卷积层后的）**pool5**的单元可视化

- pool5的单元的接受野是195x195
- pool5输出的特征图维度是256x6x6

### (3) 消融研究（Ablation Studies）

- fc6连接在pool5后，权重矩阵为4096x9216(9216=256x6x6)维，和pool5输出的特征图相乘后得到4096维
- fc7连接在fc6后，权重矩阵为4096x4096维

#### ① 不采用微调

​	实验表明，fc7产生的特征比fc6效果差（mAP更低），甚至可以去除fc6和fc7来减少网络的参数

#### ② 采用微调

​	实验表明，对fc6和fc7的提升很显著

### (4) 包围盒回归

- 为了减少定位的误差，使用增加一个简单的包围盒回归阶段，训练一个线性回归模型来预测新的检测窗口

- 经过selective search区域建议和SVM分类计算分数后，使用与类别相关的包围盒回归器来预测一个新的用于检测的包围盒
- 实验表明，该方法修正了大量的错误定位检测窗口/包围盒

## 4 ILSVRC2013检测数据集

### (1) 数据集概述

- train（395，918），val（20，121），test（40，152）
- 图像的大小不等

### (2) 区域建议（Region Proposals）

- 采用selective search
- 在执行selective search之前要将输入图像变换到固定的大小

### (3) 训练数据

​	R-CNN有三个模块需要训练数据：

- CNN 微调：val_1+train_N
  - 50k次SGD迭代
- 检测器SVM训练：val_1+train_N
- 包围盒回归器训练：val_1



## 5 R-CNN和OverFeat的关系

- OverFeat可以看作是R-CNN的特例
- OverFeat的滑动窗口不需要变换大小，**OverFeat比R-CNN的速度快了9倍**



# Fast R-CNN

- 论文 [Fast R-CNN](https://arxiv.org/abs/1504.08083)

# Faster R-CNN

- 论文 [Fatser R-CNN:Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

# Mask R-CNN

- 论文 [Mask R-CNN](https://arxiv.org/abs/1703.06870)
