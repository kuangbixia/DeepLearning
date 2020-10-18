# 学习论文 [Mask R-CNN](https://arxiv.org/abs/1703.06870)

## 摘要&介绍

### 1 Mask R-CNN

- Mask R-CNN检测图像中的目标，同时给每个实例生成高质量的分割掩码mask
- 在Faster R-CNN的基础上增加一个**并行分支**，用来**预测目标的掩码mask**
  - 在每个RoI（Region of Interest）上添加端到端预测掩码的分支
  - 新增的掩码分支是一个简单的FCN
- Faster R-CNN对输入输出图像没有设计成像素到像素对齐——RoI Pool -> 本文提出了RoI Align，修复错位，保留了精确的空间位置
  - RoI Align将mask精度提高了10%-50%
- 本文发现，将**mask预测和类别预测解耦**是很重要的
  - 对每个类别分开独立预测二元mask
  - 通过网络中的RoI分类分支来预测类别

### 2 实例分割的困难

- 实例分割与语义分割的不同在于，需要检测图像中的每个目标，还要分割每个实例
- 结合了计算机视觉两个任务：目标检测和语义分割



## 相关工作

### 1 R-CNN

- R-CNN的包围盒目标检测，生成可管理数量的候选目标区域，并且在每个RoI上通过CNN
- Fast R-CNN扩展了R-CNN，使用**RoIPool**来获取全图上所有RoI的特征
- Faster R-CNN使用**RPN**（Region Proposal Network）学习**注意力机制**
- 本文Mask R-CNN提出**掩码mask和标签的并行预测**



## Mask R-CNN

### 1 Faster R-CNN

​	Faster R-CNN对每个候选目标有两部分输出：类别标签和包围盒偏移

- 第一阶段：**RPN**，生成候选的目标包围盒
- 第二阶段：**Fast R-CNN**，使用RoIPool从每个候选包围盒RoI中提取特征，并进行分类和包围盒回归

### 2 Mask R-CNN

<img src=".\Figures\R-CNN\Mask R-CNN.JPG" style="zoom:67%;" />

​	在Faster R-CNN的基础上增加第三个分支：mask分支 -> 输出目标mask

- 增加第三个分支，与Faster R-CNN预测类别和包围盒偏移分支**并行**，对每个RoI输出一个**二元mask**

### 3 Mask表示

​	类别标签和包围盒偏移的输出是一个短向量，而mask需要通过卷积得到**像素到像素**的空间结构的输出

- 这种像素到像素的形式**要求每个RoI上的特征能很好的对齐**，以保证每个像素的空间关系 -> 产生了RoIAlign

### 4 RoIAlign

#### (1) RoIPool

- RoIPool是对每个RoI提取小特征图的标准操作
- RoIPool首先将**浮点数**RoI量化（即取整）成特征图的离散值，之后将量化后的RoI细化为**空间面元（spatial bins）**，最后（通常使用**最大池化**）整合被每个**面元**覆盖的特征值
- 量化带来了RoI和提取的特征**不对齐**的结果，不影响分类，但对像素到像素的mask预测有很大的负面影响

#### (2) RoIAlign

​	为了解决RoIPool带来的RoI和特征不对齐的问题，提出了RoIAlign，移除粗糙的量化，让提取的特征和输入保证对齐

- 使用双线性插值计算每个RoI面元中，输入特征的四个规则采样位置上的值
- （使用最大或平均池化）整合结果

### 5 网络架构

​	分为两部分：

1. 卷积backbone架构：用于对全图提取特征
2. 网络head：用于包围盒识别（即分类和回归），以及对每个RoI进行的mask预测

#### (1) backbone架构

- ResNet-50，ResNet-101

- ResNeXt-50，ResNeXt-101

- 其他：

  - 在Faster R-CNN中，采用了ResNet第四个阶段的最后一个卷积层（ResNet-50-C4）

  <img src=".\Figures\Mask R-CNN\ResNet-C4.JPG" style="zoom:67%;" />

  - 提出使用**ResNet-FPN**（Feature Pyramid Network），应用在Mask R-CNN中提取特征会有很大的提高

  <img src=".\Figures\Mask R-CNN\ResNet-FPN.JPG" style="zoom:67%;" />

#### (2) head

- 在Faster R-CNN的基础上，增加一个分支：用于预测mask的全卷积层（如上图所示）

### 6 损失函数

$$
L=L_{cls}+L_{box}+L_{mask}
$$



### 7 实现细节

- 跟之前的版本一样，当IoU与GT的IoU大于0.5，即为正的RoI；而mask损失值只考虑正的RoI
- 采用以图像为中心的训练：
  - 图像调整大小，短边为800像素
  - 每个图像有N个RoI（ResNet-50-C4中N设为64，ResNet-FPN中N设为512），其中正和负样本比例为1：3
- RPN的anchor有5种尺寸，3种空间比例，为了方便**消融**，RPN（Region Proposal Network）和Mask R-CNN分开训练，且不共享特征。但在本文中，**RPN和Mask R-CNN采用同一个backbone，它们共享权重**。

