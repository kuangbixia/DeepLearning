# 学习论文[DeepLabv3+:Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)

- 作者（2018）
  - Liang-Chieh Chen
  - Yukun Zhu
  - George Papandreou
  - Florian Schroff
  - Hartwig Adam



## 摘要&介绍

### 1 主要内容

#### (1) 肯定了SPP和Encoder-Decoder模块在深度的语义分割网络中的作用

- SPP：**编码不同尺度的上下文信息**
- Encoder-Decoder：**恢复清晰的物体边界**

#### (2) 提出将**SPP**和**Encoder-Decoder**结合

- 在DeepLabv3的基础上**增加一个简单的decoder解码器**模块，来改善分割结果，特别是**恢复物体的边界（边缘）**
  - 在DeepLabv3中，通过空洞卷积控制（encoder）输出的特征的密度，使得丰富的语义信息已经被编码到DeepLabv3的输出
  - 现在要增加decoder模块，来恢复物体的清晰的边界
- 提出全新的Encoder-Decoder结构：DeepLabv3作为encoder模块 + 简单的decoder模块
  - 与已有的Encoder-Decoder的不同：DeepLabv3+的encoder可以通过空洞卷积控制提取的特征的分辨率，权衡特征密度和运行时间
- 如下图a（ASPP，DeepLabv3）和b（Encoder-Decoder）结合构建c

<img src=".\Figures\DeepLabv3+\DeepLabv3+_model.JPG" style="zoom:67%;" />

#### (3) 进一步

- 调整Xception模型，应用到分割任务
- 将**深度可分的卷积**应用到ASPP和decoder解码器模块中
- 构成更快速更健壮的Encoder-Decoder网络



## 相关工作

​	主要讨论使用SPP和Encoder-Decoder结构的模型

### 1 SPP:Spatial Pyramid Pooling

​	PAPNet和DeepLab都是通过**利用不同尺度的信息**，生成更好的结果

#### (1) PSPNet（金字塔场景解析网络）

​	在多个网格尺度上进行**空间金字塔池化**

#### (2) DeepLab

​	构建**并行**的多个具有不同采样率的**空洞卷积层**（即ASPP）

### 2 Encoder-Decoder

#### (1) 结构

- encoder
  - 逐渐减小特征图，并捕获更高级的语义信息
- decoder
  - 逐渐恢复空间信息（增大分辨率）

#### (2) 用途

​	把DeepLabv3的模型作为encoder模块，然后增加一个简单的decoder模块来恢复边界，获得更清晰的分割。

### 3 Depthwise separable convolution 深度可分卷积

#### (1) group convolution 分组卷积

​	对输入的特征图进行分组，分别对每一组进行卷积。（如下图右，图源网络）

<img src=".\Figures\DeepLabv3+\group_convolution.png" style="zoom:67%;" />

- 假设输入特征图尺寸为C\*H\*W，输出特征图有N通道，即有N个过滤器，每个过滤器的尺寸为C\*K\*K，分成G个组进行分组卷积
- 则对G组的每一组：输入特征图尺寸为C/G\*H*W，输出特征图通道数为N/G，过滤器的个数为N/G，每个过滤器的尺寸为C/G\*K\*K
- 因此，总参数量由原本的N\*(C\*K\*K)减少到G\*(N/G*(C/G\*K\*K))，即减少到1/G倍

#### (2) depthwise convolution 深度卷积

​	当分组数量G=输入特征图通道数C，把每个输入特征图作为一组，分别卷积，生成一个输出特征图，即总的输出特征图通道数=输入特征图通道数C，则总参数量减少到1/N倍。（如下图，图源网络）

<img src=".\Figures\DeepLabv3+\depthwise_separable_convolution.png" style="zoom:67%;" />

- 减少参数量
- 减少计算成本
- 加快计算速度

#### (3) pointwise convolution 逐点卷积

​	逐点卷积的意思是遍历整个图像（C个通道）的每一个像素，过滤器的尺寸为C\*1\*1。



## 方法

### 1 具有空洞卷积的Encoder-Decoder

#### (1) 空洞卷积

- 作用

  - 控制DCNN计算的特征图的分辨率
  - 调整过滤器的接受野（调整dilation rate的值 r ），来捕获多尺度信息

- 输出特征图 y 上位置 i 的计算如下：

  - $$
    y[i]=\sum_kx[i+r*k]w[k]
    $$

#### (2) 深度可分卷积

- **深度可分卷积**由**深度卷积**和**逐点卷积**组成（如下图a和b）
- Tensorflow版本实现的深度可分卷积中，将空洞卷积应用到深度卷积（如下图c），叫**空洞可分卷积**（atrous separable convolution）

<img src=".\Figures\DeepLabv3+\atrous_separable_convolution.JPG" style="zoom:67%;" />

#### (3) DeepLabv3作为encoder编码器

- DeepLabv3采用**空洞卷积**提取任意分辨率的特征图（由DCNN计算）
- **output stride输出步长** = input（原图像）分辨率/output（最后的输出，全连接层前）分辨率
  - 如，图像分类中（VGG16），output stride为32
  - 在DeepLabv3中提到，为了提取更密集的特征，可以调整output stride的值。如，output stride=8，移除了DCNN最后两个block的下采样（将卷积的stride从2改为1），对最后的两个block采用空洞卷积（rate=2&rate=4）
- DeepLabv3中，优化了ASPP模块：通过采用多个不同采样率的空洞卷积提取不同尺度的特征，此外增加了图像级别的特征（用于解决当rate足够大时，过滤器的有效权重变少）来增强结果
- 将DeepLabv3最后的特征图作为Encoder-Decoder结构的encoder输出，有256通道且包含丰富的语义信息（如下图）
  - DeepLabv3输出的编码器特征图，output stride=16

<img src=".\Figures\DeepLabv3\ASPP.JPG" style="zoom:67%;" />

#### (4) 提出的decoder解码器

- 在DeepLabv3中，用output stride=16计算特征（如上图），最后采用简单的双线性上采样（x16）恢复分辨率，可以看作最原始的decoder模块。
  - 缺点：这个原始的decoder并不能成功恢复物体分割的细节（边界）

- 提出了一个简单且有效的decoder（如下图）

  1. 首先对encoder的特征图通过双线性插值上采样（x4，1/16->1/4）
  2. 对DCNN中低层输出的同样分辨率的特征图进行逐点卷积（改变通道数，=encoder输出的通道数256）
     - 如，ResNet-101中选择在下采样前的conv2（1/4）
  3. 将以上两个结果整合concat
  4. 采用多个3x3的卷积来细化（refine）整合的结果
  5. 最后通过简单的双线性插值上采样（x4）恢复到原图大小

  <img src=".\Figures\DeepLabv3+\DeepLabv3+.png" style="zoom: 67%;" />

- 实验证明，当output stride=8时，性能略有提高

### 2 改进对齐的Xception模型

#### (1) 对齐的Xception模型

​	Xeption模型在图像分类上计算很快，后来被MSRA团队优化进一步提高目标检测任务的性能，称作**对齐的Xeption模型**。

#### (2) 改进对齐的Xception模型

​	调整Xception模型应用到语义分割中，直接在对齐的Xception模型上做一些修改：

1. 不修改入口流网络的结构
2. **所有的最大池化操作用深度可分卷积代替**，这样可以通过采用空洞卷积来提取任意分辨率的特征图
3. 在每个**3x3的深度卷积**之后都增加**批量标准化**和**ReLU激活函数**

<img src=".\Figures\DeepLabv3+\modified_aligned_Xception.JPG" style="zoom:67%;" />