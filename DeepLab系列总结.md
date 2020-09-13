# DeepLab是什么

## 1 用途

​	进行更精细的语义分割，给输入图像的每个像素都分配一个类别标签

## 2 四个版本

#### (1) DeepLabv1:Semantic Image Segmentation with **Deep Convolutional Nets** and **Fully Connected CRFs**

​	笔记见DeepLabv1Learning.md

#### (2) DeepLabv2:Semantic Image Segmentation with Deep Convolutional Nets, **Atrous Convolution**, and Fully Connected CRFs

​	笔记见DeepLabv2Learning.md

#### (3) DeepLabv3:Rethinking Atrous Convolution for Semantic Image Segmentation

​	笔记见DeepLabv3Learning.md

#### (4) DeepLabv3+:**Encoder-Decoder** with **Atrous Separable Convolution** for Semantic Image Segmentation

​	笔记见DeepLabv3+Learning.md



|                    | DeepLabv1 | DeepLabv2       | DeepLabv3 | DeepLabv3+ |
| ------------------ | --------- | --------------- | --------- | ---------- |
| 主干网络(backbone) | VGG16     | VGG16/ResNet101 |           |            |
| 空洞卷积           |           |                 |           |            |
| ASPP模块           |           |                 |           |            |
| 改进的ASPP模块     |           |                 |           |            |
| 全连接CRF后处理    |           |                 |           |            |
|                    |           |                 |           |            |
|                    |           |                 |           |            |

## 3 DeepLabv1到DeepLabv3+的主干网络&结构

​	以下的主干网络都是ImageNet预训练过的。

### (1) DeepLabv1:VGG16（如下图）

<img src=".\Figures\FCN\VGG16.JPG" style="zoom:67%;" />

1. **全卷积化**：将VGG16的全连接层（如下图，fc6，fc7，fc8 -> conv6，conv7，conv8）转换为卷积层（接受任意分辨率的输入图像）-> **对应下一节问题3**
   
   - 特征图分辨率大幅度下降，output stride=32，提取到的特征是通过非常稀疏（粗糙）的计算得到的
   
2. 跳过最后两个最大池化层的下采样（如下图，将pool4和pool5的stride从2设为1），则output stride=8 -> **对应下一节问题1**

   <img src=".\Figures\DeepLabv1\VGG16_to_DeepLab.JPG" style="zoom:67%;" />

3. 后面两个块（conv5，conv6/fc6）的卷积层采用**空洞卷积**（input stride=2&input stride=4）

4. 用21路分类器替代原本的1000路分类器（类别数量由1000降为21）

5. 采用简单的双线性插值上采样，将特征图恢复到原图像大小（x8）

6. 在输出之前加入**全连接CRF后处理**，恢复局部结构的细节，提高边界定位的精度 -> **对应下一节问题2**

### (2) DeepLabv2:

#### ① VGG16

1. 跳过最后两个最大池化层的下采样（将pool4和pool5的stride从2设为1），则output stride=8 -> **对应下一节问题1**
2. 后面（conv5）的卷积层采用**空洞卷积**（rate=2，跟DeepLabv1的方式不一样，增大了过滤器的尺寸）
3. 将pool5后的全连接层（fc6，fc7，fc8）改造成**ASPP模块**（如下图）-> **对应下一节问题2&3**
   - 其中，将1000路分类器（fc8）替换为目标（包括背景）总数量维的分类器
   - 将四个分支的结果进行融合
   
   <img src=".\Figures\DeepLabv2\DeepLabv2_ASPP.JPG" style="zoom:67%;" />
4. 在输出之前加入**全连接CRF后处理**，恢复局部结构的细节，提高边界定位的精度 -> **对应下一节问题2**
5. 在训练时将DCNN和CRF解耦，即当设置CRF参数时假设DCNN的一元项是固定的

#### ② ResNet101

​	（实验表明，采用ResNet101比VGG16性能提高更显著，目标边界也更清晰。）

### (3) DeepLabv3:ResNet101（如下图）

<img src=".\Figures\ResNet\ResNet101.jpg" style="zoom:67%;" />

​	DeepLabv3中提出了串联和并联两种形式，具体结构在后续章节展开，点击下面的标题①②直接跳转。

#### ① [串联形式](#(3) 提出带有<u>串联</u>空洞卷积的深度网络（DeepLabv3）)

<img src=".\Figures\DeepLabv3\go_deeper_with_atrous_convolution.JPG" style="zoom:67%;" />

#### ② [并联形式（改进的ASPP）](#(7) 改进ASPP：带有<u>并联</u>空洞卷积的深度网络（DeepLabv3）——用图像级别的特征来增强)

<img src=".\Figures\DeepLabv3\ASPP.JPG" style="zoom:67%;" />

### (4) DeepLabv3+:（应用Encoder-Decoder）

​	DeepLabv3+中对ResNet101和Xception两种主干网格进行实验，它们具体结构在后续章节展开，点击下面的标题①②直接跳转。

#### ① [ResNet101](#① 主干网络是ResNet101版本)

1. 将DeepLabv3作为Encoder
2. 增加简单有效的Decoder模块

<img src=".\Figures\DeepLabv3+\DeepLabv3+.png" style="zoom: 67%;" />

#### ② [改进的对齐Xception（如下图，详见DeepLabv3+Learning.md）](#② 主干网络是Xception版本)

<img src=".\Figures\DeepLabv3+\modified_aligned_Xception.JPG" style="zoom:67%;" />

1. 将Xception的输出**连接到改进后的ASPP**（同DeepLabv3）上，作为Encoder
2. 增加简单有效的Decoder模块

​	（实验证明，改进后的对齐Xception比ResNet101效果好。）



# 解决什么问题&如何解决

- DCNN（如：ResNet）可以有效处理图像级别的分类任务（输出代表输入图像的类别的单个值，即为整个图像分配一个标签），但对**像素级别的分类**任务/**密集**的预测任务（如：语义分割，为输入图像的每一个像素分配一个标签）表现不佳
- **FCN**（全卷积网络）去掉图像分类网络（如：VGG16）中的全连接层（全连接层固定了输入图像的尺寸），替换为卷积层，再通过上采样层融合，实现较成功的密集预测，这是一个**Encoder-Decoder**结构

## 1 DCNN多个下采样降低特征图分辨率，空间信息恢复有难度

### (1) 引入空洞卷积（DeepLabv1）

#### ① DeepLabv1:

- 保持过滤器不变
- 设置input stride/dilation rate对特征图进行稀疏采样

<img src=".\Figures\DeepLabv1\hole_algorithm.JPG" style="zoom:67%;" />

#### ② DeepLabv2&v3:

- 对过滤器进行上采样，在卷积核中填充0值，增大过滤器的大小，但过滤器参数没有增加（如下图b，rate=2，则在原卷积核中每两个值之间插入一个0值）

- 实际上的计算只和过滤器中的非零值有关（和权重0相乘即为0）

  - $$
    y[i]=\sum_{k=1}^Kx[i+r\cdot k]w[k]\\
    其中，r为步长/rate
    $$

<img src=".\Figures\DeepLabv2\atrous_convolution.JPG" style="zoom:67%;" />

### (2) 引入深度可分卷积——用于Xception改进（DeepLabv3+）

​	**深度可分卷积**由**深度卷积**和**逐点卷积**组成（如下图a和b）

<img src="C:/工作/DeepLearning/Figures/DeepLabv3+/atrous_separable_convolution.JPG" style="zoom:67%;" />

## 2 DCNN平移不变性导致细节信息丢失，目标边界定位模糊

### (1) 引入全连接CRF后处理（DeepLabv1引入，DeepLabv3开始废弃）

​	在双线性插值上采样后增加全连接CRF后处理，给位置和颜色强度相近的像素打上相似的标签，让边界更清晰

![](.\Figures\DeepLabv1\FC_CRF.JPG)

### (2) 引入Encoder-Decoder结构（DeepLabv3+）

#### ① 主干网络是ResNet101版本

- encoder

  将DeepLabv3最后的特征图作为Encoder-Decoder结构的encoder输出，有<u>256通道</u>且包含丰富的语义信息

- decoder

  - **分支1：**首先对encoder输出的特征图通过双线性插值上采样（x4，1/16->1/4）
  - **分支2：**对DCNN中低层输出的同样分辨率的特征图进行逐点卷积（改变通道数，=encoder输出的<u>通道数256</u>）
    - 如，ResNet-101中选择在conv2下采样前（1/4）
  - 将以上两个结果整合concat
  - 采用多个3x3的卷积来细化（refine）整合的结果
  - 最后通过简单的双线性插值上采样（x4）恢复到原图大小

  <img src=".\Figures\DeepLabv3+\DeepLabv3+.png" style="zoom: 67%;" />

#### ② 主干网络是Xception版本

- encoder

  将Xception的输出**连接到改进后的ASPP**（同DeepLabv3，即上图的ASPP模块）上，最后的特征图作为Encoder-Decoder结构的encoder输出

- decoder

  同上（[主干网络是ResNet版本](#① 主干网络是ResNet101版本)）

## 3 处理不同尺度的输入图像

### (1) 四种方法（详见DeepLabv3Learning.md）

<img src=".\Figures\DeepLabv3\four_methods_for_multiscale_images.JPG" style="zoom:67%;" />

#### ① Image Pyramid 图像金字塔

#### ② Encoder-Decoder 编码器-解码器结构

#### ③ Deeper w. Atrous Convolution 深度空洞卷积

#### ④ SPP 空间金字塔池化

​	在DCNN中最后的卷积层和全连接层之间增加SPP，保证网络中输入任意分辨率的图像经过SPP（如下图，分别对特征图分16份、4份、1份各自池化）后，都会输出21（每一份池化得到一个值[256通道，即1x256]，16+4+1=21）维（固定长度[256通道，即21x256]）的特征图，之后输入全连接层处理

<img src=".\Figures\DeepLabv2\SPP.JPG" style="zoom:67%;" />

### (2) 引入ASPP（DeepLabv2)——编码了图像不同尺度的上下文信息，也帮助提高预测精度/问题2

- 采用多个带有不同采样率的空洞卷积层并行处理，rates=(6,12,18,24)
- 将不同分支处理后的特征图进行融合产生最后的结果

<img src=".\Figures\DeepLabv2\ASPP.JPG" style="zoom:67%;" />

### (3) 提出带有<u>串联</u>空洞卷积的深度网络（DeepLabv3）

<img src=".\Figures\DeepLabv3\go_deeper_with_atrous_convolution.JPG" style="zoom:67%;" />

#### ① 采用普通卷积（如上图a）——逐渐缩小分辨率，丢失图像的细节信息

- 对**ResNet101**原网络中最后一个块（Block4），复制成多个副本（Block5-Block7），并将它们**串联**起来

- Block4-Block6（不包括Block7）每个块中有3组（9层）卷积，共有3层3x3卷积，且<u>最后一层卷积</u>（1x1，2048）中stride=2，图像经过Block4-Block6依次缩小一半

  - Block4的结构

  $$
  \begin{bmatrix}
  1*1,512\\
  3*3,512\\
  1*1,2048
  \end{bmatrix}*3
  $$

#### ② 采用空洞卷积（如上图b）

- 基于多重网格法，对Block4-Block7的卷积层设置不同的rate

- 对Block4-Block7的每一块中的3组卷积层的单位rate定义为：

  - $$
    Multi\_Grid=(r_1,r_2,r_3)
    $$

- 每一个块中每一组卷积层的rate计算如下：

  - $$
    rates=rate\cdot Multi\_Grid
    $$

  - 如，output_stride=16, Multi_Grid=(1, 2, 4)，那么，Block4中，rate=2，则rates=2 · (1, 2, 4)=(2, 4, 8)

### (4) 改进ASPP：带有<u>并联</u>空洞卷积的深度网络（DeepLabv3）——用图像级别的特征来增强

​	ASPP确实有效捕获了图像不同尺度的上下文信息，但实验证明，采样率rate越大时，过滤器的有效权重的数量越少。当rate的值接近于图像大小，3x3过滤器并没有捕获图像的全局上下文，而是退化为1x1的过滤器（因为只有过滤器的中间的权重有效）。

<img src=".\Figures\DeepLabv3\ASPP.JPG" style="zoom:67%;" />

- ASPP（如上图a）：四组卷积
  - 一组是：<u>256个</u>1x1过滤器的卷积
  - 三组是：<u>256个</u>3x3过滤器的卷积，rates=(6, 12, 18)，output_stride=16
    - 注：当output_stride=8时，rates加倍（x2）
- Image Pooling（如上图b）
  1. 对模型输出的特征图进行**全局平均池化**
  2. 将池化后产生的**图像级别的特征**反馈到<u>256个</u>**1x1过滤器**的卷积层中
  3. 将卷积后的特征图通过**双线性上采样**恢复到指定大小

- 跟原本ASPP一样，最后concat将各个分支的结果（<u>256个通道</u>）融合。但还要经过带有<u>256个</u>1x1过滤器的卷积层，最后再经过另一个1x1卷积得到最终的logits结果



# 评价指标