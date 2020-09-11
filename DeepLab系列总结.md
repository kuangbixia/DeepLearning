# DeepLab是什么

## 1 用途

​	进行更精细的语义分割，给输入图像的每个像素都分配一个类别标签

## 2 DeepLabv1到DeepLabv3+的比较

- DeepLabv1:Semantic Image Segmentation with **Deep Convolutional Nets** and **Fully Connected CRFs**
- DeepLabv2:Semantic Image Segmentation with Deep Convolutional Nets, **Atrous Convolution**, and Fully Connected CRFs
- DeepLabv3:Rethinking Atrous Convolution for Semantic Image Segmentation
- DeepLabv3+:Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

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

### (1) DeepLabv1:ImageNet预训练的VGG16（如下图）

<img src=".\Figures\FCN\VGG16.JPG" style="zoom:67%;" />

<img src=".\Figures\DeepLabv1\VGG16_to_DeepLab.JPG" style="zoom:67%;" />

1. **全卷积化**：将VGG16的全连接层（fc6，fc7，fc8 -> conv6，conv7，conv8）转换为卷积层（接受任意分辨率的输入图像）-> **对应下一节问题3**
   - 特征图分辨率大幅度下降，output stride=32，提取到的特征是通过非常稀疏（粗糙）的计算得到的
2. 跳过最后两个最大池化层的下采样（将pool4和pool5的stride从2设为1），则output stride=8 -> **对应下一节问题1**
3. 后面两个块（conv5，conv6/fc6）的卷积层采用**空洞卷积**（input stride=2&input stride=4）
4. 用21路分类器替代原本的1000路分类器（类别数量由1000降为21）
5. 采用简单的双线性插值上采样，将特征图恢复到原图像大小（x8）
6. 在输出之前加入**全连接CRF后处理**，恢复局部结构的细节，提高边界定位的精度 -> **对应下一节问题2**

### (2) DeepLabv2:ImageNet预训练的VGG16

<img src=".\Figures\DeepLabv2\DeepLabv2_ASPP.JPG" style="zoom:67%;" />

1. 跳过最后两个最大池化层的下采样（将pool4和pool5的stride从2设为1），则output stride=8 -> **对应下一节问题1**
2. 后面（conv5）的卷积层采用**空洞卷积**（rate=2，跟DeepLabv1的方式不一样，增大了过滤器的尺寸）
3. 将pool5后的全连接层（fc6，fc7，fc8）改造成**ASPP模块**（如上图）-> **对应下一节问题2&3**
   - 其中，将1000路分类器（fc8）替换为目标（包括背景）总数量维的分类器
   - 将四个分支的结果进行融合
4. 在输出之前加入**全连接CRF后处理**，恢复局部结构的细节，提高边界定位的精度 -> **对应下一节问题2**
5. 在训练时将DCNN和CRF解耦，即当设置CRF参数时假设DCNN的一元项是固定的

### (3) DeepLabv2:ResNet101

​	实验表明，采用ResNet101比VGG16性能提高更显著，目标边界也更清晰。

### (4) DeepLabv3:ResNet101（如下图，详见ResNetLearning.md）

<img src=".\Figures\ResNet\ResNet101.jpg"  />



# 解决什么问题&如何解决

- DCNN（如：ResNet）可以有效处理图像级别的分类任务（输出代表输入图像的类别的单个值，即为整个图像分配一个标签），但对**像素级别的分类**任务/**密集**的预测任务（如：语义分割，为输入图像的每一个像素分配一个标签）表现不佳
- **FCN**（全卷积网络）去掉图像分类网络（如：VGG16）中的全连接层（全连接层固定了输入图像的尺寸），替换为卷积层，再通过上采样层融合，实现较成功的密集预测，这是一个**Encoder-Decoder**结构

## 1 DCNN多个下采样降低特征图分辨率，空间信息恢复有难度

### (1) 引入空洞卷积（DeepLabv1）

#### ① DeepLabv1:

- 保持过滤器不变
- 设置input stride/dilation rate对特征图进行稀疏采样

<img src=".\Figures\DeepLabv1\hole_algorithm.JPG" style="zoom:67%;" />

#### ② DeepLabv2:

- 对过滤器进行上采样，在卷积核中填充0值，增大过滤器的大小，但过滤器参数没有增加（如下图b，rate=2，则在原卷积核中每两个值之间插入一个0值）

- 实际上的计算只和过滤器中的非零值有关（和权重0相乘即为0）

  - $$
    y[i]=\sum_{k=1}^Kx[i+r\cdot k]w[k]\\
    其中，r为步长/rate
    $$

<img src=".\Figures\DeepLabv2\atrous_convolution.JPG" style="zoom:67%;" />

### (2) 引入深度可分卷积（DeepLabv3+）



## 2 DCNN平移不变性导致细节信息丢失，目标边界定位模糊

### (1) 引入全连接CRF后处理（DeepLabv1引入，DeepLabv3开始废弃）

​	在双线性插值上采样后增加全连接CRF后处理，给位置和颜色强度相近的像素打上相似的标签，让边界更清晰

![](.\Figures\DeepLabv1\FC_CRF.JPG)

### (2) 引入Encoder-Decoder结构（DeepLabv3+）



## 3 处理不同尺度的输入图像

### (1) Image Pyramid 图像金字塔（见DeepLabv3Learning.md）

### (2) Encoder-Decoder 编码器-解码器结构（见DeepLabv3Learning.md）

### (3) Deeper w. Atrous Convolution（见DeepLabv3Learning.md） 

### (4) SPP 空间金字塔池化

​	在DCNN中最后的卷积层和全连接层之间增加SPP，保证网络中输入任意分辨率的图像经过SPP（如下图，分别对特征图分16份、4份、1份各自池化）后，都会输出21（每一份池化得到一个值[256通道，即1x256]，16+4+1=21）维（固定长度[256通道，即21x256]）的特征图，之后输入全连接层处理

<img src=".\Figures\DeepLabv2\SPP.JPG" style="zoom:67%;" />

### (2) 引入ASPP（DeepLabv2)——编码了图像不同尺度的上下文信息，也帮助提高预测精度

- 采用多个带有不同采样率的空洞卷积层并行处理
- 将不同分支处理后的特征图进行融合产生最后的结果

<img src=".\Figures\DeepLabv2\ASPP.JPG" style="zoom:67%;" />

### (3) 带有串联空洞卷积的模块（DeepLabv3）

<img src=".\Figures\DeepLabv3\go_deeper_with_atrous_convolution.JPG" style="zoom:67%;" />

#### ① 采用普通卷积（如上图a）

- 对**ResNet101**原网络中最后一个块（Block4），复制成多个副本（Block5-Block7），并将它们**串联**起来

- Block4-Block6（不包括Block7）每个块中第一层卷积（1x1）中stride=2，图像经过Block4-Block6依次缩小一半

  - Block4的结构

  $$
  \begin{bmatrix}
  1*1,512\\
  3*3,512\\
  1*1,2048
  \end{bmatrix}*3
  $$

#### ② 采用空洞卷积（如上图b）

### (4) 带有并联空洞卷积的模块：改进的ASPP（DeepLabv3）



# 评价指标