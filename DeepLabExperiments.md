# DeepLabv2

​	from github:[deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch)

- 下载预训练好的caffe模型（DeepLab团队发布了基于COCO和PASCAL VOC两种数据集训练的版本）

- 实验中采用resnet101主干网络（如下图），使用PASCAL VOC2012数据集

<img src=".\Figures\DeepLabExps\DeepLabv2.JPG" style="zoom:67%;" />



## 1 训练

- loss曲线

## 2 评估

### (1) 未加CRF后处理

### (2) 加CRF后处理

## 3 测试

### (1) 使用VOC2012数据集

### (2) 使用网络图片



# DeepLabv3

- backbone
  - resnet50
  - **resnet101**
  - resnet152
- dataset
  - pascal_voc

## 1 base_forward()

