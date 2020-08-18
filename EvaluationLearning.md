# [深度学习模型评估指标](https://zhuanlan.zhihu.com/p/59481933)

## 1 分类评估指标

### (1) 应用

​	图像分类，将不同的图像划分为不同的类别，只考虑**单标签分类**问题（即每个图像有唯一的类别）

- 灰度图像手写数字识别mnist（10分类）
- cifar10（10分类），cifar100（100分类）
- ImageNet（2万分类）

### (2) 评估指标

- 二分类
  - 预测概率大于阈值T（一般是0.5），预测为正类，反之为负类
- 多分类
  - 预测类别为预测概率最大的那个类

1. accuracy 准确率
   $$
   acc = \frac{\sum_{i=0}^{N-1}(f(x_i)==y_i)}{N}
   $$

   - tutorials里logistics regression和CCN都是使用accuracy评价指标（mnist）

2. precision 精确率
   $$
   precision = \frac{TP}{TP+FP}
   $$

   - 表示预测（召回）为正类的样本中，有多少是真正的**正样本**

3. recall
   $$
   recall = \frac{TP}{TP+FN}
   $$

   - 表示在**正样本**中，有多少被预测（召回）为正类

4. F-score

5. PR曲线

6. ROC

7. AUC

## 2 [语义分割评估指标](https://blog.csdn.net/lingzhou33/article/details/87901365)

### (1) 应用

- 图像语义分割

### (2) 评估指标

1. IoU/IU 交并比(Intersection over Union)
   $$
   IoU=\frac{target \cap prediction}{target \cup prediction}
   $$

   - 基于类进行计算，将每一类的IoU计算后，累加计算平均值 -> mean IoU均交并比

     - $$
       MIoU=\frac{1}{k+1}\sum_{i=0}^k\frac{p_{ii}}{\sum_{j=0}^kp_{ij}+\sum_{j=0}^kp_{ji}-p_{ii}}
       $$

     - $$
       其中，p_{ii}表示TP，p_{ij}表示FN，p_{ji}表示FP的数量\\
       \sum_{j=0}^kp_{ij}相当于target中类别为i的面积，\sum_{j=0}^kp_{ji}相当于prediction中类别为i的面积，p_{ii}相当于它们相交的面积
       $$

       

2. pixcal accuracy 像素精度(PA)
   $$
   PA=\frac{\sum_{i=0}^{k}p_{ii}}{\sum_{i=0}^k \sum_{j=0}^kp_{ij}}
   $$
   

