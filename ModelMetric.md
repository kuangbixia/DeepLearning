# [深度学习模型评估指标](https://zhuanlan.zhihu.com/p/59481933)

## 0 Metric用来评估和选择模型



## 1 分类评估指标

### (1) 应用

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

     ```python
     print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
     ```
   
     
   
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
     
       
   
2. pixcal accuracy 像素精度(PA)
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

