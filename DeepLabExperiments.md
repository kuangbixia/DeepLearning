```
本周学习了DeepLabv2和DeepLabv3的代码，读懂代码，用ppt制图梳理网络的架构，并进行了实验，将结果与之前的FCN的效果进行对比。
```



# DeepLabv2

​	from github:[deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch)

- 实验中采用resnet101主干网络（如下图）

  - MSC

    - base(): DeepLabv2

    ```python
    self.add_module("layer1", _Stem(ch[0]))
    self.add_module("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1))
    self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1))
    self.add_module("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], 1, 2))
    self.add_module("layer5", _ResLayer(n_blocks[3], ch[4], ch[5], 1, 4))
    self.add_module("aspp", _ASPP(ch[5], n_classes, atrous_rates))
    ```

    - 加入scales: [0.5, 0.75]

    ```python
    		# Original
            logits = self.base(x)
            _, _, H, W = logits.shape
            interp = lambda l: F.interpolate(
                l, size=(H, W), mode="bilinear", align_corners=False
            )
    
            # Scaled
            logits_pyramid = []
            for p in self.scales:
                h = F.interpolate(x, scale_factor=p, mode="bilinear", align_corners=False)
                logits_pyramid.append(self.base(h))
    
            # Pixel-wise max
            logits_all = [logits] + [interp(l) for l in logits_pyramid]
            logits_max = torch.max(torch.stack(logits_all), dim=0)[0]
    
            if self.training:  # batch size:5
                return [logits] + logits_pyramid + [logits_max]
            else:  # batch size:1
                return logits_max
    ```

  - DenseCRF

    - 训练时不加入CRF

    ```python
    C, H, W = probmap.shape
    
    U = utils.unary_from_softmax(probmap)
    U = np.ascontiguousarray(U)
    
    image = np.ascontiguousarray(image)
    
    d = dcrf.DenseCRF2D(W, H, C)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
    d.addPairwiseBilateral(
                sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
            )
    
    Q = d.inference(self.iter_max)
    Q = np.array(Q).reshape((C, H, W))
    ```

    

<img src=".\Figures\DeepLabExps\DeepLabv2.JPG" style="zoom:67%;" />



## 1 训练

- 使用pascal voc2012数据集train，共1464张图

- loss曲线

  <img src=".\Figures\DeepLabExps\deeplabv2_loss.JPG" style="zoom:67%;" />



## 2 测试

使用PASCAL VOC2012数据集val，共1449张图

### (1) 未加CRF后处理

```json
{
    "Class IoU": {
        "0": 0.9381065566550431,
        "1": 0.8670873374879957,
        "2": 0.5648016487183284,
        "3": 0.8070445634603366,
        "4": 0.6930709296227279,
        "5": 0.7035921150971599,
        "6": 0.9253384456029814,
        "7": 0.8374863620188253,
        "8": 0.8431036935403197,
        "9": 0.3141988380185367,
        "10": 0.7785908758306208,
        "11": 0.5869767482304655,
        "12": 0.7817073223260699,
        "13": 0.7838859388344795,
        "14": 0.8096244021736647,
        "15": 0.8463193324963102,
        "16": 0.525183538939361,
        "17": 0.8227096017731864,
        "18": 0.4476325825560281,
        "19": 0.8599482528547515,
        "20": 0.7601421759002682
    },
    "Frequency Weighted IoU": 0.8930119615363026,
    "Mean Accuracy": 0.8076855870625423,
    "Mean IoU": 0.7379310124827362,
    "Pixel Accuracy": 0.9410767969470897
}
```

### (2) 加CRF后处理

```json
{
    "Class IoU": {
        "0": 0.941336572968029,
        "1": 0.8842308012956723,
        "2": 0.5618030229681028,
        "3": 0.8267684488784134,
        "4": 0.687792297282065,
        "5": 0.7118177556099816,
        "6": 0.9328776500144249,
        "7": 0.8435369819072479,
        "8": 0.8572111287188322,
        "9": 0.31793516760385143,
        "10": 0.7941212822546022,
        "11": 0.5958438527459142,
        "12": 0.7979313493096475,
        "13": 0.7968390339073101,
        "14": 0.8283894024279859,
        "15": 0.8578991536386813,
        "16": 0.5112299625356024,
        "17": 0.8444287324177158,
        "18": 0.4535167259537733,
        "19": 0.8692640781380712,
        "20": 0.7725684607615165
    },
    "Frequency Weighted IoU": 0.8982991044171963,
    "Mean Accuracy": 0.8063695749925508,
    "Mean IoU": 0.7470162791113067,
    "Pixel Accuracy": 0.9444584157491147
}
```

#### ① 使用VOC2012数据集val

<img src=".\Figures\DeepLabExps\deeplabv2_2007_000033.JPG" style="zoom:67%;" />

<img src=".\Figures\DeepLabExps\deeplabv2_2007_000042.JPG" style="zoom:67%;" />

<img src=".\Figures\DeepLabExps\deeplabv2_2007_000061.JPG" style="zoom:67%;" />

<img src=".\Figures\DeepLabExps\deeplabv2_2007_000123.JPG" style="zoom:67%;" />

<img src=".\Figures\DeepLabExps\deeplabv2_2007_000129.JPG" style="zoom:67%;" />

<img src=".\Figures\DeepLabExps\deeplabv2_2007_000175.JPG" style="zoom:67%;" />

#### ② 使用网络图片

<img src=".\Figures\DeepLabExps\deeplabv2_walkers.JPG" style="zoom:67%;" />

<img src=".\Figures\DeepLabExps\deeplabv2_walkers2.JPG" style="zoom:67%;" />

<img src=".\Figures\DeepLabExps\deeplabv2_walkers5.JPG" style="zoom:67%;" />



# DeepLabv3

​	from github:[awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)

- 实验中采用resnet101主干网络（结构如下图）

  - base_forward()

    ```python
    x = self.pretrained.conv1(x)
    x = self.pretrained.bn1(x)
    x = self.pretrained.relu(x)
    x = self.pretrained.maxpool(x)
    c1 = self.pretrained.layer1(x)
    c2 = self.pretrained.layer2(c1)
    c3 = self.pretrained.layer3(c2)
    c4 = self.pretrained.layer4(c3)
    ```

  - ASPP()

    - b0, b1, b2, b3, b4分别对应下图ASPP中的五个分支
      - b0采用1x1卷积
      - b1, b2, b3分别采用3x3空洞卷积，rates=(12, 24, 36)
      - b4采用平均池化
    - project: 最后整合五个分支输出的结果，共256x5个通道

    ```python
    self.b0 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
                nn.ReLU(True)
            )
    
    rate1, rate2, rate3 = tuple(atrous_rates)
    self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
    self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
    self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
    self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
    
    self.project = nn.Sequential(
                nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
                nn.ReLU(True),
                nn.Dropout(0.5)
            )
    ```

<img src=".\Figures\DeepLabExps\DeepLabv3.JPG" style="zoom:67%;" />

## 1 训练

- loss曲线

  <img src=".\Figures\DeepLabExps\deeplabv3_loss.JPG" style="zoom: 67%;" />

- metric变化曲线

  - mean IoU

    <img src=".\Figures\DeepLabExps\metric_mIoU.JPG" style="zoom:67%;" />

  - pixel accuracy

    <img src=".\Figures\DeepLabExps\metric_pixAcc.JPG" style="zoom:67%;" />

## 2 测试

### (1) 使用VOC2012数据集val

- 如下图，DeepLabv3分割效果确实比FCN提高了，且从下表评估指标Mean IoU大幅度提高，达到59.05%，超5过了50%
- 再跟上面DeepLabv2（带CRF后处理）实验比较，DeepLabv2的评估指标值非常可观，Mean IoU达到了74.70%，pixAcc达到了94.45%，远超DeepLabv3
  - 从上面DeepLabv2的实验效果图看到，目标的边界分割效果很不错，优于DeepLabv3（如下图）

| Methods   | Backbone  | DataSet | epochs | lr     | Mean IoU | pixAcc |
| --------- | --------- | ------- | ------ | ------ | -------- | ------ |
| FCN32s    | vgg16     | VOC2012 | 60     | 0.0001 | 46.02    | 85.88  |
| FCN16s    | vgg16     | VOC2012 | 60     | 0.0001 | 47.74    | 87.17  |
| FCN8s     | vgg16     | VOC2012 | 60     | 0.0001 | 47.38    | 87.12  |
| DeepLabv3 | resnet101 | VOC2012 | 60     | 0.0001 | 59.05    | 88.89  |
| DeepLabv2 | resnet101 | VOC2012 |        |        | 74.70    | 94.45  |

![](.\Figures\DeepLabExps\deeplabv3_voc.JPG)

### (2) 使用网络图片

![](.\Figures\DeepLabExps\deeplabv3_walkers.JPG)