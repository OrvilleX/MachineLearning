# 数据增强

数据增强是一种重要的机器学习方法之一，是基于已有的训练样本数据来生成更多的训练数据，其目的就是为了使扩增的训练数据尽可能接近真实分布的数据，从而提高检测
精度。此外，数据增强能够迫使模型学习到更多鲁棒性的特征，从而有效提高模型的泛化能力。 在实际的应用场景中，足量且高保真的优质数据集通常是可遇不可求的，这不
仅费钱费时费力，而且隐私保护和极端概率问题，数据集的获取变得尤为困难。因此，一种低成本且有效的方法便是利用数据增强来减少对训练数据的依赖，从而帮助开发人员
更好更快地构建高精度的 AI 模型，其可以起到如下的作用：

* 避免过拟合。当数据集具有某种明显的特征，例如数据集中图片基本在同一个场景中拍摄，使用Cutout方法和风格迁移变化等相关方法可避免模型学到跟目标无关的信息；
* 提升模型鲁棒性，降低模型对图像的敏感度。当训练数据都属于比较理想的状态，碰到一些特殊情况，如遮挡，亮度，模糊等情况容易识别错误，对训练数据加上噪声，掩码等方法可提升模型鲁棒性；
* 增加训练数据，提高模型泛化能力；
* 避免样本不均衡。在工业缺陷检测方面，医疗疾病识别方面，容易出现正负样本极度不平衡的情况，通过对少样本进行一些数据增强方法，降低样本不均衡比例；

# Yolo中的应用

针对下述的各类数据增强的技术，在Yolo中均需要维护对应的超参数。为此针对部分参数我们通过`hyp.scratch-high.yaml`来进行控制，其中各类超参数会根据目前训练模型所使用的
具体模型的复杂度来决定使用何种复杂度的超参数，如果需要在训练的时候指定对应的超参数，可以通过下述方式进行指定即可：  

```bash
python train.py --hyp data/hyp.scratch-high.yaml ...
```  
注意：下述各类源码具体在Yolo的datasets.py文件中

## 1. HSV增强

HSV（色相、饱和度、明度）增强是一种图像增强技术，它基于调整图像的颜色信息来改变图像的外观。HSV颜色空间由色相（Hue）、饱和度（Saturation）和明度（Value）三个参数组成，每个参数都
可以单独或联合地调整以改变图像的色彩和外观。

* 色相（Hue）：代表颜色的种类或名称，如红、绿、蓝等。在HSV增强中，调整色相可以改变图像中的整体色调，使图像看起来更加明亮或更加暗淡。  
* 饱和度（Saturation）：衡量颜色的纯度或浓度，越高表示颜色越鲜艳。增加饱和度可以使颜色更加饱和艳丽，减少则使颜色变得更加灰暗和淡化。  
* 明度（Value）：表示颜色的亮度或明暗程度。调整明度可以改变图像的亮度水平，增加明度会使图像更明亮，降低则使图像变得更暗。  

在yolo中的实现上述HSV增强的主要实现代码偏度如下：

```python
# hsv色域变换
elif method == 'hsv':
    """hsv色域增强  处理图像hsv，不对label进行任何处理
    :param img: 待处理图片BGR
    :param hgain: h通道色域参数 用于生成新的h通道
    :param sgain: s通道色域参数 用于生成新的s通道
    :param vgain: v通道色域参数 用于生成新的v通道
    """
    if hgain or sgain or vgain:
    # 随机取-1到1三个实数，乘以hyp中的hsv三通道的系数  用于生成新的hsv通道
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))  # 图像的通道拆分 h s v
    dtype = img.dtype  # uint8

    # 构建查找表
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)  # 生成新的h通道
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)  # 生成新的s通道
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)  # 生成新的v通道

    # 图像的通道合并 img_hsv=h+s+v  随机调整hsv之后重新组合hsv通道
    # cv2.LUT(hue, lut_hue)   通道色域变换 输入变换前通道hue 和变换后通道lut_hue
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    # no return needed  dst:输出图像
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
```

其对应的在yolov5中默认是没有启用的，被注释掉的。  
```python
# create_dataloader to set augment
if self.augment:
    # Albumentations
    img, labels = self.albumentations(img, labels)
    nl = len(labels)  # update after albumentations
    # HSV color-space
    # augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
```

## 2. 旋转、缩放、翻转与平移

顾名思义，就是将图片根据中点坐标进行旋转指定的角度，进行缩放，或者进行上下、左右的翻转，最后就是平移，当然上述的操作也会意味对应的
标注坐标的数据也需要同步进行更新，相关的实现代码如下所示：  

```python
# 上下垂直翻转
if method == 'flipud':
    img = np.flipud(img)
# 左右水平翻转
elif method == 'fliplr':
    img = np.fliplr(img)
# 旋转
elif method == 'rotation':
    a = random.uniform(-45, 45)
    R = cv2.getRotationMatrix2D(angle=a, center=(width / 2, height / 2), scale=1)
    img = cv2.warpAffine(img, R, dsize=(width, height), borderValue=(114, 114, 114))
# 缩放
elif method == 'scale':
    img = cv2.resize(img, dsize=(640, 640))
# 平移
elif method == 'translation':
    T = np.eye(3)
    tr = 0.1
    T[0, 2] = random.uniform(0.5 - tr, 0.5 + tr) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - tr, 0.5 + tr) * height  # y translation (pixels)
    img = cv2.warpAffine(img, T[:2], dsize=(width, height), borderValue=(114, 114, 114))
```

注意：据配置文件里的超参数发现只使用了Scale和Translation即缩放和平移

## 3. 剪切

其是图像处理中的一种几何变换，用于沿着某个方向对图像进行推移或错切。这种变换会改变图像中的像素位置，使图像中的一部分在水平或垂直方向
上发生位移，但不改变其大小或旋转角度。 在图像处理中，Shear 变换通常以一定的角度来施加。对于二维图像，Shear 可以沿着水平或垂直方向
进行，使图像的某些部分沿着某个方向相对于其他部分进行位移。 在数据增强中，Shear 变换可以帮助模型更好地学习到图像的不同视角或观察角
度，从而增加模型的鲁棒性。例如，在目标检测中，应用 Shear 变换可以使模型更好地适应目标在不同角度或方向上的出现方式，从而提高模型在
实际场景中的性能，相关的实现代码如下所示：

```python
# 剪切
# https://blog.csdn.net/LaoYuanPython/article/details/113856503
elif method == 'shear':
    S = np.eye(3)
    sh = 20.0
    S[0, 1] = math.tan(random.uniform(-sh, sh) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-sh, sh) * math.pi / 180)  # y shear (deg)
    img = cv2.warpAffine(img, S[:2], dsize=(width, height), borderValue=(114, 114, 114))
```

## 4. 透视

透视变换（Perspective Transformation）是一种图像处理中常用的几何变换方法，它可以改变图像的视角和投影，使得图像从一个透视或视
角转换到另一个透视或视角，仿佛是从不同角度观察了同一个场景。 透视变换基于投影原理，通常应用于处理三维场景投影到二维图像的情况，比如
在计算机视觉中处理相机捕获的三维场景投影到二维图像上。  
透视变换通常涉及到一些关键参数：  

* 源点和目标点： 确定变换的源图像和目标图像上的对应关系。例如，确定源图像中四个角的点与目标图像中对应的位置。
* 透视变换矩阵： 根据源点和目标点的对应关系，计算出变换矩阵。这个矩阵会应用于图像，从而完成透视变换。

透视变换的效果包括拉伸、扭曲或者仿佛从不同角度观察场景的效果。在图像处理和计算机视觉中，透视变换常用于校正图像、拼接图像、增强视角以
及仿射变换无法处理的更复杂的变换等方面。  

```python
# 透视变换
elif method == 'perspective':
    P = np.eye(3)
    pe = 0.001
    P[2, 0] = random.uniform(-pe, pe)  # x perspective (about y)
    P[2, 1] = random.uniform(-pe, pe)  # y perspective (about x)
    img = cv2.warpPerspective(img, P, dsize=(width, height), borderValue=(114, 114, 114))
```

## 5. Mixup

其是一种用于数据增强的技术，它通过在训练过程中对原始数据进行混合，产生新的训练样本。Mixup 的基本思想是将两个不同样本的特征和标签进行
线性插值，生成新的样本的特征和标签。这种方法可以减少模型在训练过程中的过拟合风险，并提高模型的泛化能力。

Mixup的工作原理如下：

* 数据混合： 从训练数据集中随机选择两个样本，每个样本包含图像和对应的标签。
* 特征和标签线性插值： 对两个样本的图像和标签进行线性组合。即对图像进行线性插值，同时对标签也进行线性插值。例如，对两个图像进行加权平均得到新的图像，对两个标签进行相同的权重平均得到新的标签。
* 生成新样本： 将线性插值后的图像作为输入，将线性插值后的标签作为输出，形成新的训练样本。 

这种混合的方式将两个不同样本的特征和标签信息融合在一起，使得模型更难以记住单个样本的特征，从而减轻过拟合问题。同时，这种混合还可以使
模型更好地学习到类别之间的关系和特征，提高模型的泛化能力。对应在yolo中的实现方式如下：  

```python
if method == 'mixup':
    # 填充到相同大小 640 × 640
    imgs[:2] = fix_shape(imgs[:2])
    img1 = imgs[0]
    img2 = imgs[1]
    # 显示原图
    htitch = np.hstack((img1, img2))
    cv2.imshow("origin images", htitch)
    cv2.waitKey(0)
    cv2.imwrite('outputs/mixup_origin.jpg', htitch)
    # mixup ratio, alpha=beta=32.0
    r = np.random.beta(32.0, 32.0)
    imgs = (img1 * r + img2 * (1 - r)).astype(np.uint8)
    return imgs
```

注意：代码中只有较大的模型才使用到了MixUp，而且每次只有10%的概率会使用到

## 6. CutOut

Cutout 是一种常用的图像数据增强技术，它通过随机遮挡图像的某些区域来产生新的训练样本。这种方法旨在提高模型对遮挡和噪声的鲁棒性，同时
增加数据的多样性，从而降低模型的过拟合风险。Cutout 的基本原理如下：

* 随机遮挡： 从图像中随机选择一个区域，并将该区域的像素值设置为某个特定的值（通常是零）或者进行随机化处理。
* 生成新样本： 将遮挡后的图像作为训练样本的一部分，输入到模型中进行训练。这个遮挡区域的大小、位置和形状通常是随机确定的，可以在训练过程中多次应用，生成不同的遮挡样本。

Cutout 的关键在于通过随机遮挡图像的部分区域来生成新的训练样本，从而使模型更具鲁棒性，能够更好地应对部分区域遮挡或缺失的情况。这有助
于模型更好地学习到数据的整体特征，而不是过度依赖于局部细节。对应在yolo中的实现方式如下：

```python
elif method == 'cutout':
    img = imgs[0]
    cv2.imshow("origin images", img)
    cv2.waitKey(0)
    height, width = img.shape[:2]
    # image size fraction
    scales = [0.5] * 1 + \
            [0.25] * 2 + \
            [0.125] * 4 + \
            [0.0625] * 8 + \
            [0.03125] * 16
    # create random masks
    for s in scales:
        # mask box shape
        mask_h = random.randint(1, int(height * s))
        mask_w = random.randint(1, int(width * s))

        # mask box coordinate
        xmin = max(0, random.randint(0, width) - mask_w // 2)  # 左上角 x坐标
        ymin = max(0, random.randint(0, height) - mask_h // 2)  # 左上角 y坐标
        xmax = min(width, xmin + mask_w)  # 右下角 x坐标
        ymax = min(height, ymin + mask_h)  # 右下角 y坐标

        # apply random color mask
        color = [random.randint(64, 191) for _ in range(3)]
        # color = [0, 0, 0]
        img[ymin:ymax, xmin:xmax] = color
    return img
```

其数据增强方式在yolov5中默认是被注释掉不启用的。  

```python
# Cutouts
# labels = cutout(img, labels, p=0.5)
```

## 7. Cutmix

CutMix是一种先进的图像数据增强技术，它结合了图像裁剪（Cutout）和样本混合（Mixup）的思想。CutMix通过将一个图像的一部分区域剪
切并粘贴到另一个图像上，同时将对应的标签也进行相应的混合，生成新的训练样本。其基本原理如下：

* 随机裁剪和粘贴： 从一张图像中随机选择一个区域，并将这个区域剪切下来。然后将这个剪切下来的区域粘贴到另一张随机选择的图像上，并填充原始区域剩余部分。
* 标签混合： 对新生成的图像使用混合标签，即根据两个图像的标签进行相应权重的混合。

通过这种方式，CutMix创造了结合了两个不同图像信息的新样本，同时合并了它们的标签信息。这种数据增强方法可以提高模型对图像局部信息
的学习能力，同时有助于防止模型对于训练数据的过拟合。对应在yolo中的实现方式如下：

```python
elif method == 'cutmix':
    # 这里未做fix_shape处理 两张图片大小不一样
    img1, img2 = imgs[0], imgs[1]
    h1, h2 = img1.shape[0], img2.shape[0]
    w1, w2 = img1.shape[1], img2.shape[1]
    # 设定lamda的值，服从beta分布
    alpha = 1.0
    lam = np.random.beta(alpha, alpha)
    cut_rat = np.sqrt(1. - lam)
    # 裁剪第二张图片
    cut_w = int(w2 * cut_rat)  # 要裁剪的图片宽度
    cut_h = int(h2 * cut_rat)  # 要裁剪的图片高度
    # uniform
    cx = np.random.randint(w2)  # 随机裁剪位置
    cy = np.random.randint(h2)

    # 限制裁剪的坐标区域不超过2张图片大小的最小值
    xmin = np.clip(cx - cut_w // 2, 0, min(w1, w2))  # 左上角x
    ymin = np.clip(cy - cut_h // 2, 0, min(h1, h2))  # 左上角y
    xmax = np.clip(cx + cut_w // 2, 0, min(w1, w2))  # 右下角x
    ymax = np.clip(cy + cut_h // 2, 0, min(h1, h2))  # 右下角y

    # 裁剪区域混合
    img1[ymin:ymax, xmin:xmax] = img2[ymin:ymax, xmin:xmax]
    return img1
```

## 8. Mosaic数据增强

Mosaic 是一种高级的图像数据增强技术，常用于目标检测模型的训练。它通过将多张图像拼接成一张大图像，使得模型能够在单个训练样
本中学习到多样的场景和目标，从而提高模型对复杂场景的理解和泛化能力。 其数据增强的基本原理如下：

* 随机选择图像： 从数据集中随机选择四张图像。
* 创建 Mosaic 图像： 将这四张图像随机地拼接在一起，形成一张大的 Mosaic 图像。这个过程包括将四张图像的某些部分剪裁并粘贴到一张新的图像中，形成一个四分之一大小的图像区域。
* 调整标签： 根据拼接后的 Mosaic 图像，相应地调整目标的位置和标签信息。例如，目标可能跨越了原始图像的边界，因此需要相应地调整目标的坐标信息。
* 
Mosaic 数据增强技术通过将多张图像拼接在一起，使得模型在单个训练样本中可以观察到多个场景和对象的组合。这样的训练方式可以帮
助模型更好地理解多样化的场景，并学习到不同目标之间的关联性，提高模型对复杂场景的识别和推断能力，在yolov5中也提供了一个9张
图的版本，其对应yolo中的实现代码如下：  

```python
img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
for i in range(len(imgs)):
    img = imgs[i]
    h, w = img.shape[:2]
    # place img in img4
    if i == 0:  # top left
        # 创建马赛克图像 [1280, 1280, 3]=[h, w, c] base image with 4 tiles
        img4 = np.full((s * 2, s * 2, imgs[0].shape[2]), 114, dtype=np.uint8)
        # xmin, ymin, xmax, ymax (large image)
        # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
        # 马赛克图像【大图】：(x1a,y1a)左上角，(x2a,y2a)右下角
        x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
        # xmin, ymin, xmax, ymax (small image)
        # 计算截取的图像区域信息(以xc,yc为第一张图像的右下角坐标填充到马赛克图像中，丢弃越界的区域)
        # 要拼接的图像【小图】：(x1b,y1b)左上角 (x2b,y2b)右下角
        x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
    elif i == 1:  # top right
        x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
        x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
    elif i == 2:  # bottom left
        x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
        x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
    elif i == 3:  # bottom right
        x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
        x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
    # img4[ymin:ymax, xmin:xmax]
    # 将截取的图像区域填充到马赛克图像的相应位置   img4[h, w, c]
    # 将图像img的【(x1b,y1b)左上角 (x2b,y2b)右下角】区域截取出来填充到马赛克图像的【(x1a,y1a)左上角 (x2a,y2a)右下角】区域
    img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
    # 计算小图填充到大图时所产生的偏移 用来计算mosaic数据增强后 标签框的位置
    padw = x1a - x1b
    padh = y1a - y1b

    # 处理图像的labels信息
    label = labels[i].copy()
    if label.size:
        # normalized xywh to pixel xyxy format
        label[:, 1:] = xywhn2xyxy(label[:, 1:], w, h, padw, padh)
    labels4.append(label)

# Concat/clip labels
# 把label4中4张小图的信息整合到一起
labels4 = np.concatenate(labels4, 0)
for x in (labels4[:, 1:]):
    np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
```

## 9. Albumentations数据增强工具包

[官方代码地址](https://github.com/albumentations-team/albumentations)  
[使用说明文档](https://albumentations.ai/docs)

注意：其需要在安装该包后才会启用对应的数据增强功能在Yolov5中

# 参考链接文档

* [YOLOv5使用的数据增强方法汇总](https://developer.aliyun.com/article/1078058)  
* [YOLOv5数据增强代码解析](https://blog.csdn.net/weixin_43799388/article/details/123830587)  
