# 网络层剖析

本文档将主要针对yolov5 6.x中核心的网络层进行介绍说明，以便熟知并了解目前框架的主要核心网络层的构成以及原理。各个网络层的主要
代码集中在`model/common.py`文件中。  

![yolo_v5_6_global_network.png](..%2F..%2Fimages%2Fyolo_v5_6_global_network.png)

## 1. Conv

网络中的标准卷积层，有2D卷积+BN层+激活函数（SiLU）组成，在之后的Bottleneck、C3、SPPF等结构中都会被调用。  

![conv.png](..%2F..%2Fimages%2Fconv.png)  

```python
# 标准卷积操作：conv2D+BN+SiLU
# 在Focus、Bottleneck、BottleneckCSP、C3、SPP、DWConv、TransformerBloc等模块中调用
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # 这里的nn.Identity()不改变input，直接return input
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    # 前向加速推理模块
    # 用于Model类的fuse函数，融合conv+bn 加速推理 一般用于测试/验证阶段
    def forward_fuse(self, x):
        return self.act(self.conv(x))
```  

其中相关Pytorch部分的内容知识介绍如下，可根据情况自行学习了解。  
* [Conv2d介绍](https://blog.csdn.net/qq_34243930/article/details/107231539)  
* [Pytorch卷积层介绍](https://pytorch.zhangxiann.com/3-mo-xing-gou-jian/3.2-juan-ji-ceng)  
* [BatchNorm2d用法详解](http://www.jokerak.com/deep-learning/4-CNN/4-BatchNormalization/#)  
* [激活函数SiLU](https://zhuanlan.zhihu.com/p/545032801)  

## 2. Focus

Focus模块是作者自己设计出来，为了减少浮点数和提高速度，而不是增加featuremap的，本质就是将图像进行切片，类似于下采样取值，将原
图像的宽高信息切分，聚合到channel通道中。

![focus.png](..%2F..%2Fimages%2Ffocus.png)

```python
class Focus(nn.Module):
    # Focus wh information into c-space
    """理论：从高分辨率图像中，周期性的抽出像素点重构到低分辨率图像中，即将图像相邻的四个位置进行堆叠，
    聚焦wh维度信息到c通道中，增大每个点的感受野，减少原始信息的丢失，该模块的设计主要是减少计算量加快速度
    Focus wh information into c-space 把宽度w和高度h的信息整合到c空间中
    1. 先做4个slice 再concat 最后再做Conv
    2. slice后 (b,c1,w,h) -> 分成4个slice 每个slice(b,c1,w/2,h/2)
    3. concat(dim=1)后 4个slice(b,c1,w/2,h/2)) -> (b,4c1,w/2,h/2)
    4. conv后 (b,4c1,w/2,h/2) -> (b,c2,w/2,h/2)
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # 假设x = [1,2,3,4,5,6,7,8,9] x[::2] = [1,3,5,7,9] 间隔2个取样
        # x[1::2] = [2, 4, 6, 8] 从第二个数据开始，间隔2个取样
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))
```

## 3. Bottleneck  

标准的bottleneck模块，用在构建BottleneckCSP和C3等模块中，包含shortcut，起到加深网络的作用。  

![bottleneck.png](..%2F..%2Fimages%2Fbottleneck.png)

```python
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
```

## 4. C3  

C3模块其实是简化版的BottleneckCSP，该部分除了Bottleneck之外，只有3个卷积模块，可以减少参数，所以取名C3。  

![c3.png](..%2F..%2Fimages%2Fc3.png)

```python
class C3(nn.Module):
    # C3() is an improved version of CSPBottleneck()
    # It is simpler, faster and lighter with similar performance and better fuse characteristics
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
```  

## 5. SPP

SPP层将更多不同分辨率的特征进行融合，在送入网络neck之前能够得到更多的信息。  

![spp.png](..%2F..%2Fimages%2Fspp.png)

```python
class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        # cv2的输入channel数，等于c_乘以4（4个不同的分辨率的feature map进行融合）
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
```  

## 6. SPPF  

SPP-Fast顾名思义就是为了保证准确率相似的条件下爱，减少计算量，以提高速度，使用3个5×5的最大池化，代替原来的5×5、9×9、13×13最大池化。  

![sppf.png](..%2F..%2Fimages%2Fsppf.png)

```python
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

```

# 参考链接  

* [YOLOv5-6.x 网络模型&源码解析](https://blog.csdn.net/weixin_43799388/article/details/123271962?spm=1001.2014.3001.5502)