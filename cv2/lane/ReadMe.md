# 直线检测

## 一、环境准备

需要安装对应OpenCV的扩展库  

```bash
pip install opencv-contrib-python
```

## 二、算法模型介绍

### 2.1 霍夫变换（Hough Transform）算法模型

霍夫变换用于直线检测的原理基于一个简单的思想：在图像空间中形成一条直线的点，在参数空间中会相交于一点。这个原理允许霍夫变换将图像空间中的
直线检测问题转换为参数空间中的峰值检测问题。其也存在以下几个缺点：  

* 计算密集，特别是对于高分辨率图像;
* 对于非线性形状或复杂形状的检测效率较低;

其算法在OpenCV中已经具备对应的算法模型库，可以直接调用进行分析，可以参考[示例代码](example.py)中的`hough_line`方法。
其中组合使用了`Canny`边缘检测算法，其是一种非常流行的边缘检测算法，用于从图像中提取有用的结构信息并减少要处理的数据量。Canny边缘检测器是一
个多阶段的算法，能够检测到图像中的宽边缘和细边缘。

```python
edges = cv2.Canny(image, threshold1, threshold2, apertureSize=L2gradient)
```
其中函数中对应的参数说明以及使用注意事项如下：  
* image：输入图像。它应该是一个灰度图像。
* threshold1：第一个阈值。
* threshold2：第二个阈值。
* apertureSize（可选）：Sobel算子的大小，默认值为3。
* L2gradient（可选）：布尔值，若为True，则使用更精确的L2范数进行计算，否则使用L1范数（默认为False）。   

阈值threshold1和threshold2用于Canny检测器的滞后阈值过程。一般来说，较小的阈值会检测到更多的边缘，但也可能包括一些噪声或不重要的边缘。选择
合适的阈值对于获得好的结果非常重要。使用过程中需要注意其需要输入的图像为灰度图，且算法对噪声较为敏感，建议采用高斯滤波等其他滤波算法降低图像噪声。
核心的霍夫变换算法的使用以及介绍如下。  

```python
lines = cv2.HoughLines(image, rho, theta, threshold)
```
* image：输入图像，这应该是一个二值图像，通常是边缘检测的结果;  
* rho：累加器的距离分辨率（以像素为单位）。这个参数决定了累加器数组中的行数;  
* theta：累加器的角度分辨率（以弧度为单位）。这个参数决定了累加器数组中的列数;  
* threshold：累加器的阈值参数。只有那些累加器值高于此阈值的直线才会被返回。增加此值可以减少检测到的直线数量，但可能错过重要的直线;  

需要注意`rho`和`theta`的选择对检测结果有显著影响。较小的`rho`和`theta`值会增加检测直线的精度，但同时也增加计算复杂度。  

### 2.2 概率霍夫变换（Probabilistic Hough Transform）

概率霍夫变换的基本原理与传统霍夫变换相似，都是将图像空间中的直线检测问题转换为参数空间中的峰值检测问题。不同之处在于，概率霍夫变换不是对图像中的
所有边缘点进行考虑，而是随机选取一部分点来估计直线参数。由于采用随机抽样，结果可能会有一定的随机性，特别是在边缘点较少的情况下。  

其算法在OpenCV中已经具有对应的算法模型库，可以直接调用进行分析，可以参考[示例代码](example.py)中的`hougnp_line`方法。

```python
lines = cv2.HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap)
```  
* image：输入图像，这应该是一个二值图像，通常是边缘检测的结果;
* rho：累加器的距离分辨率（以像素为单位）;
* theta：累加器的角度分辨率（以弧度为单位）;
* threshold：累加器的阈值参数。只有那些累加器值高于此阈值的线段才会被返回;
* minLineLength：线段的最小长度。比这个长度短的线段会被忽略;
* maxLineGap：线段上允许的最大间隙。如果两个线段之间的间隙小于这个值，它们会被认为是同一条线段;

### 2.3 LSD（Line Segment Detector）
其算法是一种高效且精确的直线检测方法，它直接在图像空间中识别和验证线段，而不依赖于传统的边缘检测和参数空间变换。通过计算图像中每个像素的梯度强度和
方向，LSD算法从高梯度区域开始，沿着一致的梯度方向生长线段，并使用几何和统计测试来验证这些线段。这种方法以子像素精度提供高度准确的线段检测，几乎不
需要用户调整参数，使其在机器视觉、图像分析以及地图制作等领域中非常有用。尽管如此，它在处理复杂背景或线段密集的图像时可能面临挑战。  

该算法在OpenCV中具备对应的算法，可以通过学习[示例代码](/example.py)中的`lsd_line`方法了解其使用，关于`createLineSegmentDetector`
的详细使用以及介绍如下。  

```python
lsd = cv2.createLineSegmentDetector(_refine, _scale, _sigma_scale, _quant, _ang_th, _log_eps, _density_th, _n_bins)
```
* refine：控制检测到的线段的细化方式;
* scale：图像的缩放比例;
* sigma_scale：用于计算图像梯度的高斯滤波器的标准差;
* quant：角度量化的精度;
* ang_th：线段的最小角度阈值;
* log_eps：检测线段时的对数精度阈值;
* density_th：线段的密度阈值;
* n_bins：用于累加器数组的箱数;

### EDLines算法模型

由于其相对C++的版本更贴合实际的场景，为了我们需要依赖[ED_Lib](https://github.com/CihanTopal/ED_Lib)并通过C++语言进行开发调试。 关于C++的编写示例可以参考[OpenCVApp](https://github.com/OrvilleX/OpenCVApp/blob/master/OpenCVApp.cpp)文
件中的`EDLinesTest`函数即可。  

### FLD（Fast Line Detector）
其是一种用于直线检测的算法，它被设计为快速且高效，特别适用于实时应用。FLD算法的主要目标是在图像中快速准确地检测出直线，同时保持对不同条件的鲁棒性，
如不同的光照条件、图像噪声等。  

该算法在OpenCV中具备对应的算法，可以通过学习[示例代码](example.py)中的`fld_line`方法了解其使用，关于`createFastLineDetector`
的详细使用以及介绍如下。  

```python
fld = cv2.ximgproc.createFastLineDetector(_length_threshold, _distance_threshold, _canny_th1, _canny_th2, _canny_aperture_size, _do_merge)
```
* _length_threshold：检测到的线段的最小长度;
* _distance_threshold：当合并线段时，线段之间允许的最大间隙;
* _canny_th1, _canny_th2：Canny边缘检测器的阈值;
* _canny_aperture_size：Canny算子的孔径大小;
* _do_merge：是否合并相似的线段;

### LSWMS算法模型  

LSWMS（Line Segment and Width Maximization Segmentation）是一种用于直线检测的算法。它主要用于图像处理和计算机视觉领域，特别是在需要从图
像中提取直线特征的场景中。LSWMS 算法的核心思想是通过局部搜索和宽度最大化来识别和提取图像中的直线段。

* 局部搜索: 从图像中的一个点开始，进行局部搜索。 它使用一个小的窗口在图像中移动，寻找可能的直线段的起点。
* 宽度最大化: 一旦确定了可能的直线段起点，算法尝试通过“宽度最大化”原则来扩展这个直线段。 它沿着预测的直线方向移动，同时调整直线的方向和长度，以最大化
覆盖该直线的像素点的数量。
* 直线段提取: 算法通过比较直线段覆盖的像素点与周围背景的对比度来评估直线段的质量。 高质量的直线段被保留下来，作为最终结果。

由于其目前尚未提供对应的Pyhton的版本，可以通过笔者其他的C++的项目进行学习了解，参考[OpenCVApp](https://github.com/OrvilleX/OpenCVApp/blob/master/OpenCVApp.cpp)
中的`LSWMSTest`方法。  

## 参考文献

* [直线检测算法汇总](https://mp.weixin.qq.com/s/eOLTnrIPaMoiRneN_CbaIA)
* [直线检测算法](https://zhuanlan.zhihu.com/p/500594323?utm_id=0)
* [车道线检测SOTA模型CLRNet复现](https://aistudio.baidu.com/projectdetail/6724057?channelType=0&channel=0)