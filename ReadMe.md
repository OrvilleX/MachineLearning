# 机器学习 

本项目以应用为主出发，结合了从基础的机器学习、深度学习到目标检测以及目前最新的大模型，采用目前成熟的
第三方库、开源预训练模型以及相关论文的最新技术，目的是记录学习的过程同时也进行分享以供更多人可以直接
进行使用。  

> 本人自己目前属于自己创业，目前要时围绕各类算法场景的应用开发，目前主要的领域为船舶、教育以及企业定制的开发  

## 一、目录  

对应每个案例将采用独立的文件夹的方式进行管理，非源码的可以参考对应的文档进行相关依赖的安装，部分存在源码的则可以
通过源码中对应的requirements.txt安装对应的依赖。  


* [TTS解决方案](#tts解决方案)
* [图片特征提取](#图片特征提取)

### 机器学习基础

* [基于numpy实现的机器学习算法](./numpy/ReadMe.md): 主要是讲述底层的算法的逻辑，实际使用中往往采用第三方库来实现  
* [基于sklearn的机器学习算法](./sklearn/ReadMe.md): 主要是讲述如何使用第三方类库快速使用成熟的算法  
* [预处理技术](./preprocessing/ReadMe.md): 其主要包含针对机器学习工程中针对数据的预处理的部分的算法  
* [特征工程](./featureengineering/ReadMe.md): 主要是围绕各类数据分析场景下针对数据的特征表示的算法  

### 数据挖掘

* [挖掘频繁项集](./frequentItemsets/ReadMe.md): 主要是采用numpy与sklearn的方式实现这类算法    

### TTS解决方案

* [Kokore适合边缘设备的TTS解决方案](./kokore/ReadMe.md)  

### 图片特征提取

* [SigLIP 图文对照模型](./siglip/ReadMe.md): 大量的多模态模型的图像特种提取必使用的模型，本文档基于目前主流的`siglip-so400m-patch14-384`模型进行编写，开发多模态大模型必须掌握的图像特征提取库

* [InternVideo2 多模态视频理解模型](./internvideo/ReadMe.md): 由于上海人工智能实验室（General Vision Team of Shanghai AI Laboratory）推出的针对视频理解的模型，目前针对视频理解的论文逐渐将其作为融合siglip来实现针对视频&图片场景的多模态大模型的基础组件  

### 目标检测技术

* [基于yolo目标检测系列](./yolo/ReadMe.md)  
* [DETR技术的应用方式](./detr/ReadMe.md)  
* [face_recognition人脸识别应用方式](./facerecognition/ReadMe.md)  

### 其他技术

* [Spark ML的使用方式](./spark/ReadMe.md): 目前该技术的应用场景逐步减少，本教程也是基于较老的版本进行编写，读者需要根据自己的使用
以及目前最新的文档结合进行对应的API调整。  


—————— 以下为未重构的老版本 ————————

## 二、文档目录

### 2.1 目标检测相关 (cnn)  

* [相关基本术语介绍](./docs/cnn/Basic.md)  
* [介绍关于各类NMS相关的概念以及对应的实现方式](./docs/cnn/NMS.md)  
* [关于Yolo模型中输入图片尺寸的影响分析](./docs/cnn/yolo/InputSize.md)  
* [针对Yolo训练结果的评估验证](./docs/cnn/yolo/Evaluation.md)  
* [数据增强技术的分析](./docs/cnn/DataAugmentation.md)  
* [边缘检测图像增强技术](./docs/cnn/Vague.md)  
* [yolo网络层剖析](./docs/cnn/yolo/Network.md)  
* [yolo各个版本的使用方式](./docs/cnn/yolo/Usage.md)

### 2.3 LLM大模型相关  

* [Transformer模型基础知识](./docs/llm/Transformer.md)  

### 2.4 机器学习基础

* [机器学习中的学习方式](./docs/ml/Learning.md)  

### 2.5 机器人基础

* [基础知识内容](./docs/ml/Robotics.md)

### 数据基础知识  

* [统计计算基础知识](https://www.math.pku.edu.cn/teachers/lidf/docs/statcomp/html/_statcompbook/index.html)

#### 正态分布

* [正态分布含义](https://www.zhihu.com/question/56891433)  
* [高斯分布](https://baijiahao.baidu.com/s?id=1621087027738177317&wfr=spider&for=pc)  

可使用`numpy.random中的randn、standard_normal和normal`返回随机正态分布的数组，其
中`normal`是[普遍使用](./normal/numpyTest.py)的方法。  


* [泊松分布](https://www.matongxue.com/madocs/858)  
* [伯努利分布](https://www.cnblogs.com/jmilkfan-fanguiju/p/10589773.html)  


## 其他算法与工具  

### 扩展算法  

1. [黎曼和估算与面积法](https://zhuanlan.zhihu.com/p/76304788)  

### 指标  

即衡量目标的单位或方法，这里我们列举几个在互联网中比较常见的指标进行说明：  

1. PV：页面浏览树数，即每天的点击数。
2. UV：独立用户数，即每天每个用户的浏览数。
3. DAU：日活跃用户数，即每天活跃的用户数量。  

当然指标不仅仅只有上面还有`MAU`、`LTV`和`ARPU`等，每个指标都要满足以下几点：

* 数字化
* 易衡量
* 意义清晰  
* 周期适当  
* 尽量客观  

### 依赖工具  

1. [matplotlib可视化](https://www.matplotlib.org.cn/)  
2. [训练模型持久化](https://github.com/joblib/joblib)  
3. [Sklearn中文文档](https://sklearn.apachecn.org/)  
4. [将模型持久化为PMML供Java应用运行](https://github.com/jpmml/sklearn2pmml)  
5. [Java运行PMML模型算法](https://github.com/jpmml/jpmml-evaluator)  

