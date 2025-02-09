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

### 挖掘频繁项集  

如果读者看过《数据挖掘 概念与技术》书，大家肯定可以看到对于数据挖掘中其中一个重要的部分就是挖掘数据其中的规律。其中比较重要的就是挖掘
其中频繁出现的数据组合以及他们的关联关系，当然这需要建立对于业务的深入了解的基础上，在此基础上我们就可以采用对应的非监督学习的算法便于
从众多的数据组合中挖掘我们感兴趣的频繁项集出来从而便于我们更好的分析数据其中的奥秘，下面我们将介绍常用的Apriori算法和FP-growth算法。  

首先介绍的Apriori算法是发现频繁项集的一种方法。该算法的三个输入参数分别是数据集、最小支持度和最小置信度。该算法首先会生成所有单个物品
的项集列表。接着扫描记录来查看哪些项集满足最小支持度要求，那些不满足最小支持度的集合会被去掉。然后，对剩下来的集合进行组合以生成包含两个
元素的项集。接下来，再重新扫描记录，去掉不满足最小支持度的项集。该过程重复进行直到所有项集都被去掉。由于sklearn并没有包含该类算法，所以
读者需要额外安装其他库`pip install efficient-apriori`进行安装，然后可以按照如下算法进行编写：  

* [Numpy原始算法](/frequentItemsets/aprioriWithRaw.py)  
* [第三方库算法](/frequentItemsets/aprioriWithLib.py)  

Apriori算法对于每个潜在的频繁项集都会扫描数据集判定给定模式是否频繁，这在小数据量的情况下并不会存在问题，但是当我们需要面对更大数据集的
时候，这个问题将会被放大。由此我们就需要一个更高效的算法，即FP-growth算法，该算法能够高效地的发现频繁项集，并发现关联规则。
FP-growth只需要对数据库进行两次扫描即可，所以整体提升的效率非常可观。由于sklearn本身没有提供该算法API，所以读者需要安装额外的库进行
支持，如`pip install pyfpgrowth`。  

* [Numpy原始算法](/frequentItemsets/fpgrowthWithRaw.py)  
* [第三方库算法](/frequentItemsets/fpgrothWithLib.py)  

## 特征工程

其核心可以理解是为了解决特定应用的最佳数据表示问题，它是数据科学家和机器学习从业者尝试解决现实世界问题时的主要任务之一。用正确的方式表示
数据，对监督模型性能的影响比所选择的精确参数还要大。  

### 分类变量  

前面的例子我们一直假设数据是由浮点数组成的二维数组，其中每一列是描述数据点的连续特征（conmtiinuous feature）。对于许多应用而言，数据
的搜集方式并不是这样。一种特别常见的特征类型就是分类特征（categorical feature），也叫离散特征（disccrete feature）。为了表示这种
类型数据，我们最常用的方法就是one-hot编码（one-hot-encoding）或N取一编码（one-out-of-N encoding），也叫虚拟变量（dummy vari
able）。  

其背后的思想就是将一个分类变量替换为一个或多个新特征，新特征取值为0和1。这里我们可以举例如公司性质特征，可以存在国有，私有，外资等类型，为了
利用one-hot来表示，我们将该特征替换为多个特征，即国有企业特征、私有企业特征和外资企业特征，对于每个数据如果属于对应类型，则对应特征值为1，
其他特征为0。下面我们将利用第三方类库来帮助我们实现这一目的。  

* [基于pands的One-Hot算法](/featureengineering/onehot.py)  

由于pandas的get_dummies函数将所有数字看作是连续的，不会为其创建虚拟变量。为了解决这个问题，你可以使用scikit-learn的OneHotEncoder，
指定哪些变量是连续、哪些变量是离散的，你也可以将数据框中的数值转换为字符串。当然利用get_dummies也是可以办到了，比如下面这样使用：  

```python
demo_frame['Integer Feature'] = demo_freame['Integer Feature'].astype(str)
pd.get_dummies(demo_frame, columns=['Integer Feature', 'Categorical Feature'])
```  

### 分箱、交互与多项式特征    

通过前面学习我们知道，线性模型只能对线性关键建模，对于单个特征的情况就是直线。决策树可以构建更为复杂的数据模型，但这强烈依赖于数据表示。
有一种方法可以让线性模型在连续数据上变得更加强大，就是使用特征分箱（binning，也叫离散化）将其划分为多个特征。比如某个特征具备输入的范围
（10~20），那么我们就可以将其划分为固定的几个箱子。如10个，那么每个箱子都是均匀的划分了这些值，具体如何使用第三方类库完成我们可以参入下
面的代码。  

* [基于pands的分箱算法](/featureengineering/binningAndFeature.py)  

通过上面示例的方法执行后我们可以看到最终模型与决策树一致，如果我们想要丰富特征表示，此时我们就需要添加原始数据的交互特征（interaction feature）
与多项式特征（polynomial feature），最终代码我们依然通过上述代码文件中进行表现，具体方法可以参考`interactionMain`方法。最后就是多项式特征
其也比较好理解，就是基于原始数据的平方、立方等组成多个特征的数据，具体可以参考`polynomialMain`方法。  


## 其他算法与工具  

### 扩展算法  

1. [黎曼和估算与面积法](https://zhuanlan.zhihu.com/p/76304788)  


### 特征归一化  

很多算法对数据存在敏感性，由于数据计量单位的不一致性很有可能会被某些数值较大的数据破坏最终的算法效果，为此
我们需要对数据进行归一化处理从而保证算法的效率。首先我们可以通过以下文档进一步的了解归一化的作用：  

* [常见归一化算法](https://blog.csdn.net/zenghaitao0128/article/details/78361038)  
* [归一化的解释](https://www.zhihu.com/question/20455227)  
* [机器学习「输出概率化」：一种无监督的方法](https://zhuanlan.zhihu.com/p/33873947)  

了解完以上的基本概念后，具体使用的算法由以下几种：  

1. 最大最小标准化方法
2. [Z-score标准化方法](https://blog.csdn.net/Orange_Spotty_Cat/article/details/80312154)
3. 非线性归一化
4. L2范数归一化方法  

对于如何使用sklearn可以参考本[示例代码](preprocessing/scaler.py)  

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

