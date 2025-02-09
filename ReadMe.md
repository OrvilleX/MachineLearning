# 机器学习 

项目整体调整中...

## 新目录

### TTS解决方案

* [Kokore适合边缘设备的TTS解决方案](./kokore/ReadMe.md)  

### 图片特征提取

* [SigLIP 图文对照模型](./siglip/ReadMe.md)  
* [InternVideo2 多模态视频理解模型](./internvideo/ReadMe.md)  


#### 注意以下文档链接可能需要访问外网，所以请保证网络的正常。  

> 对于`Spark ML`的具体教程将在[独立的文档](./spark/ReadMe.md)中进行说明，不在本教程上下进行介绍。  

## 一、项目情况  

### 前言

本教程纯属个人利用业务时间进行积累学习，同时考虑到大多数场景下以应用为主，所以直接采用Numpy
进行相关成熟算法的编写并使用并不符合高效的开发原则，所以这里采用结合了Numpy算法以及sklearn
库对比的方式进行实现，从而保证读者可以在深入算法的研究与快速应用之间选择其一。  

由于本人的数学理论基础并不是特别扎实如果存在相关理论或者说法的错误欢迎各位大神进行指点并提交
相关的PR从而使该教程更完善。  

### 安装方式

为了能够顺利的运行使用本项目的各类代码并能够进行开发测试验证，需要首先在设置好对应的虚拟环境后
通过以下指令安装所需要的各类依赖库。  

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

### 项目结构  

本项目将按照具体的类库、平台进行区分，当前主要包含如下目录。  

* cnn: 卷积神经网络相关算法  
* featureengineering: 特征工程相关  
* frequentltemsets: 频繁项集挖掘相关算法  
* raw：基于numpy实现的原始算法  
* sklearn：基于scikit-learn库快速实现  
* spark：基于Spark ML的机器学习实现  
* scipy: 基于SciPy的科学计算使用，[具体说明](./scipy/ReadMe.md)    
* utils：项目额外补充的工具类  

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

### 2.2 其他图像检测算法

* [直线检测算法集合](./docs/cnn/LaneDetection.md)  

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

## 三、监督学习  

以下算法大多数使用场景主要为分类问题，部分算法基于回归可以实现预测行为。  

### 3.1 监督学习方面



### K近邻算法(KNN)  

该算法主要采用距离算法将未分类的测试数据与样本数据进行运算得出K个距离最近的数据，并根据少数服从
多数原则返回最终的分类结果，其中相关的实现代码可以参考如下。  

* [基于numpy的knn算法](./raw/knn.py)  
* [基于sklearn的knn算法](./sklearn/knn.py)  

由于本算法的核心依赖距离算法，所以根据实际的使用场景选择适当的距离算法进行替换，当前可以利用的距离算法主要有：  

1. 欧式距离  
2. 马氏距离  
3. 曼哈顿距离  
4. 切比雪夫距离  
5. 闵可夫斯基距离  
6. 夹角余弦  
7. 汉明距离  
8. 杰卡德距离  

关于各类算法的介绍可以参考[本文章](https://www.cnblogs.com/soyo/p/6893551.html)

### 决策树  

该算法其实就是采用IF...THEN...ELSE形式进行组织，从而形成一个树结构。从人类本身的认知出发一个问题
往往会有多个选择，但是考虑计算机本身的特点以及效率等，往往会会采用二叉树。相关的算法可以通过如下文件
进行学习：  

* [Numpy原始算法](raw/tree.py)  
* [sklearn库使用](sklearn/tree.py)  
* [随机森林](raw/treePlotter.py)  

树回归（预测）相关源码  
* [Numpy原始算法](raw/regTrees.py)  

以上已经实现的ID3与CART算法，至此还剩下C4.5算法，关于三种算法的具体原理介绍可以参考如下文章：  

1. [ID3算法](https://www.infoq.cn/article/ZXig7JMhPzeH97zM5l0z)  
该算法依赖[信息嫡](https://zh.wikipedia.org/wiki/%E7%86%B5_(%E4%BF%A1%E6%81%AF%E8%AE%BA))
与 
[信息增益](https://www.zhihu.com/question/22104055)
进行特征选择从而进行树的创建。  
2. [C4.5算法](https://blog.csdn.net/zhihua_oba/article/details/70632622)  
3. [CART算法](https://blog.csdn.net/ACdreamers/article/details/44664481)

### 朴素贝叶斯  

关于该算法的基本介绍见[本文](https://zhuanlan.zhihu.com/p/26262151) 下面我们将主要介绍
其算法的实现：  

* [Numpy原始算法](raw/bayes.py)  
* [sklearn库使用](sklearn/bayes.py)  

### 逻辑回归（Logistic回归）  

首先在了解具体的逻辑回归算法前我们需要先了解几个相关的基础知识算法以便于我们更好的去
了解其源码的实现逻辑和最终应用场景的选择。  

* [sigmoid算法](https://www.jianshu.com/p/506595ec4b58)  

由于我们这里处理的是二分类问题，对于Logistic最终4计算的结果值，我们需要将其限定在一个实当的范围内
从而实现分类，其实就是计算概率，如果概率大于一半进行选择。  

* [梯度上升算法简单理解](https://www.jianshu.com/p/eb94c60015c7)  
* [梯度下降算法](https://www.cnblogs.com/sench/p/9817188.html)  

因为本身函数的特点，我们需要计算回归系数，这里我们主要采用了梯度上升算法进行计算，当然读者也可以
采用梯度下降算法。但是梯度上升算法在数据量较多时，由于其本身需要进行全数据的迭代所以我们还需要引
入一个更高效的算法，即随机梯度提升算法，可以仅用一个样本点更新回归系数。下面我们可以参考具体的实
现代码进行学习：  

* [基于numpy的逻辑回归算法](sklearn/logisticRegression.py)  
* [基于sklearn的逻辑回归算法](raw/logisticRegression.py)  

### 支持向量机（SVM）  

关于该算法比较好的解释可以参考[SVM原理](https://www.cnblogs.com/LeftNotEasy/archive/2011/05/02/basic-of-svm.html) 
文章。从个人简短的理解来说，该算法在面对非线性问题下，将采用超平面，即多维空间进行数据分类的切分。以下这个视频将可以较形式的展示这一
过程[演示视频](https://v.qq.com/x/page/k05170ntgzc.html)  

为了训练模型我们此时需要引入[SMO算法](https://www.jianshu.com/p/eef51f939ace) （序列最小优化算法）来解决二次规划问题，当然
如果读者并不想过多接触具体的核心算法逻辑，可以参考具体的实现源码进行学习应用：  

* [支持向量机](sklearn/SVM.py)  

### 线性回归  

对于了解线性回归可以参考[本文章](https://zhuanlan.zhihu.com/p/53979679) 进行相关关系，其中还包含了关于局部加权线性回归方式，
针对常规的数据这是没有问题的，但是如果样本数据中的特征多余样本数据本身那么就存在问题了，此时我们就需要通过引入岭回归、lasso与前向
逐步回归算法。  

* [基于sklearn的线性回归](sklearn/lineRegression.py)  
* [基于numpy的线性回归](raw/lineRegression.py)  

### 神经网络  

这里我们以入门的MLP（多层感知机）为例，相关的代码可以参考：  

* [基于sklearn的MLP使用](sklearn/MLP.py)  

### 元计算  

上面我们介绍了多个分类算法，为了得到最好的结果我们有时可能需要结合不同算法或相同算法不同配置进行组合，这就叫做元计算。本篇将主要介绍
同算法不同配置的情况，主要采用的是[adaboost算法](https://blog.csdn.net/px_528/article/details/72963977) ，对应的源码参考如下：  

* [numpy原始算法](raw/adaboost.py)  

## 无监督学习  

### 降维技术  

下面我们将开始通过术语介绍各类降维技术。第一种降维的方法称为主成分分析，在PCA中，数据从原来的坐标系转换到了新的坐标系，新坐标系的选择是由
数据本身决定的。第一个新坐标轴选择在的是原始数据中方差最大的方向，第二个新坐标轴的选择和第一个坐标轴正交且最大方差的方向。该过程一直重复，
重复次数为原始数据中特征的数目。我们会发现，大部分方差都包含在最前面的几个新坐标轴中。  

另外一种降维技术是因子分析。在因子分析中，我们假设在观察数据的生成中有一些观察不到隐变量。假设观察数据是这些隐变量和某些噪声的线性组合。
那么隐变量的数据可能比观察数据的数目少，也就是说通过找到隐变量就可以实现数据的降维。  
还有一种降维技术就是独立成分分析。ICA假设数据是从N个数据源生成的，这一点和因子分析有些类似。假设数据为多个数据源的混合观察结果，这些数
据源之间在统计上是相互独立的，而在PCA中只假设数据是不相关的。同因子分析一样，如果数据源的数目少于观察数据的数目，则可以实现降维过程。  

* [Numpy原始算法](preprocessing/PCAWithRaw.py)  
* [Sklearn算法](preprocessing/PCA.py)  

PCA本身也由于其算法的特点，并不能满足所有的场景。这里我们将学习另一个非负矩阵分解（NMF）算法，它主要用于提取有用的特征，它的工作原理
类似于PCA，也可以用于降维。其特点在将数据分解成非负加权求和的这个过程，对由多个独立源相加创建而成的数据特别有用，比如多人说话的音轨或
包含多种乐器的音乐，在这种情况下，NMF可以识别除组成合成数据的原始分量。如果读者希望了解更多关于NMF的知识，可以参考
[本文章](https://zhuanlan.zhihu.com/p/27460660)  

* [Sklearn算法](/preprocessing/NMF.py)  

`注意其中我们使用了图像，但是其库将会自动下载，考虑到实际网速原因，需要读者自行下载(百度fetch_lfw_people接口)
安装包然后解压到C:\Users\[用户]\scikit_learn_data目录下`  

最后我们介绍t-SNE算法。虽然PCA通常是用于变换数据的首选方法，但这一方法的性质限制了其有效性。而有一类用于可视化的算法叫做流形学习算法
，它允许进行更复杂的映射，通常也可以给出更好的可视化。  

* [Sklearn算法](/preprocessing/t-SNE.py)

最后需要介绍介绍的就是SVD算法，由于该算法本身已在numpy提供了，所以并不需要具体的实现算法进行单独的介绍，具体的使用方式可以参考如下
的方式：  

```python
from numpy import linalg
U,Sigma,VT = linalg.svd(mat(data))
```

在很多情况下，数据中的一小段携带了数据集中的大部分信息，其他信息则要么是噪声，要么就是毫不相关的信息。为了提取中其中重要的数据，我们
需要使用到矩阵分解技术，而最常见的则是SVD分解技术。SVD将原始数据集矩阵A分解成三个矩阵，从而

### K均值聚类  

k均值聚类是最简单也最常用的聚类算法之一。它试图找到代表数据特定区域的簇中心。算法交替执行如下两个步骤：将每个数据点分配给最近的簇中心，
然后将每个簇中心设置为所分配的所有数据点的平均值。如果簇的分配不再发生变化，那么算法结束。以下将列举出对应实现的方式。  

* [Numpy原始算法](/raw/KMeans.py)  
* [Sklearn算法](/sklearn/KMeanss.py)  

### 凝聚聚类  

该算法指的是许多基于相同原则构建的聚类算法，这一原则是：算法首先声明每个点是自己的簇，然后合并两个最相似的簇，直到满足某种停止准则为止。
其中sk库的停止准则是簇的个数，因此相似的簇被合并，直到仅剩下指定个数的簇。其中库实现了以下三种选择。  

* ward：默认选项。ward挑选两个簇来合并，使得所有簇中的方差增加最小。这通常会得到大小差不多相等的簇。  
* average：该链接将簇中所有点之间平均距离最小的两个簇合并。  
* complete：该链接将簇中点之间最大距离最小的两个簇合并。  

ward适用于大多数数据集，如果簇中的成员个数非常不同，那么average或complete可能效果更好。以下为具体的算法代码：  

* [Sklearn算法](/sklearn/cluster/agglomerative.py)  

### 密度聚类

该算法名叫DBSCAN（具有噪声的基于密度的空间聚类应用），该算法主要优点在于不需要用户先验地设置簇的个数，可以划分具有复杂形状的簇，还可以
找出不属于任何簇的点。DBSCAN比凝聚聚类和K均值稍慢，但仍可以扩展到相对较大的数据集。  

它主要的参数需要min_samples和eps。算法首先任意选取一个点，然后找到这个点的距离小于eps的所有的点。如果距起始点的距离在eps之内的数据点
个数小于min_samples，那么这个点被标记为噪声，也就是说它不属于任何簇。如果距离在eps之内的数据点个数大于min_samples，则这个点被标记为
核心样本，并被分配一个新的簇标签。然后访问该点的所有邻居，如果它们没有被分配一个簇，那么就分配刚刚创建的新的簇标签。如果它们是核心样本，
那么就以此访问其邻居，以此类推。  

* [Sklearn算法](/sklearn/cluster/Dbscan.py)  

### 聚类算法对比评估  

为了论证不同聚类算法的性能，为此我们需要利用一些评估方式。其中可用于聚类算法相对于真实聚类的结果，其中最重要的是调整rand指数（ARI）
和归一化互信息（NMI），二者都给出了定量的度量，其最佳值为1，0标识不相关的聚类。  

以上两种评估方式存在一个很大的问题就是需要真实值来比较结果。但是实际情况可能并没有真实值进行评估，此时我们就需要利用轮廓系数。但它们
在实践中的效果并不好。轮廓分数计算一个簇的紧致度，其值越大越好，最高分数为1。  

* [Sklearn算法](/sklearn/cluster/assessment.py)  

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

