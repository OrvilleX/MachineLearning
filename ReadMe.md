# 机器学习 

#### 注意以下文档链接可能需要访问外网，所以请保证网络的正常。  

## 前言  

本教程纯属个人利用业务时间进行积累学习，同时考虑到大多数场景下以应用为主，所以直接采用Numpy
进行相关成熟算法的编写并使用并不符合高效的开发原则，所以这里采用结合了Numpy算法以及sklearn
库对比的方式进行实现，从而保证读者可以在深入算法的研究与快速应用之间选择其一。  

由于本人的数学理论基础并不是特别扎实如果存在相关理论或者说法的错误欢迎各位大神进行指点并提交
相关的PR从而使该教程更完善。  

## 基础知识  

* [正态分布含义](https://www.zhihu.com/question/56891433)  
* [高斯分布](https://baijiahao.baidu.com/s?id=1621087027738177317&wfr=spider&for=pc)  
* [泊松分布](https://www.matongxue.com/madocs/858)  
* [伯努利分布](https://www.cnblogs.com/jmilkfan-fanguiju/p/10589773.html)   


## 监督学习  

以下算法大多数使用场景主要为分类问题，部分算法基于回归可以实现预测行为。  

### K近邻算法(KNN)  

该算法主要采用距离算法将未分类的测试数据与样本数据进行运算得出K个距离最近的数据，并根据少数服从
多数原则返回最终的分类结果，其中相关的实现代码可以参考如下。  

* [KNN算法](./KNN.py)  

由于本算法的核心依赖距离算法，所以根据实际的使用场景选择适当的距离算法进行替换，当前可以利用的距离、
算法主要有：  

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

* [Numpy原始算法](tree/treeWithRaw.py)  
* [sklearn库使用](tree/treeWithMglearn.py)  
* [随机森林](./randomtreeWithMglearn.py)  

树回归（预测）相关源码  
* [Numpy原始算法](./regTreesWithRaw.py)  

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

* [Numpy原始算法](./bayesWithRaw.py)  
* [sklearn库使用](./bayesWithMglearn.py)  

### Logistic回归  

首先在了解具体的Logistic回归算法前我们需要先了解几个相关的基础知识算法以便于我们更好的去
了解其源码的实现逻辑和最终应用场景的选择。  

* [sigmoid算法](https://www.jianshu.com/p/506595ec4b58)  

由于我们这里处理的是二分类问题，对于Logistic最终计算的结果值，我们需要将其限定在一个实当的范围内
从而实现分类，其实就是计算概率，如果概率大于一半进行选择。  

* [梯度上升算法简单理解](https://www.jianshu.com/p/eb94c60015c7)  
* [梯度下降算法](https://www.cnblogs.com/sench/p/9817188.html)  

因为本身函数的特点，我们需要计算回归系数，这里我们主要采用了梯度上升算法进行计算，当然读者也可以
采用梯度下降算法。但是梯度上升算法在数据量较多时，由于其本身需要进行全数据的迭代所以我们还需要引
入一个更高效的算法，即随机梯度提升算法，可以仅用一个样本点更新回归系数。下面我们可以参考具体的实
现代码进行学习：  

* [Numpy原始算法](./linearModelWithRaw.py)  
* [sklearn库使用](./linearModelWithMglearn.py)  

### 支持向量机（SVM）  

关于该算法比较好的解释可以参考[SVM原理](https://www.cnblogs.com/LeftNotEasy/archive/2011/05/02/basic-of-svm.html) 
文章。从个人简短的理解来说，该算法在面对非线性问题下，将采用超平面，即多维空间进行数据分类的切分。以下这个视频将可以较形式的展示这一
过程[演示视频](https://v.qq.com/x/page/k05170ntgzc.html)  

为了训练模型我们此时需要引入[SMO算法](https://www.jianshu.com/p/eef51f939ace) （序列最小优化算法）来解决二次规划问题，当然
如果读者并不想过多接触具体的核心算法逻辑，可以参考具体的实现源码进行学习应用：  

* [Numpy原始算法](./svmMLiA.py)  
* [sklearn库使用](./svmWithMglearn.py)  

### 线性回归  

对于了解线性回归可以参考[本文章](https://zhuanlan.zhihu.com/p/53979679) 进行相关关系，其中还包含了关于局部加权线性回归方式，
针对常规的数据这是没有问题的，但是如果样本数据中的特征多余样本数据本身那么就存在问题了，此时我们就需要通过引入岭回归、lasso与前向
逐步回归算法。  

* [numpy原始算法与sklearn库使用](./regressionWithRaw.py)  

### 元计算  

上面我们介绍了多个分类算法，为了得到最好的结果我们有时可能需要结合不同算法或相同算法不同配置进行组合，这就叫做元计算。本篇将主要介绍
同算法不同配置的情况，主要采用的是[adaboost算法](https://blog.csdn.net/px_528/article/details/72963977) ，对应的源码参考如下：  


## 无监督学习  



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
