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


## 监督学习  

### K近邻算法(KNN)  

该算法主要采用距离算法将未分类的测试数据与样本数据进行运算得出K个距离最近的数据，并根据少数服从
多数原则返回最终的分类结果，其中相关的实现代码可以参考如下。  

* [Numpy原始算法](./knnWithRaw.py)  
* [sklearn库使用](./knnWithMglearn.py)  

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

* [Numpy原始算法](./treeWithRaw.py)  
* [sklearn库使用](./treeWithMglearn.py)  

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


## 无监督学习  


## 其他算法与工具  

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

### 依赖工具  

1. [matplotlib可视化](https://www.matplotlib.org.cn/)  
