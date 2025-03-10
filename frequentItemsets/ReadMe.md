# 数据挖掘

### 挖掘频繁项集  

如果读者看过《数据挖掘 概念与技术》书，大家肯定可以看到对于数据挖掘中其中一个重要的部分就是挖掘数据其中的规律。其中比较重要的就是挖掘
其中频繁出现的数据组合以及他们的关联关系，当然这需要建立对于业务的深入了解的基础上，在此基础上我们就可以采用对应的非监督学习的算法便于
从众多的数据组合中挖掘我们感兴趣的频繁项集出来从而便于我们更好的分析数据其中的奥秘，下面我们将介绍常用的Apriori算法和FP-growth算法。  

首先介绍的Apriori算法是发现频繁项集的一种方法。该算法的三个输入参数分别是数据集、最小支持度和最小置信度。该算法首先会生成所有单个物品
的项集列表。接着扫描记录来查看哪些项集满足最小支持度要求，那些不满足最小支持度的集合会被去掉。然后，对剩下来的集合进行组合以生成包含两个
元素的项集。接下来，再重新扫描记录，去掉不满足最小支持度的项集。该过程重复进行直到所有项集都被去掉。由于sklearn并没有包含该类算法，所以
读者需要额外安装其他库`pip install efficient-apriori`进行安装，然后可以按照如下算法进行编写：  

* [Numpy原始算法](aprioriWithRaw.py)  
* [第三方库算法](aprioriWithLib.py)  

Apriori算法对于每个潜在的频繁项集都会扫描数据集判定给定模式是否频繁，这在小数据量的情况下并不会存在问题，但是当我们需要面对更大数据集的
时候，这个问题将会被放大。由此我们就需要一个更高效的算法，即FP-growth算法，该算法能够高效地的发现频繁项集，并发现关联规则。
FP-growth只需要对数据库进行两次扫描即可，所以整体提升的效率非常可观。由于sklearn本身没有提供该算法API，所以读者需要安装额外的库进行
支持，如`pip install pyfpgrowth`。  

* [Numpy原始算法](fpgrowthWithRaw.py)  
* [第三方库算法](fpgrothWithLib.py)  
