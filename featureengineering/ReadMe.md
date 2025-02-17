# 特征工程

其核心可以理解是为了解决特定应用的最佳数据表示问题，它是数据科学家和机器学习从业者尝试解决现实世界问题时的主要任务之一。用正确的方式表示
数据，对监督模型性能的影响比所选择的精确参数还要大。  

## 分类变量  

前面的例子我们一直假设数据是由浮点数组成的二维数组，其中每一列是描述数据点的连续特征（conmtiinuous feature）。对于许多应用而言，数据
的搜集方式并不是这样。一种特别常见的特征类型就是分类特征（categorical feature），也叫离散特征（disccrete feature）。为了表示这种
类型数据，我们最常用的方法就是one-hot编码（one-hot-encoding）或N取一编码（one-out-of-N encoding），也叫虚拟变量（dummy vari
able）。  

其背后的思想就是将一个分类变量替换为一个或多个新特征，新特征取值为0和1。这里我们可以举例如公司性质特征，可以存在国有，私有，外资等类型，为了
利用one-hot来表示，我们将该特征替换为多个特征，即国有企业特征、私有企业特征和外资企业特征，对于每个数据如果属于对应类型，则对应特征值为1，
其他特征为0。下面我们将利用第三方类库来帮助我们实现这一目的。  

* [基于pands的One-Hot算法](onehot.py)  

由于pandas的get_dummies函数将所有数字看作是连续的，不会为其创建虚拟变量。为了解决这个问题，你可以使用scikit-learn的OneHotEncoder，
指定哪些变量是连续、哪些变量是离散的，你也可以将数据框中的数值转换为字符串。当然利用get_dummies也是可以办到了，比如下面这样使用：  

```python
demo_frame['Integer Feature'] = demo_freame['Integer Feature'].astype(str)
pd.get_dummies(demo_frame, columns=['Integer Feature', 'Categorical Feature'])
```  

## 分箱、交互与多项式特征    

通过前面学习我们知道，线性模型只能对线性关键建模，对于单个特征的情况就是直线。决策树可以构建更为复杂的数据模型，但这强烈依赖于数据表示。
有一种方法可以让线性模型在连续数据上变得更加强大，就是使用特征分箱（binning，也叫离散化）将其划分为多个特征。比如某个特征具备输入的范围
（10~20），那么我们就可以将其划分为固定的几个箱子。如10个，那么每个箱子都是均匀的划分了这些值，具体如何使用第三方类库完成我们可以参入下
面的代码。  

* [基于pands的分箱算法](binningAndFeature.py)  

通过上面示例的方法执行后我们可以看到最终模型与决策树一致，如果我们想要丰富特征表示，此时我们就需要添加原始数据的交互特征（interaction feature）
与多项式特征（polynomial feature），最终代码我们依然通过上述代码文件中进行表现，具体方法可以参考`interactionMain`方法。最后就是多项式特征
其也比较好理解，就是基于原始数据的平方、立方等组成多个特征的数据，具体可以参考`polynomialMain`方法。  

## 特征归一化  

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

对于如何使用sklearn可以参考本[示例代码](scaler.py)  
