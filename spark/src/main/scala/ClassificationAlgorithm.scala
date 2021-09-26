import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.classification.NaiveBayes

/**
  * 分类算法
  */
object ClassificationAlgorithm {
    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("ScalaSparkML").setMaster("local")
        val spark = SparkSession.builder().config(conf).getOrCreate()

        val bInput = spark.read.format("parquet").load("data/binary-classification")
            .selectExpr("features", "cast(label as double) as label")
        
        /**
          * 逻辑回归是一种线性模型，为输入的每个特征赋以权重之后将他们组合在一起，从而获得该输入属于特定类的概率。
          * 
          * 超参数：
          *   family：可以设置为`multinomial`（多分类）或`binary`（二分类）
          *   elasticNetParam：从0到1的浮点值。该参数依照弹性网络正则化的方法将L1正则化和L2正则化混合（即两者的线性组合）
          *   fitIntercept：此超参数决定是否适应截距
          *   regParam：确定在目标函数中正则化项的权重，它的选择和数据集的噪声情况和数据维度有关
          *   standardization：可以为true或false，设置它用于决定在将输入数据传递到模型之前是否要对其标准化
          * 训练参数：
          *   maxIter：迭代次数
          *   tol：此值指定一个用于停止迭代的阈值
          *   weightCol：权重列的名称，用于赋予某些行更大的权重
          * 预测参数
          *   threshold：此参数事预测时的概率阈值，你可以根据需要调整此参数以平衡误报和漏报
          * 
          * 对于多项式分类模型（多分类） lrModel.coefficientMatrix和lrModel.interceptVector可以用来得到系数和截距值
          */
        // val lr = new LogisticRegression()
        // val lrModel = lr.fit(bInput)

        // println(lrModel.coefficients) // 输出 系数
        // println(lrModel.intercept) // 输出 截距

        /**
          * 决策树是一种更友好和易于理解的分类方法，因为它类似人类经常使用的简单决策模型。
          * 
          * 超参数：
          *   maxDepth：指定最大深度
          *   maxBins：确定应基于连续特征创建多少个槽，更多的槽提供更细的粒度级别
          *   impurity：不纯度表示是否应该在某叶子结点拆分的度量（信息增益），此参数可以设置为entropy或者gini
          *   minInfoGain：此参数确定可用于分割的最小信息增益
          *   minInstancePerNode：此参数确定需要在一个节点结束训练的实例最小数目
          * 训练参数：
          *   checkpointInterval：检查点是一种在训练过程中保存模型的方法，此方法可以保证当集群节点因某种原因奔溃时不影响整个训练过程
          */
        // val dt = new DecisionTreeClassifier()
        // val dtModel = dt.fit(bInput)
        // println(dtModel.explainParams())

        /**
          * 随机森林，我们只训练大量的树，然后平均他们的结果做出预测。利用梯度提升树，每棵树进行加权预测。
          * 随机森林超参数：
          *   numTrees：用于训练的树总数
          *   featureSubsetStrategy：此参数确定拆分时应考虑多少特征
          * 梯度提升树超参数：
          *   lossType：损失函数，目前仅支持logistic loss损失
          *   maxIter：迭代次数
          *   stepSize：代表算法的学习速度
          */
          // val rfClassifier = new RandomForestClassifier()
          // println(rfClassifier.explainParams())
          // val rfModel = rfClassifier.fit(bInput)
          
          // val gbtClassifier = new GBTClassifier()
          // println(gbtClassifier.explainParams())
          // val gbtModel = gbtClassifier.fit(bInput)

          /**
            * 朴素贝叶素分类是基于贝叶斯定理的分类方法，通常用于文本或文档分类，所有的输入特征碧玺为非负数。
            * 超参数：
            *   modelType：可选bernoulli或multinomial
            *   weightCol：允许对不同的数据点赋值不同的权重
            * 训练参数：
            *   smoothing：它指定使用加法平滑时的正则化量，改设置有助于平滑分两类数据
            */
            // val nb = new NaiveBayes()
            // println(nb.explainParams())
            // val nbModel = nb.fit(bInput)

          /**
            * 对于二分类，我们使用BinaryClassificationEvaluator，它支持优化两个不同的指标 areaUnderRoc和 areaUnderPR
            * 对于多分类，需要使用MulticlassClassificationEvaluator，它支持优化 f1 weightedPrecision weightedRecall accuracy
            */
    }
}