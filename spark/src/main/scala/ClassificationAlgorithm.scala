import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier

/**
  * 分类算法
  */
object ClassificationAlgorithm {
    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("classification").setMaster("local")
        val spark = SparkSession.builder().config(conf).getOrCreate()

        val df = spark.read.format("parquet").load("data/binary-classification")
            .selectExpr("features", "cast(label as double) as label")
        
        val Array(train, test) = df.randomSplit(Array(0.5, 0.5))
        
        /**
          * 逻辑回归是一种线性模型，为输入的每个特征赋以权重之后将他们组合在一起，从而获得该输入属于特定类的概率。
          */
        // val lr = new LogisticRegression().setFamily("binomial")
        //       .setElasticNetParam(0.7).setMaxIter(10)
        // val lrModel = lr.fit(train)
        // lrModel.transform(test).show()

        /**
          * 决策树是一种更友好和易于理解的分类方法，因为它类似人类经常使用的简单决策模型。
          */
        // val dt = new DecisionTreeClassifier()
        // val dtModel = dt.fit(train)
        // dtModel.transform(test).show()

        /**
          * 随机森林，我们只训练大量的树，然后平均他们的结果做出预测。利用梯度提升树，每棵树进行加权预测。
          */
        // val rfClassifier = new RandomForestClassifier()
        // val rfModel = rfClassifier.fit(train)
        // rfModel.transform(test).show()
          
        // val gbtClassifier = new GBTClassifier()
        // val gbtModel = gbtClassifier.fit(train)
        // gbtModel.transform(test).show()

        /**
          * 朴素贝叶素分类是基于贝叶斯定理的分类方法，通常用于文本或文档分类，所有的输入特征必须为非负数。
          */
        // val nb = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
        // val nbModel = nb.fit(train)
        // nbModel.transform(test).show()

        /**
          * 线性支持向量机
          */
        // val lsvc = new LinearSVC()
        // val lsvcModel = lsvc.fit(train).setFeaturesCol("features")
        // lsvcModel.transform(test).show()

        /**
          * 多层感知分类器
          */
        // 网络层数，即3个特征点，2个节点数分别为7与6的两隐藏层，输出层2个节点（即二分类）
        // val layers = Array[Int](3, 7, 6, 2)
        // val mlp = new MultilayerPerceptronClassifier().setFeaturesCol("features")
        //       .setLayers(layers).setMaxIter(100)
        // val mlpModel = mlp.fit(train)
        // mlpModel.transform(test).show()

        /**
          * 对于二分类，我们使用BinaryClassificationEvaluator，它支持优化两个不同的指标 areaUnderRoc和 areaUnderPR
          * 对于多分类，需要使用MulticlassClassificationEvaluator，它支持优化 f1 weightedPrecision weightedRecall accuracy
          */
    }
}