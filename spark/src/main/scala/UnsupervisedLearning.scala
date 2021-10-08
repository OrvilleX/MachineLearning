import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.commons.math3.ml.clustering.evaluation.ClusterEvaluator
import org.apache.spark.ml.clustering.BisectingKMeans
import org.apache.spark.ml.clustering.GaussianMixture
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.clustering.LDA

object  UnsupervisedLearning {
    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("ScalaSparkML").setMaster("local")
        val spark = SparkSession.builder().config(conf).getOrCreate()
        
        val va = new VectorAssembler()
            .setInputCols(Array("Quantity", "UnitPrice"))
            .setOutputCol("features")
        
        val sales = va.transform(spark.read.format("csv")
            .option("header", "true")
            .option("inferSchema", "true")
            .load("data/byday/*.csv")
            .limit(50)
            .coalesce(1)
            .where("Description IS NOT NULL"))
        sales.cache()

        /**
          * k-means算法中，用户在数据集中随机选择数量为k的数据点作为处理聚类的聚类中心，未分配的点基于他们与这些
          * 聚类中心的相似度（欧氏距离）倍分配到离他们最近的聚类中。分配之后，再根据被分配到一个聚类的数据点再计算
          * 聚类的新中心，并重复该过程，直到到达有限的迭代次数或直到收敛。
          * 
          * 超参数：
          *     k：指定你希望的聚类数量
          * 训练参数：
          *     initMode：初始化模式是确定质心初始位置的算法，可支持 random（随机初始化）与k-means
          *     initSteps：k-means模式初始化所需要的步数
          *     maxIter：迭代次数
          *     tol：该阈值指定质心改变到该程度后优化结束
          */
        val km = new KMeans().setK(3)
        println(km.explainParams())
        val kmModel = km.fit(sales)

        val summary = kmModel.summary
        summary.clusterSizes // 中心点数量

        // 计算数据点与每个聚类中心点的距离，可以采用ClusterEvaluator评估器

        /**
          * 二分k-means
          * 该算法为k-means变体，关键区别在于，它不是通过“自下而上”的聚类，而是自上而下的聚类方法。就是通过创建一个组
          * 然后进行拆分直到拆分到k个组。
          * 
          * 超参数：
          *     k：指定你希望的聚类数量
          * 训练参数：
          *     minDivisibleClusterSize：指定一个可分聚类中的最少数据点数
          *     maxIter：迭代次数
          */
        // val bkm = new BisectingKMeans().setK(5).setMaxIter(5)
        // println(bkm.explainParams())
        // val bkmModel = bkm.fit(sales)
        
        // val summary = bkmModel.summary
        // summary.clusterSizes

        /**
          * 高斯混合模型
          * 一种简单理解高斯混合模型的方法是，它们就像k0means的软聚类斑斑（软聚类softclustering即每个数据点可以划分到多个聚类中），而
          * k-means创建硬聚合（即每个点仅在一个聚类中），高斯混合模型GMM依照概率而不是硬性边界进行聚类。
          * 
          * 超参数：
          *     k：聚类数量
          * 训练参数：
          *     maxIter：迭代次数
          *     tol：指定一个阈值来代表将模型优化到什么程度就够了
          */
        // val gmm = new GaussianMixture().setK(5)
        // println(gmm.explainParams())
        // val model = gmm.fit(sales)
        // model.gaussiansDF.show()
        // model.summary.probability.show()

        /**
          * LDA主题模型
          * 隐含狄利克雷分布式一种通常用于对文本文档执行主体建模的分层聚类模型。LDA试图从与这些主题相关联的一系列文档和关键字
          * 中提取高层次的主题，然后它将每个文档解释为多个输入主题的组合。
          * 
          * 超参数：
          *     k：用于指定从数据中提取的主题数量
          *     docConcentration：文档分布的浓度参数向量
          *     topicConcentration：主题分布的浓度参数向量
          * 训练参数：
          *     maxIter：最大迭代次数
          *     optimizer：指定是使用EM还是在线训练方法来优化DA模型
          *     learningDecay：学习率，即指数衰减率
          *     learningOffset：一个正数值的学习参数，在前几次迭代中会递减
          *     optimizeDocConcentration：指示docConcentration是否在训练过程中进行优化
          *     subsamplingRate：在微型批量梯度下降的每次迭代中采样的样本比例
          *     seed：随机种子
          *     checkpointInterval：检查点
          * 预测参数：
          *     topicDistributionCol：将每个文档的主题混合分布输出作为一列保存起来
          */
        val tkn = new Tokenizer().setInputCol("Description").setOutputCol("DescOut")
        val tokenized = tkn.transform(sales.drop("features"))
        val cv = new CountVectorizer()
            .setInputCol("DescOut")
            .setOutputCol("features")
            .setVocabSize(500)
            .setMinTF(0)
            .setMinDF(0)
            .setBinary(true)
        val cvFitted = cv.fit(tokenized)
        val prepped = cvFitted.transform(tokenized)

        val lda = new LDA().setK(10).setMaxIter(5)
        println(lda.explainParams())
        val model = lda.fit(prepped)

        model.describeTopics(3).show()
    }
}