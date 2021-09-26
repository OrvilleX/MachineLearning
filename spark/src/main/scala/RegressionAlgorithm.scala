import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.GeneralizedLinearRegression
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.CrossValidator

object  RegressionAlgorithm {

    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("ScalaSparkML").setMaster("local")
        val spark = SparkSession.builder().config(conf).getOrCreate()

        val df = spark.read.format("parquet").load("data/regression")

        /**
          * 线性回归假定输入特征的线性组合（每个特征乘以权重的综合）将得到（有一定高斯误差的）输出结果，具体参数同分类中该算法。
          */
        // val lr = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
        // println(lr.explainParams())
        // val lrModel = lr.fit(df)

        /**
          * 广义线性回归
          * 线性回归的广义形式使你可以更细粒度地控制使用各种回归模型
          * 模型超参数
          *     family：指定在模型中使用的误差分布，支持Poisson、binomial、gamma、Caussian和tweedie。
          *     link：链接函数的名称，指定线性预测器与分布函数平均值之间的关系，支持cloglog、probit、logit、reverse、sqrt、identity和log
          *     solver：指定的优化算法。
          *     variancePower：Tweedie分布方差函数中的幂。
          *     linkPower：Tweedie分布的乘幂链接函数索引。
          * 预测参数
          *     linkPredictionCol：指定一个列名，为每个预测保存我们的链接函数。
          */
        // val glr = new GeneralizedLinearRegression().setFamily("gaussian").setLink("identity").setMaxIter(10).setRegParam(0.3).setLinkPredictionCol("linkOut")
        // println(glr.explainParams())
        // val glrModel = glr.fit(df)

        /**
          * 用于回归分析的决策树不是在每个叶子节点上输出离散的标签，而是一个连续的数值，但是可解释性和模型结构仍然适用。对应参数可以参考分类中该算法。
          */
        // val dtr = new DecisionTreeRegressor()
        // println(dtr.explainParams())
        // val dtrModel = dtr.fit(df)

        /**
          * 随机森林和梯度提升树模型可应用于分类和回归，它们与决策树具有相同的基本概念，不是训练一棵树而是很多树来做回归分析。
          */
        // val rf = new RandomForestRegressor()
        // println(rf.explainParams())
        // val rfModel = rf.fit(df)

        // val gbt = new GBTRegressor()
        // println(gbt.explainParams())
        // var gbtModel = gbt.fit(df)

        /**
          * 评估器和自动化模型校正
          * 用于回归任务的评估器称为 RegressionEvaluator，支持许多常见的回归度量标准。与分类评估器一样，
          * RegressionEvaluator 需要两项输入，一个表示预测值，另一个表示真是标签的值。
          */
    
        val glr = new GeneralizedLinearRegression()
            .setFamily("gaussian")
            .setLink("identity")

        val pipeline = new Pipeline().setStages(Array(glr))
        val params = new ParamGridBuilder().addGrid(glr.regParam, Array(0, 0.5, 1)).build()
        val evaluator = new RegressionEvaluator()
            .setMetricName("rmse")
            .setPredictionCol("prediction")
            .setLabelCol("label")
        val cv = new CrossValidator()
            .setEstimator(pipeline)
            .setEvaluator(evaluator)
            .setEstimatorParamMaps(params)
            .setNumFolds(2)
        val model = cv.fit(df)
    }
}