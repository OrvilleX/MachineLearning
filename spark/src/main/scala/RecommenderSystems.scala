import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.functions
import org.apache.spark.mllib.evaluation.RankingMetrics

/**
  * 推荐系统
  */
object RecommenderSystems {
    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("ScalaSparkML").setMaster("local")
        val spark = SparkSession.builder().config(conf).getOrCreate()

        val ratings = spark.read.textFile("data/sample_movielens_ratings.txt")
                    .selectExpr("split(value, '::') as col")
                    .selectExpr("cast(col[0] as int) as userId",
                    "(col[1] as int) as movieId",
                    "(col[2] as float) as rating",
                    "(col[3] as long) as timestamp")
        val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))
        val als = new ALS().setMaxIter(5).setRegParam(0.01).setUserCol("userId").setItemCol("movieId").setRatingCol("rating")
        println(als.explainParams())

        val alsModel = als.fit(ratings)
        val predictions = alsModel.transform(test)

        /**
          * 模型的recommendForAllUsers方法返回对应某个userId的DataFrame，包含推荐电影的数组，以及每个影片的评分。
          * recommendForAllItems返回对应某个movieId的DataFrame以及最有可能给该影片打高分的前几个用户。
          */
        alsModel.recommendForAllUsers(10)
            .selectExpr("userId", "explode(recommendations)").show()
        alsModel.recommendForAllItems(10)
            .selectExpr("movieId", "explode(recommendations)").show()
        
        /**
          * 针对训练的模型效果我们依然需要针对其进行评估，这里我们可以采用与回归算法中相同的评估器进行评估。
          */
        val evaluator = new RegressionEvaluator()
            .setMetricName("rmse")
            .setLabelCol("rating")
            .setPredictionCol("prediction")
        val rmse = evaluator.evaluate(predictions)
        println(s"Root-mean-square error = $rmse")

        /**
          * 对于度量指标我们通过回归度量指标与排名指标进行度量，首先是回归度量指标，我们可以简单的查看
          * 每个用户和项目的实际评级与预测值的接近程度
          */
        val regComparison = predictions.select("rating", "prediction")
                            .rdd.map(x => (x.getFloat(0).toDouble, x.getFloat(1).toDouble))
        val metrics = new RegressionMetrics(regComparison)
    }
}