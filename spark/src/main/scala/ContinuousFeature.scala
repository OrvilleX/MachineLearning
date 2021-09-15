import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.ml.feature.QuantileDiscretizer
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.feature.MaxAbsScaler
import org.apache.spark.ml.feature.ElementwiseProduct
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.Normalizer

/**
  * 连续特征的预处理
  */
object ContinuousFeature {
    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("ScalaSparkML").setMaster("local")
        val spark = SparkSession.builder().config(conf).getOrCreate()

        val df = spark.read.json("data/simple-ml.json")

        /**
          * 分桶，对应其他框架中的分箱行为。以下仅仅介绍较为简单的分桶方式，如果读者
          * 需要学习更高级的方式可以学习如局部敏感哈希（LSH）
          */

        // 基于Bucketizer的数据边界分桶方式
        // 为了处理null或NaN值，必须指定handlerInvalid参数，如果需要保留可以使用keep
        // 如果是需要报错则填入null或者error，以及skip跳过。
        // val bucketBorders = Array(0.0, 10.0, 20.0, 30.0, 40.0, 50.0)
        // val bucketer = new Bucketizer().setSplits(bucketBorders).setInputCol("value1")
        // bucketer.transform(df).show()

        // 基于QuantileDiscretizer的百分比拆分
        // 通过setRelativeError设置近似分位数计算的相对误差
        // val bucketer = new QuantileDiscretizer().setNumBuckets(5).setInputCol("value1").setOutputCol("val1category")
        // var bucketerModel = bucketer.fit(df)
        // bucketerModel.transform(df).show()

        /**
          * 缩放与归一化
          */
        val va = new VectorAssembler().setInputCols(Array("value1", "value2"))
          .setOutputCol("features")
        val rdf = va.transform(df)

        // 基于StandardScaler将一组特征值归一化成平均值为0而标准偏差为1的一组新值。
        // 通过withStd标志设置单位标准差，而withMean标识将使数据在缩放之前进行中心化。
        // 稀疏向量中心化非常耗时，因为一般会将它们转化为稠密向量
        // val sScaler = new StandardScaler().setInputCol("features")
        // sScaler.fit(rdf).transform(rdf).show()

        // 基于MinMaxScaler将向量中的值基于给定的最小值到最大值按比例缩放。
        // 如最小值为0且最大值为1，则所有值将介于0和1之间。
        // val minMax = new MinMaxScaler().setMin(5).setMax(10).setInputCol("features")
        // minMax.fit(rdf).transform(rdf).show()

        // 基于MaxAbsScaler将每个值除以该特征的最大绝对值来缩放数据。
        // 计算后的值将在-1与1之间
        // val maScaler = new MaxAbsScaler().setInputCol("features")
        // maScaler.fit(rdf).transform(rdf).show()

        // 基于ElementwiseProduct将一个缩放向量对某向量中的每个值以不同的尺度进行缩放
        // val scalingUp = new ElementwiseProduct()
        //   .setScalingVec(Vectors.dense(10.0, 20.0))
        //   .setInputCol("features")
        // scalingUp.transform(rdf).show()

        // 基于Normalizer用幂范数来缩放多维向量，Normalizer的作用范围是每一行。
        // 其通过参数P设置几范数，为1表示曼哈顿范数，2表示欧几里德范数等。
        // val manhattanDistance = new Normalizer().setP(1).setInputCol("features")
        // manhattanDistance.transform(rdf).show()
    }
}