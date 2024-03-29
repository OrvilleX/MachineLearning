import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.feature.OneHotEncoder

/**
  * 类型特征工程处理更新
  */
object CategoricalFeature {
    def main(args: Array[String]) {
        val sparkConf = new SparkConf().setAppName("categorical").setMaster("local")
        val spark = SparkSession.builder().config(sparkConf).getOrCreate()

        val df = spark.read.json("data/simple-ml.json")

        // 基于StringIndexer将字符串映射到不同的数字Id
        // 它必须看到所有输入之后进而确定输入到ID的映射关系
        val lblIndxr = new StringIndexer().setInputCol("lab").setOutputCol("labelInd")
        val idxRes = lblIndxr.fit(df).transform(df)
        idxRes.select("lab", "labelInd").show()

        // 将索引值转回文本
        val labelReverse = new IndexToString().setInputCol("labelInd").setOutputCol("labelOut")
        labelReverse.transform(idxRes).select("labelInd", "labelOut").show()

        // 基于One-hot（独热编码）在类别变量索引之后进行转换。
        // 其目的主要是因为经过字符串索引后就存在了一种大小关系，比如对应颜色这种类别是无意义的所以我们
        // 需要将其转换为向量中的一个布尔值元素。
        val ohe = new OneHotEncoder().setInputCol("labelInd").setOutputCol("onehot")
        ohe.transform(idxRes).select("labelInd", "onehot").show()
    }
}