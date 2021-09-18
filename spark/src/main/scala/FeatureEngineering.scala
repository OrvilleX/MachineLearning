import org.apache.spark.ml.feature.RFormula
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.SQLTransformer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorIndexer

/**
  * 特征工程
  */
object FeatureEngineering {
    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("ScalaSparkML").setMaster("local")
        val spark = SparkSession.builder().config(conf).getOrCreate()

        val df = spark.read.json("data/simple-ml.json")

        /**
          * 基于RFormula
          * 对于字符串通过独热编码(One-hot)处理字符串
          */

        // val supervised = new RFormula()
        //     .setFormula("lab ~ value1 + value2")
        // supervised.fit(df).transform(df).show()

        /**
          * 基于SQLTransformer
          * 需要注意的是不需要使用表名，只需使用关键字THIS
          */
        
        // val basicTrans = new SQLTransformer()
        //     .setStatement("""
        //         SELECT sum(value1), count(*), color
        //         FROM __THIS__
        //         GROUP BY color
        //     """)
        // basicTrans.transform(df).show()

        /**
          * 基于VectorAssembler
          * 将多个Boolean，Double或Vector类型的列做为输入组成一个大的向量
          */
        
        val va = new VectorAssembler().setInputCols(Array("value1", "value2")).setOutputCol("features")
        val vva = va.transform(df)

        /**
          * 基于VectorIndexer
          * 将一个向量列中的特征进行索引，一般用于决策树。其对于离散特征的索引是基于0开始的，但是并不保证值每次对应的索引值一致，但是对于0必然是会映射到0
          */
        val vi = new VectorIndexer().setInputCol("features").setOutputCol("indexFeatures").fit(vva)
        vi.transform(vva).show()
    }
}