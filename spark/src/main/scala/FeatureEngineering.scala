import org.apache.spark.ml.feature.RFormula
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.SQLTransformer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.Binarizer
import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.ml.feature.Imputer

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
        
        // val va = new VectorAssembler().setInputCols(Array("value1", "value2")).setOutputCol("features")
        // val vva = va.transform(df)

        /**
          * 基于VectorIndexer
          * 将一个向量列中的特征进行索引，一般用于决策树。其对于离散特征的索引是基于0开始的，但是并不保证值每次对应的索引值一致，但是对于0必然是会映射到0
          */
        // val vi = new VectorIndexer().setInputCol("features").setOutputCol("indexFeatures").fit(vva)
        // vi.transform(vva).show()

        /**
          * 基于Binarizer
          * 二值化是将数值特征阀值化为二进制（0/1）特征的过程，其将根据`threshold`阈值参数，大于该阈值的数据
          * 将二值化为1，小于则二值化为0。
          */
        // val bin = new Binarizer().setInputCol("value2")
        //       .setOutputCol("bin").setThreshold(15)
        // bin.transform(df).select("value2", "bin").show(10)

        /**
          * 欧几里德距离度量-局部敏感哈希
          * 输入是密集的(dense)或稀疏的(sparse)矢量，每个矢量表示欧几里德距离空间中的一个点，输出将是可配置维度的向量。
          */
        // val va = new VectorAssembler().setInputCols(Array("value1", "value2")).setOutputCol("features")
        // val vva = va.transform(df)
        // val brp = new BucketedRandomProjectionLSH()
        //       .setInputCol("features").setOutputCol("lsh").setBucketLength(100)
        //       .setNumHashTables(10).setNumHashTables(5).fit(vva)
        // brp.transform(vva).select("features", "lsh").show()

        /**
          * Imputer
          * 归因估算器使用缺失值所在列的平均值或中位数来完成数据集中的缺失值，输入列应为DoubleType或FloatType。
          */
        // val imputer = new Imputer().setInputCols(Array("value2"))
        //       .setOutputCols(Array("value2_imp")).setMissingValue(0)
        //       .fit(df)
        // imputer.transform(df).select("value2", "value2_imp").show()
    }
}