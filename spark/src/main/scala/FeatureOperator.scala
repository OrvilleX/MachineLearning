import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.feature.PolynomialExpansion
import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.feature.StringIndexer

object FeatureOperator {
    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("ScalaSparkML").setMaster("local")
        val spark = SparkSession.builder().config(conf).getOrCreate()
        val df = spark.read.json("data/simple-ml.json")
        val va = new VectorAssembler().setInputCols(Array("value1", "value2")).setOutputCol("features")
        val vdf = va.transform(df)

        val lblIndxr = new StringIndexer().setInputCol("lab").setOutputCol("labelInd")
        val featureDF = lblIndxr.fit(vdf).transform(vdf)
        
        // 特征操作
        // 主成分（PCA）是一种数据方法， 用于找到我们的数据中最重要的成分。
        // PCA使用参数k指定要创建的输出特征的数量，这通常应该比输入向量的尺寸小的多。
        // val pca = new PCA().setInputCol("features").setK(2)
        // pca.fit(featureDF).transform(featureDF).show()

        // 多项式扩展基于所有输入列生成交互变量。对于一个二阶多项式，Spark吧特征向量
        // 中的每个值乘以所有其他值，然后将结果存储成特征。

        // 多项式扩展会增大特征空间，从而导致高计算成本和过拟合效果，所以请效性使用。
        // val pe = new PolynomialExpansion().setInputCol("features").setDegree(2)
        // pe.transform(featureDF).show()

        // 特征选择
        // ChiSqSelector利用统计测试来确定与我们试图预测的标签无关的特征，并删除不相关的特征。
        // 其提供了以下集中方法：
        // numTopFea tures：基于p-value排序
        // percentile：采用输入特征的比例
        // fpr：设置截断p-value
        val chisq = new ChiSqSelector().setFeaturesCol("features")
            .setLabelCol("labelInd")
            .setNumTopFeatures(1)
        chisq.fit(featureDF).transform(featureDF).show()
    }
}