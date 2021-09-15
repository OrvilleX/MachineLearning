import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.classification.LogisticRegression

object MachineLearnWithSpark {

    def main(args: Array[String]) {
        val sparkConf = new SparkConf().setAppName("ScalaSparkML").setMaster("local")
        val spark = SparkSession.builder().config(sparkConf).getOrCreate()

        val df = spark.read.json("data/simple-ml.json")
        val supervised = new RFormula().setFormula("lab ~ . + color: value1 + color: value2")
            .setLabelCol("label1")
            .setFeaturesCol("features2")
        
        val model = supervised.fit(df);
        val preparedDF = model.transform(df);

        val Array(train, test) = preparedDF.randomSplit(Array(0.7, 0.3))
        val lr = new LogisticRegression().setLabelCol("label1").setFeaturesCol("features2")

        val lrModel = lr.fit(train)
        lrModel.transform(test).select("label1", "prediction").show()
    }
}