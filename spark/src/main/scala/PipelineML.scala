import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.TrainValidationSplit

object PipelineML {
    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("ScalaSparkML").setMaster("local")
        val spark = SparkSession.builder().config(conf).getOrCreate()

        val df = spark.read.json("data/simple-ml.json")
        val Array(train, test) = df.randomSplit(Array(0.7 ,0.3))

        val rForm = new RFormula();
        var lr = new LogisticRegression()
        val pipeline = new Pipeline().setStages(Array(rForm, lr))

        val params = new ParamGridBuilder()
            .addGrid(rForm.formula, Array("lab ~ . + color:value1", 
                "lab ~ . + color:value1 + color:value2"))
            .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
            .addGrid(lr.regParam, Array(0.1, 2.0)).build()

        val evaluator = new BinaryClassificationEvaluator()
            .setMetricName("areaUnderROC")
            .setRawPredictionCol("prediction")
            .setLabelCol("label")
        
        val tvs = new TrainValidationSplit()
            .setTrainRatio(0.75)
            .setEstimatorParamMaps(params)
            .setEstimator(pipeline)
            .setEvaluator(evaluator)

        var model = tvs.fit(train)
        println(evaluator.evaluate(model.transform(test)))
    }
}