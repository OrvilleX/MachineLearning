import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.classification.LogisticRegression
import org.jpmml.sparkml.PMMLBuilder
import org.jpmml.model.JAXBUtil
import javax.xml.transform.stream.StreamResult
import java.io.File
import java.io.FileWriter
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.Pipeline
import org.jpmml.evaluator.Evaluator
import org.jpmml.evaluator.LoadingModelEvaluatorBuilder
import java.util.LinkedHashMap
import org.dmg.pmml.FieldName
import org.jpmml.evaluator.FieldValue
import org.jpmml.evaluator.EvaluatorUtil
import scala.collection.JavaConversions._

object SparkWithPMML {

    def main(args: Array[String]) {
        val sparkConf = new SparkConf().setAppName("ScalaSparkML").setMaster("local")
        val spark = SparkSession.builder().config(sparkConf).getOrCreate()

        val df = spark.read.json("data/simple-ml.json")
        val iris = df.schema
        val Array(train, test) = df.randomSplit(Array(0.7, 0.3))

        val supervised = new RFormula().setFormula("lab ~ . + color: value1 + color: value2")
            .setLabelCol("label1")
            .setFeaturesCol("features2")
        val lr = new LogisticRegression().setLabelCol("label1").setFeaturesCol("features2")

        var pipeline = new Pipeline().setStages(Array(supervised, lr))
        var model = pipeline.fit(train)

        val pmml = new PMMLBuilder(iris, model).build()

        JAXBUtil.marshalPMML(pmml, new StreamResult(new File("data/model")))

        var eval = new LoadingModelEvaluatorBuilder().load(new File("data/model")).build()
        eval.verify()

        var inputFields = eval.getInputFields()

        val row = test.first()
        val arguments = new LinkedHashMap[FieldName, FieldValue]
        
        for (inputField <- inputFields) {
            var fieldName = inputField.getFieldName()
            var value = row.get(row.fieldIndex(fieldName.getValue()))
            val inputValue = inputField.prepare(value)

            arguments.put(fieldName, inputValue)
        }

        var result = eval.evaluate(arguments)
        var resultRecoard = EvaluatorUtil.decodeAll(result)
        println(resultRecoard)
    }
}