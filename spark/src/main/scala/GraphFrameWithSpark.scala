import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.graphframes.GraphFrame

/**
  * 图分析
  */
object  GraphFrameWithSpark {
    def main(args: Array[String]) {
        val sparkConf = new SparkConf().setAppName("graphFrame").setMaster("local")
        val spark = SparkSession.builder().config(sparkConf).getOrCreate()

        val v = spark.createDataFrame(List(
            ("a", "Alice", 34),
            ("b", "Bob", 36),
            ("c", "Charlie", 30)
        )).toDF("id", "name", "age")

        val e = spark.createDataFrame(List(
            ("a", "b", "friend"),
            ("b", "c", "follow"),
            ("c", "b", "follow")
        )).toDF("src", "dst", "relationship")

        val g = GraphFrame(v, e)
        g.inDegrees.show()
        g.edges.filter("relationship = 'follow' ").count()
        val results = g.pageRank.resetProbability(0.01).maxIter(20).run()
        results.vertices.select("id", "pagerank").show()
    }
}