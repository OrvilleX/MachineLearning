import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.graphframes.{examples,GraphFrame}

/**
  * 图分析
  */
object  GraphFrameWithSpark {
    def main(args: Array[String]) {
        val sparkConf = new SparkConf().setAppName("graphFrame").setMaster("local")
        val spark = SparkSession.builder().config(sparkConf).getOrCreate()

        // val v = spark.createDataFrame(List(
        //     ("a", "Alice", 34),
        //     ("b", "Bob", 36),
        //     ("c", "Charlie", 30)
        // )).toDF("id", "name", "age")

        // val e = spark.createDataFrame(List(
        //     ("a", "b", "friend"),
        //     ("b", "c", "follow"),
        //     ("c", "b", "follow")
        // )).toDF("src", "dst", "relationship")

        // val g = GraphFrame(v, e)
        // g.inDegrees.show()
        // g.edges.filter("relationship = 'follow' ").count()
        // val results = g.pageRank.resetProbability(0.01).maxIter(20).run()
        // results.vertices.select("id", "pagerank").show()

        /**
          * 模式查询
          */
        val g = examples.Graphs.friends
        // val motifs = g.find("(a)-[e]->(b); (b)-[e2]->(a)")
        // motifs.show()

        // motifs.filter("b.age > 30").show()

        /**
          * 子图
          */
        val g1 = g.filterVertices("age > 30").filterEdges("relationship = 'friend'").dropIsolatedVertices()

        val paths = g.find("(a)-[e]->(b)")
            .filter("e.relationship = 'follow'")
            .filter("a.age < b.age")
        val e2 = paths.select("e.src", "e.dst", "e.relationship")
        val g2 = GraphFrame(g.vertices, e2)
        e2.show()
    }
}