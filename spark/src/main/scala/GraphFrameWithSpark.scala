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
        spark.sparkContext.setCheckpointDir("./")

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
        // val g1 = g.filterVertices("age > 30").filterEdges("relationship = 'friend'").dropIsolatedVertices()

        // val paths = g.find("(a)-[e]->(b)")
        //     .filter("e.relationship = 'follow'")
        //     .filter("a.age < b.age")
        // val e2 = paths.select("e.src", "e.dst", "e.relationship")
        // val g2 = GraphFrame(g.vertices, e2)
        // e2.show()

        /**
          * 广度优先搜索算法（BFS）
          * 是一种盲目搜寻法，目的是系统地展开并检查图中的所有节点，以找寻结果。其主要可以
          * 找寻两点之间最短距离。
          */
        // val paths = g.bfs.fromExpr("name = 'Esther'").toExpr("age < 32").run()
        // paths.show()

        // val paths2 = g.bfs.fromExpr("name = 'Esther'").toExpr("age < 32")
        //     .edgeFilter("relationship != 'friend'")
        //     .maxPathLength(3).run()
        // paths2.show()

        /**
          * 连通分量
          */
        // val result = g.connectedComponents.run() // 强连通为 g.stronglyConnectedComponents.run()
        // result.select("id", "component").orderBy("component").show()
        
        /**
          * 标签传播算法
          * 
          * 标签传播算法是基于图的半监督学习方法，基本思路是从已标记的节点的标签信息来预测未标记的节点的标
          * 签信息，利用样本间的关系，建立完全图模型。每个节点标签按相似度传播给相邻节点，在节点传播的每一
          * 步，每个节点根据相邻节点的标签来更新自己的标签，与该节点相似度越大，其相邻节点对其标注的影响权
          * 值越大，相似节点的标签越趋于一致，其标签就越容易传播。
          */
        // val lpa = g.labelPropagation.maxIter(5).run()
        // lpa.select("id", "label").show()

        /**
          * PageRank算法
          * 
          */
        

        /**
          * 最短路径算法
          */
        // val sp = g.shortestPaths.landmarks(Seq("a", "b")).run()
        // sp.select("id", "distances").show()

        /**
          * 三角形计数算法
          */
        val tc = g.triangleCount.run()
        tc.select("id", "count").show()
    }
}