import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.graphframes.{examples,GraphFrame}
import org.apache.spark.sql.functions

/**
  * 图分析
  */
object  GraphFrameWithSpark {
    def main(args: Array[String]) {
        val sparkConf = new SparkConf().setAppName("graphFrame").setMaster("local")
        val spark = SparkSession.builder().config(sparkConf).getOrCreate()
        spark.sparkContext.setCheckpointDir("./checkpoint")

        /**
          * 为了构建一个图，我们需要两个表的数据构成。首先是顶点表，我们将人物标识符
          * 定义为id，在边表中我们将每条边的源顶点ID标记为src，目的地顶点ID为dst
          */

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

        /**
          * 采用与DataFrame一致的方式进行查询
          */
        // g.edges
        //   .where("src = 'a' OR dst = 'b'")
        //   .groupBy("src", "dst").count()
        //   .orderBy(functions.desc("count"))
        //   .show()

        /**
          * 子图
          * 即就是一个大图中的小图
          */
        // val g = examples.Graphs.friends
        // val g1 = g.filterVertices("age > 30").filterEdges("relationship = 'friend'").dropIsolatedVertices()

        // val paths = g.find("(a)-[e]->(b)")
        //     .filter("e.relationship = 'follow'")
        //     .filter("a.age < b.age")
        // val e2 = paths.select("e.src", "e.dst", "e.relationship")
        // val g2 = GraphFrame(g.vertices, e2)
        // e2.show()

        /**
          * 模式发现
          * motif是图的结构化模式的一种表现形式。当指定一个motif时，查询的是数据中的模式而不是实际
          * 的数据。在GraphFrame中，我们采用具体领域语言来指定查询。此语言允许我们指定顶点和边的组合，并给它们
          * 指定名称。
          */
        // val g = examples.Graphs.friends
        // val motifs = g.find("(a)-[e]->(b); (b)-[e2]->(a)")
        // motifs.show()
        // motifs.filter("b.age > 30").show()

        /**
          * PageRank算法
          * 通过计算网页链接的数量和质量来确定网站的重要性。它根本的假设是，重要的网站可能会被
          * 更多气压网站所连接。
          */
        // val g = examples.Graphs.friends
        // g.edges.filter("relationship = 'follow' ").count()
        // val results = g.pageRank.resetProbability(0.01).maxIter(20).run()
        // results.vertices.select("id", "pagerank").show()

        /**
          * 入度出度指标
          * 一个常见的任务是计算进出某一站点的次数。为了测量进出车站的行程数，我们将分别使用一种名为“入度”
          * 和“出度”的度量。
          */
        val g = examples.Graphs.friends
        g.inDegrees.orderBy(functions.desc("inDegree")).show(5)
        g.outDegrees.orderBy(functions.desc("outDegree")).show(5)

        /**
          * 广度优先搜索算法（BFS）
          * 是一种盲目搜寻法，目的是系统地展开并检查图中的所有节点，以找寻结果。其主要可以
          * 找寻两点之间最短距离。
          */
        // val g = examples.Graphs.friends
        // val paths = g.bfs.fromExpr("name = 'Esther'").toExpr("age < 32").run()
        // paths.show()

        // val paths2 = g.bfs.fromExpr("name = 'Esther'").toExpr("age < 32")
        //     .edgeFilter("relationship != 'friend'")
        //     .maxPathLength(3).run()
        // paths2.show()

        /**
          * 连通分量
          * 连通图存在一个连通分量，非连通图存在多个连通分量。即节点间是否存在路径可连通两点。
          */
        // val g = examples.Graphs.friends
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
        // val g = examples.Graphs.friends
        // val lpa = g.labelPropagation.maxIter(5).run()
        // lpa.select("id", "label").show()

        

        /**
          * 最短路径算法
          */
        // val g = examples.Graphs.friends
        // val sp = g.shortestPaths.landmarks(Seq("a", "b")).run()
        // sp.select("id", "distances").show()

        /**
          * 三角形计数算法
          */
        // val g = examples.Graphs.friends
        // val tc = g.triangleCount.run()
        // tc.select("id", "count").show()
    }
}