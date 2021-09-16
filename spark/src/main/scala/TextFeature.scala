import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.NGram
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.Word2Vec

/**
  * 文本数据特征处理
  */
object TextFeature {
    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("ScalaSparkML").setMaster("local")
        val spark = SparkSession.builder().config(conf).getOrCreate()

        val sales = spark.read.format("csv").option("header", "true")
            .option("inferSchema", "true")
            .load("data/sales.csv")
            .coalesce(5)
            .where("Description IS NOT NULL")
        
        // 分词是将任意格式的文本转变成一个“符号”列表或者一个单词列表的过程
        val tkn = new Tokenizer().setInputCol("Description").setOutputCol("DescOut")
        val tokenized = tkn.transform(sales.select("Description"))
        // tokenized.show()

        // 不仅可以基于空格分析，也可以使用RegexTokenizer指定的正则表达式来分词
        // val rt = new RegexTokenizer().setInputCol("Description")
        //     .setOutputCol("DescOut")
        //     .setPattern(" ")
        //     .setToLowercase(true)
        // rt.transform(sales.select("Description")).show()

        // 分词后的一个常见任务是过滤停用词，这些常用词在许多分析中没有什么意义，因此应被删除
        // val englishStopWords = StopWordsRemover.loadDefaultStopWords("english")
        // val stops = new StopWordsRemover().setStopWords(englishStopWords)
        //     .setInputCol("DescOut")
        //     .setOutputCol("StopWords")
        // stops.transform(tokenized).show()

        // 字符串分词和过滤停用词之后，会得到可作为特征的一个词集合。
        // val bigram = new NGram().setInputCol("DescOut").setN(2)
        // bigram.transform(tokenized.select("DescOut")).show()

        // 一旦有了词特征，就可以开始对单词和单词组合进行计数，以便在我们的模型中使用
        // 可以通过setMinTF来决定词库中是否包含某项，通过setVocabSize设置总的最大单词量
        // val cv = new CountVectorizer()
        //     .setInputCol("DescOut")
        //     .setOutputCol("countVec")
        //     .setVocabSize(500)
        //     .setMinTF(1)
        //     .setMinDF(2)
        // val fittedCV = cv.fit(tokenized)
        // fittedCV.transform(tokenized).show()

        // 实际上它是一个稀疏向量，包含总的词汇量、词库中某单词的索引，以及该单词的计数


        // 另一种将本文转换为数值表示的方法是使用词频-逆文档频率（TF-IDF）。最简单的情况
        // 是，TF-IDF度量一个单词在每个文档中出现的频率，并根据该单词出现过的文档数进行
        // 加权，结果是在较少文档中出现的单词比在许多文档中出现的单词权重更大。
        // val tf = new HashingTF().setInputCol("DescOut")
        //     .setOutputCol("TFOut")
        //     .setNumFeatures(10000)
        // var idf = new IDF().setInputCol("TFOut")
        //     .setOutputCol("IDFOut")
        //     .setMinDocFreq(2)
        // idf.fit(tf.transform(tokenized)).transform(tf.transform(tokenized)).show()

        // 输出显示总的词汇量、文档中出现的每个单词的哈希值，以及这些单词的权重

        // 因为机器学习只能接受数值形式，为此需要进行转换，而Word2Vec就是词嵌入的中的一种。
        // 而Spark内使用的是基于`skip-gram`模型，而该模型主要是根据输入的词语推算出上下文可能与该词组合的其他词语。
        // 如果希望学习Word2Vec则可以参考[本文档](https://zhuanlan.zhihu.com/p/26306795/)
        // val word2Vec = new Word2Vec().setInputCol("DescOut").setOutputCol("result").setVectorSize(3).setMinCount(0)
        // val model = word2Vec.fit(tokenized)
        // model.transform(tokenized).show()
    }
}