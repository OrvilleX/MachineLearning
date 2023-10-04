# Transformer

## 框架整体解读

* [Transformer在目标检测领域的开山之作DETR模型](https://mp.weixin.qq.com/s?__biz=MzIwMTE1NjQxMQ==&mid=2247585078&idx=1&sn=acf1b094e7b32fb82807e32267154445&chksm=96f1ff62a1867674b49f3b9040444e92a09b986b40ee2d72030ec21c773e6f6f39f585f08343&scene=27)  
* 

## Positional Encoding位置编码

在任何一门语言中，词语的位置和顺序对句子意思表达都是至关重要的。传统的RNN模型在处理句子时，以序列的模式逐个处理
句子中的词语，这使得词语的顺序信息在处理过程中被天然的保存下来了，并不需要额外的处理。 而对于Transformer来说，由
于句子中的词语都是同时进入网络进行处理，顺序信息在输入网络时就已丢失。因此，Transformer是需要额外的处理来告知每个
词语的相对位置的。其中的一个解决方案，就是论文中提到的Positional Encoding，将能表示位置信息的编码添加到输入中，让
网络知道每个词的位置和顺序。  

一句话概括，Positional Encoding就是句子中词语相对位置的编码，让Transformer保留词语的位置信息。

### 参考文章
* [一文教你彻底理解Transformer中Positional Encoding](https://zhuanlan.zhihu.com/p/338592312)  
* [Transformer学习笔记一：Positional Encoding（位置编码）](https://zhuanlan.zhihu.com/p/454482273)

## Object Queries

Object queries有N个（其中N是一个事先设定的、比远远大于image中object个数的一个整数），输入Transformer Decoder后分别得到 
N个decoder output embedding，经过FFN（后面会讲）处理后就得到了 N个预测的boxes和这些boxes的类别。 具体实现上，object qu
eries是N个learnable embedding，训练刚开始时可以随机初始化。在训练过程中，因为需要生成不同的boxes，object queries会被迫使
变得不同来反映位置信息，所以也可以称为leant positional encoding （注意和encoder中讲的position encoding区分，不是一个东西）。  

### 参考文章
* [用Transformer做object detection：DETR](https://zhuanlan.zhihu.com/p/267156624)  
* [Object query的理解](https://blog.csdn.net/wzk4869/article/details/129908100)  