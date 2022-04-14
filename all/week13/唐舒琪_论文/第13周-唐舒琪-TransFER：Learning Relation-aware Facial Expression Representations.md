**TransFER: Learning Relation-aware Facial Expression Representations**
**with Transformers**

[TOC]

### 一：论文摘要

* 本论文提出了Trans- FER模型，它可以学习丰富的关系感知的局部表达。
* 它主要由三个部分组成：
  * Multi- Attention Dropping (MAD)：随机丢弃一个attention map
  * ViT-FER:利用全局特征帮助在不同局部特征之间建立丰富的联系
  *  Multi-head Self- Attention Dropping (MSAD)：联合关注不同位置的不同信息子空间的特征，并且随机丢掉一个self-attention模块

### 二：知识点

#### 1.简要介绍

###### 1.1目前存在的两个挑战

*   类间相似度大
*   类内相似度小

###### 1.2目前的研究主要分为两类

* 基于全局的方法
  * 会忽略关键的部位，因此很多研究都采基于局部的方法
* 基于局部的方法，主要分为两类
  * landmarked-base： 基于面部标志位置剪裁而提取的特征，存在不够灵活处理姿势变化的问题，关键点位置检测可能会因为光照、姿势、遮挡等问题检测不够准确甚至失败。
  * attention-base ：可能会提取到冗余的特征，即关注到的位置都是相似的。因此，应该提取不同的局部特征来对不同的表情进行分类。不同的局部块之间的关系需要在全局范围内进行搜索，突出有意义的，忽略无意义的部分。

###### 1.3Trans-FER模型的提出

* 主要是为了实现上述提到的两个目标，即**在全局范围内搜寻不同局部之间的关系、提取检测重要的面部部位并抑制无用的部位**。
* Trans-FER能够学习多样化的关系感知的局部特征

###### 1.4Tran-FER简要介绍

* Multi-Attention Dropping (MAD)：随机丢弃一个attention map，这样能推动模型去探索除了最具辨别力以外的综合局部块，自适应地关注不同的局部块，**能够有效解决姿势变换和遮挡的问题**
* ViT-FER：模拟多个局部块之间的连接，使用全局范围来增强每个局部块，能够充分利用多个局部块之间存在的互补性，提高识别性能
* Multi-head Self-Attention Dropping (MSAD): 虽然multi-head self-attention能够联合VIT在不同位置关注不同信息子空间的特征，但是可能会关注到冗余的联系，因此提出了MSAD方法随机丢弃一个self-attention。当一个self-attention被丢弃之后，模型就会去从其它部分学习有用的关系，来探索不同局部块之间的关系。

###### 1.5本文的贡献

* 应用了VIT自适应地描述面部不同部位之间的关系
* 引入了MSAD方法来随机移除self-attention模块，促使模型学习不同局部块之间的丰富关系
* 设计了MAD方法随即删除一个attention map，促进模型从面部的每个部分提取全面的局部信息，而不是只关注最具辨别力的部分



#### 2.相关研究

###### 2.1. Facial Expression Recognition

* 传统手动特征：LBP、HOG、SIFT
* 深度学习方法

###### 2.2Transformers in Computer Vision

* 利用transformer decoder能够实现对象位置的推断，是一种端到端的检测方法
* 第一次表明局部块之间具有重要联系

###### 2.3正则化方法

* Dropout：将全连接层元素随机归零，缓解过拟合问题，在卷积运算中不是很有效（因为CNN中是特征相关的）
* Cutout:为解决上述问题，随机擦除输入图像中的连续区域
* DropBlock：在每一个feature map上都应用Cout来改进Cutout
* MSAD：本篇文章提出该方法来对Transformer进行规范化，以探索不同局部之间的联系



#### 3.TransFER

![1638032764165](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\1638032764165.png)

###### 3.1整体架构

* 主要由stem CNN、 Local CNNs、 Multi-head Self-Attention Dropping (MSAD)组成
* steam CNN：用以提取特征，使用了IR-50，因为它有良好的泛化性能
* MAD:帮助提取多样化的局部特征

###### 3.2Local CNN

* 利用Local CNN来提取由**MAD**引导的各种局部特征，有效解决姿势、遮挡、光照问题，提取到足够的有辨别力的局部特征
* Local CNN中分为四个步骤：
  * 首先，利用原始feature map生成多个attention maps
  * 其次，利用MAD方法。假设B个LANet会生成B个attention map, 将这B个attention maps作为MAD输入，然后随机将一个attention map设置为0（即丢弃），MAD再输出这B个attention map。
  * 然后，再将多个attention maps合并成为一个attention map，作为输出M_out
  * 然后再将刚刚得到的M_out与原始feature相乘，这样就能突出重要区域，忽略不重要区域
* 总结：Local CNN能够定位不同的局部块。即通过使用多个LANET来定位多个区分区域，并通过最大值运算将attention map聚合，然后与输入特征映射（进行元素相乘）来实现的。

###### 3.3MAD

* 它的提出受Dropout启发
* 在训练过程中，从均匀分布的输入的多个feature map中选择一个特征图，该特征图被完全设置为零，被丢弃的特征图不会在接下来的层中被激活
* 分布规律良好的面部部位可以被定位，从而形成全面的局部表征

###### 3.4MSAD

* MSAD主要是为了探索由 Local CNN产生的不同局部特征之间的联系，它主要是由Transformer encoder和**MLP 分类器（ViT-FER）**组成，而Transformer encoder又在每个在每个Multi-head Self Attention之后加入了MAD方法
* 投影：在经过Local CNN后，需要将二维序列投影到一维以适应tranformer
* Transformer encoder组成：由多个encode block组成，每个encode block包括多层Multi-head Self-Attention **(MSA)** 和 Multi-Layer Perceptron (MLP) ……
* **MSA**的设计是为了将投影嵌入它们的各自的空间，MAD防止MSA中多个自我注意模块产生冗余的投影
* MAD为样本带来了足够的随机性，能够有效防止过拟合，且只在训练时进行
* 总结：本实验只考虑在MAD丢弃了一个SA（Self-Attention）分支，丢弃2个或者两个以上并没有实际性能提高

#### 4.实验

###### 4.1数据集

* RAF-DB
* FERPlus
* AffectNet

###### 4.2实验实施

* 使用SGD优化器训练，以使交叉熵损失最小

* 数据增强：一系列方法

* RAF-DB和FERPlus：40个 epochs，AffectNet：2k epochs

* 在RAF-DB上设计了不同丢弃率的实验。记为 p1, p2分别为MAD和MSAD中的丢弃率。默认情况下，它们被设置为0.6和0.3。小的和大的p1、p2值都会降低模型的 性能。

* MAD方法的有效性

  ![1638034275322](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\1638034275322.png)

* 经过LOCAL CNN和MAD方法之后的整个框架可以专注于更具辨别力的面部区域。

###### 4.3与目前方法比较

![1638034398785](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\1638034398785.png)

###### 4.4总结

* TransFER能够学习丰富、多样的关系感知的局部特征，专注更具辨别力的面部区域
* 首先，利用MAD来指导Local CNN产生不同的局部块，使模型对**姿势变化或遮挡具有鲁棒性**
* 其次，应用ViT-FER在多个局部块上建立联系，其中重要的面部部分被赋予较高的权重，而无用的部分被赋予较小的权重。
* 最后，MSAD用以探索不同面部部位之间更丰富的关系