## Reformer

### abstract

​	大型 Transformer 模型通常会在许多方面取得最先进的结果，但训练这些模型的成本可能高得惊人，尤其是长时间序列。 我们介绍了两种技术来提高 Transformer 的效率。

​	 一方面，我们将点积注意力（dot-product attention）替换为**使用局部敏感的注意力散列（locality-sensitive hashing）**，将其复杂度从 O(L^2) 到 O(L logL)，其中 L 是长度 的序列。

​	 此外，我们使用**可逆残差层(reversible residual layers)**而不是标准残差(standard residuals)，它允许在训练过程中仅存储一次激活，而不是 N 次，其中 N 是层数。

​	处理后的模型： The Reformer，性能与 Transformer 模型相当，在长序列上内存效率更高且速度更快。 

### Introduction

以下是 Transformer 中内存使用的主要来源：

•**每一层的激活都需保存。** N 层模型的内存比单层模型的内存大 N 倍，原因是事实上，需要存储激活以进行反向传播。
• **前馈层（feedforward）深度太大。**由于中间前馈层的深度  d ff通常远大于注意力激活的 d model，它占内存使用的很大一部分。
• **注意力计算消耗大。**注意长度为 L 的序列是 O(L^2) 的计算和内存复杂度，因此即使是单个 64K 令牌序列也会耗尽加速器内存。



我们介绍了使用以下技术解决这些问题的reformer模型：
• 可逆层，首先由 Gomez 等人引入。 (2017)，只允许存储一个副本整个模型中的激活次数，因此 N 因子消失。
• 在前馈层内拆分激活并分块处理它们可以消除d ff因子并在前馈层内节省内存。
• 基于局部敏感哈希的近似注意力计算取代了 O(L2)用 O(Llog L) 考虑注意力层，因此允许对长序列进行操作。



我们研究了这些技术，并表明与标准 Transformer 相比，它们对训练过程的影响可以忽略不计。

1. 拆分激活实际上只影响实现； 它是在数值上与 Transformer 中使用的层相同。 
2. 应用可逆残差代替标准残差确实改变了模型，但对所有配置的训练影响可以忽略不计
   我们试验过。 
3. 注意力中的局部敏感哈希是一个更重大的变化。可以影响训练动态，具体取决于使用的并发散列数。 我们**学习此参数并找到一个值**，该值既可以有效使用，又可以产生非常接近 full 的结果
   注意力。

### 局部敏感哈希注意力（LSH Attention）

原来在ViT中：是采用的dot-product attention：

<img src="C:\Users\86136\AppData\Roaming\Typora\typora-user-images\image-20211220164854403.png" alt="image-20211220164854403" style="zoom: 80%;" />

Multi-head attention：

​	在多头注意力，多个注意力层平行计算并叠加。每个注意力层会线性地投影head次q，k，v

Memory-efficient attention：

​	在公式中 QK^T 这里的shape为[batch size, length, length]，很耗内存。所以，这里的处理是让Q和K保持一致，这不会影响性能。

Hashing attention：

​	需要Q=K，以及V，它们的shape都是[batch size, length, d_model]。

​	since softmax is dominated by the largest elements，所以其实对每一个q i我们只需要去关注对这个q i最近的几个K。所以这就是hash，

Locality sensitive hashing：

​	该怎么去找最近的k，这就是局部注意力hash要做的事情。

​	Reformer的论文选择了局部敏感哈希的







LSH就是让相近的hash value，且保证不相近的keys拥有不同的hash value：即两个相近的输入比相远的输入更容易获得相同的hash指。



