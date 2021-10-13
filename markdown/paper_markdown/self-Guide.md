BERT的每一层的句子表示都蕴含着不同的语义类型

模型使用两个BERT，一个fixed，一个tunable。

一个fixed一个tunable避免了两个一样的编码器出现表示退化的问题。

fixedBERT用来给tunable BERT提供学习信号，所谓的学习信号就是positive example



整体流程：

1. batch个句子$\{x_i\}_{i=1}^{N}$​扔到两个BERT中
2. 对于tunable BERT，取最后一层的[CLS] token得到sentence embedding。$\{v_i\}_{i=1}^{N}$
3. 接下来就是从fixed BERT中找到每一个$v_i$对应的positive example。具体的就是根据fixed BERT得到的12层的每一层的token embeddings根据max pooling得到每一层的sentence embedding。然后从这12个sentence embeddings中随机的取出一个sentence embedding $\hat{v}_i$作为$v_i$的positive example。
4. 最后就是计算NT-Xent loss。负样本仍然是batch内的其余句子表示。另外由于fixed BERT不更新，为了防止两个BERT距离太远，额外限制两个BERT的距离不要太远。



作者发现，inner batch内的句子是不需要作为负样本推开的，如果将第一个batch内的句子也作为负样本推开，效果反而下降。在实验中也确实出现？？？



作者的另一个改进是负样本中将第$i$个句子在fixed BERT中的所有层的sentence representation都作为其余句子的负样本。也就是增加负样本的数量。



另一个改进是未必一定要max pooling，可以随机的mean pooling或者max pooling

作者发现将back translation的句子扔进fixed BERT的效果更好。

此外lower layer的句子表示可能不适合作为语义相似度计算的sentence representation

最后，这种限制两个BERT参数距离的方式可能不如momentum encoder效果好。，即便不用队列，最起码fixed bert也应该随着tunable BERT进行slow update

