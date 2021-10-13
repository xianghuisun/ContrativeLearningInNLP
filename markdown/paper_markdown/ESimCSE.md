SimCSE这种利用dropout作为data augmentation的策略虽然有效，但是存在的问题两个positive examples**的长度是一样的，而长度的信息由于位置编码的作用会被编码在vector中，所以SimCSE可能倾向于将长度差不多的pair预测为positive**

由于训练过程中，长度信息被编码在vector中，所以长度信息也会作为一个特征，使得模型预测时出现bias，也就是偏向将长度一致的句子预测为positive，将长度不一致的句子视为negative



==ESimCSE这种方式的提升效果是很有限的，而且没什么泛化能力，不能提升SimCSE在监督任务上的表现==



对于某一个STS数据集，作者将数据集中的pairs根据长度划分，每一个pair中两个句子的长度差异小于等于3的是一组，每一个pair中两个句子的长度差异大于3的是另一组，在这两组数据上计算spearman系数，发现果然在长度差异小的那组中SimCSE的spearman系数更高，而且两组的spearman系数差异很显著



作者提出的改进方法之一就是word repetition。因为作者认为random insert和delete可能会改变整个句子的语义。word repetition是指随机的重复这个句子的某些单词。

将经过word repetition后的句子再次扔进BERT中，而不是一个句子仍两次。



**大部分的paper都表明，增加batch_size未必会有提升，而且太大的batch_size还会降低performance。但是增加negative examples的数量确实会提升performance**



所以第二个改进就是引入momentum contrast机制。

MOCO机制是对比学习架构的另一个常用架构。思想是有两个encoder，$f_q,f_k$，流程是将两个句子(original,augumented)分别的扔进两个encoder，编码得到两个vector，这两个vector就是positive关系，而只有$f_q$会进行back-propogation。$f_k$的更新策略是$f_k=\gamma f_k+(1-\gamma)f_q$，其中$\gamma$一般取0.99。

由于不会计算$f_k$的梯度，所以可以利用一个queue存储$f_k$编码的向量供下次$f_q$使用。**队列中存储的是$f_k$编码的vector，这些vector是之前的batch的句子，而当前batch的句子仍然会经过$f_q$和$f_k$，计算positive和negative，而queue中存储的所有vector都可以作为当前batch内的每一个句子的negative examples。从而也就实现了increase negative examples**



在ESimCSE的流程是：

将sentences 和word repetition的sentences经过$f_q$得到2*N个vector

同时将queue中的所有vector作为每一个句子negative examples计算NT-Xent loss
$$
-\log \frac{\exp(Sim(h_i,h_i^+)/\tau)}{\sum_{j=1}^{N}\exp(Sim(h_i,h_j)/\tau)+\sum_{k=1}^{M}\exp(Sim(h_i,h_k)/\tau)}
$$
$M$就是队列的长度，论文设置为2*batch_size

然后将当前batch内的句子通过$f_k$，论文将$f_k$关闭dropout，$f_k$编码的vector存到queue中。



和MOCO原论文有一些不同，这里的$f_k$仅仅作为负样本的编码。word repetition后的sentence和original sentence通过一个BERT。这里的momentum contrast仅仅用来increase negative examples