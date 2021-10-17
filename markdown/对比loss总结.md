不像交叉熵损失，计算的是预测的概率分布与真实标签分布的差异，目的是预测一个label。

而排序损失，目的是预测两个输入之间的距离，利用ranking loss的任务也成为度量学习。

## Contrastive loss

[Chopra et al. 2005](http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf)

最初的contrastive loss只提供一个正例和一个负例进行对比


$$
v_i=f_\theta(x_i),\hat{v}_i=f_\theta(\hat{x_{i}})\\
d(v_i,\hat{v}_i)=\left\|v_i-\hat{v}_i \right\|_2\\
L=\frac{1}{2N}\sum_{i=1}^{N}(y_id_i^2+(1-y_i)\max(0,\gamma-d_i)^2) \\
\text{其中} y_i=\left\{\begin{matrix}
1, (x_i,\hat{x_i})\in p^+\\
0, (x_i,\hat{x_i}) \in p^- 

\end{matrix}\right.
$$
$\gamma$是一个超参数，用来控制负例的距离，当anchor与负例的距离超出$\gamma$时，不再优化参数，如果与负例的距离低于$\gamma$​，那么继续推开anchor与负例的距离。

```python
distance=torch.sqrt((vector1-vector2).pow(2).sum(dim=1))#(bach_size,)
loss=(labels*distance.pow(2)+(1-labels)*F.relu(gamma-distance).pow(2)).mean()
```



## Triplet loss

[Schroff et al. 2015](https://arxiv.org/pdf/1503.03832.pdf)
$$
v_i=f_\theta(x_i),v_i^+=f_\theta(x_i^+),v_i^-=f_\theta(x_i^-) \\
d(v_i,v_i^+ )=\left\| v_i-v_i^+\right\|_2^2,d(v_i,v_i^- )=\left\| v_i-v_i^-\right\|_2^2  \\
L=\frac{1}{N}\sum_{i=1}^{N}\max(0,d(v_i,v_i^+)-d(v_i,v_i^-)+\gamma)
$$
**anchor和正例之间的距离要比anchor和负例之间的距离小$\gamma$**

```python
#假如用余弦距离度量
cos_distance_anchor_pos=1-F.cosine_similarity(v_anchor,v_pos)#1-余弦分数代表余弦距离
cos_distance_anchor_neg=1-F.cosine_similarity(v_anchor,v_neg)
L=F.relu(cos_distance_anchor_pos-cos_distance_anchor_neg+gamma).mean()
```

## NT-Xnet

normalized temperature-scaled cross entropy

### NCE

**参数化的softmax函数**
$$
p(i|q)=\frac{\exp{(q^Tw_i)}}{\sum_{j=1}^{C}\exp{(q^Tw_j)}}
$$
参数化的函数不灵活，随着类别的改变，$W$的维度也要变

**非参数化的softmax函数**
$$
Sim(q,k)=\frac{q^Tk}{\left\|q\right\|\left\|k\right\|}\\
p(k^+|q)=\frac{\exp{(Sim(q,k^+))}}{\sum_{k\in \mathcal{K}}\exp(Sim(q,k))} \\
\mathcal{K}=\{k^+,k_1^-,k_2^-,\cdots\}\quad \mathcal{K}\text{包含一个正样本和所有的负样本} \\
L=-\log p(k^+|q)
$$
**噪声对比估计就是将多分类问题转为二分类问题，判别样本是属于正常分布还是噪音分布，解决计算归一化因子计算量太大的问题**，通过和噪音分布（预先指定，比如均匀分布）对比来学习正常数据 的分布模式。

假设噪音数据的分布是**均匀分布**，给定任意一对样本$(q,k)$，它们是正样本对的概率为：
$$
\begin{align*}
p(D=1|q,k)&=\frac{p(k^+|q)p(D=1)}{p(k^+|q)p(D=1)+p(k^-|q)p(D=0)} \\ 
 &= \frac{p(k^+|q)}{p(k^+|q)+mp(k^-|q)}
\end{align*} \\
$$
**对于$p(k|q)$​，研究表明，在神经网络建模的情况下，归一化因子$\sum_{k\in \mathcal{K}}\exp(q^Tk)$​可以设为1。​**

此时NCE的loss变为：
$$
L_{NCE}=-\log p(D=1|q,k)
$$
如果只取一个负样本，那么有：
$$
L_{NCE}=-\log\frac{\exp(Sim(q,k^+))}{\exp(Sim(q,k^+))+\exp(Sim(q,k^-))}
$$
==如果不用余弦相似度作为两者相似的度量标准，而是直接用向量内积呢？==

### InfoNCE

定义$p(k_{pos}|q)$​​为给定$q$​​取出正样本$k_{pos}$的概率，也就是说正样本是要从条件分布$p(\cdot|q)$中取出，$p(k_i)$为取出负样本$k_i$的概率。（因为正样本必须在给定$q$的前提下才算正样本）

定义$\mathcal{K}=\{k_i\}_{i=1}^{N}$​​。仅包含一个正样本。
$$
p(k_{pos}|q,\mathcal{K})=\frac{p(k_{pos}|q)\prod_{j=1,2,\cdots,N,j\neq pos}p(k_j)}{\sum_{n=1}^{N}[p(k_n|q)\prod_{j=1,2,\cdots,N,j\neq n}p(k_j)]} \\
p(k_{pos}|q,\mathcal{K})=\frac{\frac{p(k_{pos}|q)}{p(k_{pos})}}{\sum_{n=1}^{N}\frac{p(k_n|q)}{p(k_n)}}
$$
定义
$$
f(k,q)=\frac{p(k|q)}{p(k)}
$$
有：
$$
p(k_{pos}|q,\mathcal{K})=\frac{f(k,q)}{\sum_{i=1}^{N}f(k_i,q)}
$$
infoNCE的目的是最大化$p(k_{pos}|q,\mathcal{K})$，也就相当于最大化$\frac{p(k|q)}{p(k)}$，而这个分式就是$q$和$k$的局部互信息。
$$
I(k,q)=\sum_{k,q}p(k,q)\log\frac{p(k|q)}{p(k)}
$$
也就表明**最大化infoNCE的目标函数就相当于最大化一对正样本对$(q,k)$之间的互信息。**

**根据非参数化的softmax形式，我们可以将给定$q,\mathcal{K}$从中选出正样本的概率写为（根据公式4和9）**
$$
p(k_{pos}|q,\mathcal{K})=\frac{\exp(Sim(q,k_{pos}))}{\sum_{i=1}^{N}\exp(Sim(q,k_i))}
$$


所以infoNCE的loss形式为：
$$
L_{infoNCE}&=-\log p(k_{pos}|q,\mathcal{K}) \\
&=-\log\frac{\exp(Sim(q,k_{pos}))}{\sum_{i=1}^{N}\exp(Sim(q,k_i))}\\
$$
==最小化loss相当于最大化$I(q,k_{pos})$==



最终推得NT-Xent的loss形式：
$$
L_{NT-Xent}=-\log\frac{\exp(Sim(q,k_{pos})/\tau)}{\sum_{i=1}^{N}\exp(Sim(q,k_i)/\tau)}
$$
再直白一点就是：

**给定一个查询语句query以及batch个句子$\{x_i\}_{i=1}^{N}$​​​​，（假设batch=N），将这N+1个句子编码为vector，得到$q$​和$\{k_i\}_{i=1}^{N}$​，目标是从这N个vector中找出与$q$​是正样本对的vector（只有一个vector与$q$​是正样本对的关系），所以训练模型的loss定义为公式13的形式**



#####################################

- 在实际中，$k^+\sim p^+(\cdot|q)$，其中$p^+(\cdot|q)$就是根据$q$构造对应的数据增强样本所得到的就是$q$的正样本。$k^-\sim p^-(\cdot|q)$，其中$p^-(\cdot|q)$就是除了根据$q$构造的增强样本外的其余样本都是$q$​的负样本。
- 以上是先取$q$，然后根据$q$构造相似分布，进而得出不相似分布。在CV中的instance discrimination任务中，$q,k$是从联合概率中一起取出的，即$(q,k)\sim p^+(\cdot,\cdot)$，而相似分布$p^+(\cdot,\cdot)$就是从一张图片经过两次数据增强得到的两个样本作为相似样本。即$q=t(x),k=t'(x)$​。
- 很多情况下不相似分布(负样本数据分布)，$p^-(\cdot,\cdot)$或者$p^-(\cdot|q)$，并不是显示给出的，而是除了从相似分布中取出的样本外，其余取出的样本均视为负样本。

#####################################



实验涉及的loss包括：

- contrastive loss
- triplet loss
- softmax loss(parametric)
- infoNCE
- NT-Xent



$$
L=-\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(Sim(z_i,z_i’)/\tau)}{\sum_{j=1}^{N}\exp(Sim(z_i,z_j')/\tau)}
$$

$$
\begin{aligned}
L_{SCL}(i)&=-\frac{1}{m}\sum_{j=1}^{N+Q}\mathbb{I}_{y_i=y_j}\log\frac{\exp(Sim(z_i,z_j)/\tau)}{\sum_{k=1}^{N+Q}\exp(Sim(z_i,z_k)/\tau)}\\
L_{SCL}&=\frac{1}{N}\sum_{i=1}^{N}L_{SCL}(i)
\end{aligned}
$$

