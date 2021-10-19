# 对比试验

## 数据集

STS-B

| 训练集 | 验证集 | 测试集 |
| ------ | ------ | ------ |
| 5231   | 1458   | 1361   |





## 参数设置

Epochs=5

optimizer_type='AdamW'

scheduler='WarmupLinear'

warmup_proportion=0.1

optimizer_params={'lr': 2e-5}

weight_decay=0.01

max_seq_len=64

batch_size=128

## 实验结果

| 模型                   | 皮尔逊系数(Pearson) | 斯皮尔曼等级系数(Spearman) |
| ---------------------- | ------------------- | -------------------------- |
| chinese-roberta-wwm    | 0.6641              | 0.6808                     |
| SimCSE                 | 0.7585              | 0.7632                     |
| ConSERT(关闭dropout)   | 0.7792              | 0.7808                     |
| ConSERT(不关闭dropout) | 0.7764              | 0.7783                     |
| SimCSE+word repetition | 0.7754              | 0.7784                     |
| ESimCSE                | 0.7900              | 0.7929                     |



## 温度参数

| SimCSE      | 皮尔逊系数(Pearson) | 斯皮尔曼等级系数(Spearman) |
| ----------- | ------------------- | -------------------------- |
| $\tau$=0.05 | 0.7585              | 0.7632                     |
| $\tau$=0.1  | 0.7721              | 0.7737                     |
| $\tau$=0.5  | 0.7048              | 0.7028                     |
| $\tau$=1    | 0.6761              | 0.6779                     |



## 温度的作用

区分较难区分的负例
$$
L(x_i)=-\log \frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{N}\exp(s_{ij}/\tau)}
$$
其中$s_{ij}=f(x_i)^Tg(x_j)$。$f$和$g$分别分别代表encoder。

定义
$$
P_{ii}=\frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{N}\exp(s_{ij}/\tau)}
$$
其中k代表的时除了i以外的其它样本，因此$x_k$和$x_i$时负样本的关系

那么$x_i$的对比损失关于其与$x_k$相似性的梯度为：
$$
\frac{\partial L(x_i)}{\partial s_{ik}}=-\frac{\partial\log P(x_{ii})}{\partial s_{ik}}=-\frac{1}{P(x_{ii})}\frac{\partial P(x_{ii})}{\partial s_{ik}}
$$

$$
\frac{\partial P(x_{ii})}{\partial s_{ik}}=-(\frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{N}\exp(s_{ij}/\tau)})*\frac{1}{\sum_{j=1}^{N}\exp(s_{ij}/\tau)}*\frac{\partial\exp(s_{ik}/\tau)}{\partial s_{ik}}
$$

$$
\frac{\partial L(x_i)}{\partial s_{ik}}=\frac{1}{\tau}\frac{\exp(s_{ik}/\tau)}{\sum_{j=1}^{N}\exp(s_{ij}/\tau)}
$$

**所以可以看出，样本$x_i$关于其负样本$x_k$的梯度为公式5的形式，因此，如果两个样本的相似性越高，那么梯度就越大，对$x_k$这个样本的惩罚也就越大。越能够推开$x_i$和$x_k$之间的距离**

