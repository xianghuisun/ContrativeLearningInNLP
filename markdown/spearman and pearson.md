相关系数用来衡量两个变量之间的相关性。

皮尔逊相关系数用来衡量两个变量之间的线性相关性。
$$
\rho_{XY}=\frac{cov(X,Y)}{\sigma(X)\sigma(Y)}=\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}}
$$
皮尔逊相关系数是有一些限制条件对变量的：

- 变量的标准差不能是0
- **两个变量之间是线性关系，都是连续数据**
- 每对观测值之间是相互独立的

斯皮尔曼等级相关系数也是一种衡量两个变量之间相关程度的评估指标，**与pearson系数不同的是，spearman采用等级rank来衡量。
$$
\rho_{XY}=1-\frac{6*\sum_{i=1}^{n}d_i^2}{n*(n^2-1)}
$$




==所谓等级就是每一个数值在变量(变量指的是X或者Y)内的排序顺序，不关系具体数值，只关注的是排序的顺序是否与真实标签的排序顺序一致==。

例如：

```python
X=[0.9,0.7,0.3,0.8]
Y=[5,4,3,2]#Y的数值可以仍认为是相似的程度，5表示特别相似

rank_X=[4,2,1,3]
rank_Y=[4,3,2,1]

d_i=[0,-1,-1,2]
d_i2=[0,1,1,4]
numerator=36
denominator=60
spearmanr=0.4
```

**关注的是数值在变量中的排序顺序指的是要想提升上例的spearmanr，就需要提升第二个样本的分数，使其顺序增加一位，第三个样本同理。减少第四个样本的分数，使其降低两位**

if X=[0.9,0.8,0.3,0.2]

then **spearmanr==1.0**

**需要注意的是如果出现了排序数值相等的case，那么排序的数值等于它们的位置的平均**

比如

```python
X=[0.9,0.9,0.3.0.8]
Y=[5,4,3,2]#Y的数值可以仍认为是相似的程度，5表示特别相似

rank_X=[3.5,3.5,1,2]#0.35=(3+4)/2
rank_Y=[4,3,2,1]

d_i=[-0.5,0.5,-1,1]
d_i2=[0.25,0.25,1,1]
numerator=15
denominator=60
spearmanr=0.75
```

**以文本匹配数据集计算spearson和pearson**

```python
X=[0.9,0.9,0.5,0.8,0.2,0.1]
Y=[1,1,0,1,0,0]

rank_X=[5.5,5.5,3,4,2,1]#5.5=(5+6)/2
rank_Y=[5,5,2,5,2,2]#2=(1+2+3)/3,5=(4+5+6)/3

d_i=[0.5,0.5,1,-1,0,-1]
d_i2=[0.25,0.25,1,1,0,1]
numerator=21
denominator=210
spearmanr=0.9

mu_X=0.57
mu_Y=0.5
numerator=0.0
for i in range(6):
    numerator+=(scores[i]-0.57)*(labels[i]-0.5)

X_sigma=0.0
Y_sigma=0.0
for i in range(6):
    X_sigma+=(scores[i]-0.57)**2
    Y_sigma+=(labels[i]-0.5)**2
X_sigma=math.sqrt(X_sigma)
Y_sigma=math.sqrt(Y_sigma)
denominator=X_sigma*Y_sigma
pearsonr=numerator/denominator
pearsonr=0.9233
```

==spearmanr不关心预测两个句子相似的具体分数数值，关注的是预测的分数数值的排序是否和标准答案分数的排序一致==



