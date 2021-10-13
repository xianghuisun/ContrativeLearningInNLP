# 协方差矩阵

方差刻画单个变量的离散程度

协方差刻画两个变量的相关程度

任何的协方差矩阵是对称和半正定的，主对角线包含方差（the covariance of each element with itself）

每一个句子看作一个$x_i$，也就是一个随机变量。那么所有句子就是$X=(X_1,X_2,\cdots,X_n)^T$
$$
\begin{align*}
E[(x-E(x))(x-E(x))^T] &= E[x^2-2xE(x)+E(x)E(x)^T]\\ 
 &= E(x^2)-2E^2(x)+E^2(x) \\
 &= E(x^2)-E^2(x)
\end{align*}
$$
**原始数据的协方差**
$$
\Sigma =E(X^2)-E^2(X)
$$
定义转换:
$$
\widetilde{x}_i=(x_i-\mu)W
$$


转换后的协方差
$$
\widetilde{\Sigma}=E(\widetilde{X}^2)-0=E(((x_i-\mu)W)^2)=W^T\Sigma W
$$
目标是让转换后的协方差变为单位阵，也就是方差是1
$$
W^T\Sigma W=I
$$

$$
\Sigma=(W^T)^{-1}W^{-1}
$$

**协方差矩阵是半正定对称阵**

证明协方差矩阵是半正定的：
$$
\begin{align*}
y^T\Sigma y&=E[y^T(x-\mu)(x-\mu)^Ty]\\ 
&=E[((x-\mu)^Ty)^T((x-\mu)^Ty)]\\ 
&=E[((x-\mu)^Ty)^2]\geqslant 0\\
\end{align*}
$$

- 半正定矩阵的特征值大于等于0
- 实对称矩阵的特征分解：$A=U\Sigma U^T$，其中$\Sigma$是由$A$的特征值组成的对角阵，$U$是由特征向量组成的
- 如果把$A$的所有特征向量进行标准化，那么A的所有规范化后的特征向量构成了一组标准正交基，那么$U$也就自然是正交矩阵
- 由于协方差矩阵是半正定的，所以$\Sigma$的对角线元素全部大于等于0



# 正交矩阵

一组向量称为规范正交基：($\alpha_1,\alpha_2,\cdots.\alpha_n$)，如果满足当$i==j$时,向量内积$<\alpha_i,\alpha_j>=1$，$i\neq j$时，向量内积是0。那么A称为正交矩阵，$A=[\alpha_1,\alpha_2,\cdots,\alpha_n]$.



**所以我们可以看到，根据标准正交基，那么$AA^T=E$，于是$A^T=A^{-1}$**





已经证明原始数据的协方差矩阵$\Sigma$可以分解为$U\Lambda  U^T$以及等式$\Sigma=(W^T)^{-1}W^{-1}$

所以有:
$$
\begin{align*}
(W^T)^{-1}W^{-1} &=U\Lambda U^T \\ 
 &= U\sqrt{\Lambda^T}\sqrt{\Lambda}U^T\\ 
 W^{-1}&= \sqrt{\Lambda}U^T \\
W&=U\sqrt{\Lambda^{-1}}
\end{align*}
$$




# whitening

也就是说取出来所有训练数据的句向量，每一个句向量看作一个随机变量$X_i$，$X=[X_1,X_2,\cdots,X_n]$。计算$X$的$\mu$和协方差矩阵$\Sigma$，然后对协方差矩阵进行特征分解，得到$U,\Lambda$，也就是得到了$\Sigma$的特征值和特征向量，所谓的whitening操作就是将$X$线性变换到另一个空间：
$$
\widetilde{x}_i=(x_i-\mu)U\sqrt{\Lambda^{-1}}
$$
接下来计算余弦相似度的时候，使得这些向量在一个基底下比较，从而余弦相似度更加合理



## 降维

$U$的作用是变换向量，$\Lambda$则控制变换的幅度，因为特征值的作用就是控制线性变换的幅度，所以特征值较小的那些特征值忽略掉，也就是PCA降维