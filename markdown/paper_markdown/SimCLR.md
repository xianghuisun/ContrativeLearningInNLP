# 模型（SimCLR用来学习无监督表示）

## 模型总览

- 将数据x两次通过同一个数据增强模块得到x的两个稍有不同的版本$x_1$和$x_2$。
- 将这两个版本通过同一个特征提取层，得到两版的向量表示$h_1$和$h_2$。
- 将两版的向量表示通过一个非线性全连接层($W_2ReLU(W_1*h)$)得到两个向量表示$g_1$和$g_2$
- 利用$g_1$和$g_2$计算NT-Xnet。(Normalized Temperature-scaled cross entroy loss)
- ==计算loss的时候增加了负例，本分支的N-1个样本也作为负例，总计2*(N-1)个负例，而且(i,j)和(j,i)视为两次不同的计算，也就是同一个pair的两个句子视为同等的==。
- ==训练后，丢弃projection层，将编码器输出的向量作为下游任务的输入，丢弃的原因暂时没读懂论文的意思==

## 结论

- 在编码器和CL之间添加一个非线性映射可以提高向量表示的质量（==这个质量指的是编码器输出的向量表示的质量==）
- 数据增强的方式很重要（这里的数据增强是指构造对比样本的数据处理方法）
- 使用对比交叉熵loss的表示学习得益于**normalized embeddings**以及合适的temperature(**这里的temperature很重要，如果是正常等于1的case下，模型是不太容易训练的，实践证明，这种case下loss可能不会下降。温度的值越高，分布越平缓，模型越不容易学习。温度的值越低，分布越尖锐，模型越容易学习,SBERT官方给出的温度值是0.05,也就是在logits上乘以20即可**)
- 无监督对比学习获得的收益在以下的case大于监督学习：
  - 更大的batch_size和training steps(epochs)
  - stronger 数据增强策略（也就是说数据增强对于监督学习的收益甚微）
  - 更大、更深的网络





## 代码

```python
hidden1=hidden1.normalize(p=2,dim=-1)
hidden2=hidden2.normalize(p=2,dim=-1)
#normaized后计算余弦相似度只需要向量相乘
hidden1_aug=hidden1
hidden2_aug=hidden2
mask=torch.nn.functional.one_hot(torch.arange(batch_size),num_classes=batch_size)
#temperature可以取0.1或者0.05
logits_aa=torch.matmul(hidden1,hidden1_aug.transpose(0,1))/temperature
logits_aa=logits_aa-mask*1e9
logits_ab=torch.matmul(hidden1,hidden2_aug.transpose(0,1))/temperature
logits_bb=torch.matmul(hidden2,hidden2_aug.transpose(0,1))/temperature
logits_bb=logits_bb-mask*1e9
logits_ba=torch.matmul(hidden2,hidden1_aug.transpose(0,1))/temperature
#logits_aa和logits_bb本身就是对称的
#logits_ab和logits_ba是互为对称的关系
labels=torch.nn.functional.one_hot(torch.arange(batch_size),num_classes=batch_size*2)
loss_a=nn.CrossEntropy(torch.cat([logits_ab,logits_aa]),labels)
loss_b=nn.CrossEntropy(torch.cat([logits_ba,logits_bb]),labels)
cl_loss=loss_a+loss_b
```



## loss

==The final loss is computed across all positive pairs==

假设输入两个句子a和b(batch_size=2)，分别得到对应的增强表示$a1,a2,b1,b2$
$$
cl\_loss(a1,a2)=-\log(\frac{e^{sim(a1,a2)}}{e^{sim(a1,a2)}+e^{sim(a1,b1)}+e^{sim(a1,b2)}})
$$

$$
cl\_loss(a2,a1)=-\log(\frac{e^{sim(a2,a1)}}{e^{sim(a2,a1)}+e^{sim(a2,b1)}+e^{sim(a2,b2)}})
$$

$$
cl\_loss(b1,b2)=-\log(\frac{e^{sim(b1,b2)}}{e^{sim(b1,b2)}+e^{sim(b1,a1)}+e^{sim(b1,a2)}})
$$

$$
cl\_loss(b2,b1)=-\log(\frac{e^{sim(b2,b1)}}{e^{sim(b2,b1)}+e^{sim(b2,a1)}+e^{sim(b2,a2)}})
$$

最终的loss就是：
$$
cl\_loss=(cl\_loss(a1,a2)+cl\_loss(a2,a1)+cl\_loss(b1,b2)+cl\_loss(b2,b1))/(2*2)
$$
**第二个2指的是batch_size**



| labels形式 | a2   | a1   | b2   | b1   |
| ---------- | ---- | ---- | ---- | ---- |
| **a1**     | 1    | 0    | 0    | 0    |
| **a2**     | 0    | 1    | 0    | 0    |
| **b1**     | 0    | 0    | 1    | 0    |
| **b2**     | 0    | 0    | 0    | 1    |

