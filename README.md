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

