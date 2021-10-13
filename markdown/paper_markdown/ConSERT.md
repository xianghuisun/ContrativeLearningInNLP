# 修改了models

## Transfromer

```python
attention_probs_dropout_prob=0.0,hidden_dropout_prob=0.0
config.attention_probs_dropout_prob=attention_probs_dropout_prob
config.hidden_dropout_prob = hidden_dropout_prob
self.auto_model=AutoModel.from_pretrained(model_name_or_path,config=config)#从这里可以选择是否在forward过程中使用dropout
self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
```

另外Transformer.py中会更新传进来的features，添加上all_layer_embeddings，这个取决于auto_model.config.output_hidden_states，然而默认是False的(就算是从官网上下载也是False)

```python
output_states=self.auto_model(**features)
if self.auto_model.config.output_hidden_states:
    all_layer_idx=2
    features.update({'all_layer_embeddings': output_states[all_layer_idx]})
```

Transformer的输入是{'input_ids','token_type_ids','attention_mask'}

Transformer的输出是{'input_ids','token_type_ids','attention_mask',‘token_embeddings’,'cls_token_embeddings','all_layer_embeddings'}

### AutoModel或者说BertModel的返回值

返回的是BaseModelOutputWithPooling的一个实例，这个实例有四个element

- last_hidden_state
- pooler_output
- hidden_states，就是所有层的hidden_state
- attentions

## Pooling

pooling中增加了四个选择(但是main函数中根本没有用到)

```python
pooling_mode_mean_last_2_tokens: bool = False,
pooling_mode_mean_first_last_tokens: bool = False, 
pooling_mode_pad_max_tokens: bool = False,
pooling_mode_pad_mean_tokens: bool = False,
```

假如用到了pooling_mode_mean_last_2_tokens和pooling_mode_mean_first_last_tokens

```python
input_mask_expanded=attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
sum_mask=input_mask_expanded.sum(1)#在seq_length上进行sum，因为pad的就是seq_length
if self.pooling_mode_mean_last_2_tokens and "all_layer_embeddings" in features:
    token_embeddings_last1=features["all_layer_embeddings"][-1]
    sum_embeddings_last1=(token_embeddings_last1*input_mask_expanded).sum(1)/sum_mask#(batch_size,hidden_size)
    token_embeddings_last2=features["all_layer_embeddings"][-2]
    sum_embeddings_last2=(token_embeddings_last2*input_mask_expanded).sum(1)/sum_mask#(batch_size,hidden_size)
    output_vectors.append((token_embeddings_last1+token_embeddings_last2)/2)
if self.pooling_mode_mean_first_last_tokens and "all_layer_embeddings" in features:
    token_embeddings_last=features["all_layer_embeddings"][-1]
    sum_embeddings_last=(token_embeddings_last*input_mask_expanded).sum(1)/sum_mask#(batch_size,hidden_size)
    token_embeddings_first=features["all_layer_embeddings"][0]
    sum_embeddings_first=(token_embeddings_first*input_mask_expanded).sum(1)/sum_mask#(batch_size,hidden_size)
    output_vectors.append((sum_embeddings_first + sum_embeddings_last) / 2)

output_vector=torch.cat(output_vectors,1)#(batch_size,hidden_size*multiple_dims)
features.update({"sentence_embedding":output_vector})
```

Pooling的输入是{'input_ids','token_type_ids','attention_mask',‘token_embeddings’,'cls_token_embeddings','all_layer_embeddings'}

Pooling的输出是{'input_ids','token_type_ids','attention_mask',‘token_embeddings’,'cls_token_embeddings','all_layer_embeddings','sentence_embedding'}



# 数据部分

## no_pair

**如果no_pair设置为True，那么传进去的就是单句子而不是句子对pair。no pair texts only used when contrastive loss only**

## cl_loss_only

如果设置cl_loss_only那么此时只能是no_pair（也不一定，只不过ConSERT代码这么设置的）

```python
if args.no_pair:
    assert args.cl_loss_only, "no pair texts only used when contrastive loss only"
    train_samples.append(InputExample(texts=[row['sentence1']]))
    train_samples.append(InputExample(texts=[row['sentence2']]))
    #单句子输入
else:
    train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
    #sentence pair as input
if is_chinese_data:
    for line in lines:
        sent1, sent2, label = line.strip().split("\t")
        if split == "train":
            all_samples.append(InputExample(texts=[sent1]))
            all_samples.append(InputExample(texts=[sent2]))
        else:
            all_samples.append(InputExample(texts=[sent1, sent2], label=float(label)))
     #也就是说对于中文数据集，训练的时候就是单句子传入？？？
```



### SentenceDataset

```python
class SentencesDataset(Dataset):
    '''
    each batch is only padded to its longest sequence instead of padding all sequences to the max length
    '''
    def __init__(self,examples: List[InputExample], model:SentenceTransformer):
        self.model = model
        self.examples = examples
        self.label_type = torch.long if isinstance(self.examples[0].label, int) else torch.float
    def __getitem__(self,index):
        #每一次都会传出一个example
        label = torch.tensor(self.examples[index].label, dtype=self.label_type)
        self.examples[index].texts_tokenized=[self.model.tokenize(text) for text in self.examples[index].texts]
        #texts是一个列表，如果是单句子任务，那就形如[sentence]。如果是句子对任务，那就形如[sentence1,sentence2]
        return self.examples[index].texts_tokenized, label
    #返回的self.examples[index].texts_tokenized就是将texts进行convert_tokens_to_ids的结果
    #如果是单句子任务，返回的就是形如[[1085,1456,2587,...,145]],size()==(1,seq_length)
    #如果是句子对任务，返回的列表有两个元素，这两个元素的长度不同
    #注意此时没有[CLS]和[SEP]在里面
```



```python
train_dataset = SentencesDataset(train_samples, model=model)
train_dataloader = DataLoader(train_dataset, shuffle=not args.no_shuffle,batch_size=train_batch_size)
#此时还不能enumerate train_dataloader，因为train_dataset各个样本的长度不同
#each element in the list of batch should be of equal size
```



# AdvCLSoftmaxLoss

```python
#_reps_to_output的作用就是根据rep_a和rep_b结合之前的限定条件如concatenation_sent_rep，concatenation_sent_difference，concatenation_sent_multiplication等等
#然后最后通过classifier，输出的output.size()==(batch_size,num_classes)
```

Tensor.detach(), return a new tensor, detached from the current graph,**the result will never require gradient.**

- normal_loss_stop_grad #对于传统的交叉熵loss是否加stop_grad，如果加的话，是加在rep_b上的 rep_b=rep_b.detach()
- data_augmentation_strategy_final_1，最终的五种数据增强方法（none,shuffle,token-cutoff,feature-cutoff,dropout），1代表生成第一个view

#### 存在一个问题，就是no_pairs的case下，在fit的493行

#### sentence_feature_a, sentence_feature_b = sentence_features是不对的



##### 代码中需要注意的

```python
embedding_output_a = self.model[0].auto_model.get_most_recent_embedding_output()
embedding_output_b = self.model[0].auto_model.get_most_recent_embedding_output()
#model[0]就是word_embedding_model，而需要注意的是word_embedding_model的auto_model=AutoModel.from_pretrained(xxxx)3
#此时默认调用的是官方的transformers的BertModel，而这个模型是没有get_most_recent_embedding_output()的
#解决办法是
word_embedding_model = models.Transformer(model_name_or_path)
word_embedding_model.auto_model=BertModel.from_pretrained(model_name_or_path)#此时的BertModel来自ConSERT
```



**两个batch的长度不是统一的**

![image-20210701193343602](D:\sunxianghui002\AppData\Roaming\Typora\typora-user-images\image-20210701193343602.png)



# 对抗训练

**If you indeed want the gradient for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor**

```python
#if not use embedding_output_a.retain_grad(), then normal_loss.backward(retain_graph=True) cannot access non-leaf tensor grad
embedding_output_a.retain_grad()
embedding_output_b.retain_grad()
normal_loss.backward(retain_graph=True)
noise_embedding_a=embedding_output_a+epsilon*normalize(embedding_output_a.grad.detach_())
#需要1e-10，不然真的全是nan
model[0].auto_model.set_flag("noise_embedding", noise_embedding_a)
```

**ori_feature_keys是用来保留传进model之前的keys，包括{'input_ids','token_type_ids','attention_mask'}, record the keys since the features will be updated**, 

features--->loss_model.forward--->Transformer(BertModel.forward(features))---

```python
self.embeddings = BertEmbeddings(config)
embedding_output = self.embeddings(input_ids=input_ids)
embedding_output = self._replace_embedding_output(embedding_output, attention_mask) 
def _replace_embedding_output(self, embedding_output, attention_mask):
    bsz, seq_len, emb_size = embedding_output.shape
    if self.exists_flag("data_aug_adv"):
        noise_embedding = self.get_flag("noise_embedding")
        assert noise_embedding.shape == embedding_output.shape
        self.unset_flag("noise_embedding")
        self.unset_flag("data_aug_adv")
        return noise_embedding
```



# args

- add_cl store_true assign true
- adv_training store_true assign true
- model_name_or_path assign path
- continue_training  assign False
- model_save_path assign path
- tensorboard_log_dir assign path
- force_del assign True
- adv_loss_stop_grad False



# cutoff和shuffling策略

```python
position_ids = self._replace_position_ids(input_ids, position_ids, attention_mask)
#传进去的position_ids是None，传出来的position_ids的每一个element都被shuffle了

```

