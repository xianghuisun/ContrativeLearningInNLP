contrastive learning with **semantic negative examples**

众所周知，NN对于对抗样本来说是比较vulnerable的，目前很多提升NN robustness的研究工作主要集中在对抗训练上。对抗训练主要让模型可以识别出与原始样本虽然些许不同但是语义相近的样本。

**而本文提出这些工作只集中于adversarial examples，而忽视了与原始文本语义不同甚至语义相反的样本**

==论文指出，通过让模型对比语义相近的样本和语义不相近的样本，（论文认为语义相近指的是adversarial examples，语义不相近指的是contrastive examples，这两种样本都是经过small perturbations得到的，只不过语义与原句子一个相近一个相反）可以提升模型识别这两类样本的能力，也就是让模型对语义更加sensitive==



作者实验发现，经过对抗训练后的模型，对contrastive examples的识别率下降了。说明经过对抗训练后的模型对于语义不敏感了。



作者指出，如SimCSE等基于对比学习训练句向量的模型中，**负样本的构造过于简单**，**并没有考虑semantic negative examples，也就是与原始句子语义相反但是形似相近的样本**。作者认为，这种基于对抗样本和对比样本结合的对比学习方式，**可以同步的提升模型的robustness和semantic sensitivity**。



**论文中所谓的adversarial examples其实只是将原句子中的部分单词替换为同义词而已，未必使得标签真的改变了，所以可以在写专利的时候指出positive examples的生成方式可以采用textfooler等黑盒攻击的方式。**



- 论文中positive examples的生成方式是将原句子中的部分单词进行（同义词、上义词、或者名词变动词、动词变形容词等）替换
- 论文中negative examples的生成方式是将原句子中的部分单词进行（反义词、随机单词）替换



论文的loss分为三部分，除了MLM和contrastive loss，**模型还要预测adversarial sentence and contrastive sentence中哪一个token是被replace的**

这种通过同义词反义词替换同时利用textfooler的方式构造对抗对比样本的思路在实际中是很难实现的，尤其是中文，不过写专利倒是不错的选择，可以和ConSERT、SimCSE等进行对比。

指出之前的各种基于对比学习的文本表示模型的差异只是positive example的生成方式不同。而negative examples都是采用令一个batch内的其余句子作为negative examples。而忽视了与原始句子形似但是语义不同甚至相反的hard neg examples。



最后需要注意的是，这篇论文仅生成一个contrastive example作为negative example，所以contrastive loss形如：
$$
-\log\frac{\exp(Sim(q,k_{adv}))}{\exp(Sim(q,k_{adv}))+\exp(Sim(q,k_{con}))}
$$


怎么实现多个呢？

难道起名为多样化negative examples的表示学习嘛？？？