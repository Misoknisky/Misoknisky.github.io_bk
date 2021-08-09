---
layout: post
title:  "MRC比赛总结"
categories: MRC
tags: NLP MRC  
author: admin
---

* content
{:toc}


###  前言

MRC（机器阅读理解）是NLP（自然语言处理）领域极具挑战性的研究课题。MRC的任务形式多种多样：生成式问答、多跳式问答、对话式问答、开放领域问答、抽取式问答、知识库问答等等。生成式问答主要特点是答案不是文中的连续片段，要求模型能够结合问题和文章内容生成一句或者一段连贯的文字作为最终的答案；多跳式问答强调的模型的推理能力，要求模型具备句间或者文章之间的逻辑推理能力；对话形式的问答是将阅读理解和对话系统结合的产物，表现形式是给定一段文字，要求模型能够根据文字内容进行一段简单的多轮对话，难点在于如何解决指代消解、如何结合历史信息推理；开放领域问答特点：数据集中的文章是一篇完整结构的文字（以往大多数文章都是单段落的），模型需要具备篇章级文字处理能力，预测答案的复杂度也随之增加；抽取式问答：答案是文章中的一个连续片段；知识库问答即KBQA；

###   MRC
> ####  CoQA会话阅读理解
>> [CoQA](https://stanfordnlp.github.io/coqa/)  
>> [Baseline](https://github.com/stanfordnlp/coqa-baselines)  
>> [论文](https://arxiv.org/abs/1808.07042)  
> #### HotpotQA推理式阅读理解
>> [HotpotQA](https://hotpotqa.github.io/)  
>> [Baseline](https://github.com/hotpotqa/hotpot)  
>> [论文](https://arxiv.org/pdf/1809.09600.pdf)  
> #### Natural Questions开放领域阅读理解
>> [BERTBaseline](https://github.com/google-research/language/tree/master/language/question_answering)  
>> [github utils](https://github.com/google-research-datasets/natural-questions)  
>> [论文](https://ai.google/research/pubs/pub47761)  
> #### Squad Dataset 抽取形式
>> [Squad1.0 and Squad2.0](https://rajpurkar.github.io/SQuAD-explorer/)  
>> [Baseline](https://github.com/aswalin/SQuAD)  
>> [BertBaseline](https://github.com/google-research/bert(tensorflow))  
>> [BertBaseline](https://github.com/gitkangkangliu/pytorch-transformers(pytorch))  
>> [论文squad1.0](https://arxiv.org/abs/1606.05250)  
>> [论文squad2.0](https://arxiv.org/abs/1806.03822)  
> #### MS MARCO生成式
>> [MS MARCO](http://www.msmarco.org/dataset.aspx)  
>> [Baseline](https://github.com/microsoft/MSMARCO-Question-Answering)  
>> [论文](https://arxiv.org/abs/1611.09268)  
> #### TriviaQa（偏抽取）
>> [TriviaQa](http://nlp.cs.washington.edu/triviaqa/)  
>> [github utils](https://github.com/mandarjoshi90/triviaqa)  
>> [论文](https://arxiv.org/abs/1705.03551)  
> #### DuReader抽取形式
>> [DuReader](http://ai.baidu.com/broad/leaderboard)  
>> [Baseline](https://github.com/baidu/DuReader)  
>> [论文](https://www.aclweb.org/anthology/W18-2605/)  

### 数据集下载
> ####  Download GloVe
``` 
mkdir wordvecs
wget -P wordvecs http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip
unzip -d wordvecs wordvecs/glove.42B.300d.zip
wget -P wordvecs http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip
<<<<<<< HEAD
unzip -d wordvecs wordvecs/glove.840B.300d.zip ```
```
> #### Download COQA Data
```
mkdir data
wget -P data https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json
wget -P data https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json
```
> #### Download Hotpot Data
```
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
```
> #### Download Squad Data
```
mkdir data
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
```
> #### Download Download TriviaQa
```
wget http://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz
wget http://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz
```
> #### Download MS MARCO
```
进入页面下载　http://www.msmarco.org/dataset.aspx
```
> #### Download DuReader
```
进入页面下载 http://ai.baidu.com/broad/download
```
> ####  Download Natural Questions
```
#kaggle
```








```

```