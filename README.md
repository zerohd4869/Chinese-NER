# Chinese NER Using Neural Network

## 任务简介
命名实体识别 (Named Entity Recognition, NER) 涉及实体边界的确定和命名实体识别类别的识别，是自然语言处理 (NLP) 领域的一项基础性工作。本项目针对 Chinese NER 任务，目前已复现 BiLSTM-CRF、Lattice LSTM 和 LR-CNN 等基线模型。

## 项目运行

### 环境要求
Pytorch v0.4.1 </br>
Python v3.6.2 </br>
numpy </br>
tqdm </br>

### 数据准备

Resume 开源数据集是Yue等人在 Sina Finance 采集的简历数据集，主要包括来自中国股票市场上市公司的高级管理人员的简历数据，可在 paper[4] 中获取，并将其放入目录```./data/resume```下。

**数据统计**：

Typing| Train | Dev |Test
:-:|:-|:-:|:-:
Sentence  | 3.8k | 0.46k | 0.48k
Char   | 124.1k | 13.9k | 15.1k

**标注策略**：BMEO

**分割方式**: '\t' (吴\tB-NAME)  

**标注具体类型：**

该数据集使用 YEDDA System（Yang et al., 2018）手动注释了8种命名实体。

Tag | Meaning | Train | Dev |Test
:-:|:-|:-:|:-:|:-:
CONT | Country                  | 260 | 33  | 28
EDU  | Educational Institution  | 858 | 106 | 112
LOC  | Location                 | 47  | 2   | 6
NAME | Personal Name            | 952 | 110 | 112
ORG  | Organization             | 4611| 523 | 553
PRO  | Profession               | 287 | 18  | 33 
RACE | Ethnicity Background     | 115 | 15  | 14
TITLE| Job Title                | 6308| 690 | 772
Total Entity |---               |13438| 1497| 1630

详见目录```data/resume```。

### 加载预训练 Embeddings


预训练 Embeddings 使用了分词器 [RichWordSegmentor](https://github.com/jiesutd/RichWordSegmentor)（Yang et al.,2017a）的 baseline。其中，```gigaword_chn.all.a2b.uni.ite50.vec```, ```gigaword_chn.all.a2b.bi.ite50.vec``` 和 ```ctb.50d.vec``` 分别对应的是 char, bichar 和 word embeddings，这三个 ```*.vec``` 文件均可在 RichWordSegmentor 项目中获取，并将其放入目录```./data/```下。


### 模型训练

参数配置文件是 ./*.conf , 其中 lrcnn_ner.conf 为默认配置文件，配置了 LR-CNN 模型默认参数。同样的，lattice_ner.conf 是配置了 Lattice LSTM 模型默认参数，charbl_ner.conf 是基于char的 BiLSTM-CRF 基线模型配置文件， charbl_ner.conf 是基于 char 和 bichar 的 BiLSTM-CRF 模型配置文件。

使用 LR-CNN 模型进行训练时，在配置文件 ./lrcnn_ner.conf 中修改参数 status 为 train （训练），其它参数可进行对应修改（或使用其默认值），然后运行以下命令： 
``` bash
python main.py --conf_path ./lrcnn_ner.conf # conf_path 配置文件地址

```

### 模型评估与预测

    在模型的对应配置文件 ./*.conf 中修改参数 status 为 test （性能评估及预测）。运行以下命令：

``` bash
python main.py --conf_path ./lrcnn_ner.conf

```

## 性能说明

### 实验结果
在 Resume 数据集下的结果如下表：

Models | ACC | P | R |F1
:-|:-:|:-:|:-:|-
BiLSTM-CRF [Lample et al., 2016]            | 0.9564 | 0.9335| 0.9323 | 0.9329
BiLSTM-CRF + bichar [Yang et al., 2017a]| 0.9599| 0.9393| 0.9405| 0.9399
Lattice LSTM [Yang et al., 2018]             | 0.9662 | 0.9378| **0.9429** | 0.9403
LR-CNN [Gui et al., 2019]                  | **0.9689** | **0.9499** | 0.9417 | **0.9458** 

结果表明，在 F1指标下，引入了反思机制的 LR-CNN 模型的 F1 值为 0.9458，明显优于基于 char 和基于 char 和 bichar 的 BiLSTM-CRF 模型。此外，通过更好的整合词典信息，LR-CNN 优于 Lattice LSTM 模型，在该 NER 任务上取得了最好的效果。同时，LR-CNN 方法的训练速度比 LatticeLSTM 方法快近3倍。

### 结果分析

以上四个基于 char 的神经网络模型, 不仅都可以有效地捕捉上下文信息, 而且均可以避免词粒度编码时的分词错误带来的影响。

其中，加入 bichar 的 BiLSTM-CRF 模型充分利用了字粒度信息，效果略优于加 BiLSTM-CRF 传统基线模型。对于 Lattice LSTM 中文基线模型，相较于前两者，将字符级别序列信息和该序列对应的词信息同时编码供模型自动取用，加入的词信息更加丰富了语义表达，且它的门控循环单元允许模型从一个句子中选择最相关的字符和单词，进而可以取得更好的效果。这也反映了词典在字符级的中文NER任务中起着重要作用。

而引入了反思机制的 LR-CNN 模型比 Lattice LSTM 等上述三个模型取得了更快更好的效果，这说明了利用反思机制解决匹配相同字符的潜在词之间的冲突的方法，可以进一步提高词典信息的有效利用。而利用 CNN 结构把句子里的所有字符以及所有字符对应所有可能的词语全部并行地进行处理，以更充分的利用 GPU 的性能，因此训练速度会比RNN快很多。

## 总结

在该 Resume 数据集上，分别使用了 BiLSTM-CRF 传统基线模型、加入 bichar 的 BiLSTM-CRF 模型、Lattice LSTM 中文基线模型和引入反思机制的 LR-CNN 模型这四个神经网络方法来进行中文命名实体的识别。

实验结果表明，引入反思机制的 LR-CNN 方法与 BiLSTM-CRF、加入 bichar 的 BiLSTM-CRF 和 Lattice LSTM 这三个方法相比，该方法可以显著地提高模型性能和训练速度，进而更好的完成了该中文 NER 的任务。

**参考文献**

[1] Chiu, J. P. C. , & Nichols, E. . (2015). Named entity recognition with bidirectional lstm-cnns. Computer Science.

[2] Jie Yang, Yue Zhang, Linwei Li, and Xingxuan Li. 2018. Yedda: A lightweight collaborative text span annotation tool. In ACL. Demonstration.

[3] Jie Yang, Zhiyang Teng, Meishan Zhang, and Yue Zhang. 2016. Combining discrete and neural fea- tures for sequence labeling. In CICLing.

[4] Kim, Y. . (2014). Convolutional neural networks for sentence classification. Eprint Arxiv.

[5] Lample, G. , Ballesteros, M. , Subramanian, S. , Kawakami, K. , & Dyer, C. . (2016). Neural architectures for named entity recognition.

[6] Li, X., Jie, Z., Feng, J. , Liu, C. , & Yan, S. (2017). Learning with rethinking: recurrently improving convolutional neural networks through feedback. Pattern Recognition, 79.

[7] Ma, X. , & Hovy, E. . (2016). End-to-end sequence labeling via bi-directional lstm-cnns-crf.
Strubell, E., Verga, P. , Belanger,D. , & Mccallum, A. . (2017). Fast and accurate entity recognition with iterated dilated convolutions.

[8] Tao Gui, Ruotian Ma, Qi Zhang, Lujun Zhao, Yu-Gang Jiang, & Xuanjing Huang. 2019. CNN-Based Chinese NER with Lexicon Rethinking, In Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI 2019), August 10-16.

[9] Yang, J. , Zhang, Y. , & Dong, F. . (2017). Neural word segmentation with rich pretraining.

[10] Zhang, Y. , & Yang, J. . (2018). Chinese NER Using Lattice LSTM.