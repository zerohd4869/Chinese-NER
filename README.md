# Chinese NER Using Neural Network

## 任务简介
命名实体识别 (Named Entity Recognition, NER) 涉及实体边界的确定和命名实体识别类别的识别，是自然语言处理 (NLP) 领域的一项基础性工作。本项目针对 Chinese NER 任务，已复现 BiLSTM-CRF、Lattice LSTM、LR-CNN、WC-LSTM 等模型。另外 图模型 LGN 源码实现见 [LGN code](https://github.com/RowitZou/LGN)，序列模型 SLK-NER 代码实现见 [SLK-NER code](https://github.com/zerohd4869/SLK-NER)。

## 项目运行

### 环境要求
Pytorch v0.4.1 </br>
Python v3.6.2 </br>
numpy </br>
tqdm </br>

### 数据准备

Resume 开源数据集是Yue等人在 Sina Finance 采集的简历数据集，主要包括来自中国股票市场上市公司的高级管理人员的简历数据，可在 [Yang et al., 2018] 中获取，并将其放入目录```./data/resume```下。

**数据统计**：

Typing| Train | Dev |Test
:-:|:-|:-:|:-:
Sentence  | 3.8k | 0.46k | 0.48k
Char   | 124.1k | 13.9k | 15.1k

**标注策略**：BMEO

**分割方式**: '\t' (吴 \t B-NAME)  

**标注具体类型：**

该数据集使用 YEDDA System [Yang et al.,2018] 手动注释了8种命名实体。

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


预训练 Embeddings 使用了分词器 [RichWordSegmentor](https://github.com/jiesutd/RichWordSegmentor) [Yang et al.,2017a] 的 baseline。其中，```gigaword_chn.all.a2b.uni.ite50.vec```, ```gigaword_chn.all.a2b.bi.ite50.vec``` 和 ```ctb.50d.vec``` 分别对应的是 char, bichar 和 word embeddings，这三个 ```*.vec``` 文件均可在 RichWordSegmentor 项目中获取，并将其放入目录```./data/```下。


### 模型训练

参数配置文件是 ./*.conf, 运行实例： 

<!--, 其中 wclstm_ner.conf 为默认配置文件，配置了 WC-LSTM 模型的默认参数。同样的，lrcnn_ner.conf是 LR-CNN 的模型配置文件，lattice_ner.conf 是 Lattice LSTM 的模型配置文件，charbl_ner.conf 是基于char的 BiLSTM-CRF 基线模型配置文件， charbl_ner.conf 是基于 char 和 bichar 的 BiLSTM-CRF 模型配置文件。
使用 WC-LSTM 模型进行训练时，在配置文件 ./wclstm_ner.conf 中修改参数 status 为 train （训练），其它参数可进行对应修改（或使用其默认值），然后运行以下命令：-->

``` bash
python main.py --conf_path ./wclstm_ner.conf # conf_path 配置文件地址

```

### 模型评估与预测

在模型的对应配置文件 ./*.conf 中修改参数 status 为 test （性能评估及预测）。运行实例：

``` bash
python main.py --conf_path ./wclstm_ner.conf

```

<!--## 性能说明 -->

### 实验结果
在 Resume 数据集下的结果如下表：

Models  | P | R |F1
:-|:-:|:-:|-
BiLSTM-CRF [Lample et al., 2016]            | 93.7    | 93.3    | 93.5
BiLSTM-CRF + bichar [Yang et al., 2017a]    | 93.9    | 94.1    | 94.0
CAN [Zhu et al., 2019]                     | 95.1    | 94.8    | 94.9
BERT [Devlin et al., 2019]                         | 94.2    | 95.8    | 95.0
Lattice LSTM [Yang et al., 2018]            | 94.8    | 94.1    | 94.5
LR-CNN [Gui et al., 2019]                   | **95.4**| 94.8    | 95.1 
WC-LSTM [Liu et al., 2019]                  | 95.3    | 95.2    | 95.2
LGN [Gui et al., 2019]                          | 95.3    | 95.5    | 95.4
SLK-NER [Hu et al., 2020]                  | 95.2    | **96.4** | **95.8**

<!-- 
### 结果分析
以上四个基于 char 的神经网络模型, 不仅都可以有效地捕捉上下文信息, 而且均可以避免词粒度编码时的分词错误带来的影响。
其中，加入 bichar 的 BiLSTM-CRF 模型充分利用了字粒度信息，效果略优于加 BiLSTM-CRF 传统基线模型。对于 Lattice LSTM 中文基线模型，相较于前两者，将字符级别序列信息和该序列对应的词信息同时编码供模型自动取用，加入的词信息更加丰富了语义表达，且它的门控循环单元允许模型从一个句子中选择最相关的字符和单词，进而可以取得更好的效果。这也反映了词典在字符级的中文NER任务中起着重要作用。
引入了反思机制的 LR-CNN 模型比 Lattice LSTM 等上述三个模型取得了更快更好的效果，这说明了利用反思机制解决匹配相同字符的潜在词之间的冲突的方法，可以进一步提高词典信息的有效利用。而利用 CNN 结构把句子里的所有字符以及所有字符对应所有可能的词语全部并行地进行处理，以更充分的利用 GPU 的性能，因此训练速度会比RNN快很多。
最后，WC-LSTM 效果目前最好。 -->

**参考文献**

[1] Jie Yang, Yue Zhang, Linwei Li, and Xingxuan Li. 2018. Yedda: A lightweight collaborative text span annotation tool. In ACL. Demonstration.

[2] Jie Yang, Zhiyang Teng, Meishan Zhang, and Yue Zhang. 2016. Combining discrete and neural features for sequence labeling. In CICLing.

[3] Ma, Xuezhe, and Eduard Hovy. End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF.
Strubell, E., Verga, P. , Belanger,D. , & Mccallum, A. . (2017). Fast and accurate entity recognition with iterated dilated convolutions.

[4] Lample, Guillaume, et al. Neural Architectures for Named Entity Recognition. Proceedings of NAACL-HLT. 2016.

[5] Yang, Jie, Yue Zhang, and Fei Dong. Neural Word Segmentation with Rich Pretraining. Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2017.

[6] Yuying Zhu and Guoxin Wang. Can-ner: Convolutional attention network for chinese named entity recognition. In NAACL, pages 3384–3393, 2019.

[7] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidi-rectional transformers for language understanding. In NAACL, pages 4171–4186, Minneapolis, June 2019.

[8] Zhang, Yue, and Jie Yang. Chinese NER Using Lattice LSTM. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2018.

[9] Tao Gui, Ruotian Ma, Qi Zhang, Lujun Zhao, Yu-Gang Jiang, & Xuanjing Huang. 2019. CNN-Based Chinese NER with Lexicon Rethinking, In Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI 2019), August 10-16.

[10] Liu, Wei, et al. An Encoding Strategy Based Word-Character LSTM for Chinese NER. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). 2019.

**[11] Tao Gui, Yicheng Zou, Qi Zhang, Minlong Peng, Jinlan Fu, Zhongyu Wei, and Xuan-Jing Huang. A lexicon-based graph neural network for chinese ner. In EMNLP- IJCNLP, pages 1039–1049, 2019.

**[12] Dou Hu and Lingwei Wei. ”SLK-NER: Exploiting Second-order Lexicon Knowledge for Chinese NER.” The 32st International Conference on Software & Knowledge Engineering. 2020.
