![DiaParser](docs/source/logo.png)
# DiaParser

[![build](https://github.com/Unipisa/diaparser/workflows/build/badge.svg)](https://github.com/Unipisa/diaparser/actions)
[![docs](https://readthedocs.org/projects/diaparser/badge/?version=latest)](https://diaparser.readthedocs.io/en/latest)
[![release](https://img.shields.io/github/v/release/Unipisa/diaparser)](https://github.com/Unipisa/diaparser/releases)
[![downloads](https://pepy.tech/badge/diaparser)](https://pepy.tech/project/diaparser)
[![LICENSE](https://img.shields.io/github/license/Unipisa/diaparser)](https://github.com/Unipisa/diaparser/blob/master/LICENSE)

`DiaParser` is a **state-of-the-art dependency parser**, that extends the architecture of the Biaffine Parser ([Dozat and Manning, 2017](#dozat-2017-biaffine)) by exploiting both embeddings and attentions provided by **transformers**.

By exploiting the rich hidden linguistic information in contextual embeddings from transformers, `DiaParser` can avoid using intermediate annotations like POS, lemma and morphology, used in traditional parsers.
Therefore the only stages in the parsing pipeline are tokenization and parsing.

The parser may also work **directly on plain text**.
The parser automatically dowloads pretrained models as well as tokenizers and produces dependency parsing trees, as detailed in [Usage](#Usage).

Exploiting attentions from transformer heads provides improvements in accuracy, without resorting to fine tuning or training its own attention.
Overall, this simplifies the architecture and lowers the cost of resources needed during training, especially memory, and allows the parser to improve as new versions of transformers become available.
The parser uses the [HuggingFace Transformers API](https://huggingface.co/transformers/) and in particular the generic [AutoClasses interface](https://huggingface.co/transformers/model_doc/auto.html) to access the transformer models avaiable.

We plan to track the improvements in transformer technology and to realease models of the parser that incorporate these models.
Currently we provide **pretrained models** for 18 languages.

We encourage anyone to contribute trained models for other combinatiobns of language/transformer pairs, that we will publish.
We will soon provide a web form where to upload new models.

You can also train your own models and contribute them to the repository, to share with others.

`DiaParser` uses pretrained contextual embeddings for representing input from models in [`transformers`](https://github.com/huggingface/transformers).

Pretrained tokenizers are provided by [Stanza](https://stanfordnlp.github.io/stanza/).

Alternatively to contextual embeddings, `DiaParser` also allows to utilize CharLSTM layers to produce character/subword-level features.
Both BERT and CharLSTM avoid the need of generating POS tags.

`DiaParser` is derived from [`SuPar`](https://github.com/yzhangcs/parser), which provides additional variants of dependency and constituency parsers.

## Contents

* [Contents](#contents)
* [Installation](#installation)
* [Performance](#performance)
* [Usage](#usage)
  * [Training](#training)
  * [Evaluation](#evaluation)
* [TODO](#todo)
* [References](#references)

## Installation

`DiaParser` can be installed via `pip`:
```sh
$ pip install -U diaparser
```
Installing from sources is also possible:
```sh
$ git clone https://github.com/Unipisa/diaparser && cd diaparser
$ python setup.py install
```

The package has the following requirements:
* `python`: >= 3.6
* [`pytorch`](https://github.com/pytorch/pytorch): >= 1.4
* [`transformers`](https://github.com/huggingface/transformers): >= 3.1
* optional tokenizers [`stanza`](https://stanfordnlp.github.io/stanza/): >= 1.1.1

## Performance

`DiaParser` provides pretrained models for English, Chinese and other 21 languages from the Universal Dependencies treebanks [v2.6](https://universaxsldependencies.org).
English models are trained on the Penn Treebank (PTB) with Stanford Dependencies, with 39,832 training sentences, while Chinese models are trained on Penn Chinese Treebank version 7 (CTB7) with 46,572 training sentences.

The accuracy and parsing speed of these models are listed in the following tables.
The first table shows the result of parsing starting from gold tokenized text.
Notably, punctuation is ignored in the evaluation metrics for PTB, but included in all the others.
The numbers in bold represent state-of-the-art values.

<table>
  <thead>
    <tr>
      <th>Language</th>
      <th align="center">Corpus</th>
      <th align="center">Name</th>
      <th align="center">UAS</th>
      <th align="center">LAS</th>
      <th align="right">Speed (Sents/s)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>English</td>
      <td>PTB</td>
      <td><code>en_ptb.electra</code></td>
      <td align="center"><b>96.03</b></td>
      <td align="center"><b>94.37</b></td>
      <td align="right">352</td>
    </tr>
    <tr>
      <td>Chinese</td>
      <td>CTB</td>
      <td><code>zh_ptb.hfl</code></td>
      <td align="center">92.14</td>
      <td align="center">85.74</td>
      <td align="right">319</td>
    </tr>
    <tr>
      <td>Catalan</td>
      <td>AnCora</td>
      <td><code>ca_ancora.mbert</code></td>
      <td align="center"><b>95.55</b></td>
      <td align="center"><b>93.78</b></td>
      <td align="right">249</td>
    </tr>
    <tr>
      <td>German</td>
      <td>HDT</td>
      <td><code>de_htd.dbmdz-bert-base</code></td>
      <td align="center"><b>97.97</b></td>
      <td align="center"><b>96.97</b></td>
      <td align="right">184</td>
    </tr>
    <tr>
      <td>Japanese</td>
      <td>GSD</td>
      <td><code>ja_gsd.mbert</code></td>
      <td align="center"><b>95.41</b></td>
      <td align="center"><b>93.98</b></td>
      <td align="right">397</td>
    </tr>
    <tr>
      <td>Latin</td>
      <td>ITTB, LLCT</td>
      <td><code>la_ittb_llct.mbert</code></td>
      <td align="center"><b>94.03</b></td>
      <td align="center"><b>91.70</b></td>
      <td align="right">139</td>
    </tr>
    <tr>
      <td>Norwegian</td>
      <td>Nynorsk</td>
      <td><code>no_nynorsk.mbert</code></td>
      <td align="center"><b>92.50</b></td>
      <td align="center"><b>90.13</b></td>
      <td align="right"></td>
    </tr>
    <tr>
      <td>Romanian</td>
      <td>RRT</td>
      <td><code>ro_rrt.mbert</code></td>
      <td align="center"><b>93.03</b></td>
      <td align="center"><b>87.18</b></td>
      <td align="right">286</td>
    </tr>
    <tr>
      <td>Spanish</td>
      <td>AnCora</td>
      <td><code>es_ancora.mbert</code></td>
      <td align="center"><b>96.03</b></td>
      <td align="center"><b>94.37</b></td>
      <td align="right">352</td>
    </tr>
  </tbody>
</table>

Below are the results on the dataset of the [IWPT 2020 Shared Task on Enhanced Dependencies](), where the tokenization was done by the parser itself:
<table>
  <thead>
    <tr>
      <th>Language</th>
      <th align="center">Corpus</th>
      <th align="center">Name</th>
      <th align="center">UAS</th>
      <th align="center">LAS</th>
      <th align="right">Speed (Sents/s)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Arabic</td>
      <td>PADT</td>
      <td><code>ar_padt.bert</code></td>
      <td align="center"><b>87.75</b></td>
      <td align="center"><b>83.25</b></td>
      <td align="right">99</td>
    </tr>
    <tr>
      <td>Bulgarian</td>
      <td>BTB</td>
      <td><code>bg_btb.DeepPavlov</code></td>
      <td align="center">95.02</td>
      <td align="center">92.20</td>
      <td align="right">479</td>
    </tr>
    <tr>
      <td>Czech</td>
      <td>PDT, CAC, FicTree</td>
      <td><code>cs_pdt_cac_fictree.DeepPavlov</code></td>
      <td align="center"><b>94.02</b></td>
      <td align="center"><b>92.06</b></td>
      <td align="right">403</td>
    </tr>
    <tr>
      <td>English</td>
      <td>EWT</td>
      <td><code>en_ewt.electra</code></td>
      <td align="center"><b>91.66</b></td>
      <td align="center"><b>89.51</b></td>
      <td align="right">397</td>
    </tr>
    <tr>
      <td>Estonian</td>
      <td>EDT, EWT</td>
      <td><code>et_edt_ewt.mbert</code></td>
      <td align="center">86.39</td>
      <td align="center">82.44</td>
      <td align="right">247</td>
    </tr>
    <tr>
      <td>Finnish</td>
      <td>TDT</td>
      <td><code>fi_tdt.turkunlp</code></td>
      <td align="center"><b>94.28</b></td>
      <td align="center"><b>92.56</b></td>
      <td align="right">364</td>
    </tr>
    <tr>
      <td>French</td>
      <td>sequoia</td>
      <td><code>fr_sequoia.camembert</code></td>
      <td align="center"><b>92.81</b></td>
      <td align="center"><b>89.55</b></td>
      <td align="right">200</td>
    </tr>
    <tr>
      <td>German</td>
      <td>HDT</td>
      <td><code>de_hdt.dbmdz-bert-base</code></td>
      <td align="center"><b>97.97</b></td>
      <td align="center"><b>96.97</b></td>
      <td align="right">381</td>
    </tr>
    <tr>
      <td>Italian</td>
      <td>ISDT</td>
      <td><code>it_isdt.dbmdz</code></td>
      <td align="center"><b>95.40</b></td>
      <td align="center"><b>93.78</b></td>
      <td align="right">379</td>
    </tr>
    <tr>
      <td>Latvian</td>
      <td>LVBT</td>
      <td><code>lv_lvtb.mbert</code></td>
      <td align="center">87.46</td>
      <td align="center">83.51</td>
      <td align="right">290</td>
    </tr>
    <tr>
      <td>Lithuanian</td>
      <td>ALKSNIS</td>
      <td><code>lt_alksnis.mbert</code></td>
      <td align="center">80.09</td>
      <td align="center">75.14</td>
      <td align="right">290</td>
    </tr>
    <tr>
      <td>Dutch</td>
      <td>Alpino, LassySmall</td>
      <td><code>nl_alpino_lassysmall.wietsedv</code></td>
      <td align="center">90.80</td>
      <td align="center">88.34</td>
      <td align="right">367</td>
    </tr>
    <tr>
      <td>Polish</td>
      <td>PDB, LFG</td>
      <td><code>pl_pdb_lfg.dkleczek</code></td>
      <td align="center"><b>94.38</b></td>
      <td align="center"><b>91.70</b></td>
      <td align="right">563</td>
    </tr>
    <tr>
      <td>Russian</td>
      <td>SynTagRus</td>
      <td><code>ru_syntagrus.DeepPavlov</code></td>
      <td align="center"><b>94.97</b></td>
      <td align="center"><b>93.72</b></td>
      <td align="right">445</td>
    </tr>
    <tr>
      <td>Slovak</td>
      <td>SNK</td>
      <td><code>sk_snk.mbert</code></td>
      <td align="center">93.11</td>
      <td align="center">90.44</td>
      <td align="right">381</td>
    </tr>
    <tr>
      <td>Swediskh</td>
      <td>Talbanken</td>
      <td><code>sv_talbanken.KB</code></td>
      <td align="center"><b>90.79</b></td>
      <td align="center"><b>88.08</b></td>
      <td align="right">491</td>
    </tr>
    <tr>
      <td>Tamil</td>
      <td>TTB</td>
      <td><code>ta_ttb.mbert</code></td>
      <td align="center">74.20</td>
      <td align="center">66.49</td>
      <td align="right">175</td>
    </tr>
    <tr>
      <td>Ukrainian</td>
      <td>IU</td>
      <td><code>uk_iu.TurkuNLP</code></td>
      <td align="center">90.39</td>
      <td align="center">87.61</td>
      <td align="right">301</td>
    </tr>
  </tbody>
</table>

These results were obtained on a server with Intel(R) Xeon(R) Gold 6132 CPU @ 2.60GHz
and Nvidia T4 GPU.

## Usage

`DiaParser` is simple to use: you can just download a pretrained model and run syntactic parsing over sentences with a few lines of code:
```py
>>> from diaparser.parsers import Parser
>>> parser = Parser.load('en_ewt-electra')
>>> dataset = parser.predict([['She', 'enjoys', 'playing', 'tennis', '.']], prob=True)
```
The call to `parser.predict` will return an instance of `diaparser.utils.Dataset` containing the predicted syntactic trees.
You can access each sentence within the `dataset`:
```py
>>> print(dataset.sentences[0])
1       She     _       _       _       _       2       nsubj   _       _
2       enjoys  _       _       _       _       0       root    _       _
3       playing _       _       _       _       2       xcomp   _       _
4       tennis  _       _       _       _       3       dobj    _       _
5       .       _       _       _       _       2       punct   _       _
```

To parse plain text just requires specifying the language code:
```py
>>> dataset = parser.predict('She enjoys playing tennis.', text='en')
```

You may also provide the input in a file in CoNLL-U format.

Further examples of how to use the parser and experiment with it can be found in this [notebook](demo/DiaParser.ipynb).


### Training

To train a model from scratch, it is preferred to use the command-line option, which is more flexible and customizable.
Here are some training examples:
```sh
# Biaffine Dependency Parser
# some common and default arguments are stored in config.ini
$ python -m diaparser.cmds.biaffine_dependency train -b -d 0  \
    -c config.ini  \
    -p exp/en_ptb.char/model  \
    -f char
# to use BERT, `-f` and `--bert` (default to bert-base-cased) should be specified
$ python -m diaparser.cmds.biaffine_dependency train -b -d 0  \
    -p exp/en_ptb.bert-base/model  \
    -f bert  \
    --bert bert-base-cased
```

**Warning**. There is currently a limit of 500 to the length of tokenized sentences, due to the maximum size of embeddings in most pretrained trnsformer models.

For further instructions on training, please type `python -m diaparser.cmds.<parser> train -h`.

Alternatively, `DiaParser` provides an equivalent command entry points registered in `setup.py`:
`diaparser`.
```sh
$ diaparser train -b -d 0 -c config.ini -p exp/en_ptb.electra-base/model -f bert --bert google/electra-base-discriminator
```

For handling large models, distributed training is also supported:
```sh
$ python -m torch.distributed.launch --nproc_per_node=4 --master_port=10000  \
    -m parser.cmds.biaffine_dependency train -b -d 0,1,2,3  \
    -p exp/en_ptb.electra-base/model  \
    -f bert --bert google/electra-base-discriminator
```
You may consult the PyTorch [documentation](https://pytorch.org/docs/stable/notes/ddp.html) and [tutorials](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for more details.

### Evaluation

The evaluation process resembles prediction:
```py
>>> parser = Parser.load('biaffine-dep-en')
>>> loss, metric = parser.evaluate('data/ptb/test.conllx')
2020-07-25 20:59:17 INFO Loading the data
2020-07-25 20:59:19 INFO
Dataset(n_sentences=2416, n_batches=11, n_buckets=8)
2020-07-25 20:59:19 INFO Evaluating the dataset
2020-07-25 20:59:20 INFO loss: 0.2326 - UCM: 61.34% LCM: 50.21% UAS: 96.03% LAS: 94.37%
2020-07-25 20:59:20 INFO 0:00:01.253601s elapsed, 1927.25 Sents/s
```

## TODO

- [ ] Provide a repository where to upload models, like HuggingFace.


## References

* Giuseppe Attardi, Daniele Sartiano, Yu Zhang. 2021. DiaParser attentive dependency parser. Submitted for publication.
* <a id="dozat-2017-biaffine"></a>
Timothy Dozat and Christopher D. Manning. 2017. [Deep Biaffine Attention for Neural Dependency Parsing](https://openreview.net/forum?id=Hk95PK9le).
* <a id="wang-2019-second"></a>
Xinyu Wang, Jingxian Huang, and Kewei Tu. 2019. [Second-Order Semantic Dependency Parsing with End-to-End Neural Networks](https://www.aclweb.org/anthology/P19-1454/).
