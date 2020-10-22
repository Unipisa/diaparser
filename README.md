# DiaParser

[![build](https://github.com/Unipisa/diaparser/workflows/build/badge.svg)](https://github.com/Unipisa/diaparser/actions)
[![docs](https://readthedocs.org/projects/diaparser/badge/?version=latest)](https://diaparser.readthedocs.io/en/latest)
[![release](https://img.shields.io/github/v/release/Unipisa/diaparser)](https://Unipisa/diaparser/releases)
[![downloads](https://pepy.tech/badge/diaparser)](https://pepy.tech/project/diaparser)
[![LICENSE](https://img.shields.io/github/license/Unipisa/diaparser)](https://github.com/Unipisa/diaparser/blob/master/LICENSE)

`DiaParser` provides a state-of-the-art direct attentive dependency parser based onthe Biaffine Parser ([Dozat and Manning, 2017](#dozat-2017-biaffine)) architecture.

The parser can work directly on plain text or on tokenized text.
The parser automatically dowloads pretrained models as well as tokenizers and produces dependency parsing trees, as detailed in [Usage](#Usage).

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

`DiaParser` provides pretrained models for English, Chinese and other 17 languages of the IWPT 2020 Shared task.
English models are trained on the Penn Treebank (PTB) with Stanford Dependencies, with 39,832 training sentences, while Chinese models are trained on Penn Chinese Treebank version 7 (CTB7) with 46,572 training sentences.
The other languages are trained on the Universal Dependencies treebanks [v2.5](https://universaxsldependencies.org).

The performance and parsing speed of these models are listed in the following table.
Notably, punctuation is ignored in all evaluation metrics for PTB, but included in all the others.
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
      <td>PDT</td>
      <td><code>cs_pdt.DeepPavlov</code></td>
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
      <td><code>et_edt.mbert</code></td>
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
      <td>Italian</td>
      <td>ISDT</td>
      <td><code>it_isdt.dbmdz</code></td>
      <td align="center"><b>95.40</b></td>
      <td align="center"><b>93.78</b></td>
      <td align="right"></td>
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
      <td>Alpino</td>
      <td><code>nl_alpino.wietsedv</code></td>
      <td align="center">90.80</td>
      <td align="center">88.34</td>
      <td align="right">367</td>
    </tr>
    <tr>
      <td>Polish</td>
      <td>PDB, LFG</td>
      <td><code>pl_pdb.dkleczek</code></td>
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
    <tr>
      <td>Chinese</td>
      <td>CTB</td>
      <td><code>zh_ptb.hfl</code></td>
      <td align="center">92.14</td>
      <td align="center">85.74</td>
      <td align="right">319</td>
    </tr>
  </tbody>
</table>

These results were obtained on a server with Intel(R) Xeon(R) Gold 6132 CPU @ 2.60GHz
and Nvidia T4 GPU.

## Usage

`DiaParser` is very easy to use. You can download a pretrained model and run syntactic parsing over sentences with a few lines of code:
```py
>>> from diaparser.parsers import Parser
>>> parser = Parser.load('en_ewt-electra')
>>> dataset = parser.predict([['She', 'enjoys', 'playing', 'tennis', '.']], prob=True, verbose=False)
100%|####################################| 1/1 00:00<00:00, 85.15it/s
```
The call to `parser.predict` will return an instance of `diaparser.utils.Dataset` containing the predicted syntactic trees.
You can either access any sentence within the `dataset` or an individual field of all the tokens.
```py
>>> print(dataset.sentences[0])
1       She     _       _       _       _       2       nsubj   _       _
2       enjoys  _       _       _       _       0       root    _       _
3       playing _       _       _       _       2       xcomp   _       _
4       tennis  _       _       _       _       3       dobj    _       _
5       .       _       _       _       _       2       punct   _       _

>>> print(f"arcs:  {dataset.arcs[0]}\n"
          f"rels:  {dataset.rels[0]}\n"
          f"probs: {dataset.probs[0].gather(1,torch.tensor(dataset.arcs[0]).unsqueeze(1)).squeeze(-1)}")
arcs:  [2, 0, 2, 3, 2]
rels:  ['nsubj', 'root', 'xcomp', 'dobj', 'punct']
probs: tensor([1.0000, 0.9999, 0.9642, 0.9686, 0.9996])
```
Probabilities can be returned along with the results if `prob=True`.

If there are plenty of sentences to parse, `DiaParser` also supports loading them from file, and saving the results to a file specified with option `pred`.
```py
>>> dataset = parser.predict('data/ptb/test.conllx', pred='pred.conllx')
2020-07-25 18:13:50 INFO Loading the data
2020-07-25 18:13:52 INFO
Dataset(n_sentences=2416, n_batches=13, n_buckets=8)
2020-07-25 18:13:52 INFO Making predictions on the dataset
100%|####################################| 13/13 00:01<00:00, 10.58it/s
2020-07-25 18:13:53 INFO Saving predicted results to pred.conllx
2020-07-25 18:13:54 INFO 0:00:01.335261s elapsed, 1809.38 Sents/s
```

Please make sure the file is in CoNLL-X or CoNLL-U format. If some fields are missing, you can use underscores as placeholders.
An interface is provided for converting a list of tokens to a string in CoNLL-X format.
```py
>>> from diaparser.utils import CoNLL
>>> print(CoNLL.toconll(['She', 'enjoys', 'playing', 'tennis', '.']))
1       She     _       _       _       _       _       _       _       _
2       enjoys  _       _       _       _       _       _       _       _
3       playing _       _       _       _       _       _       _       _
4       tennis  _       _       _       _       _       _       _       _
5       .       _       _       _       _       _       _       _       _

```

The CoNLL-U format for Universal Dependencies (UD) is also supported, with comments and extra annotations preserved and restored in the output.
```py
>>> import os
>>> import tempfile
>>> text = '''# text = But I found the location wonderful and the neighbors very kind.
1\tBut\t_\t_\t_\t_\t_\t_\t_\t_
2\tI\t_\t_\t_\t_\t_\t_\t_\t_
3\tfound\t_\t_\t_\t_\t_\t_\t_\t_
4\tthe\t_\t_\t_\t_\t_\t_\t_\t_
5\tlocation\t_\t_\t_\t_\t_\t_\t_\t_
6\twonderful\t_\t_\t_\t_\t_\t_\t_\t_
7\tand\t_\t_\t_\t_\t_\t_\t_\t_
7.1\tfound\t_\t_\t_\t_\t_\t_\t_\t_
8\tthe\t_\t_\t_\t_\t_\t_\t_\t_
9\tneighbors\t_\t_\t_\t_\t_\t_\t_\t_
10\tvery\t_\t_\t_\t_\t_\t_\t_\t_
11\tkind\t_\t_\t_\t_\t_\t_\t_\t_
12\t.\t_\t_\t_\t_\t_\t_\t_\t_

'''
>>> path = os.path.join(tempfile.mkdtemp(), 'data.conllx')
>>> with open(path, 'w') as f:
...     f.write(text)
...
>>> print(parser.predict(path, verbose=False).sentences[0])
100%|####################################| 1/1 00:00<00:00, 68.60it/s
# text = But I found the location wonderful and the neighbors very kind.
1       But     _       _       _       _       3       cc      _       _
2       I       _       _       _       _       3       nsubj   _       _
3       found   _       _       _       _       0       root    _       _
4       the     _       _       _       _       5       det     _       _
5       location        _       _       _       _       6       nsubj   _       _
6       wonderful       _       _       _       _       3       xcomp   _       _
7       and     _       _       _       _       6       cc      _       _
7.1     found   _       _       _       _       _       _       _       _
8       the     _       _       _       _       9       det     _       _
9       neighbors       _       _       _       _       11      dep     _       _
10      very    _       _       _       _       11      advmod  _       _
11      kind    _       _       _       _       6       conj    _       _
12      .       _       _       _       _       3       punct   _       _

```

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

* <a id="dozat-2017-biaffine"></a>
Timothy Dozat and Christopher D. Manning. 2017. [Deep Biaffine Attention for Neural Dependency Parsing](https://openreview.net/forum?id=Hk95PK9le).
* <a id="wang-2019-second"></a>
Xinyu Wang, Jingxian Huang, and Kewei Tu. 2019. [Second-Order Semantic Dependency Parsing with End-to-End Neural Networks](https://www.aclweb.org/anthology/P19-1454/).
