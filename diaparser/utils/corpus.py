# -*- coding: utf-8 -*-

from collections import namedtuple
from collections.abc import Iterable
from .field import Field

from tokenizer.tokenizer import Tokenizer

import sys
import math
from ..utils.logging import logger

if 'dedfaults' in namedtuple.__code__.co_varnames[:namedtuple.__code__.co_argcount]:
    CoNLL = namedtuple(typename='CoNLL',
                       field_names=['ID', 'FORM', 'LEMMA', 'CPOS', 'POS',
                                    'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL'],
                       defaults=[None]*10)
else:
    CoNLL = namedtuple(typename='CoNLL',
                       field_names=['ID', 'FORM', 'LEMMA', 'CPOS', 'POS',
                                    'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL'])
    CoNLL.__new__.__defaults__ = (None,) * 10


class Sentence():

    def __init__(self, fields, lines):
        self.annotations = dict()
        values = []
        for i, line in enumerate(lines):
            if line.startswith('#'):
                self.annotations[-i-1] = line
            else:
                value = line.split('\t')
                if value[0].isdigit():
                    values.append(value)
                    self.annotations[int(value[0])] = ''  # placeholder
                else:
                    self.annotations[-i] = line
        for field, value in zip(fields, list(zip(*values))):
            if isinstance(field, Iterable):
                for j in range(len(field)):
                    setattr(self, field[j].name, value)
            else:
                setattr(self, field.name, value)
        self.fields = fields

    @property
    def values(self):
        for field in self.fields:
            if isinstance(field, Iterable):
                yield getattr(self, field[0].name)
            else:
                yield getattr(self, field.name)

    def __len__(self):
        return len(next(iter(self.values)))

    def __repr__(self):
        merged = {**self.annotations,
                  **{i+1: '\t'.join(map(str, line))
                     for i, line in enumerate(zip(*self.values))}}
        return '\n'.join(merged.values()) + '\n'


class Corpus(object):

    def __init__(self, fields, sentences):
        super(Corpus, self).__init__()

        self.fields = fields
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __repr__(self):
        return '\n'.join(str(sentence) for sentence in self)

    def __getitem__(self, index):
        return self.sentences[index]

    def __getattr__(self, name):
        if not hasattr(self.sentences[0], name):
            raise AttributeError
        for sentence in self.sentences:
            yield getattr(sentence, name)

    def __setattr__(self, name, value):
        if name in ['fields', 'sentences']:
            self.__dict__[name] = value
        else:
            for i, sentence in enumerate(self.sentences):
                setattr(sentence, name, value[i])

    @classmethod
    def load(cls, path, fields, max_sent_length=math.inf):
        sentences = []
        fields = [field if field is not None else Field(str(i))
                  for i, field in enumerate(fields)]
        with open(path, 'r') as f:
            lines = []
            for line in f:
                line = line.strip()
                if not line:
                    if len(lines) > max_sent_length:
                        logger.info('Discarded sentence longer than max_sent_length:',
                                    len(lines), file=sys.stderr)
                        lines = []
                        continue
                    sentences.append(Sentence(fields, lines))
                    lines = []
                else:
                    lines.append(line)

        return cls(fields, sentences)

    def save(self, path):
        if path:
            with open(path, 'w') as f:
                f.write(f"{self}\n")
        else:
            print(self)


class TextCorpus(Corpus):
    """
    Class for tokenizing and MW splitting plain text.
    """
    @classmethod
    def load(cls, path, fields, tokenizer_lang, tokenizer_dir, verbose=True, max_sent_length=math.inf):
        tokenizer = Tokenizer(lang=tokenizer_lang, dir=tokenizer_dir, verbose=verbose)

        sentences = []
        fields = [field if field is not None else Field(str(i))
                  for i, field in enumerate(fields)]

        with open(path, 'r') as f:
            lines = []
            for line in tokenizer.format(tokenizer.predict(f.read())):
                line = line.strip()
                if not line:
                    if len(lines) > max_sent_length:
                        logger.info('Discarded sentence longer than max_sent_length:',
                                    len(lines), file=sys.stderr)
                        lines = []
                        continue
                    sentences.append(Sentence(fields, lines))
                    lines = []
                else:
                    if not line.startswith('#'):
                        # append empty columns
                        line += '\t_' * (len(CoNLL._fields) - len(line.split('\t')))
                    lines.append(line)

        return cls(fields, sentences)
