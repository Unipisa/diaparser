# -*- coding: utf-8 -*-

import argparse

# import pdb;pdb.set_trace()      # DEBUG

from ..parsers import BiaffineDependencyParser
from .cmd import parse
import os


def main():
    parser = argparse.ArgumentParser(description='Create Biaffine Dependency Parser.')
    parser.add_argument('--tree', action='store_true', help='whether to ensure well-formedness')
    parser.add_argument('--proj', action='store_true', help='whether to projectivise the data')
    parser.add_argument('--partial', action='store_true', help='whether partial annotation is included')
    parser.set_defaults(Parser=BiaffineDependencyParser)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--feat', '-f', choices=['tag', 'char', 'bert'], help='choices of additional features')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--max-len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--train', required=True, help='path to train file')
    subparser.add_argument('--dev', required=True, help='path to dev file')
    subparser.add_argument('--test', help='path to test file')
    subparser.add_argument('--embed', help='path to pretrained embeddings')
    subparser.add_argument('--unk', default='unk', help='unk token in pretrained embeddings')
    # SUPPRESS or else it cannot be set through config.ini
    subparser.add_argument('--n-word-embed', default=argparse.SUPPRESS, type=int, help='dimension of embeddings')
    subparser.add_argument('--bert', default=argparse.SUPPRESS, help='which bert model to use')
    subparser.add_argument('--attention-head', default=argparse.SUPPRESS, type=int,
                           help='attention head')
    subparser.add_argument('--attention-layer', default=argparse.SUPPRESS, type=int,
                           help='attention layer')
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/ptb/test.conllx', help='path to dataset')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/ptb/test.conllx', help='path to dataset')
    subparser.add_argument('--pred', default='pred.conllx', help='path to predicted result')
    subparser.add_argument('--text', metavar='LANGUAGE', default=None,
                           help='parse plain text in the given language.')
    subparser.add_argument('--cache-dir', default=os.path.expanduser('~/.cache/diaparser'),
                           help='path to saved parser/tokenizer models')
    parse(parser)


if __name__ == "__main__":
    main()
