# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
from diaparser import Model
from diaparser.cmds.cmd import CMD
from diaparser.utils.corpus import Corpus, TextCorpus
from diaparser.utils.data import TextDataset, batchify
from diaparser.utils.logging import logger

import torch


class Predict(CMD):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Use a trained model to make predictions.'
        )
        subparser.add_argument('--prob', action='store_true',
                               help='whether to output probs')
        subparser.add_argument('fdata',
                               help='path to input file')
        subparser.add_argument('--fpred',
                               help='path where to print predicted result, stdout if missing')
        subparser.add_argument('--text', metavar='LANGUAGE', default=None,
                               help='parse plain text in the given language rather than CoNLLU files.')
        subparser.add_argument('--tokenizer-dir', default='~/.tokenizer-models',
                               help='path to saved tokenizer models')

        return subparser

    def __call__(self, args):
        logger.info("Load the model")
        self.model = Model.load(args.model)
        # override from CLI args
        args = self.model.args.update(vars(args))

        super().__call__(args)

        logger.info("Load the dataset")
        if args.prob:
            self.fields = self.fields._replace(PHEAD=Field('probs'))
        if args.text:
            corpus = TextCorpus.load(args.fdata, self.fields, args.text, args.tokenizer_dir, use_gpu=args.device != 1)
        else:
            corpus = Corpus.load(args.fdata, self.fields)
        dataset = TextDataset(corpus, [self.WORD, self.FEAT], args.buckets)
        # set the data loader
        dataset.loader = batchify(dataset, args.batch_size)
        logger.info(f"{len(dataset)} sentences, "
              f"{len(dataset.loader)} batches")

        logger.info("Make predictions on the dataset")
        start = datetime.now()
        pred_arcs, pred_rels, pred_probs = self.predict(dataset.loader)
        total_time = datetime.now() - start
        # restore the order of sentences in the buckets
        indices = torch.tensor([i
                                for bucket in dataset.buckets.values()
                                for i in bucket]).argsort()
        corpus.arcs = [pred_arcs[i] for i in indices]
        corpus.rels = [pred_rels[i] for i in indices]
        if args.prob:
            corpus.probs = [pred_probs[i] for i in indices]
        logger.info(f"Save the predicted result to {args.fpred}")
        corpus.save(args.fpred)
        logger.info(f"{total_time}s elapsed, "
              f"{len(dataset) / total_time.total_seconds():.2f} Sents/s")
