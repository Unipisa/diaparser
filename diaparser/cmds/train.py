# -*- coding: utf-8 -*-

import argparse
from datetime import datetime, timedelta
import math
from diaparser.utils.config import Config
from diaparser import Model
from diaparser.cmds.cmd import CMD
from diaparser.utils.corpus import Corpus
from diaparser.utils.data import TextDataset, batchify
from diaparser.utils.metric import Metric
from diaparser.utils.logging import logger

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from transformers.optimization import AdamW, get_linear_schedule_with_warmup



class TransparentDataParallel(nn.DataParallel):
    """custom class so that I can have other attributes than modules."""

    def __init__(self, module, **kwargs):
        super().__init__(module, **kwargs)

    def __getattr__(self, name):
        if name is not 'module':
            try:
                return getattr(self.module, name)
            except AttributeError:
                pass

        return super(nn.DataParallel, self).__getattr__(name)


class Train(CMD):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        # allow --conf only in train to avoid overriding model parameters in predict
        subparser.add_argument('--conf', '-c', default='config.ini',
                               help='path to config file (default config.ini)')
        # use SUPPRESS, or else it will have a default that will override the value in config file
        subparser.add_argument('--bert-model', '-m', default=argparse.SUPPRESS,
                               help='pretrained BERT model')
        subparser.add_argument('--punct', action='store_true',
                               help='whether to include punctuation')
        subparser.add_argument('--ftrain', required=True,
                               help='path to train file')
        subparser.add_argument('--fdev', required=True,
                               help='path to dev file')
        subparser.add_argument('--ftest', default='',
                               help='path to test file')
        subparser.add_argument('--fembed', default='',
                               help='path to pretrained embeddings')
        subparser.add_argument('--lower', action='store_true',
                               help='whether to turn words to lowercase')
        subparser.add_argument('--unk', default='[UNK]',
                               help='unk token in pretrained embeddings')
        subparser.add_argument('--max-sent-length', default=512, type=int,
                               help='max tokenized sentence length (longer ones are discarded)')
        subparser.add_argument('--feat', default='bert',
                               choices=['tag', 'char', 'bert'],
                               help='choices of additional features')
        subparser.add_argument('--batch-size', default=5000, type=int,
                               help='batch size')
        subparser.add_argument('--buckets', default=32, type=int,
                               help='max num of buckets to use')
        subparser.add_argument('--attention-layer', default=8, type=int,
                               help='attention head')

        return subparser

    def __call__(self, args):
        # override config from CLI parameters
        args = Config(args.conf).update(vars(args))

        # loads train corpus into self.trainset
        super().__call__(args)

        logger.info(f"Configuration parameters:\n{args}")

        #train = Corpus.load(args.ftrain, self.fields, args.max_sent_length)
        train = self.trainset
        dev = Corpus.load(args.fdev, self.fields, args.max_sent_length)
        if args.ftest:
            test = Corpus.load(args.ftest, self.fields, args.max_sent_length)

        train = TextDataset(train, self.fields, args.buckets)
        dev = TextDataset(dev, self.fields, args.buckets)
        if args.ftest:
            test = TextDataset(test, self.fields, args.buckets)
        # set the data loaders
        train.loader = batchify(train, args.batch_size, True)
        dev.loader = batchify(dev, args.batch_size)
        if args.ftest:
            test.loader = batchify(test, args.batch_size)
        logger.info(f"{'train:':6} {len(train):5} sentences, "
              f"{len(train.loader):3} batches, "
              f"{len(train.buckets)} buckets")
        logger.info(f"{'dev:':6} {len(dev):5} sentences, "
              f"{len(dev.loader):3} batches, "
              f"{len(train.buckets)} buckets")
        if args.ftest:
            logger.info(f"{'test:':6} {len(test):5} sentences, "
                  f"{len(test.loader):3} batches, "
                  f"{len(train.buckets)} buckets")

        logger.info("Create the model")
        self.model = Model(args, mask_token_id=self.FEAT.mask_token_id)
        if self.WORD:
            self.model.load_pretrained(self.WORD.embed)
        self.model = self.model.to(args.device)
        if torch.cuda.device_count() > 1:
            self.model = TransparentDataParallel(self.model)
        logger.info(f"{self.model}\n")
        if args.optimizer == 'adamw':
            self.optimizer = AdamW(self.model.parameters(),
                                   args.lr,
                                   (args.mu, args.nu),
                                   args.epsilon,
                                   args.decay)
            training_steps = len(train.loader) // self.args.accumulation_steps \
                             * self.args.epochs
            warmup_steps = math.ceil(training_steps * self.args.warmup_steps_ratio)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps,
                num_training_steps=training_steps)
        else:
            self.optimizer = Adam(self.model.parameters(),
                                  args.lr,
                                  (args.mu, args.nu),
                                  args.epsilon)
            self.scheduler = ExponentialLR(self.optimizer,
                                           args.decay**(1/args.decay_steps))

        total_time = timedelta()
        best_e, best_metric = 1, Metric()

        for epoch in range(1, args.epochs + 1):
            start = datetime.now()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            loss, train_metric = self.train(train.loader)
            logger.info(f"{'train:':6} Loss: {loss:.4f} {train_metric}")
            loss, dev_metric = self.evaluate(dev.loader)
            logger.info(f"{'dev:':6} Loss: {loss:.4f} {dev_metric}")
            if args.ftest:
                loss, test_metric = self.evaluate(test.loader)
                logger.info(f"{'test:':6} Loss: {loss:.4f} {test_metric}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric and epoch > args.patience//10:
                best_e, best_metric = epoch, dev_metric
                if hasattr(self.model, 'module'):
                    self.model.module.save(args.model)
                else:
                    self.model.save(args.model)
                logger.info(f"{t}s elapsed (saved)\n")
            else:
                logger.info(f"{t}s elapsed\n")
            total_time += t
            if epoch - best_e >= args.patience:
                break
        self.model = Model.load(args.model)
        if args.ftest:
            loss, metric = self.evaluate(test.loader)

        logger.info(f"max score of dev is {best_metric.score:.2%} at epoch {best_e}")
        if args.ftest:
            logger.info(f"the score of test at epoch {best_e} is {metric.score:.2%}")
        logger.info(f"average time of each epoch is {total_time / epoch}s")
        logger.info(f"{total_time}s elapsed")
