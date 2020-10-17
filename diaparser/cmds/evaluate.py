# -*- coding: utf-8 -*-

from datetime import datetime
from diaparser import Model
from diaparser.cmds.cmd import CMD
from diaparser.utils.corpus import Corpus
from diaparser.utils.data import TextDataset, batchify
from diaparser.utils.logging import logger


class Evaluate(CMD):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Evaluate the specified model and dataset.'
        )
        subparser.add_argument('--punct', action='store_true',
                               help='whether to include punctuation')
        subparser.add_argument('fdata',
                               help='path to dataset')

        return subparser

    def __call__(self, args):
        super(Evaluate, self).__call__(args)

        logger.info("Load the dataset")
        corpus = Corpus.load(args.fdata, self.fields)
        dataset = TextDataset(corpus, self.fields, args.buckets)
        # set the data loader
        dataset.loader = batchify(dataset, args.batch_size)
        logger.info(f"{len(dataset)} sentences, "
              f"{len(dataset.loader)} batches, "
              f"{len(dataset.buckets)} buckets")

        logger.info("Load the model")
        self.model = Model.load(args.model)
        logger.info(f"{self.model}\n")

        logger.info("Evaluate the dataset")
        start = datetime.now()
        loss, metric = self.evaluate(dataset.loader)
        total_time = datetime.now() - start
        logger.info(f"Loss: {loss:.4f} {metric}")
        logger.info(f"{total_time}s elapsed, "
              f"{len(dataset) / total_time.total_seconds():.2f} Sents/s")
