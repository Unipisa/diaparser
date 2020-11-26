# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta

import torch
import torch.distributed as dist
from .. import parsers
from tokenizer.tokenizer import Tokenizer
from ..catalog import select
from ..utils import Config, Dataset
from ..utils.field import Field, BertField
from ..utils.logging import init_logger, logger
from ..utils.metric import Metric
from ..utils.parallel import DistributedDataParallel as DDP
from ..utils.parallel import is_master
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


class Parser():

    MODEL = None

    def __init__(self, args, model, transform):
        self.args = args
        self.model = model
        self.transform = transform

    def train(self, train, dev, test,
              buckets=32,
              batch_size=5000,
              lr=2e-3,
              mu=.9,
              nu=.9,
              epsilon=1e-12,
              clip=5.0,
              decay=.75,
              decay_steps=5000,
              epochs=5000,
              patience=100,
              verbose=True,
              **kwargs):
        r"""
        Args:
            lr (float): learnin rate of adam optimizer. Default: 2e-3.
            mu (float): beta1 of adam optimizer. Default: .9.
            nu (float): beta2 of adam optimizer. Default: .9.
            epsilon (float): epsilon of adam optimizer. Default: 1e-12.
            buckets (int): number of buckets. Default: 32.
            epochs (int): number of epochs to train: Default: 5000.
            patience (int): early stop after these many epochs. Default: 100.
        """

        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        if dist.is_initialized():
            args.batch_size = args.batch_size // dist.get_world_size()
        logger.info(f"Load the datasets\n"
                    f"{'train:':6} {train}\n"
                    f"{'dev:':6} {dev}\n")
        train = Dataset(self.transform, args.train, **args)
        train.build(args.batch_size, args.buckets, True, dist.is_initialized())
        logger.info(f"{'train:':6} {len(train):5} sentences, "
                    f"{len(train.loader):3} batches, "
                    f"{len(train.buckets)} buckets")
        dev = Dataset(self.transform, args.dev)
        dev.build(args.batch_size, args.buckets)
        logger.info(f"{'dev:':6} {len(dev):5} sentences, "
                    f"{len(dev.loader):3} batches, "
                    f"{len(train.buckets)} buckets")
        if args.test:
            test = Dataset(self.transform, args.test)
            test.build(args.batch_size, args.buckets)
            logger.info(f"{'test:':6} {len(test):5} sentences, "
                        f"{len(test.loader):3} batches, "
                        f"{len(train.buckets)} buckets")
        else:
            test = None

        logger.info(f"Model\n{self.model}\n")
        if dist.is_initialized():
            self.model = DDP(self.model,
                             device_ids=[dist.get_rank()],
                             find_unused_parameters=True)
        self.optimizer = Adam(self.model.parameters(),
                              args.lr,
                              (args.mu, args.nu),
                              args.epsilon)
        self.scheduler = ExponentialLR(self.optimizer, args.decay**(1/args.decay_steps))

        elapsed = timedelta()
        best_e, best_metric = 1, Metric()

        for epoch in range(1, args.epochs + 1):
            start = datetime.now()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            self._train(train.loader)
            loss, dev_metric = self._evaluate(dev.loader)
            logger.info(f"{'dev:':6} - loss: {loss:.4f} - {dev_metric}")
            if test:
                loss, test_metric = self._evaluate(test.loader)
                logger.info(f"{'test:':6} - loss: {loss:.4f} - {test_metric}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                if is_master():
                    self.save(args.path)
                logger.info(f"{t}s elapsed (saved)\n")
            else:
                logger.info(f"{t}s elapsed\n")
            elapsed += t
            if epoch - best_e >= args.patience:
                break

        logger.info(f"Epoch {best_e} saved")
        logger.info(f"{'dev:':6} - {best_metric}")
        if test:
            loss, metric = self.load(args.path)._evaluate(test.loader)
            logger.info(f"{'test:':6} - {metric}")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

    def evaluate(self, data, buckets=8, batch_size=5000, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        logger.info("Loading the data")
        dataset = Dataset(self.transform, data)
        dataset.build(args.batch_size, args.buckets)
        logger.info(f"\n{dataset}")

        logger.info("Evaluating the dataset")
        start = datetime.now()
        loss, metric = self._evaluate(dataset.loader)
        elapsed = datetime.now() - start
        logger.info(f"loss: {loss:.4f} - {metric}")
        logger.info(f"{elapsed}s elapsed, {len(dataset)/elapsed.total_seconds():.2f} Sents/s")

        return loss, metric

    def predict(self, data, pred=None, buckets=8, batch_size=5000, prob=False, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.eval()
        if args.prob:
            self.transform.append(Field('probs'))
        if hasattr(args, 'text') and args.text \
           or hasattr(args, 'lang') and args.lang: # PATCH: back compatibility
            self.transform.reader = Tokenizer(args.text, dir=args.cache_dir).reader()

        logger.info("Loading the data")
        dataset = Dataset(self.transform, data)
        dataset.build(args.batch_size, args.buckets)
        logger.info(f"\n{dataset}")

        logger.info("Making predictions on the dataset")
        start = datetime.now()
        preds = self._predict(dataset.loader)
        elapsed = datetime.now() - start

        for name, value in preds.items():
            setattr(dataset, name, value)
        if pred is not None and is_master():
            logger.info(f"Saving predicted results to {pred}")
            self.transform.save(pred, dataset.sentences)
        logger.info(f"{elapsed}s elapsed, {len(dataset) / elapsed.total_seconds():.2f} Sents/s")

        return dataset

    def _train(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _evaluate(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _predict(self, loader):
        raise NotImplementedError

    @classmethod
    def build(cls, path, **kwargs):
        raise NotImplementedError

    @classmethod
    def load(cls, name_or_path, cache_dir=os.path.expanduser('~/.cache/diaparser'), **kwargs):
        r"""
        Loads a parser from a pretrained model.

        Args:
            name_or_path (str):
                - a string with the shortcut name of a pretrained parser listed in ``resource.json``
                  to load from cache or download, e.g., ``'en_ptb.electra-base'``.
                - a path to a directory containing a pre-trained parser, e.g., `./<path>/model`.
            cache_dir (str):
                Directory where to cache models. The default value is `~/.cache/diaparser`.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations and initiate the model.

        Examples:
            >>> parser = Parser.load('en_ewt.electra-base')
            >>> parser = Parser.load('./ptb.biaffine.dependency.char')
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if os.path.exists(name_or_path):
            state = torch.load(name_or_path)
        else:
            url = select(name=name_or_path, **kwargs)
            if url is None:
                raise Exception(f'Could not find a model matching name {name_or_path}')
            verbose = kwargs.get('verbose', True)
            state = torch.hub.load_state_dict_from_url(url, model_dir=cache_dir,
                                                       progress=verbose)
        cls = getattr(parsers, state['name'])
        args = state['args'].update(args)
        model = cls.MODEL(**args)
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(args.device)
        transform = state['transform']
        if args.feat == 'bert':
            tokenizer = BertField.tokenizer(args.bert)
            transform.FORM[1].tokenize = tokenizer.tokenize
        return cls(args, model, transform)

    def save(self, path):
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        args = model.args
        args.pop('Parser')      # dont save parser class object
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        pretrained = state_dict.pop('pretrained.weight', None)
        if args.feat == 'bert':
            tokenize = self.transform.FORM[1].tokenize  # save it
            self.transform.FORM[1].tokenize = None
        state = {'name': type(self).__name__,
                 'args': args,
                 'state_dict': state_dict,
                 'pretrained': pretrained,
                 'transform': self.transform}
        torch.save(state, path)
        if args.feat == 'bert':
            self.transform.FORM[1].tokenize = tokenize  # restore
