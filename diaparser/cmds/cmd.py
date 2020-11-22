# -*- coding: utf-8 -*-

import torch
from ..utils import Config
from ..utils.logging import init_logger, logger
from ..utils.parallel import init_device
from ..parsers.biaffine_dependency import BiaffineDependencyParser as Parser


def parse(argparser):
    argparser.add_argument('--conf', '-c', help='path to config file')
    argparser.add_argument('--path', '-p', help='model name or path to model file')
    argparser.add_argument('--device', '-d', default='-1', help='ID of GPU to use')
    argparser.add_argument('--seed', '-s', default=1, type=int, help='seed for generating random numbers')
    argparser.add_argument('--threads', '-t', default=16, type=int, help='max num of threads')
    argparser.add_argument('--batch-size', default=5000, type=int, help='batch size')
    argparser.add_argument("--local_rank", type=int, default=-1, help='node rank for distributed training')
    argparser.add_argument('--quiet', '-q', dest='verbose', action='store_false',
                           help='suppress verbose logs')
    args, unknown = argparser.parse_known_args()
    args, _ = argparser.parse_known_args(unknown, args)
    args = Config(**vars(args))

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    init_device(args.device, args.local_rank)
    init_logger(logger, f"{args.path}.{args.mode}.log", verbose=args.verbose)
    logger.info('Configuration:\n' + str(args))

    if args.mode == 'train':
        parser = Parser.build(**args)
        parser.train(**args)
    elif args.mode == 'evaluate':
        parser = Parser.load(args.path)
        parser.evaluate(**args)
    elif args.mode == 'predict':
        parser = Parser.load(args.path, **args)
        parser.predict(**args)
