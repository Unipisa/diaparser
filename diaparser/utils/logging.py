# -*- coding: utf-8 -*-

import logging
import os

from ..utils.parallel import is_master
from tqdm import tqdm


def get_logger(name=None):
    if name is not None:
        return logging.getLogger(name)
    else:
        return logging.getLogger()


def init_logger(logger,
                path=None,
                mode='w',
                level=None,
                handlers=None,
                verbose=True):
    level = level or logging.WARNING
    if not handlers:
        handlers = [logging.StreamHandler()]
        if path:
            if os.path.dirname(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            handlers.append(logging.FileHandler(path, mode))
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=level,
                        handlers=handlers)
    logger.setLevel(logging.INFO if is_master() and verbose else logging.WARNING)


def progress_bar(iterator,
                 ncols=None,
                 bar_format='{l_bar}{bar:36}| {n_fmt}/{total_fmt} {elapsed}<{remaining}, {rate_fmt}{postfix}',
                 leave=None):   # disable on non-TTY (FIXHIM: not working)
    return tqdm(iterator,
                ncols=ncols,
                bar_format=bar_format,
                ascii=True,
                disable=True,  # (not (logger.level == logging.INFO and is_master())), FIXME: with or not verbose
                leave=leave)


logger = get_logger()
