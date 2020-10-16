# -*- coding: utf-8 -*-

from .parsers import BiaffineDependencyParser, Parser

__all__ = ['Parser', 'BiaffineDependencyParser']
__version__ = '1.0.0'

PRETRAINED = {
    'en-ptb': 'https://github.com/attardi/parser/releases/download/v1.0.0/en-ptb-electra-base-discriminator',
    'zh-ctb': 'https://github.com/attardi/parser/releases/download/v1.0.0/zh-ctb-chinese-electra-base-discriminator',
}
