# -*- coding: utf-8 -*-

from .parsers import BiaffineDependencyParser, Parser

__all__ = ['Parser', 'BiaffineDependencyParser']
__version__ = '1.0.0'

PRETRAINED = {
    'en_ptb.electra-base': 'https://github.com/Unipisa/diaparser/releases/download/v1.0.0/en_ptb.electra-base',
    'it_isdt.dbmdz-xxl': 'https://github.com/Unipisa/diaparser/releases/download/v1.0.0/it_isdt.dbmdz-xxl',
    'zh_ctb.hfl': 'https://github.com/Unipisa/diaparser/releases/download/v1.0.0/zh_ctb.hfl',
}
