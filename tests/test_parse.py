# -*- coding: utf-8 -*-

from diaparser.parsers import Parser

def test_parse():
    sentence = ['The', 'dog', 'chases', 'the', 'cat', '.']
    model = 'en_ewt.electra-base'
    parser = Parser.load(model)
    parser.predict([sentence], prob=True)
