import shutil
import os
import argparse
import unittest

from tokenizer.tokenizer import Tokenizer

class TestTokenizer(unittest.TestCase):

    MODEL_DIR = '/tmp/stanza_models'
    
    def setUp(self):
        self.args = {
            'lang': 'it',
            'dir': self.MODEL_DIR,
            'verbose': True,
            'use_gpu': True
        }

    def tearDown(self):
        shutil.rmtree(self.MODEL_DIR)
        
    def test_download_resources(self):
        self.assertTrue(not os.path.exists(self.MODEL_DIR))
        tokenizer = Tokenizer(**self.args)
        self.assertTrue(os.path.exists(self.args['dir']) and not os.path.isfile(self.args['dir']))
        self.assertTrue(os.path.exists(os.path.join(self.args['dir'], self.args['lang'])))
        self.assertTrue(os.path.exists(os.path.join(self.args['dir'], self.args['lang'], 'tokenize')))
    
    def test_tokenize(self):
        tokenizer = Tokenizer(**self.args)
        sentences = tokenizer.predict('Domani vorrei andare al mare.Speriamo faccia bel tempo.')
        self.assertEqual(len(sentences), 2)

    def test_corpus_load(self):
        tokenizer = Tokenizer(**self.args)
        raw_text_file = '/project/piqasso/Collection/IWPT20/train-dev/UD_Italian-ISDT/it_isdt-ud-dev.txt'
        
        with open(raw_text_file) as fin:
            for line in tokenizer.format(tokenizer.predict(fin.read())):
                if line and not line.startswith('#'):
                    assert len(line.split('\t')) == 2, line
