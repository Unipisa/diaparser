import shutil
import os
import argparse
import unittest
import io

from tokenizer.tokenizer import Tokenizer

class TestTokenizer(unittest.TestCase):

    MODEL_DIR = '/tmp/stanza_models'
    
    def setUp(self):
        self.args = {
            'lang': 'it',
            'dir': self.MODEL_DIR,
            'verbose': True
        }

    def tearDown(self):
        shutil.rmtree(self.MODEL_DIR)
        
    def test_download_resources(self):
        self.assertTrue(not os.path.exists(self.MODEL_DIR))
        tokenizer = Tokenizer(**self.args)
        self.assertTrue(os.path.exists(self.args['dir']) and not os.path.isfile(self.args['dir']))
        self.assertTrue(os.path.exists(os.path.join(self.args['dir'], 'tokenizer', self.args['lang'])))
        self.assertTrue(os.path.exists(os.path.join(self.args['dir'], 'tokenizer', self.args['lang'], 'tokenize')))
    
    def test_tokenize(self):
        tokenizer = Tokenizer(**self.args)
        sentences = tokenizer.predict('Ha chiamato il dr. Rossi.Vuole salutarti.')
        self.assertEqual(len(sentences), 2)

    def test_corpus_load(self):
        tokenizer = Tokenizer(**self.args)
        sin = io.StringIO("Un corazziere contro Scalfaro. L'attore le disse baciami o torno a riprendermelo.")
        
        for line in tokenizer.format(tokenizer.predict(sin.read())):
            if line and not line.startswith('#'):
                assert len(line.split('\t')) == 10, line
