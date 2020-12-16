import shutil
import os
import argparse
import unittest
import io

from tokenizer.tokenizer import Tokenizer

class TestTokenizer(unittest.TestCase):

    MODEL_DIR = os.path.expanduser('~/.cache/diaparser')
    
    def setUp(self):
        self.args = {
            'lang': 'it',
            'verbose': True
        }
        
    def test_download_resources(self):
        tokenizer = Tokenizer(**self.args)
        
        self.assertTrue(os.path.isdir(self.MODEL_DIR))
        self.assertTrue(os.path.exists(os.path.join(self.MODEL_DIR, 'tokenizer', self.args['lang'])))
        self.assertTrue(os.path.exists(os.path.join(self.MODEL_DIR, 'tokenizer', self.args['lang'], 'tokenize')))
    
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
