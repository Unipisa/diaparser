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
        tokenizer = Tokenizer(self.args['lang'])
        
        self.assertTrue(os.path.isdir(self.MODEL_DIR))
        self.assertTrue(os.path.exists(os.path.join(self.MODEL_DIR, 'tokenizer', self.args['lang'])))
        self.assertTrue(os.path.exists(os.path.join(self.MODEL_DIR, 'tokenizer', self.args['lang'], 'tokenize')))
    
    def test_tokenize(self):
        tokenizer = Tokenizer(self.args['lang'])
        sentences = tokenizer.predict("L'iphone del dr. Rossi è bello. Ma è troppo costoso.")
        self.assertEqual(len(sentences), 2)

    def test_corpus_load(self):
        tokenizer = Tokenizer(self.args['lang'])
        sin = io.StringIO("Un corazziere contro Scalfaro. L'attore le disse baciami o torno a riprendermelo.")
        
        for line in tokenizer.format(tokenizer.predict(sin.read())):
            if line and not line.startswith('#'):
                # CoNLL-U format has 10 tsv:
                assert len(line.split('\t')) == 10, line
