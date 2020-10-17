
import stanza
import torch
import json
from contextlib import contextmanager

# reference https://github.com/stanfordnlp/stanza/blob/master/stanza/utils/prepare_tokenizer_data.py

class Tokenizer:
    """
    Interface to Stanza tokenizers.
    Args.
    lang (str): conventional language identifier.
    dir (str): directory for caching models.
    verbose (Bool): print download progress.
    """

    PROCESSORS = 'tokenize,mwt'

    def __init__(self, lang, dir, verbose=False):
        stanza.download(lang, dir=dir, processors=self.PROCESSORS, verbose=verbose)
        use_gpu = torch.cuda.is_available()
        self.pipeline = stanza.Pipeline(lang, dir=dir, processors=self.PROCESSORS, verbose=verbose, use_gpu=use_gpu)

    def predict(self, text):
        return self.pipeline(text).sentences

    def format(self, sentences):
        """
        Convert sentences to TSV format.
        """
        for i, sentence in enumerate(sentences):
            yield f'# sent_id = {i+1}'
            sent_text = sentence.text.replace("\n", " ")
            yield f'# text = {sent_text}'
            for token in sentence.tokens:
                # multiword
                if len(token.words) > 1:
                    token_ids = '-'.join([str(id) for id in token.id])
                    yield f'{token_ids}\t{token.text}'
                    for word in token.words:
                        yield f'{word.id}\t{word.text}'
                else:
                    yield f'{token.id[0]}\t{token.text}'
            yield ''

    def reader(self):
        """
        Reading function that returns a generator of CoNLL-U sentences.
        """
        @contextmanager
        def generator(data):
            with open(data) as f:
                yield self.format(self.predict(f.read()))
        return generator

