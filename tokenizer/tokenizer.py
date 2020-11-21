
import stanza
import torch
import json
import os
from contextlib import contextmanager
from diaparser.catalog import available_processors, download_processors

# reference https://github.com/stanfordnlp/stanza/blob/master/stanza/utils/prepare_tokenizer_data.py

class Tokenizer:
    """
    Interface to Stanza tokenizers.
    Args.
    lang (str): conventional language identifier.
    dir (str): directory for caching models.
    verbose (Bool): print download progress.
    """

    def __init__(self, lang, dir=os.path.expanduser('~/.cache/diaparser'), verbose=True):
        dir += '/tokenizer'
        # check for custom processors
        avail_processors = available_processors(lang, dir)
        avail_preprocessors = avail_processors.keys() & ('tokenize', 'mwt')
        if avail_preprocessors:
            processors = {p: avail_processors[p] for p in avail_preprocessors}
            cached_paths = download_processors(lang, processors, dir)
            processors = ','.join(avail_preprocessors)
        else:
            cached_paths = {}
            processors='tokenize'
            stanza.download(lang, dir=dir, processors=processors, verbose=verbose)
            try:
                stanza.download(lang, dir=dir, processors='mwt', verbose=verbose)
                processors += ',mwt'
            except:
                pass
        use_gpu = torch.cuda.is_available()
        self.pipeline = stanza.Pipeline(lang, dir=dir, processors=processors, verbose=verbose,
                                        use_gpu=use_gpu, **cached_paths)

    def predict(self, text):
        return self.pipeline(text).sentences

    def format(self, sentences):
        """
        Convert sentences to CoNLL format.
        """
        empty_fields = '\t_' * 8
        for i, sentence in enumerate(sentences):
            yield f'# sent_id = {i+1}'
            sent_text = sentence.text.replace("\n", " ")
            yield f'# text = {sent_text}'
            for token in sentence.tokens:
                # multiword
                if len(token.words) > 1:
                    token_range = f'{token.id[0]}-{token.id[-1]}'
                    yield f'{token_range}\t{token.text + empty_fields}'
                    for word in token.words:
                        yield f'{word.id}\t{word.text + empty_fields}'
                else:
                    yield f'{token.id[0]}\t{token.text + empty_fields}'
            yield ''

    def reader(self):
        """
        Reading function that returns a generator of CoNLL-U sentences.
        """
        @contextmanager
        def generator(data):
            """
            Args:
                data (str): could be a filename or the text to tokenize.
            Returns:
                a context manager that can be used in a `with` contruct,
                yielding each line of the tokenized `data`.
            """
            if not os.path.exists(data):
                yield self.format(self.predict(data))
            else:
                with open(data) as f:
                    yield self.format(self.predict(f.read()))
        return generator

