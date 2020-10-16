import stanza

# reference https://github.com/stanfordnlp/stanza/blob/master/stanza/utils/prepare_tokenizer_data.py

class Tokenizer:
    PROCESSORS = 'tokenize,mwt'
    def __init__(self, lang, dir, verbose, use_gpu):
        stanza.download(lang, dir=dir, processors=self.PROCESSORS, verbose=verbose)
        self.pipeline = stanza.Pipeline(lang, dir=dir, processors=self.PROCESSORS, verbose=verbose, use_gpu=use_gpu)

    def predict(self, text):
        return self.pipeline(text).sentences

    def format(self, sentences):
        for i, sentence in  enumerate(sentences):
            yield f'# sent_id = {i+1}'
            sent_text = sentence.text.replace("\n", " ")
            yield f'# text = {sent_text}'
            for token in sentence.tokens:
                token_text = token.text
                # multiword
                if len(token.words) > 1:
                    token_ids = '-'.join([str(id) for id in token.id])
                    yield f'{token_ids}\t{token.text}'
                    for word in token.words:
                        yield f'{word.id}\t{word.text}'
                else:
                    yield f'{token.id[0]}\t{token.text}'
            yield ''
