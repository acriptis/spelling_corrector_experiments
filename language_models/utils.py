def tokenize_sentence_batch(self, sentences_batch, wrap_s=True):
    """
    input sentences (list of strings)
    ouputs list of lists of tokens
    """
    assert isinstance(sentences_batch, list)

    # wrap with S tokens
    if wrap_s:
        tok_sents = [['<S>'] + sent.split() + ['</S>'] for sent in sentences_batch]
    else:
        tok_sents = [sent.split() for sent in sentences_batch]
    return tok_sents

def tokenize_sentence(self, sentence, wrap_s=True):
    """
    """
    tok_sent = sentence.split()
    # wrap with S tokens
    if wrap_s:
        tok_sent = ['<S>'] + tok_sent + ['</S>']
    return tok_sent