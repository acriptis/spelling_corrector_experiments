class BaseELMOLM():
    """
    Class which is base for all ELMOLM Family
    """

    @staticmethod
    def chunk_generator(items_list, chunk_size):
        """
        Method to slice batches into chunks of minibatches
        """
        for i in range(0, len(items_list), chunk_size):
            yield items_list[i:i + chunk_size]

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

    def analyze_sentence(self, sentence):
        """
        Returns elmo's parsing of a sentence
        """
        # tokenize

        tok_wrapped = self.tokenize_sentence(sentence)
        # analyze the sentence:
        data = self.elmo_lm([tok_wrapped])[0]
        return data

    def get_word_idx(self, word):
        """
        Get a word's index from word string
        if no word found in dictionary returns None
        """
        word_idx = self.word_index.get(word)
        return word_idx

    def get_word_idx_or_unk(self, word):
        """
        Get a word's index from word string
        if no word found in dictionary returns UNK index and boolean flag that it is unknown word

        return tuple: (word_index, is_unk) - word_index is an index to be used for the word,
            is_unk is boolean flag if word will be interpreted as unknown word
        """
        is_unk=False
        idx = self.get_word_idx(word)
        if not idx:
            idx = self.word_index.get("<UNK>")
            is_unk = True

        return idx, is_unk

    def estimate_likelihood_batch(self, sentences_batch, preserve_states=True, batch_size=10):
        """
        Method estimates a likelihood of the batch of sentences with slicing the batch into minibatches
        to avoid memory error
        """
        if len(sentences_batch) > batch_size:
            batch_gen = self.chunk_generator(sentences_batch, batch_size)
            output_batch = []
            for mini_batch in batch_gen:
                likelihoods_mini = self._estimate_likelihood_minibatch(mini_batch,
                                                                       preserve_states=preserve_states)
                output_batch.extend(likelihoods_mini)
        else:
            output_batch = self._estimate_likelihood_minibatch(sentences_batch,
                                                               preserve_states=preserve_states)
        return output_batch
