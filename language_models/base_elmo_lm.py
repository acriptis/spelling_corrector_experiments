import numpy as np


class BaseELMOLM():
    """
    Class which is base for all ELMOLM Family
    """
    # discount number to reduce probability of UNKNOWN words
    UNK_DISCOUNTER = 1e-6

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

    def estimate_likelihood(self, sentence):
        """
        Estimates a likelihood of a sentence
        """
        tok_wrapped = self.tokenize_sentence(sentence)

        elmo_data = self.elmo_lm([tok_wrapped])[0]

        probas = self.trace_sentence_probas_in_elmo_data(elmo_data, tok_wrapped)
        logit_probas = np.log10(probas)
        #         products = np.prod(probas, axis=1)
        products = np.sum(logit_probas, axis=1)
        print(products)
        likelihood = np.mean(products)
        return likelihood

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

    def trace_sentence_probas_in_elmo_data(self, elmo_data, tokenized_sentence):
        """
        Given elmo data (results of estimation the sentence by ELMO LM) and tokenized sentence
        (which may differ from original sentence) it searches for probabilities of new sentence
        :param elmo_data: ndarray from ELMOLM
        :param tokenized_sentence: list of tokens of sentence to analyze (which are wrapped with
            <s> </s> tokens as first and last tokens)
        :return: array with dims [N, 2], where N is a length of sentence

        """
        left_probas = []
        right_probas = []

        for num, each_tok in enumerate(tokenized_sentence[:-1]):
            if num == 0:
                continue

            idx = self.get_word_idx(each_tok)
            if idx:
                #                 print("get_index for word %s is %d" % (each_tok.upper(), idx))
                magic_multiplicator = 1.0
            else:
                idx = self.word_index.get("<UNK>")
                magic_multiplicator = self.UNK_DISCOUNTER
                print("UNK: %s| " % each_tok, end='')

            #                 print("Word %s is UNKNOWN. idx=%d" % (each_tok.upper(), idx))
            left_p, right_p = elmo_data[num, :, idx]

            left_probas.append(left_p * magic_multiplicator)
            right_probas.append(right_p * magic_multiplicator)
        # print()
        return np.array([left_probas, right_probas])

    def trace_sentence_probas_in_elmo_datas_batch(self, elmo_datas, tokenized_sentences):
        """
        Given an ELMO's parsing data and tokenized sentence the method retrieves
        left and right probabilities of each token of the tokenized sentence

        : return is a list of numpy arrays with left and right probas for each word in sentence
        list has length of sentences batch, and each array has length of particular sentence
        and 2 probas for each token


        """
        results_batch = []

        for sentence_idx, each_elmo_data in enumerate(elmo_datas):
            left_probas = []
            right_probas = []
            tokenized_sentence = tokenized_sentences[sentence_idx]
            for num, each_tok in enumerate(tokenized_sentence[:-1]):
                if num == 0:
                    continue

                idx = self.get_word_idx(each_tok)
                if idx:
                    #                 print("get_index for word %s is %d" % (each_tok.upper(), idx))
                    magic_multiplicator = 1.0
                    pass
                else:
                    idx = self.word_index.get("<UNK>")
                    # reduce probability of UNK tokens
                    magic_multiplicator = self.UNK_DISCOUNTER

                #                 print("Word %s is UNKNOWN. idx=%d" % (each_tok.upper(), idx))
                left_p, right_p = each_elmo_data[num, :, idx]

                left_probas.append(left_p * magic_multiplicator)
                right_probas.append(right_p * magic_multiplicator)
            #             print(left_p)
            #             print(right_p)
            #             print(elmo_data[num,:,idx])
            #             print("_")
            results_batch.append(np.array([left_probas, right_probas]))
        return results_batch

    def retrieve_logits_of_particular_token(self, elmo_data, sentence_position_index, token_str):
        """
        Utility method to retrieve a pair of logit probabilities (left and right) for particular
        token at particular position in sentence.

        Assume we have a sentence: мама мыла раму мылом. And we have analyzed with ELMOLM this
        sentence and calculated elmo_data matrix

        No we can retrieve logit probabilit of word папа instead of мама:
            папа мыла раму мылом
        To estimate probabilities of word папа at 0's position we can cal this method with args:
        retrieve_logits_of_particular_token(elmo_data, 0, "папа") -> [-7.232, -5.134]

        With this method
        :param sentence_position_index:
        :param token_str: string of token to lookup. If token is unknown by ELMO,
        then it will return normalized probability for unknown tokens
        :return: tuple of left_logit and right_logit score
        """
        # get index of token in ELMO's dictionary
        idx = self.get_word_idx(token_str)
        if idx:
            magic_multiplicator = 1.0
        else:
            idx = self.word_index.get("<UNK>")
            # reduce probability of UNK tokens
            magic_multiplicator = self.UNK_DISCOUNTER

        left_p, right_p = elmo_data[sentence_position_index, :, idx]
        left_logit = np.log10(left_p * magic_multiplicator)
        right_logit = np.log10(right_p * magic_multiplicator)
        return left_logit, right_logit
