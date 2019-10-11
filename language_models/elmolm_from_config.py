import numpy as np
from deeppavlov import build_model
from language_models.base_elmo_lm import BaseELMOLM


class ELMOLM(BaseELMOLM):
    # TODO merge with ELMO_inference!
    # discount number to reduce probability of UNKNOWN words
    UNK_DISCOUNTER = 1e-6

    def __init__(self, config_dict):
        #         tf.compat.v1.random.set_random_seed(1234)
        # self.elmo_lm = build_model(config_dict, download=True)
        self.elmo_lm = build_model(config_dict, download=False)
        self.words = self.elmo_lm.pipe[-1][-1].get_vocab()
        self.word_index = {word: i for i, word in enumerate(self.words)}
        self.INIT_STATE_OF_ELMO = self.elmo_lm.pipe[-1][-1].init_states

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

    def _estimate_likelihood_minibatch(self, sentences_batch, preserve_states=True):
        """
        Estimates likelihood of the batch of sentence without check of batch size (may raise memory error)
        """
        tok_sents_wrapped = self.tokenize_sentence_batch(sentences_batch)
        #         print("Sentences are tokenized...")
        #         print(self.elmo_lm.pipe[-1][-1].init_states)

        elmo_datas = self.elmo_lm(tok_sents_wrapped)
        #         print("ELMO probas are calculated")
        probas = self.trace_sentence_probas_in_elmo_datas_batch(elmo_datas, tok_sents_wrapped)

        likelihoods = []

        for each_sent_probas in probas:
            logit_probas = np.log10(each_sent_probas)
            #         products = np.prod(probas, axis=1)
            products = np.sum(logit_probas, axis=1)
            #             print(products)
            likelihood = np.mean(products)
            likelihoods.append(likelihood)
        if preserve_states:
            self.elmo_lm.pipe[-1][-1].init_states = self.INIT_STATE_OF_ELMO
        return likelihoods

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
