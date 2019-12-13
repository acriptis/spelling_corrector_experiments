import numpy as np
from deeppavlov import build_model
from language_models.base_elmo_lm import BaseELMOLM


class ELMOLM(BaseELMOLM):
    # TODO merge with ELMO_inference!

    def __init__(self, config_dict):
        #         tf.compat.v1.random.set_random_seed(1234)
        # self.elmo_lm = build_model(config_dict, download=True)
        self.elmo_lm = build_model(config_dict, download=False)
        self.words = self.elmo_lm.pipe[-1][-1].get_vocab()
        self.word_index = {word: i for i, word in enumerate(self.words)}
        self.INIT_STATE_OF_ELMO = self.elmo_lm.pipe[-1][-1].init_states

        # index of unknown token:
        self.IDX_UNK_TOKEN = self.word_index.get("<UNK>")

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
