
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import scipy
import scipy.special
from language_models.base_elmo_lm import BaseELMOLM
from bilm.data import UnicodeCharsVocabulary
# import sys
import os
#
SELF_DIR = os.path.dirname(os.path.abspath(__file__))
# /home/alx/Cloud/spell_corr/py_spelling_corrector:
ROOT_DIR = os.path.dirname(SELF_DIR)

class ELMOLMTFHub(BaseELMOLM):
    """
    This ELMO LM tries to reimplement faster TF ELMO from TF hub
    """
    def __init__(self):
        self._elmo = hub.Module(
            ROOT_DIR + "/bidirectional_lms/elmo_ru_news/tf_hub_model_epoch_n_3/",
            trainable=True)
        base_path = ROOT_DIR + "/bidirectional_lms/elmo_ru_news"
        ckpt_prefixed_path = base_path + "/model.ckpt-0003"
        # metafile_path = base_path + "/model.ckpt-0003.meta"
        # ckpt_prefixed_path = base_path + "/model.ckpt-1327437"
        # metafile_path = base_path + "/model.ckpt-1327437.meta"

        self.softmax_w = tf.train.load_variable(ckpt_prefixed_path, 'lm/softmax/W')
        self.softmax_bias = tf.train.load_variable(ckpt_prefixed_path, 'lm/softmax/b')

        # read vocabulary
        path_to_vocab = base_path + "/tokens_set.txt"
        with open(path_to_vocab, "r") as vocab_file:
            self.n_tokens_vocab = vocab_file.readlines()

        # TODO finish me
        self._lm_vocab = UnicodeCharsVocabulary(path_to_vocab, 200)

        self.words = self._lm_vocab._id_to_word
        self.word_index = {word: i for i, word in enumerate(self.words)}

        # index of unknown token:
        self.IDX_UNK_TOKEN = self.word_index.get("<UNK>")

    def elmo_lm(self, tokenized_sentences):
        """

        :param tokenized_sentences: Ex.: ["<S>", "мама", "мыла", "раму", "</S>"], ["<S>", "мама", "</S>"]
        :return:
        """
        # find_the longest sentence and padd other sentences with  "" to allign batch
        # tokenized_sentences = [["<S>", "мама", "мыла", "раму", "</S>"]]
        lengths = [len(each_sent) for each_sent in tokenized_sentences]
        max_len = max(lengths)

        # make padding:
        padded_sentences = []
        for idx, each_sent in enumerate(tokenized_sentences):
            if lengths[idx] < max_len:
                pad_count = max_len - lengths[idx]
                padded_sent = each_sent + [""]* pad_count
                padded_sentences.append(padded_sent)
            else:
                padded_sentences.append(each_sent)
        # #################################################################################
        # new model from n3 tf hub checkpont:
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # tokens_length = [5]
        embeddings = self._elmo(inputs={
            "tokens": padded_sentences,
            "sequence_len": lengths
        }, signature="tokens", as_dict=True)["lstm_outputs2"]
        elmo_data = sess.run(embeddings)
        # #################################################################################

        # now we need to postprocess outputs to remove zeros from short sentences
        # (sents that are shorter than max_len)
        padless_elmo_datas = []
        for idx, each_elmo_data in enumerate(elmo_data):
            if lengths[idx] < max_len:
                padless_elmo_data = each_elmo_data[:lengths[idx]]
                padless_elmo_datas.append(padless_elmo_data)
            else:
                padless_elmo_datas.append(each_elmo_data)

        # #################################################################################
        # feed forward and softmax:
        result_elmo_datas = []
        # TODO optimize if this work
        for each_result in padless_elmo_datas:
            #
            right_results= np.dot(each_result[:, 512:], self.softmax_w.transpose()) + self.softmax_bias
            left_results = np.dot(each_result[:, :512], self.softmax_w.transpose()) + self.softmax_bias
            right_probas = scipy.special.softmax(right_results, axis=1)
            left_probas = scipy.special.softmax(left_results, axis=1)
            sent_array = np.array([left_probas, right_probas])
            # gives a shape like (2, 5, 1000000): (2, tokens_num, 1000000)

            # we need to swap axes so the tokens axis is the first:
            result_elmo_datas.append(np.swapaxes(sent_array, 0, 1))

        return result_elmo_datas

    def _estimate_likelihood_minibatch(self, sentences_batch, preserve_states=True):
        """
        Estimates likelihood of the batch of sentence without check of batch size (may raise memory error)
        """
        # TODO this method duplicates torch method!
        tok_sents_wrapped = self.tokenize_sentence_batch(sentences_batch)

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
        return likelihoods
