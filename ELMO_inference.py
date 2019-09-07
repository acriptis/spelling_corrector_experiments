from typing import List

import numpy as np
import json
from tqdm import tqdm

from deeppavlov.models.bidirectional_lms import elmo_bilm
from deeppavlov.models.tokenizers.lazy_tokenizer import LazyTokenizer

class ELMOLM(object):
    """
    Class that estimates likelihood of sentence in russian language
    """

    def __init__(self,
                 model_dir: str,
                 scores_vocab_path: str,
                 preserve_states: bool):
        self.elmo_lm = elmo_bilm.ELMoEmbedder(model_dir=model_dir)
        with open(scores_vocab_path, 'r') as f:
            self.scores_of_elmo_vocab_by_kenlm = np.array(json.load(f))
        self.tokenizer = LazyTokenizer()
        self.token2idx = dict(zip(self.elmo_lm.get_vocab(), range(len(self.elmo_lm.get_vocab()))))
        self.UNK_INDEX = self.token2idx.get("<UNK>")
        self.PENALTY_UNK = 10e-6
        if preserve_states:
            self.INIT_STATE = self.elmo_lm.init_states

    def _softmax(self, a, axis):
        numerator = np.exp(a - np.max(a))
        denominator = np.expand_dims(np.sum(numerator, axis=axis), 2)
        return numerator / denominator

    def _unite_distr(self, elmo_distr):
        elmo_distr = np.log(elmo_distr)
        elmo_distr = np.sum(elmo_distr, axis=1)
        elmo_distr = elmo_distr - self.scores_of_elmo_vocab_by_kenlm
        return self._softmax(elmo_distr, axis=1)

    @staticmethod
    def chunk_generator(items_list, chunk_size):
        """
        Method to slice batches into chunks of minibatches
        """
        for i in range(0, len(items_list), chunk_size):
            yield items_list[i:i + chunk_size]

    def estimate_likelihood_batch(self, sentences_batch, batch_size=10):
        """
        Method estimates a likelihood of the batch of sentences with slicing the batch into minibatches
        to avoid memory error
        """
        batch_gen = self.chunk_generator(sentences_batch, batch_size)
        output_batch = []
        for mini_batch in tqdm(batch_gen):
            likelihoods_mini = self._estimate_likelihood_minibatch(mini_batch)
            output_batch.extend(likelihoods_mini)
        return output_batch

    def wrap_in_spec_symbols(self, batch: List[List[str]]):
        return [['<S>'] + sent + ['</S>'] for sent in batch]

    def _estimate_prob_minibatch(self, elmo_distr_united, minibatch: List[List[str]]):
        idx_minibatch = [[self.token2idx.get(token, self.UNK_INDEX) for token in sent] for sent in minibatch]
        p_minibatch = []
        for num_sent, idx_sent in enumerate(idx_minibatch):
            p_sent = []
            for num_token, idx_token in enumerate(idx_sent):
                multiplier = self.PENALTY_UNK if idx_token == self.UNK_INDEX else 1
                p_sent.append(multiplier * elmo_distr_united[num_sent][num_token, idx_token])
            p_minibatch.append(np.sum(np.log(p_sent)))
        return p_minibatch

    def _estimate_likelihood_minibatch(self, minibatch: List[List[str]], is_wrap_spec_sym: bool = True):
        if is_wrap_spec_sym:
            minibatch = self.wrap_in_spec_symbols(minibatch)
        elmo_distr = self.elmo_lm(minibatch)
        elmo_distr_united = [self._unite_distr(distr_sent) for distr_sent in elmo_distr]
        likelihood_minibatch = self._estimate_prob_minibatch(elmo_distr_united, minibatch)
        return likelihood_minibatch


class ELMO_LM_one_track(ELMOLM):

    def __init__(self,
                 model_dir: str,
                 scores_vocab_path: str,
                 preserve_states: bool):
        super(ELMO_LM_one_track).__init__(model_dir, scores_vocab_path, preserve_states)

    def save_distr_from_sentence(self, sentences_batch: List[List[str]], batch_size=10, is_wrap_spec_sym: bool = True):
        """

        :param sentences_batch: - list of tokenized text, for example: [[it, is, cool], [i, am, know]]
        :param batch_size:
        :param is_wrap_spec_sym:
        :return:
        """
        batch_gen = self.chunk_generator(sentences_batch, batch_size)
        output_batch = []
        for mini_batch in batch_gen:
            if is_wrap_spec_sym:
                mini_batch = self.wrap_in_spec_symbols(mini_batch)
            elmo_distr = self.elmo_lm(mini_batch)
            output_batch.extend([self._unite_distr(distr_sent) for distr_sent in elmo_distr])
        self.saved_elmo_distr = output_batch

    def estimate_prob_minibatch_on_saved_distr(self, minibatch: List[List[List[str]]]):
        """

        :param minibatch: list of hyp, for example: [[[it, is, you], [at is you], [is it you]], [[stop this], [Stop this], [stop, him]]]
        :return: list[list[float]], for example: [[123, 34, 44], [1233, 4343, 5555555]]
        """

        idx_minibatch = [[[ self.token2idx.get(token, self.UNK_INDEX)
                            for token in hyp]
                                for hyp in hyps]
                                    for hyps in minibatch]
        p_minibatch = []
        for num_sent, idx_hyps in enumerate(idx_minibatch):
            p_hyps = []
            for num_hyp, idx_hyp in enumerate(idx_hyps):
                p_hyp = []
                for num_token, idx_token in enumerate(idx_hyp):
                    multiplier = self.PENALTY_UNK if idx_token == self.UNK_INDEX else 1
                    p_hyp.append(multiplier * self.saved_elmo_distr[num_sent][num_hyp][num_token, idx_token])
                p_hyps.append(np.sum(np.log(p_hyp)))
            p_minibatch.append(p_hyps)
        return p_minibatch











