from typing import List

import numpy as np
import json
from tqdm import tqdm
import kenlm

from deeppavlov.models.bidirectional_lms import elmo_bilm
from deeppavlov.models.tokenizers.lazy_tokenizer import LazyTokenizer

class ELMOLM(object):
    """
    Class that estimates likelihood of sentence in russian language

    Args:
        model_dir - path where is contained pretrainde ELMo
        freq_vocab_path - path where is contained frequency dictionary of elmo vocabulary
        penalty_for_unk_token - multiplier for token <UNK> when we estimate likelihood of sentence
        method_of_uniting_distr - {0, 1},
            0 -- method P(word|united_contex) = (P(word|left_contex) * P(word|right_contex)) / P(word)
            0 -- method P(word|united_contex) = log(P(word|left_contex) + log(P(word|right_contex)) / P(word)
    """

    def __init__(self,
                 model_dir: str,
                 penalty_for_unk_token: float=10e-6,
                 method_of_uniting_distr=0,
                 freq_vocab_path: str='./lm_models/freq_rusvector_elmo_vocab.json'):
        self.elmo_lm = elmo_bilm.ELMoEmbedder(model_dir=model_dir)
        self.words = self.elmo_lm.get_vocab()
        self.token2idx = dict(zip(self.elmo_lm.get_vocab(), range(len(self.elmo_lm.get_vocab()))))
        self.IDX_UNK_TOKEN = self.token2idx.get("<UNK>")
        self.PENALTY_FOR_UNK_TOKEN = penalty_for_unk_token
        self.INIT_STATE_OF_ELMO = self.elmo_lm.init_states
        if method_of_uniting_distr == 0:
            self._unite_distr = self._unite_distr_with_freq_of_words
            with open(freq_vocab_path, 'r') as f:
                freq_of_words = self._adjust_freq_of_words(json.load(f))
                self.freq_of_words = np.array(freq_of_words)
        elif method_of_uniting_distr == 1:
            self._unite_distr = self._unite_distr_log_sum

    def _adjust_freq_of_words(self, l: List[float]):
        """
        Utility method, that adjusts freq of words received from kenlm.
        words with freq -8.125675(unknown token) get new freq = 1, because it doesn't work well with unknown tokens
        """
        supp = max(l)
        return [i if i != l[0] else supp for i in l]

    def _softmax(self, a, axis):
        """
        softmax implementation
        """
        numerator = np.exp(a - np.max(a))
        denominator = np.expand_dims(np.sum(numerator, axis=axis), 2)
        return numerator / denominator

    def _unite_distr_with_freq_of_words(self, elmo_distr):
        """
        utility method, that unites distribution from forward pass lm and from backward pass lm
        based on formula: (P(word|left_contex) * P(word|right_contex)) / P(word)
        P(word) is taken from self.freq_of_words attribute
        """
        elmo_distr = np.log(elmo_distr)
        elmo_distr = np.sum(elmo_distr, axis=1)
        elmo_distr = elmo_distr - self.freq_of_words
        return self._softmax(elmo_distr, axis=1)

    def _unite_distr_log_sum(self, elmo_distr):
        """
        utility method, that unites distribution from forward pass lm and from backward pass lm
        based on formula: (P(word|left_contex) * P(word|right_contex))
        """
        elmo_distr = np.log(elmo_distr)
        elmo_distr = np.sum(elmo_distr, axis=1)
        return self._softmax(elmo_distr, axis=1)

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

    def trace_sentence_probas_in_elmo_data(self, elmo_data, tokenized_sentence):
        """
        Given elmo data (results of estimation the sentence by ELMO LM) and tokenized sentence
        (which may differ from original sentence) it searches for probabilities of new sentence
        :param elmo_data: ndarray from ELMOLM
        :param tokenized_sentence: list of tokens of sentence to analyze (which are wrapped with
            <s> </s> tokens as first and last tokens)
        :return: array with dims [N], where N is a length of sentence
        """
        idx_sentence = [self.get_word_idx_or_unk(token) for token in tokenized_sentence]
        probas = []
        for num_token_in_sent, idx_word_in_voc in enumerate(idx_sentence):
            multiplier = self.PENALTY_FOR_UNK_TOKEN if idx_word_in_voc == self.IDX_UNK_TOKEN else 1
            probas.append(multiplier * elmo_data[num_token_in_sent, idx_word_in_voc])
        return probas

    def trace_sentence_probas_in_elmo_datas_batch(self, elmo_datas, tokenized_sentences):
        """
        Given an ELMO's parsing data and tokenized sentence the method retrieves
        left and right probabilities of each token of the tokenized sentence
        
        : return is a list of numpy arrays with left and right probas for each word in sentence
        list has length of sentences batch, and each array has length of particular sentence
        and 2 probas for each token
        """
        return [self.trace_sentence_probas_in_elmo_data(elmo_data, tokenized_sent)
                for elmo_data, tokenized_sent in zip(elmo_datas, tokenized_sentences)]

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
        idx = self.get_word_idx_or_unk(token_str)
        multiplicator = self.PENALTY_FOR_UNK_TOKEN if idx == self.IDX_UNK_TOKEN else 1
        return multiplicator * elmo_data[sentence_position_index, idx]

    def get_word_idx(self, word):
        return self.token2idx.get(word)

    def get_word_idx_or_unk(self, word):
        return self.token2idx.get(word, self.IDX_UNK_TOKEN)

    def estimate_likelihood_batch(self, sentences_batch, preserve_states=True, batch_size=10):
        batch_gen = self.chunk_generator(sentences_batch, batch_size)
        output_batch = []
        for mini_batch in batch_gen:
            likelihoods_mini = self._estimate_likelihood_minibatch(mini_batch,
                                                                   preserve_states=preserve_states)
            output_batch.extend(likelihoods_mini)
        return output_batch

    def _estimate_likelihood_minibatch(self, sentences_batch, preserve_states=True):
        """
        Estimates likelihood of the batch of sentence without check of batch size (may raise memory error)
        """
        tok_sents_wrapped = self.tokenize_sentence_batch(sentences_batch)
        init_states_bak = self.elmo_lm.init_states
        elmo_datas = self.elmo_lm(tok_sents_wrapped)
        elmo_datas = [self._unite_distr(elmo_distr) for elmo_distr in elmo_datas]
        probas = self.trace_sentence_probas_in_elmo_datas_batch(elmo_datas, tok_sents_wrapped)

        likelihoods = []
        for each_sent_probas in probas:
            logs = np.log(each_sent_probas)
            likelihood = np.mean(logs)
            likelihoods.append(likelihood)
        if preserve_states:
            self.elmo_lm.init_states = init_states_bak
        return likelihoods

    def estimate_likelihood(self, sentence, preserve_states=True):
        """
        Estimates a likelihood of a sentence
        """
        tok_wrapped = self.tokenize_sentence(sentence)
        init_states_bak = self.elmo_lm.init_states
        elmo_datas = self.elmo_lm([tok_wrapped])
        elmo_datas = [self._unite_distr(elmo_distr) for elmo_distr in elmo_datas][0]
        probas = self.trace_sentence_probas_in_elmo_data(elmo_datas, tok_wrapped)

        logs = np.log(probas)
        likelihood = np.mean(logs)
        if preserve_states:
            self.elmo_lm.init_states = init_states_bak
        return likelihood













