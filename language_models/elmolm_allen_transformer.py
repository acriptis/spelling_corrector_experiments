from language_models.base_elmo_lm import BaseELMOLM
from bilm.data import UnicodeCharsVocabulary
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers.token import Token
from allennlp.modules.elmo import batch_to_ids
import torch
import numpy as np
import os
#
SELF_DIR = os.path.dirname(os.path.abspath(__file__))
# /home/alx/Cloud/spell_corr/py_spelling_corrector:
ROOT_DIR = os.path.dirname(SELF_DIR)


class AllenElmoTransformerLM2(BaseELMOLM):
    """
    Implementation of ELMO LM on torch, AllenNLP and with Transformers layer
    Here we actually load the model from model.tar.gz file
    """
    def __init__(self):
        self.load_model()

        # read vocabulary
        self.path_to_vocab = SELF_DIR + "/elmo_transformer_pretrained_models/vocabulary/tokens.txt"
        self._lm_vocab = UnicodeCharsVocabulary(self.path_to_vocab, 200)

        self.words = self._lm_vocab._id_to_word
        self.word_index = {word: i for i, word in enumerate(self.words)}

        # index of unknown token:
        self.IDX_UNK_TOKEN = self.word_index.get("<UNK>")

    def load_model(self):
        from allennlp.models.archival import load_archive

        path_to_model_targz = SELF_DIR + "/elmo_transformer_pretrained_models/model_2.tar.gz"
        archive_obj = load_archive(path_to_model_targz)
        self._elmo_model = archive_obj.model

        self._ff = torch.nn.Linear(512, 1000001)
        # ff.cuda()
        self._ff.load_state_dict(
            {'weight': self._elmo_model._softmax_loss._modules['softmax_w']._parameters['weight'],
             'bias': self._elmo_model._softmax_loss._modules['softmax_b']._parameters['weight'][:,
                     0]
             },
            strict=False)
        self._softmax_fn = torch.nn.Softmax(dim=3)

    def elmo_lm(self, tokenized_sentences):
        """
        Main method which returns an ELMO matrix
        :param tokenized_sentences: list of tokenized sentences.
        :return: tensor BATCH_SIZE x TOKENS_NUM x 2 x 1000001
        """

        character_indices = batch_to_ids(tokenized_sentences)

        indices_tensor = torch.LongTensor(character_indices)

        res = self._elmo_model({'token_characters': indices_tensor})

        forward_embeddings, backward_embeddings = res['lm_embeddings'].chunk(2, -1)

        left_results = self._ff(forward_embeddings)
        right_results = self._ff(backward_embeddings)
        stacked_output = torch.stack((left_results, right_results), dim=2)
        softmaxed_output = self._softmax_fn(stacked_output)

        return softmaxed_output.detach().numpy()

    def _estimate_likelihood_minibatch(self, sentences_batch, preserve_states=True):
        """
        Estimates likelihood of the batch of sentence without check of batch size (may raise memory error)
        """
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
        # if preserve_states:
        #     self.elmo_lm.pipe[-1][-1].init_states = self.INIT_STATE_OF_ELMO
        return likelihoods

    ###############################################################################################
    # TODO refactor parametrization of UNK!
    # Thank to Allen they renamed <UNK> into @@UNKNOWN@@
    def get_word_idx_or_unk(self, word):
        """
        Get a word's index from word string
        if no word found in dictionary returns UNK index and boolean flag that it is unknown word

        return tuple: (word_index, is_unk) - word_index is an index to be used for the word,
            is_unk is boolean flag if word will be interpreted as unknown word
        """
        is_unk = False
        idx = self.get_word_idx(word)
        if not idx:
            idx = self.word_index.get("@@UNKNOWN@@")
            is_unk = True

        return idx, is_unk

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
            idx = self.word_index.get("@@UNKNOWN@@")
            # reduce probability of UNK tokens
            magic_multiplicator = self.UNK_DISCOUNTER

        left_p, right_p = elmo_data[sentence_position_index, :, idx]
        left_logit = np.log10(left_p * magic_multiplicator)
        right_logit = np.log10(right_p * magic_multiplicator)
        return left_logit, right_logit

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
                idx = self.word_index.get("@@UNKNOWN@@")
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
                    idx = self.word_index.get("@@UNKNOWN@@")
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
