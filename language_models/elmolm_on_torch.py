import torch
from allennlp.modules.elmo import _ElmoBiLm
from language_models.allennlp_elmo.allennlp_elmobilm import ELMOBiLM
from allennlp.modules.elmo import batch_to_ids
import tensorflow as tf
import numpy as np
from language_models.base_elmo_lm import BaseELMOLM
from bilm.data import UnicodeCharsVocabulary
import sys
import os

SELF_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SELF_DIR)
# sys.path.append(ROOT_DIR)
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ruler_bot.settings")

class ELMOLMTorch(BaseELMOLM):
    """
    Implementation of ELMOLM on torch it is faster than deeppavlov's one
    """
    def __init__(self):
        self.load_model()

        # read vocabulary
        path_to_vocab = ROOT_DIR + "/bidirectional_lms/elmo_ru_news/tokens_set.txt"
        # with open(path_to_vocab, "r") as vocab_file:
        #     self.n_tokens_vocab = vocab_file.readlines()
        # print(self.n_tokens_vocab)
        self._lm_vocab = UnicodeCharsVocabulary(path_to_vocab, 200)

        self.words = self._lm_vocab._id_to_word
        self.word_index = {word: i for i, word in enumerate(self.words)}

        # index of unknown token:
        self.IDX_UNK_TOKEN = self.word_index.get("<UNK>")

    def load_model(self):
        # options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        # options_file = "http://files.deeppavlov.ai/lang_models/sexy_elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        options_file = "http://files.deeppavlov.ai/lang_models/sexy_elmo/options.json"
        # weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        # custom weights
        # weight_file = "http://files.deeppavlov.ai/lang_models/sexy_elmo/weights.hdf5"
        # weight_file = "http://files.deeppavlov.ai/lang_models/sexy_elmo/weights_epoch_n_2.hdf5"
        weight_file = "http://files.deeppavlov.ai/lang_models/sexy_elmo/weights_epoch_n_3.hdf5"
        # weight_file = "http://files.deeppavlov.ai/lang_models/sexy_elmo/weights_epoch_n_4.hdf5"

        # allennlp realizatioon with updating states
        # self._elmobilm = _ElmoBiLm(options_file, weight_file)
        # realizatioon without updating states
        self._elmobilm = ELMOBiLM(options_file, weight_file)
        self._elmobilm.cuda()

        # TODO load head:
        # self._ff = torch.nn.Linear(1024, 1000000)
        self._ff = torch.nn.Linear(512, 1000000)
        self._ff.cuda()

        # TODO refactor
        # Load checkpoint of TF:
        base_path = ROOT_DIR + "/bidirectional_lms/elmo_ru_news"
        ckpt_prefixed_path = base_path + "/model.ckpt-0003"
        # metafile_path = base_path + "/model.ckpt-0003.meta"

        # tf.train.list_variables(ckpt_prefixed_path)

        # matrix which holds embedding into words projection
        emb2words_w_matrix = tf.train.load_variable(ckpt_prefixed_path, 'lm/softmax/W')

        # torch_w = torch.from_numpy(np.concatenate((softmax_w, softmax_w), axis=1))
        torch_w = torch.from_numpy(emb2words_w_matrix)

        emb2words_bias = tf.train.load_variable(ckpt_prefixed_path, 'lm/softmax/b')
        self._ff.load_state_dict(
            {'weight': torch_w, 'bias': torch.from_numpy(emb2words_bias)},
            strict=False)

        self._softmax_fn = torch.nn.Softmax(dim=3)

    def elmo_lm(self, tokenized_sentences):
        """
        Main method which returns an ELMO matrix
        :param tokenized_sentences: list of tokenized sentences.
        :return: tensor BATCH_SIZE x TOKENS_NUM x 2 x 1000000
        """
        # TODO clarify usage of s, /s
        # if tokenized_sentences[0][0].lower() == "<s>":
        #     # wrapped tokenization, then we need to unwrap it because torchy ELMO wraps itself
        #     tokenized_sentences = [ts[1:len(ts)-1] for ts in tokenized_sentences]

        # print("tokenized_sentences")
        # print(tokenized_sentences)
        # TODO grab states
        # use batch_to_ids to convert sentences to character ids
        #     sentences = [['First', 'sentence', '.'], ['Another', '.']]
        character_ids = batch_to_ids(tokenized_sentences)
        # print(character_ids.shape)
        # print(character_ids)
        character_ids = character_ids.cuda()
        elmo_output = self._elmobilm(character_ids)
        # TODO check correctness:
        last_layer_activations = elmo_output['activations'][2]
        # results = self._ff(last_layer_activations)
        # print("last_layer_activations:")
        # print(last_layer_activations)
        # print(last_layer_activations.shape)
        # pre_last_layer_activations = elmo_output['activations'][1]
        # print("pre_last_layer_activations:")
        # print(pre_last_layer_activations)
        # print(pre_last_layer_activations.shape)

        # base_layer_activations = elmo_output['activations'][0]
        # print("base_layer_activations :")
        # print(base_layer_activations )
        # print(base_layer_activations .shape)
        # print(lstm_outputs[0][0].shape)

        # left_activations, right_activations = torch.split(last_layer_activations, 512, dim=1)
        splitted_tensors = torch.split(last_layer_activations, 512, dim=2)
        # import ipdb; ipdb.set_trace()

        assert len(splitted_tensors) == 2
        # print("len(splitted_tensors)")
        # print(len(splitted_tensors))
        left_activations = splitted_tensors[0]
        right_activations = splitted_tensors[1]
        left_results = self._ff(left_activations)
        right_results = self._ff(right_activations)
        stacked_output = torch.stack((left_results, right_results), dim=2)
        softmaxed_output = self._softmax_fn(stacked_output)

        # outputs:
        the_first_half = last_layer_activations[:, :, :512]
        the_right_half = last_layer_activations[:, :, 512:]

        tfh = the_first_half.cpu().detach().numpy()
        trh = the_right_half.cpu().detach().numpy()
        # print('min, max')
        # print(min(tfh), max(tfh))
        # print('std, mean, sum')
        # print(np.std(tfh),
        #       np.mean(tfh), np.sum(tfh))
        # print("the_first_half:")
        # print(tfh)
        # print(tfh.shape)
        # print("the_right_half:")
        # print(trh)
        # print(trh.shape)
        # # lla = last_layer_activations.cpu().detach().numpy()
        # print("the_right_half Activations")
        # lla_ravel = trh.ravel()
        # for each_num in lla_ravel:
        #     print(each_num, end=", ")
        # print("___")
        # # just test if it print all values
        # print(lla_ravel)
        # print("___")

        return softmaxed_output.cpu().detach().numpy()

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
