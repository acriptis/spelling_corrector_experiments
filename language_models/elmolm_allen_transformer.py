from language_models.base_elmo_lm import BaseELMOLM
from bilm.data import UnicodeCharsVocabulary
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers.token import Token
from allennlp.modules.elmo import batch_to_ids
import torch
import numpy as np


class AllenElmoTransformerLM(BaseELMOLM):
    """
    Implementation of ELMO LM on torch, AllenNLP and with Transformers layer
    """

    def __init__(self):
        self.load_model()

        # read vocabulary
        #         path_to_vocab = ROOT_DIR + "/bidirectional_lms/elmo_ru_news/tokens_set.txt"
        self.path_to_vocab = "/home/alx/Cloud/spell_corr/allennlp_lms/vocabulary/tokens.txt"
        self._lm_vocab = UnicodeCharsVocabulary(self.path_to_vocab, 200)

        self.words = self._lm_vocab._id_to_word
        self.word_index = {word: i for i, word in enumerate(self.words)}

        # index of unknown token:
        self.IDX_UNK_TOKEN = self.word_index.get("<UNK>")

    def load_model(self):
        import os
        pwd = "/home/alx/Cloud/spell_corr/allennlp_lms"
        os.environ['BIDIRECTIONAL_LM_DATA_PATH'] = "%s'/large_ru_texts_dataset'" % pwd
        os.environ['BIDIRECTIONAL_LM_TRAIN_PATH'] = "$BIDIRECTIONAL_LM_DATA_PATH'/train/*'"
        os.environ['BIDIRECTIONAL_LM_VOCAB_PATH'] = "%s/vocabulary" % pwd

        from allennlp.common import Params
        from allennlp.training.trainer_pieces import TrainerPieces
        params = Params.from_file(
            "/home/alx/Cloud/spell_corr/allennlp_lms/configs/tranformer_lm_bidirectional_language_model.jsonnet")
        pieces = TrainerPieces.from_params(params, "/home/alx/Cloud/spell_corr/allennlp_lms/temp")
        self._pieces_model = pieces.model

        self._ff = torch.nn.Linear(512, 1000001)
        # ff.cuda()
        self._ff.load_state_dict(
            {'weight': self._pieces_model._softmax_loss._modules['softmax_w']._parameters['weight'],
             'bias': self._pieces_model._softmax_loss._modules['softmax_b']._parameters['weight'][:,
                     0]
             },
            strict=False)
        self._softmax_fn = torch.nn.Softmax(dim=3)

    #         print(pieces.model)
    #         pieces.model._softmax_loss._modules['softmax_w']._parameters['weight']
    #         pieces.model._softmax_loss._modules['softmax_w']._parameters['weight'].shape
    #         # /bias exists but it is zero
    #         pieces.model._softmax_loss._modules['softmax_b']._parameters['weight']
    # options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    # options_file = "http://files.deeppavlov.ai/lang_models/sexy_elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    #         options_file = "http://files.deeppavlov.ai/lang_models/sexy_elmo/options.json"
    # weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    # custom weights
    # weight_file = "http://files.deeppavlov.ai/lang_models/sexy_elmo/weights.hdf5"
    # weight_file = "http://files.deeppavlov.ai/lang_models/sexy_elmo/weights_epoch_n_2.hdf5"
    #         weight_file = "http://files.deeppavlov.ai/lang_models/sexy_elmo/weights_epoch_n_3.hdf5"
    # weight_file = "http://files.deeppavlov.ai/lang_models/sexy_elmo/weights_epoch_n_4.hdf5"

    # allennlp realizatioon with updating states
    # self._elmobilm = _ElmoBiLm(options_file, weight_file)
    # realizatioon without updating states
    #         self._elmobilm = ELMOBiLM(options_file, weight_file)
    #         self._elmobilm.cuda()

    #         # TODO load head:
    #         # self._ff = torch.nn.Linear(1024, 1000000)
    #         self._ff = torch.nn.Linear(512, 1000000)
    #         self._ff.cuda()

    #         # TODO refactor
    #         # Load checkpoint of TF:
    #         base_path = ROOT_DIR + "/bidirectional_lms/elmo_ru_news"
    #         ckpt_prefixed_path = base_path + "/model.ckpt-0003"
    #         # metafile_path = base_path + "/model.ckpt-0003.meta"

    #         # tf.train.list_variables(ckpt_prefixed_path)

    #         # matrix which holds embedding into words projection
    #         emb2words_w_matrix = tf.train.load_variable(ckpt_prefixed_path, 'lm/softmax/W')

    #         # torch_w = torch.from_numpy(np.concatenate((softmax_w, softmax_w), axis=1))
    #         torch_w = torch.from_numpy(emb2words_w_matrix)

    #         emb2words_bias = tf.train.load_variable(ckpt_prefixed_path, 'lm/softmax/b')
    #         self._ff.load_state_dict(
    #             {'weight': torch_w, 'bias': torch.from_numpy(emb2words_bias)},
    #             strict=False)

    #         self._softmax_fn = torch.nn.Softmax(dim=3)

    def elmo_lm(self, tokenized_sentences):
        """
        Main method which returns an ELMO matrix
        :param tokenized_sentences: list of tokenized sentences.
        :return: tensor BATCH_SIZE x TOKENS_NUM x 2 x 1000001
        """

        #         lm_model_file = "/home/alx/Cloud/spell_corr/allennlp_lms/pretrained_models/model.tar.gz"

        #         sentence = "Дождь идет на улице ."
        #         tokens = [Token(word) for word in sentence.split()]

        #         lm_embedder = BidirectionalLanguageModelTokenEmbedder(
        #             archive_file=lm_model_file,
        #             dropout=0.2,
        #             bos_eos_tokens=["<S>", "</S>"],
        #             remove_bos_eos=True,
        #             requires_grad=False
        #         )

        indexer = ELMoTokenCharactersIndexer()
        #         vocab = lm_embedder._lm.vocab
        vocab = self._lm_vocab
        character_indices = []

        character_indices = batch_to_ids(tokenized_sentences)

        #         for each_tok_sent in tokenized_sentences:
        #             torch_tokenization = [Token(word) for word in each_tok_sent]
        #             character_indices.append(
        #                 indexer.tokens_to_indices(torch_tokenization, vocab, "elmo")["elmo"])

        # Batch of size 1
        #         print(tokenized_sentences)
        #         print(character_indices)

        indices_tensor = torch.LongTensor(character_indices)

        res = self._pieces_model({'token_characters': indices_tensor})

        forward_embeddings, backward_embeddings = res['lm_embeddings'].chunk(2, -1)

        left_results = self._ff(forward_embeddings)
        right_results = self._ff(backward_embeddings)
        stacked_output = torch.stack((left_results, right_results), dim=2)
        softmaxed_output = self._softmax_fn(stacked_output)

        return softmaxed_output.detach().numpy()

    #         character_ids = batch_to_ids(tokenized_sentences)
    #         # print(character_ids.shape)
    #         # print(character_ids)
    #         character_ids = character_ids.cuda()
    #         elmo_output = self._elmobilm(character_ids)
    #         # TODO check correctness:
    #         last_layer_activations = elmo_output['activations'][2]
    #         # results = self._ff(last_layer_activations)
    #         # print("last_layer_activations:")
    #         # print(last_layer_activations)
    #         # print(last_layer_activations.shape)
    #         # pre_last_layer_activations = elmo_output['activations'][1]
    #         # print("pre_last_layer_activations:")
    #         # print(pre_last_layer_activations)
    #         # print(pre_last_layer_activations.shape)

    #         # base_layer_activations = elmo_output['activations'][0]
    #         # print("base_layer_activations :")
    #         # print(base_layer_activations )
    #         # print(base_layer_activations .shape)
    #         # print(lstm_outputs[0][0].shape)

    #         # left_activations, right_activations = torch.split(last_layer_activations, 512, dim=1)
    #         splitted_tensors = torch.split(last_layer_activations, 512, dim=2)
    #         # import ipdb; ipdb.set_trace()

    #         assert len(splitted_tensors) == 2
    #         # print("len(splitted_tensors)")
    #         # print(len(splitted_tensors))
    #         left_activations = splitted_tensors[0]
    #         right_activations = splitted_tensors[1]
    #         left_results = self._ff(left_activations)
    #         right_results = self._ff(right_activations)
    #         stacked_output = torch.stack((left_results, right_results), dim=2)
    #         softmaxed_output = self._softmax_fn(stacked_output)

    #         # outputs:
    #         the_first_half = last_layer_activations[:, :, :512]
    #         the_right_half = last_layer_activations[:, :, 512:]

    #         tfh = the_first_half.cpu().detach().numpy()
    #         trh = the_right_half.cpu().detach().numpy()
    #         # print('min, max')
    #         # print(min(tfh), max(tfh))
    #         # print('std, mean, sum')
    #         # print(np.std(tfh),
    #         #       np.mean(tfh), np.sum(tfh))
    #         # print("the_first_half:")
    #         # print(tfh)
    #         # print(tfh.shape)
    #         # print("the_right_half:")
    #         # print(trh)
    #         # print(trh.shape)
    #         # # lla = last_layer_activations.cpu().detach().numpy()
    #         # print("the_right_half Activations")
    #         # lla_ravel = trh.ravel()
    #         # for each_num in lla_ravel:
    #         #     print(each_num, end=", ")
    #         # print("___")
    #         # # just test if it print all values
    #         # print(lla_ravel)
    #         # print("___")

    #         return softmaxed_output.cpu().detach().numpy()

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


class AllenElmoTransformerLM2(AllenElmoTransformerLM):
    """
    Here we actually load the model from model.tar.gz file
    """
    def __init__(self):
        self.load_model()

        # read vocabulary
        #         path_to_vocab = ROOT_DIR + "/bidirectional_lms/elmo_ru_news/tokens_set.txt"
        self.path_to_vocab = "/home/alx/Cloud/spell_corr/allennlp_lms/vocabulary/tokens.txt"
        self._lm_vocab = UnicodeCharsVocabulary(self.path_to_vocab, 200)

        self.words = self._lm_vocab._id_to_word
        self.word_index = {word: i for i, word in enumerate(self.words)}

        # index of unknown token:
        self.IDX_UNK_TOKEN = self.word_index.get("<UNK>")

    def load_model(self):
        from allennlp.models.archival import load_archive
        path_to_model_targz = "/home/alx/Cloud/spell_corr/allennlp_lms/pretrained_models/model_1.tar.gz"
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
