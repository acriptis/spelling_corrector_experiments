################# Universal Import ###################################################
import sys
import os

SELF_DIR = os.path.dirname(os.path.abspath(__file__))
PREROOT_DIR = os.path.dirname(SELF_DIR)
ROOT_DIR = os.path.dirname(PREROOT_DIR)

print(ROOT_DIR)
sys.path.append(ROOT_DIR)
# #####################################################
from lettercaser import LettercaserForSpellchecker
from language_models.elmolm_from_config import ELMOLM
from dp_components.levenshtein_searcher import LevenshteinSearcherComponent
import numpy as np
import re
from copy import deepcopy
DATA_PATH = "/home/alx/Cloud/spell_corr/py_spelling_corrector/data/"

def clean_dialog16_sentences_from_punctuation(sentences):
    """
    # we need to remove:
    # ?,() !;"
    # . if it not in the middle of word
    # - if it not in the middle of word
    # : if it not in the middle of word

    """
    output_sentences = []
    for each_s in sentences:
        sentence = each_s.translate(str.maketrans("", "", '?,()!;"'))
        # clean from "..."
        sentence = re.sub(r'\.\.\.', '', sentence)
        if sentence[-1] == ".":
            sentence = sentence[:-1]
        # clean from "."
        tokens = sentence.split()

        postprocessed_tokens = []
        for tok_idx, each_tok in enumerate(tokens):
            if each_tok[-1] in ":-":
                postprocessed_tokens.append(each_tok[:-1])
            else:
                postprocessed_tokens.append(each_tok)

        reassembled_sentence = " ".join(postprocessed_tokens)

        # finally we need to remove excessive spaces?
        reassembled_sentence = re.sub(" {2,}", " ", reassembled_sentence)
        output_sentences.append(reassembled_sentence)
    return output_sentences

class ELMO40inSpellingCorrector():
    """
    Spelling corrector based on ELMO language model
    """

    def __init__(self, language_model=None, spelling_correction_candidates_generator=None,
                 fix_treshold=10.0, max_num_fixes=5):
        self._lettercaser = LettercaserForSpellchecker()
        if language_model:
            self.lm = language_model
        else:
            self.lm = self._init_elmo()

        if spelling_correction_candidates_generator:
            self.sccg = spelling_correction_candidates_generator
        else:
            self.sccg = self._init_sccg()

        # max allowed number of fixes in a sentence
        self.max_num_fixes = max_num_fixes

        # minimal likelihood advantage treshold for fixing the sentence
        self.fix_treshold = fix_treshold

    def _init_elmo(self):
        """
        Initilize default ELMO LM if no specification was provided in configuration
        :return: ELMOLM instance
        """
        # TODO: azat substitute please with ELMO_inference component
        elmo_config = {
            "chainer": {
                "in": [
                    "sentences"
                ],
                "pipe": [
                    {
                        "in": ["sentences"],
                        "class_name": "lazy_tokenizer",
                        "out": ["tokens"]
                    },
                    {
                        "class_name": "elmo_bilm",
                        "mini_batch_size": 10,
                        "in": [
                            "tokens"
                        ],
                        "model_dir": "bidirectional_lms/elmo_ru_news",
                        "out": [
                            "pred_tokens"
                        ]
                    }
                ],
                "out": [
                    "pred_tokens"
                ]
            },
            "metadata": {
                "requirements": [
                    "../dp_requirements/tf.txt",
                    "../dp_requirements/elmo.txt"
                ],
                "download": [
                    {
                        "url": "http://files.deeppavlov.ai/deeppavlov_data/lm_elmo_ru_news.tar.gz",
                        "subdir": "bidirectional_lms/"
                    }
                ]
            }
        }
        instance = ELMOLM(elmo_config)
        return instance

    def _init_sccg(self):
        """
        Initilizes spelling correction candidates generator to generate correction candidates
        :return: instance of spelling correction candidates generator
        """
        # TODO refactor with dynamic dictionary
        path_to_dictionary = DATA_PATH + "compreno_wordforms.txt"

        # path_to_dictionary = DATA_PATH + "russian_words_vocab.dict"

        with open(path_to_dictionary, "r") as dict_file:
            words_dict = dict_file.read().splitlines()
        lsc = LevenshteinSearcherComponent(words=words_dict)
        return lsc

    def preprocess_sentence(self, sentence):
        # lowercase
        lowercased_sentence = sentence.lower()
        # TODO depunctuate
        return lowercased_sentence

    def fix_sentence(self, sentence):
        """
        given a sentence as string anlyze it, fix it and output the best hypothesis
        :param sentence: str
        :return: str, sentence with corrections
        """
        # preprocess
        preprocessed_sentence = self.preprocess_sentence(sentence)

        # analyse sentence
        analysis_dict = self.elmo_analysis_with_probable_candidates_reduction_dict_out(preprocessed_sentence)

        # implement the best fixes
        output_sentence = self.fixes_maker(analysis_dict, max_num_fixes=self.max_num_fixes,
                                           fix_treshold=self.fix_treshold)

        # restore capitalization:
        output_sentence = self._lettercaser([sentence.split()], [output_sentence.split()])

        return output_sentence

    def elmo_analysis_with_probable_candidates_reduction_dict_out(self, sentence):
        """
        Given a sentence this method analyzes it and returns an analysis dictionary
        with hypotheses of the best substitutions (as scored lists for each token).

        The analysis dictionary allows to make parametrized hypothesis selection at the next stage.

        Example of Output:
        {
            "input_sentence": "...",
            "tokenized_input_sentence": ['<S>',
                                              'обломно',
                                              'но',
                                              'не',
                                              'сдал',
                                              'горбачева',
                                              'но',
                                              'хочу',
                                              'сдать',
                                              'последний',
                                              'экзам',
                                              'на',
                                              '5',
                                              'тогда',
                                              'буит',
                                              'возможно',
                                              'хоть',
                                              'ченить',
                                              'выловить',
                                              'на',
                                              'горбачеве',
                                              '</S>']
            "word_substitutions_candidates": [
                {'tok_idx': 0,
                'top_k_candidates': [
                        {'token_str': обломно,
                        'advantage_score': 20.0
                        },
                        {'token_str': лапа,
                        'advantage_score': 21.0
                        }

                    ]
                },
                {'tok_idx': 2,
                'top_k_candidates': [
                        {'token_str': но,
                        'advantage_score': 20.1
                        },
                        {'token_str': калал,
                        'advantage_score': 21.3
                        }

                    ]
                }

            ]

        """
        result_data_dict = {
            'input_sentence': sentence
        }
        tok_wrapped = self.lm.tokenize_sentence(sentence)
        #     toks_unwrapped = tok_wrapped[1:-1]
        result_data_dict['tokenized_input_sentence'] = tok_wrapped

        elmo_data = self.lm.analyze_sentence(sentence)
        # elmo data array contains a ndarray of size: [1, len(sentence tokens), 1000000]
        candidates_lists = self.sccg([tok_wrapped])
        # find the best substitutions in sentence from candidates sets
        candidates_list_for_sentence = candidates_lists[0]
        base_scores = self.lm.trace_sentence_probas_in_elmo_datas_batch([elmo_data], [tok_wrapped])
        log_probas_base = np.log(base_scores)
        # summated_probas_base = log_probas_base.sum(axis=1)
        # TODO check if it is not necessary?
        summated_probas_base = log_probas_base.sum()

        # for each candidate_list by levenshtein find top_k hypothese of susbstitutions in ELMO data
        word_substitutions_candidates = [{'tok_idx': idx, 'top_k_candidates': []} for idx, _ in
                                         enumerate(candidates_list_for_sentence)]

        #     for candi_idx, each_candidates_list in enumerate(candidates_list_for_sentence):
        #         # find scores in elmo data
        #         pass

        for tok_idx, input_token in enumerate(tok_wrapped):
            if tok_idx == 0:
                continue

            # retieve the best candidates
            # 1. retrive best from levenshtein list
            levenshtein_candidates_for_current_token = candidates_list_for_sentence[tok_idx]
            base_left_logit, base_right_logit = self.lm.retrieve_logits_of_particular_token(elmo_data,
                                                                                         tok_idx,
                                                                                         tok_wrapped[
                                                                                             tok_idx])
            base_summa = base_left_logit + base_right_logit
            # 2. retrieve absolute best for the position
            for each_candidate in levenshtein_candidates_for_current_token:
                # retrieve advantage
                #             print(each_candidate)
                candidate_str = each_candidate[1]
                # error score in logits for substitution input into corrected hypohesis:
                error_score = each_candidate[0]
                # TODO use  error_score for SCCG which can generate distant fixes

                left_logit, right_logit = self.lm.retrieve_logits_of_particular_token(elmo_data,
                                                                                   tok_idx,
                                                                                   candidate_str)
                # todo add error score?
                advantage_score = -base_summa + left_logit + right_logit

                if advantage_score >= 0:
                    word_substitutions_candidates[tok_idx]['top_k_candidates'].append({
                        "advantage": advantage_score,
                        "token_str": candidate_str
                    })

        result_data_dict['word_substitutions_candidates'] = word_substitutions_candidates
        return result_data_dict

    @staticmethod
    def fixes_maker(analysis_data, max_num_fixes=5, fix_treshold=10.0, remove_s=True):
        """
        Function which actually makes spelling correction based on analysis of sentence with data
        about candidates.

        Outputs corrected sentence as a string.

        :param remove_s: if true then output string contains no <s> and </s> markers in output
        """

        tokens = deepcopy(analysis_data['tokenized_input_sentence'])

        best_substitutions_list = [
            {'tok_idx': tok_idx, 'best_candidate': each_tok, 'advantage': 0.0}
            for tok_idx, each_tok in enumerate(tokens)]

        # for all correction candidates find top-k fixes
        for tok_idx, each_candidates_list in enumerate(
                analysis_data['word_substitutions_candidates']):
            if len(each_candidates_list['top_k_candidates']) > 0:
                sorted_candidates_list = sorted(each_candidates_list['top_k_candidates'],
                                                key=lambda x: x['advantage'], reverse=True)
                best_candidate_dict = sorted_candidates_list[0]

                # filter by treshold
                if best_candidate_dict['advantage'] > fix_treshold:
                    best_substitutions_list[tok_idx]['best_candidate'] = best_candidate_dict[
                        'token_str']
                    best_substitutions_list[tok_idx]['advantage'] = best_candidate_dict['advantage']

        # now we have advantages
        # we should select top-k
        sorted_best_substitutions = sorted(best_substitutions_list, key=lambda x: x['advantage'],
                                           reverse=True)
        top_k_substitutions = sorted_best_substitutions[:max_num_fixes + 1]

        for each_substitution_element in top_k_substitutions:
            tokens[each_substitution_element['tok_idx']] = each_substitution_element[
                'best_candidate']

        if remove_s:
            output_str = " ".join(tokens[1:-1])
        else:
            output_str = " ".join(tokens)
        return output_str

    ##############################################################################################
    def __call__(self, input_sentences_batch):
        # TODO make optimized parallelization
        # optimization must be done at stage of ELMO calculation + analysis_dict construction
        #
        return [self.fix_sentence(each_sentence) for each_sentence in input_sentences_batch]


if __name__ == '__main__':

    sc = ELMO40inSpellingCorrector()
    print(sc(['Мама мыла раду']))
    # print(sc(['Тут есть КТО НИБУДЬ', 'тут есть кто-нибудь']))
    # print(sc(['Это происходит По сейдень', 'это происходит посей день']))
    # print(sc(['По-моему', 'по моему']))