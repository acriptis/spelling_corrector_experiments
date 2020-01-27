from .elmo_40in2_spelling_corrector import ELMO40in2SpellingCorrector
from reranker.reranker_40in import ReRanker40inRegressor
from language_models.utils import detokenize
################# Universal Import ###################################################
import sys
import os

SELF_DIR = os.path.dirname(os.path.abspath(__file__))
PREROOT_DIR = os.path.dirname(SELF_DIR)
ROOT_DIR = os.path.dirname(PREROOT_DIR)
# print(ROOT_DIR)
sys.path.append(ROOT_DIR)
###########################################################


class ELMO40in2RerankingSpellingCorrector(ELMO40in2SpellingCorrector):
    """This Spelling corrector reimplements algorithm for making fixes by using trainable reranker
    instead of simple summation of lm_score and err_score

    Model fallbacks to simple summation decisioning if reranker model is not available

    ReRanker model is applicable in conjunction with error model and language model.
    """
    def __init__(self, language_model=None, spelling_correction_candidates_generator=None,
                 fix_treshold=10.0, max_num_fixes=5, data_path=None, mini_batch_size=None,
                 frozen_words_regex_patterns=['тинькоф+', "греф", "писец", "кв\.", "м\.", "квортира", "пирдуха"],
                 reranker_model=None, *args, **kwargs):
        super().__init__(language_model, spelling_correction_candidates_generator,
                         fix_treshold, max_num_fixes, data_path, mini_batch_size,
                         frozen_words_regex_patterns, *args, **kwargs)

        #####################################################################################
        ############### set up reranker >##########################################################
        if not reranker_model:
            # init ReRanker

            # load model of reranker

            # relative to root of project:
            if self.lm.__class__.__name__ == "ELMOLMTorch":
                path_to_reranker_weights = ROOT_DIR + "/.rerankers/elmo_torch/elmo_torch_reranker_classifier.pkl"
            elif self.lm.__class__.__name__ == "ELMOLM":
                path_to_reranker_weights = ROOT_DIR + "/.rerankers/elmo_torch/elmo_kuz_reranker_classifier.pkl"
            else:
                path_to_reranker_weights = None
            self.reranker = ReRanker40inRegressor()

            try:
                self.reranker.load(path_to_reranker_weights)
                print("Rernaker is loaded succesfully!")
            except Exception as e:
                # no reranker model on local machine try to download.
                # download or init with summator?

                # 1. variant to download rernaker for pytorch model?
                from deeppavlov.core.data.utils import download
                # but we should assure that language model is pytorch model then.
                # TODO do we need a factory to produce objects?
                # hack:
                if self.lm.__class__.__name__=="ELMOLMTorch":
                    # load rernaker for it:
                    RERANKER_WEIGHTS_URL = "http://files.deeppavlov.ai/spelling_correctors/rerankers/elmo_torch_reranker_classifier.pkl"
                    download(path_to_reranker_weights, RERANKER_WEIGHTS_URL)
                    self.reranker.load(path_to_reranker_weights)
                    print("Rernaker for ELMOLMTorch initialized.")
                elif self.lm.__class__.__name__=="ELMOLM":
                    RERANKER_WEIGHTS_URL = "http://files.deeppavlov.ai/spelling_correctors/rerankers/elmo_kuz_reranker_classifier.pkl"
                    download(path_to_reranker_weights, RERANKER_WEIGHTS_URL)
                    self.reranker.load(path_to_reranker_weights)
                    print("Rernaker for ELMOLM Kuz initialized.")
                else:
                    # we don't know where to get the weights of the model
                    print("Can not load rernaker model at path: %s !" % path_to_reranker_weights)
                    print("Using basic summator as rernaker")
                    self.make_fixes = ELMO40in2SpellingCorrector.make_fixes

        #####################################################################################
        ############### END reranker set up  ######################################################

    def make_fixes(self, analysis_dict, *args, **kwargs):
        """
        Method that actually makes fixes of anlyzed sentence with ReRanker
        :param analysis_dict:
        :param args:
        :param kwargs:
        :return: string with corrected sentence
        """
        output_sentence_tokens = self.reranker.predict_fixes_tokens(analysis_dict)

        # restore capitalization:
        tokenized_input_sentence_with_s_wrap = analysis_dict['tokenized_input_sentence']
        tokenized_input_sentence = tokenized_input_sentence_with_s_wrap[1:len(tokenized_input_sentence_with_s_wrap)-2]
        output_sentence_tokens = self._lettercaser([tokenized_input_sentence],
                                                   [output_sentence_tokens])[0]

        output_sentence = detokenize(output_sentence_tokens)
        return output_sentence
