from .elmo_40in2_spelling_corrector import ELMO40in2SpellingCorrector
from reranker.reranker_40in import ReRanker40inRegressor
from language_models.utils import detokenize


class ELMO40in2RerankingSpellingCorrector(ELMO40in2SpellingCorrector):
    """This Spelling corrector reimplements algorithm for making fixes by using trainable reranker
    instead of simple summation of lm_score and err_score"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # init ReRanker

        # load model
        self.reranker = ReRanker40inRegressor()

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
