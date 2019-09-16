from .elmo_40in_spelling_corrector import ELMO40inSpellingCorrector

class ELMO40in2SpellingCorrector(ELMO40inSpellingCorrector):
    """
    This class is featured with improved hypothesis generation which may
    merge 2 tokens into one
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_sentence(self, sentence):
        """
        Interface method for sentence correction.
        Given a sentence as string anlyze it, fix it and output the best hypothesis
        :param sentence: str
        :return: str, sentence with corrections
        """
        # preprocess
        preprocessed_sentence = self.preprocess_sentence(sentence)

        # analyse sentence, atomic (token-token) hypotheses generation:
        analysis_dict = self.elmo_analysis_with_probable_candidates_reduction_dict_out(
            preprocessed_sentence)


        # TODO add support of Nto1 Hypotheses generator which updates analysis dict
        # multi-token - token hypotheses generation
        merged_tokens_hypotheses_dict = self.generate_Nto1_hypotheses(
            analysis_dict['tokenized_input_sentence'])

        # TODO merge analysis_dict with merged_tokens_hypotheses_dict

        # TODO reimplement fixes maker!
        # implement the best fixes
        output_sentence = self.fixes_maker(analysis_dict, max_num_fixes=self.max_num_fixes,
                                           fix_treshold=self.fix_treshold)

        # restore capitalization:
        output_sentence = self._lettercaser([sentence.split()], [output_sentence.split()])

        return output_sentence

    def _merge_dicts_with_atomic_hypotheses(self, dict1, dict2):
        """
        Code which helps to merge hypotheses dicts with careful handling of
        word_substitutions_candidates

        :param dict1:
        :param dict2:
        :return:
        """

        # we assert that key indexes does not overlap. So dict1 has tok_idx that are integers and
        # dict2 has tok_idx that are tuples of integers (2to1 merges)
        dict1['word_substitutions_candidates']


    def generate_Nto1_hypotheses(self, wrapped_tokenized_sentence):
        """
        Given a tokenized sentence this method makes variation of tokens by merging two tokens
        into one and then populating dictionary with token hypothesys with specific
        token_span if it is a dictionary token.

        Another case is to make second variation by LevenshteinSearcherComponent

        """
        token_hypotheses_dicts = []

        MAX_MERGE = 3
        for tok_idx, each_tok in enumerate(wrapped_tokenized_sentence):
            if tok_idx <= 1 or tok_idx == len(wrapped_tokenized_sentence) - 1:
                # the 0's token is <s> the last is </s>
                continue

            merge_hypothesis_str = wrapped_tokenized_sentence[tok_idx - 1] + \
                                   wrapped_tokenized_sentence[tok_idx]
            # TODO check if merged word in vocabulary
            idx, is_unk = self.lm.get_word_idx_or_unk(merge_hypothesis_str)
            if is_unk is False:
                # merged token is known to dictionary, nice hypothesis
                print("Merged TokenHypothesis is in dictionary!")
                out_dict = {
                    'tok_idx': (tok_idx - 1, tok_idx),
                    'tok_idx_start': tok_idx-1,
                    'tok_idx_fin': tok_idx,

                    'top_k_candidates': [
                        {
                            'token_str': merge_hypothesis_str,
                            'token_merges': 1,
                            #                         'advantage': None,
                            'error_score': -4.0

                        }
                    ]
                }

                token_hypotheses_dicts.append(out_dict)
            else:
                # merged hypothesis is not in dictionary. Any actions?
                # TODO if no we can variate it with Levenshtein?
                # TODO apply rule based statistical substitutions? чтонить -> что-нибудь etc
                pass

        return token_hypotheses_dicts

    @staticmethod
    def fixes_maker(analysis_data, max_num_fixes=5, fix_treshold=10.0, remove_s=True):
        raise Exception("Implement me!")


class WordSubstitutionCandidatesManager():
    def find_by_tok_index(self, tok_index):
        pass

    def is_span_in_index(self, span):
        pass