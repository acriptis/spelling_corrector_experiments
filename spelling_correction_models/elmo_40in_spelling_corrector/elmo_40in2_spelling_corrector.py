import numpy as np
from .elmo_40in_spelling_corrector import ELMO40inSpellingCorrector
from .helper_fns import estimate_the_best_s_hypotheses

# increment of the logit for merging 2tokens->1token:
ERROR_SCORE_FOR_MERGE = -2.0


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
        analysis_dict = self.prepare_analysis_dict_for_sentence(sentence)
        # Atomic Hypotheses space is constructed.
        # now we should compose them into sentence hypotheses with reduction of bad hypotheses
        output_sentence = self.make_fixes(analysis_dict, min_advantage_treshold=4.0)

        return output_sentence

    def make_fixes(self, analysis_dict, min_advantage_treshold=4.0):

        #  we may start construction of sentence hypotheses and reduct them
        hypotheses = estimate_the_best_s_hypotheses(analysis_dict,
                                                    min_advantage_treshold=min_advantage_treshold)
        # the_best:
        output_sentence = hypotheses[0].text

        # restore capitalization:
        output_sentence_tokens = self._lettercaser([analysis_dict['input_sentence'].split()],
                                                   [output_sentence.split()])[0]
        output_sentence = " ".join(output_sentence_tokens)
        return output_sentence

    def prepare_analysis_dict_for_sentence(self, sentence):
        """
        The method which produces analysis dictionary of the sentence, it generates
        substitution candidates of the segments of the input sentence.

        :param sentence: str
        :return: dict SentenceAnalysisDictionary
        """
        # preprocess
        preprocessed_sentence = self.preprocess_sentence(sentence)

        # calculate elmo data for the input sentence
        elmo_data = self.lm.analyze_sentence(sentence)

        # analyse sentence, atomic (token-token) hypotheses generation:
        analysis_dict = self.elmo_analysis_with_probable_candidates_reduction_dict_in_dict_out(
            {'input_sentence': preprocessed_sentence,
             'tokenized_input_sentence': self.lm.tokenize_sentence(preprocessed_sentence)},
            elmo_data)
        # TODO phonetic hypothese generation?

        # multi-token - token hypotheses generation
        merged_tokens_hypotheses_dict = self.generate_Nto1_hypotheses(
            analysis_dict['tokenized_input_sentence'], elmo_data)


        analysis_dict['word_substitutions_candidates'] += merged_tokens_hypotheses_dict
        return analysis_dict

    def generate_Nto1_hypotheses(self, wrapped_tokenized_sentence, elmo_data):
        """
        Given a tokenized sentence this method makes variation of tokens by merging two tokens
        into one and then populating dictionary with token hypothesys with specific
        token_span if it is a dictionary token.

        Another case is to make second variation by LevenshteinSearcherComponent

        TODO: make N->1 support. Now only 2->1 merges are hypothesised
        """
        token_hypotheses_dicts = []

        for tok_idx, each_tok in enumerate(wrapped_tokenized_sentence):
            if tok_idx <= 1 or tok_idx == len(wrapped_tokenized_sentence) - 1:
                # the 0's token is <s> the last is </s>
                continue

            if len(wrapped_tokenized_sentence[tok_idx-1])==1 or len(wrapped_tokenized_sentence[tok_idx])==1:
                # RULE of THUMB: don't merge words consisting of 1letter
                continue

            # simple merge hypothesis:
            merge_hypothesis_str = wrapped_tokenized_sentence[tok_idx - 1] + \
                                   wrapped_tokenized_sentence[tok_idx]
            source_segment_str = wrapped_tokenized_sentence[tok_idx-1] +" "+ wrapped_tokenized_sentence[tok_idx]
            ################################################################################
            # variate merged variant by levenshtein
            # print("Variate merged hypothesis: %s" % merge_hypothesis_str)
            candidates_lists = self.sccg([[merge_hypothesis_str]])

            candidates_list_for_token = candidates_lists[0][0]
            # print("candidates_list_for_token")
            # print(candidates_list_for_token)
            for each_merge_candidate_err_score, each_merge_candidate_str in candidates_list_for_token:
                idx, is_unk = self.lm.get_word_idx_or_unk(each_merge_candidate_str)
                if is_unk is False:
                    # merged token is known to dictionary, nice hypothesis
                    # print("Merged TokenHypothesis is in dictionary!")
                    token_start_index = tok_idx - 1
                    token_fin_index = tok_idx
                    # estimate probas for merged token
                    logit_probas = self.get_logit_probas_of_merged_token(elmo_data,
                                                                         each_merge_candidate_str,
                                                                         token_start_index,
                                                                         token_fin_index)
                    #################################
                    # Calculate base score of the span.
                    # Base score is a cumulative likelihood score of the input tokens which are
                    # related to merged one.
                    base_scores = []
                    for eac_tok_idx in range(token_start_index, token_fin_index + 1):
                        base_left_logit, base_right_logit = self.lm.retrieve_logits_of_particular_token(
                            elmo_data, eac_tok_idx, wrapped_tokenized_sentence[eac_tok_idx])
                            # elmo_data, tok_idx, wrapped_tokenized_sentence[tok_idx])
                        base_scores.append([base_left_logit, base_right_logit])
                    base_scores = np.array(base_scores)
                    summated_base_scores = base_scores.sum(axis=0)
                    # merge and variation may produce erroneous score: when merge and insertion
                    # occurs in the same position. So we calc true_levenshtein_distance
                    # from source segment.
                    # TODO And may be we need to rescore?
                    # TODO recalculate scores to avoid overscoring fixes like:
                    #   что нибудь -> (что-нибудь -6.0) because of merge + error score
                    true_lev_distance = self.sccg.searcher.transducer.distance(each_merge_candidate_str, source_segment_str)
                    error_score = ERROR_SCORE_FOR_MERGE + each_merge_candidate_err_score
                    #################################

                    lm_advantage = logit_probas.sum() - summated_base_scores.sum()
                    # advantage = lm_advantage + error_score * 2.0
                    advantage = lm_advantage + error_score
                    out_dict = {
                        'tok_idx': (token_start_index, token_fin_index),
                        'tok_idx_start': token_start_index,
                        'tok_idx_fin': token_fin_index,
                        # string of source
                        'source_segment_str': source_segment_str,

                        'top_k_candidates': [
                            {
                                'token_str': each_merge_candidate_str,
                                'token_merges': 1,
                                # 'error_score': ERROR_SCORE_FOR_MERGE,
                                'levenshtein_variation_err_score': each_merge_candidate_err_score,
                                # distance to source segment:
                                'levenshtein_distance': true_lev_distance,
                                # cumulative error (merges and levenshtein)
                                'error_score': error_score,
                                'lm_scores_list': logit_probas.tolist(),
                                'base_scores': base_scores.tolist(),
                                'summated_base_scores': summated_base_scores.tolist(),

                                'advantage': advantage,
                                'lm_advantage': lm_advantage
                            }
                        ]
                    }

                    token_hypotheses_dicts.append(out_dict)
                else:
                    # print("Skipping OOV hypothesis: %s" % each_merge_candidate_str)
                    pass
            # END variate merged variant by levenshtein
            ################################################################################

        return token_hypotheses_dicts

    def get_logit_probas_of_merged_token(self, elmo_data, token_str, token_start_index, token_fin_index):
        """
        allows to estimate likelihood of a token which constructs from a merge in a sentence
        (sentence is represented by ELMO data matrixretrieved from ELMO LM)
        """
        w_idx, is_unk = self.lm.get_word_idx_or_unk(token_str)
        if is_unk:
            print("Warning: measuring likelihood of the token which is Out-of-vocabulary for "
                  "ELMO LM! This may reduce precision! Token: %s" % token_str)
        # TODO check if right and left probas are correctly located:
        left_logit_prob = np.log10(elmo_data[token_start_index, 0, w_idx])
        right_logit_prob = np.log10(elmo_data[token_fin_index, 1, w_idx])
        # TODO add support for multiple spans of merges
        return np.array([left_logit_prob, right_logit_prob])

    @staticmethod
    def fixes_maker(analysis_data, max_num_fixes=5, fix_treshold=10.0, remove_s=True):
        raise Exception("Implement me!")

