import datetime as dt
import numpy as np
from rusenttokenize import ru_sent_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from .elmo_40in_spelling_corrector import ELMO40inSpellingCorrector
from .helper_fns import estimate_the_best_s_hypotheses
from language_models.utils import detokenize
# increment of the logit for merging 2tokens->1token:
ERROR_SCORE_FOR_MERGE = -2.0

# Size of batch in elmo lm:
ELMO_BATCH_SIZE = 8

# batch size measured in tokens count in sentences of the batch
ELMO_BATCH_TOKEN_SIZE = 500
# TODO fix duplicated parameter. Need tor etrireve batch size from elmo lm model?


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
        # output_sentence_tokens = self._lettercaser([analysis_dict['input_sentence'].split()],
        #                                            [output_sentence.split()])[0]
        output_sentence_tokens = self._lettercaser([word_tokenize(analysis_dict['input_sentence'])],
                                                   [word_tokenize(output_sentence)])[0]


        # output_sentence = " ".join(output_sentence_tokens)
        output_sentence = detokenize(output_sentence_tokens)
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

    # #######################################################################################
    # BATCHY OPTIMIZED METHODS
    @staticmethod
    def chunk_generator(items_list, chunk_size):
        """
        Method to slice batches into chunks of minibatches
        """
        for i in range(0, len(items_list), chunk_size):
            yield items_list[i:i + chunk_size]

    @staticmethod
    def chunk_generator_token_weighted(tokenized_sentences_list, max_tokens_count=200):
        """
        Method to slice batches into chunks of minibatches weighted by cumulative tokens count.

        In contrast to basic chunk generator which just yelds batches of the same number of
        sentences this method yelds batches of variable number of sentences but with approximately
        the same number of tokens.

        Algorithm:
        given a max number of tokens in batch we populate a batch with sentences and accumulating
        total number of tokens, if next sentence overflows numbsdsd...
        """
        tokens_accumulator = 0
        d_batch = []
        # TODO check that one sentence is not longer than max_tokens_count

        for each_sent in tokenized_sentences_list:
            if len(each_sent) + tokens_accumulator > max_tokens_count:
                # yeld!
                if len(d_batch) > 0:
                    print("dynamic batch size: " + str(len(d_batch)))
                    yield d_batch
                    d_batch = [each_sent]
                    tokens_accumulator = len(each_sent)

                else:
                    print(each_sent)
                    d_batch = [each_sent]
                    print("Sentence is longer than max_tokens_count?")
                    yield d_batch
                    tokens_accumulator = 0
                    d_batch = []

                    # raise Exception("Sentence is longer than max_tokens_count? Empty batch!")
            else:
                # if we fit constraints then append the sentence to the dynamic batch
                d_batch.append(each_sent)
                tokens_accumulator += len(each_sent)
        if len(d_batch)>0:
            print("dynamic batch size: " + str(len(d_batch)))
            yield d_batch

    def process_sentences_batch(self, sentences, min_advantage_treshold=1.0, max_tokens_count=None,
                                token_length_sorting=True, supply_anal_dict=False,
                                multisentences=False):
        """
        Interface method for batchy estimation of corrections
        Given a batch of sentences it returns a batch of corrections
        :param sentences: list of input sentences
        :param supply_anal_dict: if true then returns anal dicts for all sentences, used for
            debugging and analysis of errors
        :param multisentences: if true then each element of batch may be a multiple sentence string,
            so we preprocess them by splitting into sentences.
        :return: list of corrected sentences
        """

        if not max_tokens_count:
            max_tokens_count=ELMO_BATCH_TOKEN_SIZE
        # ###############################################################################
        if multisentences:
            # TODO make as function decorator?
            # we expect each input string is a multisentence, we need to split them into sentences
            # and then process it, after that we need to restore sentence structure by joining
            # sentences that occured in the same input string.

            # list of elementary sentences which are retrieved from input by sentence splitting:
            flat_sents_list = []
            # list which stores how many consequent sentences are must be joined into one input string
            list_of_lengths = []
            for each_string in sentences:
                el_sents = ru_sent_tokenize(each_string)
                list_of_lengths.append(len(el_sents))

                flat_sents_list += el_sents

            sentences = flat_sents_list
        # ###############################################################################
        if token_length_sorting:
            # TODO make as function decorator?
            # sort sentences by token length
            # and save initial order
            # TODO use nltk tokenizer insted of split? - no speed advantage
            # results = sorted(enumerate(sentences), key=lambda x: len(x[1].split()), reverse=True)
            results = sorted(enumerate(sentences), key=lambda x: len(word_tokenize(x[1])), reverse=True)
            source_idxs, sentences = zip(*results)

        # ###############################################################################

        start_dt = dt.datetime.now()
        anal_dicts = self.prepare_analysis_dict_for_sentences_batch(
            sentences, max_tokens_count=max_tokens_count)
        middle_dt = dt.datetime.now()
        print("datetimes. calculation of elmo analysis dicts: %s" % (str(middle_dt - start_dt)))
        output_sentences = []

        for sent_idx, each_data in enumerate(anal_dicts):

            sentence_hypothesis = self.make_fixes(each_data, min_advantage_treshold)
            # Letter caser:
            # restore capitalization:
            # output_sentence_tokens = self._lettercaser([each_data['input_sentence'].split()],
            #                                            [sentence_hypothesis.split()])[0]
            output_sentence_tokens = self._lettercaser([word_tokenize(each_data['input_sentence'])],
                                                       [word_tokenize(sentence_hypothesis)])[0]

            # output_sentence = " ".join(output_sentence_tokens)
            output_sentence = detokenize(output_sentence_tokens)

            # output_sentences.append(sentence_hypothesis)
            output_sentences.append(output_sentence)
        fin_dt = dt.datetime.now()
        print("datetimes. making_fixes: %s" % (str(fin_dt-middle_dt)))
        print("datetimes. total calculation time: %s" % str(fin_dt-start_dt))
        # ###############################################################################
        # restore ordering:
        # to return results in the same order
        if token_length_sorting:
            output_sentences_2 = [None]*len(sentences)
            anal_dicts_sorted = [None]*len(sentences)
            for idx, each_sentence in enumerate(output_sentences):
                output_sentences_2[source_idxs[idx]] = each_sentence
                anal_dicts_sorted[source_idxs[idx]] = anal_dicts[idx]
            anal_dicts = anal_dicts_sorted
            output_sentences = output_sentences_2

        # ###############################################################################
        if multisentences:

            merged_sentences = []
            merged_anal_dicts = []
            sent_offset = 0
            for each_length in list_of_lengths:
                # join each_length  of output sentences into one and continue
                merged_sentence = " ".join(output_sentences[sent_offset:sent_offset+each_length])

                merged_anal_dict_for_merged_sentences = anal_dicts[sent_offset:sent_offset+each_length]
                sent_offset += each_length
                merged_sentences.append(merged_sentence)
                merged_anal_dicts.append(merged_anal_dict_for_merged_sentences)
            output_sentences = merged_sentences
            if supply_anal_dict:
                anal_dicts = merged_anal_dicts

        # ###############################################################################
        if supply_anal_dict:
            # TODO: anal_dicts has different sorting! Fix it?

            return output_sentences, anal_dicts
        else:
            return output_sentences

    def prepare_analysis_dict_for_sentences_batch(self, sentences, max_tokens_count=None):
        """
        The method which produces analysis dictionary of the sentence, it generates
        substitution candidates of the segments of the input sentence.

        :param sentence: str
        :return: dict SentenceAnalysisDictionary
        """

        # preprocess
        preprocessed_sentences = [self.preprocess_sentence(sentence) for sentence in sentences]

        # tokenize sentences
        tokenized_sentences = [self.lm.tokenize_sentence(sentence) for sentence in preprocessed_sentences]
        tokenized_sentences_cased = [self.lm.tokenize_sentence(sentence) for sentence in sentences]

        # offset from the start of the batch
        batch_offset = 0
        # batch_gen = self.chunk_generator(tokenized_sentences, ELMO_BATCH_SIZE)
        if not max_tokens_count:
            max_tokens_count=ELMO_BATCH_TOKEN_SIZE
        batch_gen = self.chunk_generator_token_weighted(tokenized_sentences,
                                                        max_tokens_count=max_tokens_count)

        analysis_dicts = []

        for mini_batch_tokenized_sents in batch_gen:
            # minibatch start:
            start_dt = dt.datetime.now()
            elmo_datas_mini_batch = self.lm.elmo_lm(mini_batch_tokenized_sents)
            middle_dt = dt.datetime.now()
            # now we consequently execute hypotheses generation
            for relative_offset, each_elmo_data in enumerate(elmo_datas_mini_batch):
                absolute_offset = batch_offset + relative_offset
                # analyse sentence, atomic (token-token) hypotheses generation:
                try:
                    # analysis_dict = self.elmo_analysis_with_probable_candidates_reduction_dict_in_dict_out(
                    #     {
                    #         # 'input_sentence': preprocessed_sentences[absolute_offset],
                    #         'input_sentence': sentences[absolute_offset],
                    #         'tokenized_input_sentence': tokenized_sentences[absolute_offset]},
                    #     each_elmo_data)

                    analysis_dict = self.elmo_analysis_with_probable_candidates_reduction_dict_in_dict_out( {
                        'input_sentence': sentences[absolute_offset],
                        'tokenized_input_sentence': tokenized_sentences[absolute_offset],
                        'tokenized_cased_input_sentence': tokenized_sentences_cased[absolute_offset]

                    }, each_elmo_data)
                except Exception as e:
                    print(e)
                    print(absolute_offset)
                    print(len(sentences))
                    import ipdb; ipdb.set_trace()
                    print("1")


                # multi-token - token hypotheses generation
                merged_tokens_hypotheses_dict = self.generate_Nto1_hypotheses(
                    analysis_dict['tokenized_input_sentence'], each_elmo_data)

                analysis_dict['word_substitutions_candidates'] += merged_tokens_hypotheses_dict
                analysis_dicts.append(analysis_dict)
            # increment offset with size of batch
            batch_offset += len(mini_batch_tokenized_sents)
            fin_dt = dt.datetime.now()
            print("Calc elmo matricies in minibatch: %s, generating_hypotheses: %s" % (
            str(middle_dt - start_dt), str(fin_dt - middle_dt)))
        return analysis_dicts

    def make_fixes_batch(self, analysis_dicts, min_advantage_treshold=4.0):
        results=[]
        for each_anal_dict in analysis_dicts:
            results.append(self.make_fixes(each_anal_dict, min_advantage_treshold=min_advantage_treshold))

        return results

    def __call__(self, *args, **kwargs):
        return self.process_sentences_batch(*args, **kwargs)
