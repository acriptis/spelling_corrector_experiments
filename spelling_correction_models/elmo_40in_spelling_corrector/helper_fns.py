from copy import deepcopy
# maximum number of hypotheses in hypotheses hub for correction of one sentence
HYPOHUB_MAX_ALLOWED_SIZE = 1000

def estimate_the_best_s_hypotheses(data_analysis_dict, min_advantage_treshold=0.0):
    """
    Given a dictionary with analysis of sentence with all allowed token-candidate hypotheses
    constructs sentences hypotheses.
    :param data_analysis_dict:
    :param min_advantage_treshold: minimal value of advantage for the hypothesis to pass through
    :return:
    """
    hypotheses_hub = HypothesesHub()
    for current_token_idx, each_tok in enumerate(data_analysis_dict['tokenized_input_sentence']):
        if current_token_idx == 0:
            # skip the first token which is <s>
            continue
        if current_token_idx == len(data_analysis_dict['tokenized_input_sentence']) - 1:
            # skip the last token which is </s>
            continue
        # print(each_tok)
        #  suffixes_hypotheses = data_analysis_dict['word_substitutions_candidates'].filter(start_idx=current_token_idx)
        suffixes_hypotheses = SCAnalysisDictManager.filter_by_start_index(
            data_anal_dict=data_analysis_dict, start_index=current_token_idx)
        # print("suffixes_hypotheses")
        # print(suffixes_hypotheses)

        # suffixes hypotheses is a list of dicts with hypotheses that start at index: current_token_idx and finish at current_token_idx or later        
        # expect that each s_hypothesis in hub has finish index (pointer of the final index position).
        # Ex. suffixes_hypotheses: 
        #         [{'tok_idx': 1,
        #            'top_k_candidates': [{'advantage': 0.0, 'token_str': 'мамо'},
        #             {'advantage': 0.8290032949621251, 'token_str': 'мама'},
        #             {'advantage': 0.0682681172786106, 'token_str': 'маме'},
        #             {'advantage': 1.4654415043774653, 'token_str': 'мало'},
        #             {'advantage': 0.5788053066965881, 'token_str': 'мимо'}]},

        #          {'tok_idx': (1, 3),
        #              'tok_idx_start': 2,
        #              'tok_idx_fin': 3,
        #              'top_k_candidates': [{'token_str': 'мыла',
        #                                    'token_merges': 1,
        #                                    'error_score': -4.0,
        #                                    'lm_scores_list': array([-4.99906474, -6.26263737]),
        #                                    'summated_base_scores': array([-13.19048357, -11.76141742]),
        #                                    'advantage': 9.690198885299584}]}
        #         ]

        #########################################################################################
        # TODO filter suffixes which has low likelihood? (or top-1 suffix?)
        # TODO add filtering by treshold?
        # TODO but propogate base hypothesis
        suffixes_hypotheses = deepcopy(suffixes_hypotheses)
        for each_suffix_group_dict in suffixes_hypotheses:

            filtered_top_k_candidates = []
            for each_top_candidate in each_suffix_group_dict['top_k_candidates']:
                if 'zero_hypothesis' in each_top_candidate and each_top_candidate['zero_hypothesis'] is True:
                    filtered_top_k_candidates.append(each_top_candidate)
                elif each_top_candidate['advantage'] < min_advantage_treshold:
                    continue
                else:
                    # non zero hypothesis thta has minimal advantage for propagation
                    filtered_top_k_candidates.append(each_top_candidate)
            each_suffix_group_dict['top_k_candidates'] = filtered_top_k_candidates
        #########################################################################################
        # print("suffixes_hypotheses after filtering")
        # print(suffixes_hypotheses)
        hypotheses_hub = hypotheses_hub.fork_for_suffixes_segment_hypotheses(suffixes_hypotheses)
        # print(str(current_token_idx) + " len(hypotheses_hub) " + str(len(hypotheses_hub)))

        if len(hypotheses_hub)>HYPOHUB_MAX_ALLOWED_SIZE:
            # prune hypotheses hub from bad hypotheses
            hypotheses_hub.filter_the_best_hypotheses(top_k=HYPOHUB_MAX_ALLOWED_SIZE)
        # TODO prune hypotheses that has finish_index==current_token_idx, but dont prune hypotheses that are longer

    # TODO implement method:
    the_best_sentence_hypothesis = hypotheses_hub.filter_the_best_hypotheses(top_k=1)
    the_best_sentence_hypothesis = sorted(the_best_sentence_hypothesis,
                                          key=lambda x: x.calc_advantage_score(), reverse=True)
    return the_best_sentence_hypothesis


class SCAnalysisDictManager():
    """
    Class which wraps operations for dict with sentence analysis of spelling corrector. 

    Example artifact:
    {'input_sentence': 'мамо мы ла рабу',
     'tokenized_input_sentence': ['<S>', 'мамо', 'мы', 'ла', 'рабу', '</S>'],
     'word_substitutions_candidates': [{'tok_idx': 0, 'top_k_candidates': []},
      {'tok_idx': 1,
       'top_k_candidates': [{'advantage': 0.0, 'token_str': 'мамо'},
        {'advantage': 0.8290032949621251, 'token_str': 'мама'},
        {'advantage': 0.0682681172786106, 'token_str': 'маме'},
        {'advantage': 1.4654415043774653, 'token_str': 'мало'},
        {'advantage': 0.5788053066965881, 'token_str': 'мимо'}]},
      {'tok_idx': 2, 'top_k_candidates': [{'advantage': 0.0, 'token_str': 'мы'}]},
      {'tok_idx': 3,
       'top_k_candidates': [{'advantage': 8.881784197001252e-16,
         'token_str': 'ла'},
        {'advantage': 1.1416171799843715, 'token_str': 'л'},
        {'advantage': 4.534746617501758, 'token_str': 'ли'},
        {'advantage': 0.2392642375027938, 'token_str': 'ло'},
        {'advantage': 1.5435339098199536, 'token_str': 'ль'},
        {'advantage': 0.6130232420520505, 'token_str': 'ля'},
        {'advantage': 0.5685809466324443, 'token_str': 'лаю'},
        {'advantage': 5.609125288265215, 'token_str': 'а'},
        {'advantage': 1.3843852873087856, 'token_str': 'ва'},
        {'advantage': 4.591302278044582, 'token_str': 'да'},
        {'advantage': 6.869433134510696, 'token_str': 'за'},
        {'advantage': 2.116203203263707, 'token_str': 'ка'},
        {'advantage': 0.894093864484022, 'token_str': 'ма'},
        {'advantage': 7.909878868209844, 'token_str': 'на'},
        {'advantage': 1.3567950485204125, 'token_str': 'та'},
        {'advantage': 0.8312294913850709, 'token_str': 'ела'},
        {'advantage': 2.2575124093097294, 'token_str': 'зла'},
        {'advantage': 2.1699851412306375, 'token_str': 'шла'}]},
      {'tok_idx': 4,
       'top_k_candidates': [{'advantage': 0.0, 'token_str': 'рабу'},
        {'advantage': 0.2985749735125065, 'token_str': 'раб'},
        {'advantage': 0.9318240256975585, 'token_str': 'раба'},
        {'advantage': 1.338874848056081, 'token_str': 'рабы'},
        {'advantage': 1.0092864240096588, 'token_str': 'разу'},
        {'advantage': 0.6805089544958989, 'token_str': 'рыбу'},
        {'advantage': 1.0757820984060213, 'token_str': 'бабу'}]},
      {'tok_idx': 5,
       'top_k_candidates': [{'advantage': 0.0, 'token_str': '</S>'}]},
      {'tok_idx': (2, 3),
       'tok_idx_start': 2,
       'tok_idx_fin': 3,
       'top_k_candidates': [{'token_str': 'мыла',
         'token_merges': 1,
         'error_score': -4.0,
         'lm_scores_list': array([-4.99906474, -6.26263737]),
         'summated_base_scores': array([-13.19048357, -11.76141742]),
         'advantage': 9.690198885299584}]}]}
    """

    @staticmethod
    def filter_by_start_index(data_anal_dict, start_index):
        """Given a dict with data analysis it filters out word spans substitution candidates which start at specific token index 
        (integer measured in input sentence token space)"""
        assert "word_substitutions_candidates" in data_anal_dict
        # results list is a list of dicts of hypotheses sets for particular spans (1token spans and 2-token spans, AS-IS)
        results_list = []
        for each_dict_set_of_candidates in data_anal_dict['word_substitutions_candidates']:
            # here is a complex condition that iteratively checks that substitution candidates dict set starts at 
            # specific token position. Then it filters out the best candidates subset from each set merges that sets and returns a result 
            # of the query.
            # candidate set dict describes set of candidates which can substitute particular span it 
            # may be 1token-1token substitution or may be Ntokens->1token substitution.
            # for the firsts tok_idx key is integer holding the start_index
            # for the latter tok_idx key is tuple holding the start_index and the last index

            if each_dict_set_of_candidates['tok_idx'] == start_index or \
                    (isinstance(each_dict_set_of_candidates['tok_idx'], tuple) \
                     and each_dict_set_of_candidates['tok_idx'][0] == start_index):
                results_list.append(each_dict_set_of_candidates)

        return results_list


class SentenceHypothesis():
    def __init__(self, text):
        # TODO refactor clean code
        if text != "":
            print("SentenceHypothesis must be init with empty string!")
        self.text = ""
        # token hypotheses of the sentence
        self.token_hypotheses = []

        # index of the last token in hypothesis
        self.finish_idx = -1
        # -1 for empty hypothesis and tokens length for other

    def calc_advantage_score(self):
        adv_score = 0.0
        for each_token_hypothesis in self.token_hypotheses:
            adv_score += each_token_hypothesis['advantage']
        return adv_score

    def fork_for_each_suffix(self, suffixes_dicts_list):
        """Given a list of suffixes strings it forks the current hypotheses into several
        hypotheses for each suffix.

        : suffixes_dicts_list:
        Ex.:
        [
            {'tok_idx': 4,
               'top_k_candidates': [{'advantage': 0.0, 'token_str': 'рабу'},
                                    {'advantage': 0.2985749735125065, 'token_str': 'раб'},
                                    {'advantage': 0.9318240256975585, 'token_str': 'раба'},
                                    {'advantage': 1.338874848056081, 'token_str': 'рабы'},
                                    {'advantage': 1.0092864240096588, 'token_str': 'разу'},
                                    {'advantage': 0.6805089544958989, 'token_str': 'рыбу'},
                                    {'advantage': 1.0757820984060213, 'token_str': 'бабу'}]},
            {'tok_idx': (4, 5),
               'tok_idx_start': 4,
               'tok_idx_fin': 5,
               'top_k_candidates': [{'token_str': 'мыла',
                 'token_merges': 1,
                 'error_score': -4.0,
                 'lm_scores_list': array([-4.99906474, -6.26263737]),
                 'summated_base_scores': array([-13.19048357, -11.76141742]),
                 'advantage': 9.690198885299584}]}
         ]

        """
        hypotheses_list = []

        for idx, each_suffix_dict in enumerate(suffixes_dicts_list):
            # iteration over list of dicts with segments-candidates. First dict is a dict with 1token-1token substitution candidates:
            if isinstance(each_suffix_dict['tok_idx'], int):
                start_tok_idx = each_suffix_dict['tok_idx']
            elif isinstance(each_suffix_dict['tok_idx'], tuple):
                start_tok_idx = each_suffix_dict['tok_idx'][0]
            else:
                raise Exception("Wrong format! data: %s" % each_suffix_dict)

            for candidate_idx, each_candidate_dict in enumerate(
                    each_suffix_dict['top_k_candidates']):
                # TODO assert that each new candidate has lower advantage
                new_sentence_hypothesis = deepcopy(self)
                # ############################################################
                # add a segment suffix to the sentence hypothesis:
                if start_tok_idx==1:
                    # the first index is the first token in the sentence (0s: <s>) so we dont need
                    # to put a space before it
                    new_sentence_hypothesis.text += each_candidate_dict['token_str']
                else:
                    new_sentence_hypothesis.text += " " + each_candidate_dict['token_str']

                new_sentence_hypothesis.token_hypotheses.append(each_candidate_dict)
                # retrieve fin_tok_index from candidate span:
                if isinstance(each_suffix_dict['tok_idx'], int):
                    fin_tok_idx = each_suffix_dict['tok_idx']
                elif isinstance(each_suffix_dict['tok_idx'], tuple):
                    fin_tok_idx = each_suffix_dict['tok_idx'][1]
                else:
                    raise Exception("Wrong format! data: %s" % each_suffix_dict)

                new_sentence_hypothesis.finish_idx = fin_tok_idx
                # ############################################################

                hypotheses_list.append(new_sentence_hypothesis)

        return hypotheses_list

    def __repr__(self):
        advantage_score = self.calc_advantage_score()
        text = "hypothesis: %s | advantage: %0.5f" % (self.text, advantage_score)
        return text


class HypothesesHub():
    def __init__(self):
        # init with null hypothesis:
        self.hypotheses = [SentenceHypothesis("")]

    def fork_for_suffixes_segment_hypotheses(self, segment_candidates_dicts_list):
        """
        For each hypothesis in the hub it appends all candidates
        :param partial_candidates_dicts_list:
        Ex.: 
        [
            {'tok_idx': 4,
               'top_k_candidates': [{'advantage': 0.0, 'token_str': 'рабу'},
                                    {'advantage': 0.2985749735125065, 'token_str': 'раб'},
                                    {'advantage': 0.9318240256975585, 'token_str': 'раба'},
                                    {'advantage': 1.338874848056081, 'token_str': 'рабы'},
                                    {'advantage': 1.0092864240096588, 'token_str': 'разу'},
                                    {'advantage': 0.6805089544958989, 'token_str': 'рыбу'},
                                    {'advantage': 1.0757820984060213, 'token_str': 'бабу'}]},
            {'tok_idx': (4, 5),
               'tok_idx_start': 4,
               'tok_idx_fin': 5,
               'top_k_candidates': [{'token_str': 'мыла',
                 'token_merges': 1,
                 'error_score': -4.0,
                 'lm_scores_list': array([-4.99906474, -6.26263737]),
                 'summated_base_scores': array([-13.19048357, -11.76141742]),
                 'advantage': 9.690198885299584}]}
         ]



        :return: updated self
        """

        # validation that start indexes for candidates dicts are aligned
        assert isinstance(segment_candidates_dicts_list[0]['tok_idx'], int)
        suffixes_start_idx = segment_candidates_dicts_list[0]['tok_idx']
        if len(segment_candidates_dicts_list) > 1:
            assert isinstance(segment_candidates_dicts_list[1]['tok_idx'], tuple)
            assert 'tok_idx_start' in segment_candidates_dicts_list[1]
            assert isinstance(segment_candidates_dicts_list[1]['tok_idx_start'], int)
            assert segment_candidates_dicts_list[1]['tok_idx_start'] == suffixes_start_idx

        new_hypotheses = []
        for each_s_hypothesis in self.hypotheses:
            if each_s_hypothesis.finish_idx < suffixes_start_idx:

                hypos = each_s_hypothesis.fork_for_each_suffix(segment_candidates_dicts_list)
                new_hypotheses += hypos
            else:
                # we have a hypothesis that merged multiple tokens:
                assert each_s_hypothesis.finish_idx >= suffixes_start_idx
                # so add this hypothesis as is (without suffix appending):
                new_hypotheses.append(each_s_hypothesis)

        self.hypotheses = new_hypotheses
        return self

    #     def prune_hypotheses(self, top_k=100):
    #         prunes_hypotheses to restricted amount

    def filter_the_best_hypotheses(self, top_k=-1, advantage_treshold=None, pass_zero_hypothesis=True):
        """
        sorts and filters hyptheses space for top_k BEST hypotheses.

        :param top_k: -1 or positive int: fixes number of hyyptheses to be propogated
        :param advantage_treshold: None or float, minimal advantage required for the item to
            be selected
        :param pass_zero_hypothesis: if true then zero hypothesis (base input) is propogated as well
        :return:
        """
        # TODO implement me
        # TODO how to select base hypothesis?
        # TODO assure that we prune hypotheses of the shortest length (long hypotheses shouldnt
        # be pruned)
        the_best_sentence_hypothesis = sorted(self.hypotheses,
                                              key=lambda x: x.calc_advantage_score(), reverse=True)
        if top_k>0:
            the_best_sentence_hypothesis = the_best_sentence_hypothesis[:top_k]
        self.hypotheses = the_best_sentence_hypothesis
        return self.hypotheses

    def __len__(self):
        return len(self.hypotheses)
