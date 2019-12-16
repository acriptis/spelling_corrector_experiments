import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import pprint
from spelling_correction_models.elmo_40in_spelling_corrector.helper_fns import SCAnalysisDictManager
from sklearn.externals import joblib


class ReRanker40inRegressor():
    """
    Reranker is responsible for decision making in spelling corrector.

    In inference mode it receives a feature representation of each candidate of fix and scores it.
    Then decides according to max score.
    """

    #     def __init__(self, cls=LogisticRegression, cls_params=None):

    #         self.cls_params = cls_params if cls_params is not None else dict(warm_start=True)
    #         self.reg = cls(**self.cls_params)
    def __init__(self):
        self.reg = LogisticRegression(warm_start=True, solver='lbfgs')

    #         self.reg = LogisticRegression(warm_start=True, solver='newton-cg')

    # TODO saving loading functionality

    def predict_fixes_tokens(self, sentence_data_analysis_dict):
        """
        given data anlysis dict it outputs sentence hypothesis as sequence of tokens
        :param sentence_data_analysis_dict: DataAnalysisDict object (dict)
        :return: list of token strings
        """
        winning_tokens = []
        total_length = len(sentence_data_analysis_dict['tokenized_input_sentence'])
        for current_token_idx, each_tok in enumerate(
                sentence_data_analysis_dict['tokenized_input_sentence']):
            if current_token_idx == 0: continue
            if current_token_idx == total_length - 1:
                # last token is </S>. skip it
                continue

            suffixes_hypotheses = SCAnalysisDictManager.filter_by_start_index(
                data_anal_dict=sentence_data_analysis_dict, start_index=current_token_idx)

            #################################################################
            # TODO check that we are not under multitoken hypothesis segment ?
            flat_hypotheses_list = []
            for each_segment_hypotheses_hub in suffixes_hypotheses:
                flat_hypotheses_list.extend(each_segment_hypotheses_hub['top_k_candidates'])

            #################################################################
            if len(flat_hypotheses_list) > 1:
                preprocessed_features = list(
                    map(ReRanker40inRegressor.preprocess_feature_dict, flat_hypotheses_list))
                #             print("preprocessed_features")
                #             pprint.pprint(preprocessed_features)
                # binarize features
                # binarized_features = list(map(ReRanker40inRegressor.binarize_features, preprocessed_features))
                binarized_features = np.array(
                    list(map(ReRanker40inRegressor.binarize_features, preprocessed_features)))
                #                 predictions = np.array(map(self.reg.predict, binarized_features))
                predictions = self.reg.decision_function(binarized_features)

                print("predictions at %d" % current_token_idx)
                print(predictions)
                #                 import ipdb; ipdb.set_trace()
                max_pred_idx = np.argmax(predictions)
                print("max_pred_idx")
                print(max_pred_idx)
                print("winner:")
                print(flat_hypotheses_list[max_pred_idx])
                winning_tokens.append(flat_hypotheses_list[max_pred_idx]['token_str'])
            elif len(flat_hypotheses_list) == 1:
                winning_tokens.append(flat_hypotheses_list[0]['token_str'])
            else:
                raise Exception("Zero hypotheses?")
        # todo add check of multispan!?
        return winning_tokens

    def predict_fixes(self, sentence_data_analysis_dict):
        """
        Given data anlysis dict it outputs sentence hypothesis

        Uses simple detokenization
        """

        winning_tokens = self.predict_fixes_tokens(sentence_data_analysis_dict)
        final_str = " ".join(winning_tokens)
        print(final_str)
        return final_str

    def _prepare_token_front_data(self, list_of_feature_lists, etalon_idx):
        """Given a list of feature lists and index of etalon we make mini-batch for training
        return X and y attachable to big dataset
        """
        etalon_feature_list = list_of_feature_lists[etalon_idx]
        training_data = []
        training_labels = []
        for hypo_idx, feature_list in enumerate(list_of_feature_lists):
            if hypo_idx == etalon_idx:
                continue

            diff = feature_list - etalon_feature_list
            training_data.extend((diff, -diff))
            training_labels.extend((1, -1))
        return training_data, training_labels

    def fit_token_front(self, list_of_feature_lists, etalon_idx):
        """Given a list of feature lists and index of etalon list we make training operation"""
        training_data, training_labels = self._prepare_token_front_data(list_of_feature_lists,
                                                                        etalon_idx)
        self.reg = self.reg.fit(training_data, training_labels)
        #         self.reg.partial_fit(training_data, training_labels)
        self.coef_ = self.reg.coef_
        print("self.coef_")
        print(self.coef_)
        return self.reg

    # deprecated
    def fit_from_sorokin(self, data):
        """
        data is list of SentenceData elements

        each SentenceData element is a list of hypotheses positioned by token indexes.

        For each token index position we have a list of hypotheses starting from the position.

        At learning phase we need to iterate each token index, select front of hypotheses
        strating from this index.
        Select the etalon hypotheses and then calculate gradients for all alternatives
        (according to their feature represenations).

        Special Case is when multitoken hypothesis wins:
        Multitoken hypothesais spans over next token index, this means that we should skip
        training/inference process for the next step if multitoken hypothesis won.
        Another variant is to use alternatives as negative examples
        Fit the data for a sentence. Hypotheses for all token segements.

        """
        # TODO finish
        training_data, training_labels = [], []
        for elem in data:
            first = np.array(elem[0], dtype=float)
            for other in elem[1:]:
                other = np.array(other, dtype=float)
                diff = first - other
                training_data.extend((diff, -diff))
                training_labels.extend((1, -1))
        self.reg = self.reg.fit(training_data)
        self.coef_ = self.reg.coef_

        return self

    # ##################################################################################
    # sentence level
    def prepare_dataset_from_data_anal_dicts(self, annotated_data_anal_dicts):
        features = []
        labels = []
        for each_dad in annotated_data_anal_dicts:
            minibatch_features, minibatch_labels = self._prepare_sentence_training_data(each_dad)
            features.extend(minibatch_features)
            labels.extend(minibatch_labels)
        return features, labels

    def _prepare_sentence_training_data(self, sentence_data_analysis_dict):
        """
        Method for preparation of training dataset of the sentence of the Regressor according to
        sentence data analsysis dict and information about etalons

        sentence_data: analysis dict with etalons markup?
        """
        features = []
        labels = []

        for current_token_idx, each_tok in enumerate(
                sentence_data_analysis_dict['tokenized_input_sentence']):
            if current_token_idx == 0: continue

            suffixes_hypotheses = SCAnalysisDictManager.filter_by_start_index(
                data_anal_dict=sentence_data_analysis_dict, start_index=current_token_idx)

            #################################################################
            # TODO check that we are not under multitoken hypothesis segment ?
            flat_hypotheses_list = []
            for each_segment_hypotheses_hub in suffixes_hypotheses:
                flat_hypotheses_list.extend(each_segment_hypotheses_hub['top_k_candidates'])
            # check that etalon exists in set, and find the best fitting etalon (it possible that we have many etalons on the stage,
            # but they must be of different token length).
            etalon_hypothesis_feature_dict = None
            for idx, each_item in enumerate(flat_hypotheses_list):
                if 'etalon_ref' in each_item:
                    if etalon_hypothesis_feature_dict:
                        # not none!
                        print("flat_hypotheses_list:")
                        print(pprint.pprint(flat_hypotheses_list))
                        print("sentence_data_analysis_dict")
                        print(sentence_data_analysis_dict)
                        if isinstance(etalon_hypothesis_feature_dict['etalon_ref'],
                                      list) and isinstance(each_item['etalon_ref'], int):
                            pass
                        elif isinstance(each_item['etalon_ref'], list) and isinstance(
                                etalon_hypothesis_feature_dict['etalon_ref'], int):
                            # select new etalon which is longer:
                            etalon_hypothesis_feature_dict = each_item
                    #                         raise Exception("Multiple etalons for the step %d" % (current_token_idx))

                    # TODO select the longest
                    # debug me?
                    else:
                        etalon_hypothesis_feature_dict = each_item
                etalon_index = idx
            if not etalon_hypothesis_feature_dict:
                print("No Etalon Found current_token_index = %d, each_tok= %s" %
                      (current_token_idx, each_tok))
                print("skipping training step")
                continue
                # can not fit/ skip sentence?
            # ok etalon found we can train
            if len(flat_hypotheses_list) <= 1:
                continue
            #################################################################
            preprocessed_features = list(
                map(ReRanker40inRegressor.preprocess_feature_dict, flat_hypotheses_list))
            # binarize features
            binarized_features = list(
                map(ReRanker40inRegressor.binarize_features, preprocessed_features))

            print("fitting")
            print("binarized_features:")
            print(binarized_features)
            tokfront_features, tokfront_labels = self._prepare_token_front_data(binarized_features,
                                                                                etalon_index)
            features.extend(tokfront_features)
            labels.extend(tokfront_labels)
        return features, labels

    @staticmethod
    def preprocess_feature_dict(feature_dict):
        """Givna feature dictionary in format SC Analysis Dictionary it converts it into compatible with
        ML algorithms form

        input example:

        {'advantage': 2.0,
          'comment': '2letter '
                     'word',
          'error_score': 0.0,
          'lm_advantage': 0.0,
          'token_merges': 0,
          'token_splits': None,
          'token_str': 'по',
          'zero_hypothesis': True}

        output:
        {
        'lm_advantage': 0.0,
        'levenshtein_distance': 0.0,
        'error_score': 0.0,
        'token_merges': 0,
        'token_splits': 0,
        '1letter_word': 0,
        '2letter_word': 1,
        'is_abbrev': 0,
        'zero_hypothesis': 1,
        'has_digit': 0,
        'short_word_with_punctuation': 0,
        'capitalize': 0,
        'upper': 0,
        }
        """
        output_feature_dict = {
            'advantage': feature_dict['advantage'],
            'lm_advantage': feature_dict['lm_advantage'],
            'error_score': feature_dict['error_score'],
            'token_merges': feature_dict['token_merges'],
            'token_splits': 0,
            '1letter_word': 0,
            '2letter_word': 0,
            'has_digit': 0,
            'short_word_with_punctuation': 0,

            'is_abbrev': 0,

            'levenshtein_distance': 0.0,

            'zero_hypothesis': 0,

            'capitalize': 0,
            'upper': 0,
        }

        if 'token_splits' in feature_dict and feature_dict['token_splits']:
            output_feature_dict['token_splits'] = feature_dict['token_splits']

        if 'zero_hypothesis' in feature_dict:
            output_feature_dict['zero_hypothesis'] = int(feature_dict['zero_hypothesis'])

        if 'comment' in feature_dict:
            if '1letter_word' in feature_dict['comment']:
                output_feature_dict['1letter_word'] = 1

            if '2letter_word' in feature_dict['comment']:
                output_feature_dict['2letter_word'] = 1

            if 'has digit' in feature_dict['comment']:
                output_feature_dict['has_digit'] = 1

            if 'short word with punctuation' in feature_dict['comment']:
                output_feature_dict['short_word_with_punctuation'] = 1

        if 'is_abbrev' in feature_dict:
            output_feature_dict['is_abbrev'] = 1

        #         temp_feat_dict = {
        #             'lm_advantage': output_feature_dict['lm_advantage'],
        #             'error_score': output_feature_dict['error_score'],
        #             'token_merges':output_feature_dict['token_merges'],

        #         }
        #         return temp_feat_dict
        # TODO add is_dictionary feature
        # TODO add levenshtein distance feature
        return output_feature_dict

    @staticmethod
    def binarize_features(feature_dict):
        """Just remove labels"""
        return np.array([each_v for each_k, each_v in feature_dict.items()])

    def save(self, filename='/tmp/reranker_40in.joblib.pkl'):
        """
        save model of reranker
        :return:
        """
        return joblib.dump(self.reg, filename, compress=9)

    def load(self, filename='/tmp/reranker_40in.joblib.pkl'):
        """
        Load model of rernaker
        :return:
        """
        reg = joblib.load(filename)
        self.reg = reg
        return self
