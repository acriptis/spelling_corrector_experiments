DATA_PATH = "/home/alx/Cloud/spell_corr/py_spelling_corrector/data/"
from dp_components.levenshtein_searcher_component import LevenshteinSearcherComponent


class SpellingCorrectionCandidatesGenerator():
    """
    Generates candidates for words (words) with error scores
    """
    # decrement for frequent errors
    FREQUENT_ERRORS_DECREMENT_SCORE = -4.0

    # decrement for excessive space
    EXCESSIVE_SPACE_ERROR_DECREMENT_SCORE = -2.5

    MISSED_HYPHEN_ERROR_DECREMENT_SCORE = -2.0

    def __init__(self, path_to_dictionary=None):

        words_dict = []
        if not path_to_dictionary:
            path_to_dictionary = DATA_PATH + "compreno_wordforms.txt"
        #             path_to_dictionary = DATA_PATH + "russian_words_vocab.dict"

        with open(path_to_dictionary, "r") as dict_file:
            words_dict = dict_file.read().splitlines()
        self.lsc = LevenshteinSearcherComponent(words=words_dict)

    def gen_candidates(self, token):
        """
        Given a token string generates candidates with error scores
        :param token: string with token
        :return:
        """
        # TODO black list support: some tokens should not variate (SB requirement)
        scored_candidates = self.lsc([[token]])[0][0]
        scores, w_forms = zip(*scored_candidates)
        w_forms = list(w_forms)
        scores = list(scores)

        # ############################################################################################
        # here is rule based/statistical substitutions with distant levenshtein can be applied:

        if token == "нить":
            w_forms.append("нибудь")
            scores.append(self.FREQUENT_ERRORS_DECREMENT_SCORE)
        elif token in ["оч"]:
            w_forms.append("очень")
            scores.append(self.FREQUENT_ERRORS_DECREMENT_SCORE)
        elif token in ["што", "шо", "чо", "чё", "че"]:
            w_forms.append("что")
            scores.append(self.FREQUENT_ERRORS_DECREMENT_SCORE)
        elif token in ["ваще", "воще"]:
            w_forms.append("вообще")
            scores.append(self.FREQUENT_ERRORS_DECREMENT_SCORE)
        elif token in ["вообщем"]:
            w_forms.append("в общем")
            scores.append(self.FREQUENT_ERRORS_DECREMENT_SCORE)
        elif token in ["писят"]:
            w_forms.append("пятьдесят")
            scores.append(self.FREQUENT_ERRORS_DECREMENT_SCORE)
        elif token in ["аццкий"]:
            w_forms.append("адский")
            scores.append(self.FREQUENT_ERRORS_DECREMENT_SCORE)
        elif token in ["помойму", "помоиму"]:
            w_forms.append("по-моему")
            scores.append(self.FREQUENT_ERRORS_DECREMENT_SCORE)
        elif token in ["тыщ", "тыщь"]:
            w_forms.append("тысяч")
            scores.append(self.FREQUENT_ERRORS_DECREMENT_SCORE)
        elif token in ["щас", "щаз", "счас", "счаз"]:
            w_forms.append("сейчас")
            scores.append(self.FREQUENT_ERRORS_DECREMENT_SCORE)

        return scores, w_forms

    # deprecated
    def variate_with_prefixes(self, candidates, error_scores):
        """
        Given a tokens candidates this method enriches the space of candidates with prefixed variants
        by default it prepends prefixes of the space and hyphen to tokens.

        So ["то"] -> ["то", "-то", " то"]

        :param candidates: list of candidate strings
        :param prefixes: list of possible prefixes
        :return: list of candidates enriched with prefixed versions
        """

        #         prefixes = [" ", "-"]
        result_candidates = []
        result_scores = []
        for idx, each_candidate in enumerate(candidates):
            # add candidate produced by erroneous space problem.
            # Ex.: "при вет" -> "привет":
            result_candidates.append(each_candidate)
            result_scores.append(error_scores[idx] + self.EXCESSIVE_SPACE_ERROR_DECREMENT_SCORE)

            # add space candidate (no fix)
            result_candidates.append(" " + each_candidate)
            result_scores.append(error_scores[idx] + 0.0)

            # add hyphen candidates conditionally:
            # TODO improve heuristics for hyphen adding?
            # TODO add hyphen after "по"
            if each_candidate in ["то", "таки", "нибудь", "моему", "нашему", "твоему", "любому",
                                  "за", "другому", "как",
                                  "русски", "разному"]:
                result_candidates.append("-" + each_candidate)
                result_scores.append(error_scores[idx] + self.MISSED_HYPHEN_ERROR_DECREMENT_SCORE)
        #         print("result_scores, result_candidates")
        #         print(result_scores, result_candidates)
        return result_scores, result_candidates
