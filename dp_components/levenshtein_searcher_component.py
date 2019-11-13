import string
from math import log10
from typing import Iterable, List, Tuple

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

# from deeppavlov.models.spelling_correction.levenshtein.levenshtein_searcher import LevenshteinSearcher
from .levenshtein_searcher import LevenshteinSearcher


class LevenshteinSearcherComponent(Component):
    """Component that finds replacement candidates for tokens at a set Damerau-Levenshtein distance

    Args:
        words: list of every correct word
        max_distance: maximum allowed Damerau-Levenshtein distance between source words and candidates
        error_probability: assigned probability for every edit
        oov_penalty: OutOfVocabulary penalty (negative float or zero) - penalty in logits for out
            of vocabulary words

    Attributes:
        max_distance: maximum allowed Damerau-Levenshtein distance between source words and candidates
        error_probability: assigned logarithmic probability for every edit
        vocab_penalty: assigned logarithmic probability of an out of vocabulary token being the correct one without
         changes
    """

    _punctuation = frozenset(string.punctuation)

    def __init__(self, words: Iterable[str], max_distance: float=1, error_probability: float=1e-4,
                 alphabet=None, operation_costs=None, oov_penalty=None,
                 *args, **kwargs):
        words = list({word.strip().lower().replace('ё', 'е') for word in words})
        if not alphabet:
            alphabet = sorted({letter for word in words for letter in word})
        self.max_distance = max_distance
        self.error_probability = log10(error_probability)

        if oov_penalty:
            self.vocab_penalty = oov_penalty
        else:
            # default case:
            #         self.vocab_penalty = self.error_probability * 2
            self.vocab_penalty = 0.0

        if not operation_costs:
            operation_costs = generate_operation_costs_dict(alphabet=alphabet)
        self.searcher = LevenshteinSearcher(alphabet, words, allow_spaces=True, euristics=2,
                                            operation_costs=operation_costs)

    def _infer_instance(self, tokens: Iterable[str]) -> List[List[Tuple[float, str]]]:
        candidates = []
        for word in tokens:
            if word in self._punctuation:
                candidates.append([(0, word)])
            else:
                c = {candidate: self.error_probability * distance
                     for candidate, distance in self.searcher.search(word, d=self.max_distance)}
                c[word] = c.get(word, self.vocab_penalty)
                candidates.append([(score, candidate) for candidate, score in c.items()])
        return candidates

    def __call__(self, batch: Iterable[Iterable[str]], *args, **kwargs) -> List[List[List[Tuple[float, str]]]]:
        """Propose candidates for tokens in sentences

        Args:
            batch: batch of tokenized sentences
            Ex.: [["все","смешалось","кони","люди"]]

        Returns:
            batch of lists of probabilities and candidates for every token
            Ex.:[
                    [
                        [
                            (-0.0, 'все'),(-4.0, 'вес'), (-4.0, 'вс'), (-4.0, 'всг'),(-4.0, 'вси'),
                            (-4.0, 'вск'),(-4.0, 'всл'),(-4.0, 'овсе')],
                        [
                            (-0.0, 'смешалось'),(-4.0, 'смешало ь'),(-4.0, 'мешалось'),
                            (-4.0, 'вмешалось'),(-4.0, 'с мешалось')],
                        [
                            (-0.0, 'кони'),(-4.0, 'кон'),(-4.0, 'кона'),(-4.0, 'конв'),
                            (-4.0, 'коне'),(-4.0, 'конн'),(-4.0, 'коно'),(-4.0, 'клони')],
                        [
                            (-0.0, 'люди'),(-4.0, 'люд'),(-4.0, 'леди'),(-4.0, 'лю ди'),
                            (-4.0, 'блюди')]
                    ]
                ]
        """
        return [self._infer_instance(tokens) for tokens in batch]

def generate_operation_costs_dict(alphabet):
    from dp_components.levenshtein_searcher import SegmentTransducer
    from utils.recursive_dict_merge import recursive_dict_merge
    from utils.karta_slov_helper_fns import generate_karta_slov_costs_dict

    # make default costs dict without language specific substrings subtitution costs
    ops_costs = SegmentTransducer.make_default_operation_costs(alphabet)

    karta_slov_costs_dict = generate_karta_slov_costs_dict()

    ops_costs = recursive_dict_merge(ops_costs, karta_slov_costs_dict)

    # Let's show how deep the rabbit hole goes:
    distant_substitutions_costs = {
        "адский": {
            "аццкий": 0.7,

        },
        "ться": {
            "цца": 0.7,
            "ццо": 0.7,
            "ца": 0.7,
            "тся": 0.3,
        },
        "тся": {
            "цца": 0.7,
            "ца": 0.7,
            "ться": 0.3,
        },
        "нибудь": {
            "нить": 0.9,
        },
        "а": {
            "aa": 0.8,
            "aaa": 0.9,
            "aaaа": 0.9,
        },
        "е": {
            "еее": 1.0,
        },
        "о": {
            "оо": 1.0,
            "ооо": 1.0,
            "оооо": 1.0,
        },
        "очень": {
            "оч": 0.6
        },

        "пятьдесят": {
            "писят": 0.8
        },
        "ч": {
            "чч": 1.0,
            "ччч": 1.0,
            "чччч": 1.0,
        },
        "что": {
            "што": 0.3,
            "шо": 0.3,
            "чо": 0.2,
            # "чё": 0.5,
            "че": 0.2,
        },
        "в": {
            "ф": 0.8,
            "фф": 0.9,
        },
        "вообще": {
            "ваще": 0.8,
            "воще": 0.9,
        },
        "в общем": {
            "вообщем": 0.8,
            "вопщем": 0.5,
        },
        "по-моему": {
            "помойму": 0.8,
            "помоиму": 0.8,

        },
        "сч": {
            "щщ": 0.8,
            "щ": 0.9
        },
        "зч": {
            "щщ": 0.7,
            "щ": 0.8
        },
        "ик": {
            "ег": 1.0
        },
        "жч": {
            "щщ": 1.0
        },
        "вт": {"фф": 1.0},

        "сегодня": {
            "седня": 1.0
        },
        "сейчас": {
            "щаз": 1.0,
            "щас": 1.0,
            "счас": 0.8,
            "счаз": 0.8
        },
        "конечно": {
            "канешна": 1.0
        },

        "только": {
            "тока": 1.0
        },
        "тысяч": {
            "тыщ": 1.0,
            "тыщь": 1.0
        },
        "когда": {
            "када": 1.0
        },
        "собственно": {
            "собстно": 1.0
        },

        "здник": {
            "знег": 1.0
        },
        "не мог": {
            "немог": 0.5
        },
        "не зна": {
            "незна": 0.5
        },
        "из-за": {
            "изза": 0.3,
        }


    }

    ops_costs = recursive_dict_merge(ops_costs, distant_substitutions_costs)

    return ops_costs
