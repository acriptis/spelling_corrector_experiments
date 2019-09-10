import string
from math import log10
from typing import Iterable, List, Tuple

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger

from deeppavlov.models.spelling_correction.levenshtein.levenshtein_searcher import LevenshteinSearcher

class LevenshteinSearcherComponent(Component):
    """Component that finds replacement candidates for tokens at a set Damerau-Levenshtein distance

    Args:
        words: list of every correct word
        max_distance: maximum allowed Damerau-Levenshtein distance between source words and candidates
        error_probability: assigned probability for every edit

    Attributes:
        max_distance: maximum allowed Damerau-Levenshtein distance between source words and candidates
        error_probability: assigned logarithmic probability for every edit
        vocab_penalty: assigned logarithmic probability of an out of vocabulary token being the correct one without
         changes
    """

    _punctuation = frozenset(string.punctuation)

    def __init__(self, words: Iterable[str], max_distance: int=1, error_probability: float=1e-4, *args, **kwargs):
        words = list({word.strip().lower().replace('ё', 'е') for word in words})
        alphabet = sorted({letter for word in words for letter in word})
        self.max_distance = max_distance
        self.error_probability = log10(error_probability)
#         self.vocab_penalty = self.error_probability * 2
        self.vocab_penalty = 0
        self.searcher = LevenshteinSearcher(alphabet, words, allow_spaces=True, euristics=2)

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