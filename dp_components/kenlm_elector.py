# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import List, Tuple

import kenlm

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger


class KenlmElector(Component):
    """Component that chooses a candidate with the highest product of base and language model probabilities

    Args:
         load_path: path to the kenlm model file
         beam_size: beam size for highest probability search

    Attributes:
        lm: kenlm object
        beam_size: beam size for highest probability search
    """

    def __init__(self, load_path: Path, beam_size: int = 4, *args, **kwargs):
        self.lm = kenlm.Model(str(expand_path(load_path)))
        self.beam_size = beam_size

    def __call__(self, batch: List[List[List[Tuple[float, str]]]]) -> List[List[str]]:
        """Choose the best candidate for every token

        Args:
            batch: batch of probabilities and string values of candidates for every token in a sentence.
            Ex.:
            [
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

        Returns:
            batch of corrected tokenized sentences
        """
        return [self._infer_instance(candidates) for candidates in batch]

    def _infer_instance(self, candidates: List[List[Tuple[float, str]]]):
        candidates = candidates + [[(0, '</s>')]]
        state = kenlm.State()
        self.lm.BeginSentenceWrite(state)
        beam = [(0, state, [])]
        for sublist in candidates:
            new_beam = []
            for beam_score, beam_state, beam_words in beam:
                for score, candidate in sublist:
                    prev_state = beam_state
                    c_score = 0
                    cs = candidate.split()
                    for candidate in cs:
                        state = kenlm.State()
                        c_score += self.lm.BaseScore(prev_state, candidate, state)
                        prev_state = state
                    new_beam.append((beam_score + score + c_score, state, beam_words + cs))
            new_beam.sort(reverse=True)
            beam = new_beam[:self.beam_size]
        score, state, words = beam[0]
        return words[:-1]

    ##########################################################################
    def _tokenize(self, sentence):
        return sentence.split()

    def estimate_pure_likelihood(self, sentence):
        """Given a sentence it estimates its likelihood without spelling correction fixes"""
        return self.lm.score(sentence)

    def score_sentences(self, sentences):
        """
        Scores batch of sentences
        """
        return [self.lm.score(sentence) for sentence in sentences]

    def estimate_likelihood_with_correction_scores(self, tokenized_sentence_with_correction_scores):
        """Given a sentence it estimates its likelihood with spelling correction fixes"""
        # TODO

    def score_sentences_hypotheses(self, hypotheses):
        candidates = candidates + [[(0, '</s>')]]
        state = kenlm.State()
        self.lm.BeginSentenceWrite(state)
        beam = [(0, state, [])]
        for sublist in candidates:
            new_beam = []
            for beam_score, beam_state, beam_words in beam:
                for score, candidate in sublist:
                    prev_state = beam_state
                    c_score = 0
                    cs = candidate.split()
                    for candidate in cs:
                        state = kenlm.State()
                        c_score += self.lm.BaseScore(prev_state, candidate, state)
                        prev_state = state
                    new_beam.append((beam_score + score + c_score, state, beam_words + cs))
            new_beam.sort(reverse=True)
            beam = new_beam[:self.beam_size]
        score, state, words = beam[0]
        return words[:-1]
