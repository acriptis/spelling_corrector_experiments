from evaluate import align_sents
from typing import List
from functools import partial

class Lettercaser(object):
    """It defines lettercases of tokens and can restore them.
    By default it detects only ['lower', 'upper', 'capitalize'] lettercases,
    but there is opportunity to expand that list with 'cases' argument
    Args:
        cases: dictionary that describes map,
               name of lettercase -> func that takes str and convert it in certain lettercase
        default_case: func: str->str that define transformation of string,
                      when lettercase was not be detected in 'put_in_case' func
    Attributes:
        cases: dictionary that describes map,
               name of lettercase -> func that takes str and convert it in certain lettercase
        default_case: func: str->str that define transformation of string,
                      when lettercase was not be detected in 'put_in_case' func
    """

    def __init__(self, cases: dict = None, default_case = None):
        if default_case is None:
            self.default_case = lambda x: x.lower()
        else:
            self.default_case = default_case
        if cases is None:
            self.cases = {
                "lower": lambda x: x.lower(),
                "capitalize": lambda x: x.capitalize(),
                "upper": lambda x: x.upper()
            }
        else:
            self.cases = cases

    def determine_lettercase(self, token):
        """It detemines case of token with 'cases' attribute
        Args:
            token: token lettercases of that have been detected
        """
        for case in self.cases:
            if token == self.cases[case](token):
                return case
        return None

    def put_in_lettercase(self, token: str, case: str):
        """It restore lettercases of tokens according to 'case' arg,
        if lettercase is not detected (case==None), 'default_case' func will be used
        Args:
            tokens: token that will be put in case
            case: name of lettercase
        Return:
            tokens in certain lettercases
            if lettercase was not detected then 'default_case'would be used
        """
        if case is None:
            return self.default_case(token)
        return self.cases[case](token)

class LettercaserForSpellchecker(Lettercaser):

    def __init__(self, cases: dict = None, default_case = None):
        super().__init__(cases, default_case)
        self.aligment_func = partial(align_sents, replace_cost=1.9)

    def correct_cases(self, source, correct):
        alignment = self.aligment_func(source=source, correct=correct)
        source_cases = [self.determine_lettercase(token) for token in source]
        correct_cases = ['lower'] * len(correct)
        for s_border, c_border in alignment:
            if len(range(*c_border)) == 1 and\
                    len(range(*s_border)) == 1: # one by one
                correct_cases[c_border[0]] = source_cases[s_border[0]]
            elif len(range(*c_border)) == 1 and\
                    len(range(*s_border)) > 1: # one by many
                for c_idx in range(*c_border):
                    correct_cases[c_idx] = source_cases[s_border[0]]
            elif len(range(*c_border)) > 1:
                if all([source_cases[i] == 'upper' for i in range(*s_border)]):
                    for c_idx in range(*c_border):
                        correct_cases[c_idx] = 'upper'
                if source_cases[s_border[0]] == 'capitalize':
                    correct_cases[c_border[0]] = 'capitalize'
        return correct_cases

    def rest_cases(self, source: List[str], correct: List[str]):
        correct_cases = self.correct_cases(source, correct)
        return [self.put_in_lettercase(token, case) for token, case in zip(correct, correct_cases)]

    def __call__(self, source: List[List[str]], corrections: List[List[str]]):
        ziped = zip(source, corrections)
        return [self.rest_cases(s, c) for s, c in ziped]

if __name__ == '__main__':
    letter = LettercaserForSpellchecker()
    print(letter(['Тут есть КТО НИБУДЬ'.split()], ['тут есть кто-нибудь'.split()]))
    print(letter(['Это происходит По сейдень'.split()], ['это происходит посей день'.split()]))
    print(letter(['По моему'.split()], ['по-моему'.split()]))