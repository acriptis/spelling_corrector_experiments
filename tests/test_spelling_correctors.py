import unittest
# import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print(sys.path)
from utilities.recursive_dict_merge import recursive_dict_merge
from spelling_correction_models.elmo_40in_spelling_corrector.elmo_40in2_spelling_corrector import \
    ELMO40in2SpellingCorrector
from spelling_correction_models.elmo_40in_spelling_corrector.elmo_40in2_reranking_spelling_corrector import \
    ELMO40in2RerankingSpellingCorrector



class TestELMO40inKuzSpellingCorrector(unittest.TestCase):
    def setUp(self):
        # self.ws = WeatherSkill()
        pass

    def test_spelling_corrector(self):
        """
        Test initilization of spelling corrector
        download weights if it absent
        :return:
        """
        elmo40in = ELMO40in2SpellingCorrector()
        # elmo40in = ELMO40in2RerankingSpellingCorrector()
        result = elmo40in(["Мама мыло раму"])
        print(result)


if __name__ == '__main__':

    unittest.main()
