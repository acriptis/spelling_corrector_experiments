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
from language_models.elmolm_on_torch import ELMOLMTorch


class TestSpellingCorrectors(unittest.TestCase):
    def setUp(self):
        # tf model
        self.elmo40in_deepkuz = ELMO40in2SpellingCorrector()

        # share sccg because it loads several minutes:
        self.sccg = self.elmo40in_deepkuz.sccg

        # pytorch model:
        torch_lm = ELMOLMTorch()
        self.elmo40in_torch = ELMO40in2SpellingCorrector(
            language_model=torch_lm,
            spelling_correction_candidates_generator=self.sccg,
            mini_batch_size=1)

    # def test_spelling_corrector_ELMO40inKuz(self):
    #     """
    #     Test initilization of spelling corrector
    #     download weights if it absent
    #     :return:
    #     """
    #     result = self.elmo40in_deepkuz(["Мама мыло раму"])
    #     self.assertEqual(result[0], "Мама мыла раму")
    #     print(result)


    def test_spelling_corrector_TorchELMO40in(self):
        """
        Test initilization of spelling corrector
        download weights if it absent
        :return:
        """
        result = self.elmo40in_torch(["Мама мыло раму"])
        self.assertEqual(result[0], "Мама мыла раму")
        print(result)


if __name__ == '__main__':

    unittest.main()
