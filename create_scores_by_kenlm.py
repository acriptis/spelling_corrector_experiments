import json
import sys

import kenlm

from deeppavlov.models.bidirectional_lms import elmo_bilm

def estimate_prob_words_by_kenlm(kenlm_path: str,
                                 elmo_path: str,
                                 output_file: str):
    klm = kenlm.Model(kenlm_path)
    elmo_lm = elmo_bilm.ELMoEmbedder(model_dir=elmo_path)
    scores = [klm.score(token, bos=False, eos=False) for token in elmo_lm.get_vocab()]
    with open(output_file, 'w') as fw:
        json.dump(scores, fw)

if __name__ == '__main__':
    kenlm_path = sys.argv[1]
    elmo_path = sys.argv[2]
    output_path = sys.argv[3]
    estimate_prob_words_by_kenlm(kenlm_path, elmo_path, output_path)