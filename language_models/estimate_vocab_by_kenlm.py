import json
import sys
from pathlib import Path

import kenlm

from deeppavlov.models.bidirectional_lms import elmo_bilm

def estimate_prob_words_by_kenlm(kenlm_path: str,
                                 elmo_path: str,
                                 output_file: str):
    klm = kenlm.Model(kenlm_path)
    #elmo_lm = elmo_bilm.ELMoEmbedder(model_dir=elmo_path)
    with open(Path(elmo_path) / 'tokens_set.txt', 'r') as f:
        tokens = f.readlines()
        tokens = [i[:-1] for i in tokens if i != '\n']
    scores = [klm.score(token, bos=False, eos=False) for token in tokens]
    with open(output_file, 'w') as fw:
        json.dump(scores, fw)

if __name__ == '__main__':
    kenlm_path = sys.argv[1]
    elmo_path = sys.argv[2]
    output_path = sys.argv[3]
    estimate_prob_words_by_kenlm(kenlm_path, elmo_path, output_path)