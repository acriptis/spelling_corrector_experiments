from nltk.tokenize.moses import MosesDetokenizer
import re

# def tokenize_sentence_batch(self, sentences_batch, wrap_s=True):
#     """
#     input sentences (list of strings)
#     ouputs list of lists of tokens
#     """
#     assert isinstance(sentences_batch, list)
#
#     # wrap with S tokens
#     if wrap_s:
#         tok_sents = [['<S>'] + sent.split() + ['</S>'] for sent in sentences_batch]
#     else:
#         tok_sents = [sent.split() for sent in sentences_batch]
#     return tok_sents
#
# def tokenize_sentence(self, sentence, wrap_s=True):
#     """
#     """
#     tok_sent = sentence.split()
#     # wrap with S tokens
#     if wrap_s:
#         tok_sent = ['<S>'] + tok_sent + ['</S>']
#     return tok_sent
#
# Methods of the Manager of ELMO DATA Matrix:
# def find_top_k_word_idxes_for_elmo_mat(elmo_mat, col_idx, top_k=10):
#     """
#     Given elmo matrix and index of the word in sentence it returns indexes of the most probable candidates
#     :param elmo_mat:
#     :param col_idx:
#     :param top_k:
#     :return:
#     """
#     column = elmo_mat[col_idx]
#     sorted_indexes = column.argsort()
#     print(sorted_indexes[-top_k:])
#     return sorted_indexes[-top_k:]
#
#
# def find_top_k_words_for_elmo_mat(elmo_mat, col_idx, words, top_k=10):
#     sorted_indexes = find_top_k_word_idxes_for_elmo_mat(elmo_mat, col_idx, top_k=10)
#
#     return [words[idx] for idx in sorted_indexes]
#
#
# find_top_k_words_for_elmo_mat(out_mat, 3, torch_lm.words)

def detokenize(tokens):
    """Given a list of tokens it returns merged string"""
    detokenized_str = MosesDetokenizer().detokenize(tokens, return_str=True)
    detokenized_str = re.sub(r"\( (.+?)", r"(\1", detokenized_str)
    detokenized_str = re.sub(r"(.+?) »", r"\1»", detokenized_str)
    detokenized_str = re.sub(r"« (.+?)", r"«\1", detokenized_str)
    detokenized_str = re.sub(r": -\)", r":-)", detokenized_str)
    # detokenized_str = re.sub(r": D", r":D", detokenized_str)
    return detokenized_str

def yi_substitutor(sentence):
    """Substitutes й letter with и"""
    yi_pat = re.compile("й")

    return yi_pat.sub("и", sentence)

def yo_substitutor(sentence):
    """Substitutes ё letter with е"""
    yo_pat = re.compile("ё")

    return yo_pat.sub("е", sentence)

def yo_substitutor_batch(sentences):
    """Substitutes ё letter with е"""
    yo_pat = re.compile("ё")

    return [yo_pat.sub("е", each_sent) for each_sent in sentences]
