import torch

ru_lm = torch.hub.load('pytorch/fairseq',
                       'transformer_lm.wmt19.ru', tokenizer='moses',
                       bpe='fastbpe')
                       # bpe='fastbpe', force_reload=True)
print("Model is loaded")
output = ru_lm.sample('DeepPavlov - это ', beam=1, sampling=True, sampling_topk=10, temperature=0.8)
print(output)
output = ru_lm.sample('Для Алексея Сорокина эта неделя будет', beam=1, sampling=True, sampling_topk=10, temperature=0.8)
print(output)
output = ru_lm.sample('Для Сергея Селиверстова эта неделя будет ', beam=1, sampling=True, sampling_topk=10, temperature=0.8)
print(output)
output = ru_lm.sample('Для Михаила Бурцева эта неделя будет ', beam=1, sampling=True, sampling_topk=10, temperature=0.8)
print(output)
output = ru_lm.sample('Для Дмитрия Карпова эта неделя будет ', beam=1, sampling=True, sampling_topk=10, temperature=0.8)
print(output)
# output = ru_lm.sample('Дмитрий Карпов - это ', beam=1, sampling=True, sampling_topk=10, temperature=0.6)
# print(output)
# output = ru_lm.sample('Михаил Бурцев - это ', beam=1, sampling=True, sampling_topk=10, temperature=0.7)
# print(output)
# output = ru_lm.sample('Диляра Баймурзина - это ', beam=1, sampling=True, sampling_topk=10, temperature=0.8)
# print(output)
import ipdb; ipdb.set_trace()

print(output)


#
# ################################################################
from fairseq.models.transformer_lm import TransformerLanguageModel
custom_lm = TransformerLanguageModel.from_pretrained('/home/alx/Cloud/spell_corr/py__spelling_corrector/language_models/fairseq_transformer_lm', 'checkpoint100.pt', tokenizer='moses', bpe='fastbpe')
custom_lm.sample('Barack Obama', beam=5)
# Sample from the language model
# ru_lm.sample('Barack Obama', beam=1, sampling=True, sampling_topk=10, temperature=0.8)
# "Barack Obama is coming to Sydney and New Zealand (...)"
# #
# # # The same interface can be used with custom models as well
# # from fairseq.models.transformer_lm import TransformerLanguageModel
# # custom_lm = TransformerLanguageModel.from_pretrained('/path/to/model/dir', 'checkpoint100.pt', tokenizer='moses', bpe='fastbpe')
# # custom_lm.sample('Barack Obama', beam=5)