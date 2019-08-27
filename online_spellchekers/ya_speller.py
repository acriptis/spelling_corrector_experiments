# https://github.com/kiddick/speller
import requests


class Speller(object):
    service = 'http://speller.yandex.net/services/spellservice.json/checkText'

    def __init__(self, text, options=None, lang=None, format_text=None):
        self.text = text
        self.options = options
        self.lang = lang
        self.format_text = format_text
        self._answer = None

    def check(self):
        data = {'text': self.text}
        if self.options:
            data['options'] = self.options
        if self.lang:
            data['lang'] = self.lang
        if self.format_text:
            data['format'] = self.format_text
        answer = requests.post(url=self.service, data=data).json()
        return answer

    @property
    def answer(self):
        if self._answer is None:
            self._answer = self.check()
        return self._answer

    @property
    def correct(self):
        return not self.answer

    @property
    def spellsafe(self):
        raise NotImplementedError("Subclasses should implement this!")


class Word(Speller):

    @property
    def variants(self):
        if self.correct:
            return
        return self.answer[0]['s']

    @property
    def spellsafe(self):
        if self.correct:
            return
        return self.variants[0]


class Text(Speller):

    @property
    def spellsafe(self):
        changes = {el['word']: el['s'][0] for el in self.answer if len(el['s']) > 0}
        result = self.text
        for wrong, fixed in changes.items():
            result = result.replace(wrong, fixed)
        return result

    @property
    def errors(self):
        return [el['word'] for el in self.answer]

print(Text('42 is a cUl maagic namber').spellsafe)

import pandas as pd
from tqdm import tqdm
import sys

def spellcheck_dataframe(dataframe, text_field='text', lang=None):
    fixed_texts = []

    total = len(dataframe)
    for idx, line in tqdm(dataframe.iterrows(), total=total, leave=False):
        fixed_text = Text(line[text_field], lang=lang).spellsafe
        fixed_texts.append({
            'text_spellchecked': fixed_text,
            'text_original': line.text
        })

    return fixed_texts

if __name__ == '__main__':
    name_file = sys.argv[1]
    data_mail = pd.read_csv(name_file)
    data_mail_sp = pd.DataFrame(spellcheck_dataframe(data_mail, lang='ru'))
    data_mail_sp.to_csv('ya_'+name_file)