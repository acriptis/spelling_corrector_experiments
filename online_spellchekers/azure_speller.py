import requests
import json
import time
import pandas as pd
from tqdm import tqdm
import sys

API_KEY = "e8c2a097b59940eb806cf41b6471e840"
ENDPOINT = "https://noteastynotfree.cognitiveservices.azure.com/bing/v7.0/spellcheck"

def get_json_response(text: str):
    data = {'text': text}
    params = {
        'mode': 'spell',
        'setLang': 'ru'
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Ocp-Apim-Subscription-Key': API_KEY,
    }
    response = requests.post(ENDPOINT, headers=headers, params=params, data=data)
    return response.json()

def fix_mistake(text: str, mistake: dict, sent_offset: int) -> tuple:
    offset = mistake['offset']
    left_border = offset + sent_offset
    right_border = left_border + len(mistake['token'])
    if mistake['type'] == 'UnknownToken':
        best_suggestion = mistake['suggestions'][0]['suggestion']
        fixed_text = text[:left_border] + best_suggestion + text[right_border:]
    elif mistake['type'] == 'RepeatedToken':
        fixed_text = text[:left_border] + text[right_border+1:]
    return fixed_text, sent_offset

def correct_sent(text: str):
    json_response = get_json_response(text)
    if not json_response["flaggedTokens"]:
        return text
    sent_offset = 0
    for mistake in json_response["flaggedTokens"]:
        text, sent_offset = fix_mistake(text, mistake, sent_offset)
    return text

def spellcheck_dataframe(dataframe, text_field='text', lang='ru'):
    fixed_texts = []

    total = len(dataframe)
    for idx, line in tqdm(dataframe.iterrows(), total=total, leave=False):
        if line.get('text_spellchecked'):
            continue
        try:
            fixed_text = correct_sent(line[text_field])
            fixed_texts.append({
                'text_spellchecked': fixed_text,
                'text': line[text_field]
            })
            time.sleep(1)
        except:
            fixed_texts.append({
                'text_spellchecked': None,
                'text': line[text_field],
            })
            time.sleep(0.01)
    return fixed_texts

if __name__ == '__main__':
    source_file = sys.argv[1]
    output_file = sys.argv[2]
    data_mail = pd.read_csv(source_file)
    data_mail_sp = pd.DataFrame(spellcheck_dataframe(data_mail, lang='ru'))
    data_mail_sp.to_csv(output_file)