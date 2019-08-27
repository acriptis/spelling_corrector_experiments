from typing import List

def prepare_sections(names_of_models):
    names = ['source', 'true'] + names_of_models
    lens = [len(i) for i in names]
    right_border = max(lens) + 1
    sections = [f"{name}:" + " "*(right_border - len(name)) for name in names]
    return sections

def find_example_where_model_mistaked(models_input, true_sent ,predicts_of_models):
    result = []
    predicts = list(zip(*predicts_of_models))
    ziped = zip(models_input, true_sent, predicts)
    for source, true, predicts in ziped:
        if any([i != true for i in predicts]):
            result.append((source, true, predicts))
    return result

def write_mistake_to_file(mistakes, sections, output_file):
    with open(output_file, 'w') as fw:
        for idx, z in enumerate(mistakes):
            source, true, predicts = z
            fw.write(f'example: {idx}\n')
            fw.write(f'{sections[0]}{source}\n')
            fw.write(f'{sections[1]}{true}\n')
            for idx, predict in enumerate(predicts):
                fw.write(f'{sections[idx+2]}{predict}\n')
            fw.write(f'\n')

def where_is_mistake(models_input:List[str],
                     true_sent: List[str],
                     predicts_of_models: List[List[str]],
                     names_of_models: List[str],
                     output_file: str):
    sections = prepare_sections(names_of_models)
    with_mistakes = find_example_where_model_mistaked(models_input, true_sent, predicts_of_models)
    write_mistake_to_file(with_mistakes, sections, output_file)

if __name__ == '__main__':
    with open('./evaluate_models/dialog_testset.txt', 'r') as fsource:
        source = fsource.readlines()
        source = [i[:-1] for i in source if i != '']

    with open('./evaluate_models/true_dialog_testset.txt', 'r') as ftrue:
        true = ftrue.readlines()
        true = [i[:-1] for i in true if i != '']

    with open('./evaluate_models/ya_dialog_test.txt', 'r') as fya:
        ya = fya.readlines()
        ya = [i[:-1] for i in ya if i != '']

    with open('./evaluate_models/azure_dialog_test.txt', 'r') as fazure:
        azure = fazure.readlines()
        azure = [i[:-1] for i in azure if i != '']

    with open('./evaluate_models/levenshtein_predicts_testset.txt', 'r') as fleven:
        leven = fleven.readlines()
        leven = [i[:-1] for i in leven if i != '']

    with open('./evaluate_models/brillmore_dialog_testset.txt', 'r') as fbrill:
        brill = fbrill.readlines()
        brill = [i[:-1] for i in brill if i != '']

    where_is_mistake(source, true, [ya, azure, leven, brill], ['yandex', 'azure', 'leven_deep', 'brillmore'], 'diff_sp_models')
