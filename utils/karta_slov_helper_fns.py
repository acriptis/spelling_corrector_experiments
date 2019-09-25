import pandas as pd


def convert_transition_probas_matrix_2_cost_matrix(tr_matrix_df, min_dist=0.5, max_dist=1.01,
                                                   skip_yo=True):
    """
    Method converts matrix of probabilities of transductions into distance value
    (which is inverse multiplication of probability)
    """
    dist_matrix = tr_matrix_df.copy()
    costs_dict = {each_key: {} for each_key in tr_matrix_df.index}

    for each_row_idx, row in dist_matrix.iterrows():
        if each_row_idx == 'ё' and skip_yo:
            continue
        for each_col_idx in row.keys():
            if each_col_idx == 'ё' and skip_yo:
                continue
            if each_row_idx == each_col_idx:
                # the same transition has distance 0
                dist = 0.0
            else:
                proba = row[each_col_idx]
                #             print(proba)
                dist = max_dist - proba * (max_dist - min_dist)
            #                 print("Dist of transition from letter %s to letter %s:" % (each_row_idx, each_col_idx))
            #                 print(dist)
            dist_matrix.loc[each_row_idx, each_col_idx] = dist
            costs_dict[each_row_idx][each_col_idx] = dist
    return dist_matrix, costs_dict


def generate_karta_slov_costs_dict():
    """
    Helper method that generates a levenshtein costs dict from karta slov letters
    transitions statistics
    :return:
    """
    df = pd.read_csv(
        'https://raw.githubusercontent.com/dkulagin/kartaslov/master/dataset/orfo_and_typos/letter.matrix.csv',
        sep=';', index_col='INDEX_LETTER')
    dist_matrix, karta_slov_costs_dict = convert_transition_probas_matrix_2_cost_matrix(df)
    return karta_slov_costs_dict
