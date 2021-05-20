import pandas as pd
import numpy as np
from ast import literal_eval
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import ttest_rel
import argparse


def get_success(file):
    data = pd.read_csv(file)

    res = np.zeros(data.shape[0], dtype=np.bool_)

    for id, row in data.iterrows():
        user_id, item_id, topk, counterfactual, predicted_scores, replacement = row[:6]
        if not isinstance(counterfactual, str) or not isinstance(row['actual_scores_avg'], str):
            continue
        topk = literal_eval(topk)
        assert item_id == topk[0]
        actual_scores = literal_eval(row['actual_scores_avg'])

        replacement_rank = topk.index(replacement)
        if actual_scores[replacement_rank] > actual_scores[0]:
            res[id] = True
    return res


def get_size(file, mask):
    data = pd.read_csv(file)
    data = data[mask].reset_index(drop=True)

    res = np.zeros(data.shape[0], dtype=np.int32)
    for id, row in data.iterrows():
        user_id, item_id, topk, counterfactual, predicted_scores, replacement = row[:6]
        counterfactual = literal_eval(counterfactual)
        assert isinstance(counterfactual, set)
        res[id] = len(counterfactual)

    return res


def compare_algo(file, file2):
    cont_table = np.zeros((2, 2))
    res = get_success(file)
    res2 = get_success(file2)
    for u, v in zip(res, res2):
        cont_table[1 - u, 1 - v] += 1
    print(f'{file}: {np.mean(res)}')
    print(f'{file2}: {np.mean(res2)}')
    print(cont_table)
    print(f'mcnemar: {mcnemar(cont_table).pvalue / 2}')

    both_ok = res & res2
    algo_size = get_size(file, both_ok)
    algo2_size = get_size(file2, both_ok)
    print(f'{file} size: {np.mean(algo_size)}')
    print(f'{file2} size: {np.mean(algo2_size)}')
    print(f't-test: {ttest_rel(algo_size, algo2_size)[1] / 2}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    parser.add_argument("--file2")
    args = parser.parse_args()
    compare_algo(args.file, args.file2)
