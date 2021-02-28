from ast import literal_eval

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.contingency_tables import mcnemar
from helper import parse_args


def evaluate(ks):
	args = parse_args()
	input_files = [f"{args.algo}_{k}.csv" for k in ks]

	for file in input_files:
		print(file)
		data = pd.read_csv(file)

		swap = 0
		set_size = 0

		for id, row in data.iterrows():
			user_id, item_id, topk, counterfactual, predicted_scores, replacement = row[:6]
			if not isinstance(counterfactual, str) or not isinstance(row['actual_scores_avg'], str):
				continue
			topk = literal_eval(topk)
			counterfactual = literal_eval(counterfactual)
			assert item_id == topk[0]
			actual_scores = literal_eval(row['actual_scores_avg'])

			replacement_rank = topk.index(replacement)
			if actual_scores[replacement_rank] > actual_scores[0]:
				swap += 1
				set_size += len(counterfactual)

		print('swap', swap, swap / data.shape[0])
		print('size', set_size / swap)


def get_success(file, use_swap):
	print(file, 'swap' if use_swap else 'topk')
	data = pd.read_csv(file)

	res = np.zeros(data.shape[0], dtype=np.bool_)

	for id, row in data.iterrows():
		user_id, item_id, topk, counterfactual, predicted_scores, replacement = row[:6]
		if not isinstance(counterfactual, str) or not isinstance(row['actual_scores_avg'], str):
			continue
		topk = literal_eval(topk)
		assert item_id == topk[0]
		actual_scores = literal_eval(row['actual_scores_avg'])

		if use_swap:
			replacement_rank = topk.index(replacement)
			if actual_scores[replacement_rank] > actual_scores[0]:
				res[id] = True
		else:
			if max(actual_scores) != actual_scores[0]:
				res[id] = True

	print(np.mean(res))
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

	print(np.mean(res))
	return res


def compare(file, file2, use_swap):
	a = get_success(file, use_swap)
	b = get_success(file2, use_swap)
	cont_table = np.zeros((2, 2))
	for u, v in zip(a, b):
		cont_table[1 - u, 1 - v] += 1
	print(cont_table)
	print(mcnemar(cont_table).pvalue / 2)


def compare_algo(algo, algo2):
	for k in [5, 10, 15, 20]:
		print(k)
		file = f"{algo}_{k}.csv"
		file2 = f"{algo2}_{k}.csv"
		is_strong = True
		compare(file, file2, use_swap=is_strong)
		algo_ok = get_success(file, use_swap=is_strong)
		algo2_ok = get_success(file2, use_swap=is_strong)
		both_ok = algo_ok & algo2_ok
		print(np.mean(both_ok))
		algo_size = get_size(file, both_ok)
		algo2_size = get_size(file2, both_ok)
		print(ttest_rel(algo_size, algo2_size)[1] / 2)
		print("--------------------")



if __name__ == "__main__":
	evaluate(ks=[5, 10, 20])
