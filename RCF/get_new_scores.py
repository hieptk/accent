import os
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import Dataset
from helper import parse_args, get_pretrained_RCF_model
from retrain_counterfactual import counterfactual2path


def get_scores(idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement, item2scores, home_dir):
	key = counterfactual2path(user_id, counterfactual)
	if key in item2scores:
		return item2scores[key]
	if not Path(f'{home_dir}/{key}/').exists():
		print('missing', user_id, key)
		return None
	subfolders = sorted([f.path for f in os.scandir(f'{home_dir}/{key}/') if f.is_dir()])
	if len(subfolders) != 5:
		print('missing', user_id, key, len(subfolders))
		return None

	data = Dataset(ignored_user=user_id, ignored_items=counterfactual)
	args = parse_args()
	args.pretrain = -1

	new_scores = np.zeros(shape=(5, data.num_items))
	for i, path in enumerate(subfolders):
		model = get_pretrained_RCF_model(data, args, path)
		print('begin scoring', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement, i, path)

		new_scores[i] = model.get_scores_per_user(user_id, data, args)
	item2scores[key] = new_scores
	return new_scores


def get_topk_scores(idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement, item2scores, home_dir):
	scores = get_scores(idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement, item2scores, home_dir)
	if scores is None:
		return None

	res = np.zeros((5, len(topk)))
	for i in range(5):
		res[i] = [scores[i][item] for item in topk]

	return res


def get_new_scores(ks):
	args = parse_args()
	input_files = [f"{args.algo}_{k}.csv" for k in ks]

	home_dir = str(Path.home()) + '/pretrain-rcf-counterfactual'

	item2scores = dict()

	for file in input_files:
		print('begin file', file)
		inputs = pd.read_csv(file)
		for row in inputs.itertuples():
			idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement = row[:7]
			topk = literal_eval(topk)
			if not isinstance(counterfactual, str):
				print('skip', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement)
				continue
			counterfactual = literal_eval(counterfactual)
			if isinstance(predicted_scores, str):
				predicted_scores = literal_eval(predicted_scores)
			else:
				predicted_scores = None
			assert item_id == topk[0]
			print('begin idx', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement)

			scores = get_topk_scores(idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement,
									 item2scores, home_dir)
			if scores is None:
				print('bad scores', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement)
				continue
			assert len(scores) == 5

			for i in range(5):
				inputs.at[idx, f'actual_scores_{i}'] = str(list(scores[i]))
			s = np.mean(scores, axis=0)
			inputs.at[idx, f'actual_scores_avg'] = str(list(s))

			print('avg new scores', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement, s)

		inputs.to_csv(file, index=False)


if __name__ == "__main__":
	get_new_scores([5, 10, 20])
