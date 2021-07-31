from pathlib import Path

import numpy as np

from RCF.src.dataset import Dataset
from RCF.src.helper import parse_args, get_pretrained_RCF_model
from commons.helper import counterfactual2path, prepare_new_scores, get_new_scores_main


def get_scores(idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement, item2scores, home_dir):
	"""
	get scores of all items after retrained
	Args:
		idx: test number
		user_id: ID of user
		item_id: ID of item
		topk: the top-k items
		counterfactual: the counterfactual set
		predicted_scores: the predicted scores
		replacement: the replacement item
		item2scores: a dict for caching
		home_dir: the directory where trained models are stored

	Returns:
		a 2d array where each row is the scores of all items in one retrain.
	"""
	key = counterfactual2path(user_id, counterfactual)
	if key in item2scores:  # if cached
		return item2scores[key]

	subfolders = prepare_new_scores(user_id, key, home_dir)
	if subfolders is None:
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


def get_new_scores(ks):
	"""
	get new scores after retrained for the given values of k
	Args:
		ks: values of k to consider
	"""
	args = parse_args()
	input_files = [f"{args.algo}_{k}.csv" for k in ks]

	home_dir = str(Path.home()) + '/pretrain-rcf-counterfactual'
	get_new_scores_main(home_dir, input_files, get_scores)


if __name__ == "__main__":
	get_new_scores([5, 10, 20])
