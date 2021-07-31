from pathlib import Path
from time import time

import numpy as np
import pandas as pd

from RCF.src.dataset import Dataset
from RCF.src.helper import parse_args, get_new_RCF_model
from commons.helper import read_row_from_result_file, prepare_path


def retrain(ks):
	"""
	retrain models without counterfactual sets for given values of k.
	Trained models are saved to user's home directory
	Args:
		ks:	values of k to consider
	"""
	args = parse_args()
	inputs = []
	input_files = [f"{args.algo}_{k}.csv" for k in ks]
	for file in input_files:
		inputs.append(pd.read_csv(file))
	inputs = pd.concat(inputs, ignore_index=True)
	print(inputs)

	home_dir = str(Path.home()) + '/pretrain-rcf-counterfactual'
	np.random.seed(1802)
	seeds = np.random.randint(1000, 10000, 5)
	seeds[0] = 2512

	for row in inputs.itertuples():
		idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement = read_row_from_result_file(row)
		if counterfactual is None:
			continue

		data = Dataset(ignored_user=user_id, ignored_items=counterfactual)
		args = parse_args()
		args.pretrain = -1

		for i, seed in enumerate(seeds):
			path = prepare_path(home_dir, user_id, counterfactual, seed)
			model = get_new_RCF_model(data, args, save_file=path + f'ml1M_{args.hidden_factor}')
			print('begin retraining', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement, i, seed)
			begin = time()
			model.train(data, args, seed=seed)
			print(f"done retraining {time() - begin}")


if __name__ == "__main__":
	retrain([5, 10, 20])
