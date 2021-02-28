import numpy as np
import pandas as pd

import helper
from dataset import Dataset


def generate_cf(ks):
	data = Dataset()
	args = helper.parse_args()
	model = helper.get_pretrained_RCF_model(data, args, path='pretrain-rcf')

	np.random.seed(2512)
	# n_samples = 100
	# user_ids = np.random.choice(data.num_users, size=n_samples, replace=False)
	user_ids = list(range(data.num_users))
	n_samples = data.num_users

	all_results = []
	for _ in ks:
		all_results.append(
			{
				'user': [],
				'item': [],
				'topk': [],
				'counterfactual': [],
				'predicted_scores': [],
				'replacement': []
			}
		)

	if args.algo == 'pure_att':
		from pure_attention import find_counterfactual_multiple_k
	elif args.algo == 'attention':
		from attention import find_counterfactual_multiple_k
	elif args.algo == 'pure_fia':
		from pure_fia import find_counterfactual_multiple_k
	elif args.algo == 'fia':
		from fia import find_counterfactual_multiple_k
	else:
		from accent import find_counterfactual_multiple_k

	for i, user_id in enumerate(user_ids):
		print(f'testing user {i}/{n_samples}: {user_id}')
		res = find_counterfactual_multiple_k(user_id, ks, model, data, args)
		for j in range(len(ks)):
			all_results[j]['user'].append(user_id)
			counterfactual, rec, topk, predicted_scores, repl = res[j]
			all_results[j]['item'].append(rec)
			all_results[j]['topk'].append(topk)
			all_results[j]['counterfactual'].append(counterfactual)
			all_results[j]['predicted_scores'].append(predicted_scores)
			all_results[j]['replacement'].append(repl)

			print('k =', ks[j])
			if not counterfactual:
				print(f"Can't find counterfactual set for user {user_id}")
			else:
				print(f"Found a set of size {len(counterfactual)}: {counterfactual}")
				print("Old top k: ", topk)
				print("Replacement: ", repl, predicted_scores)

	for j in range(len(ks)):
		df = pd.DataFrame(all_results[j])
		df.to_csv(f'{args.algo}_{ks[j]}.csv', index=False)


if __name__ == "__main__":
	generate_cf([5, 10, 20])
