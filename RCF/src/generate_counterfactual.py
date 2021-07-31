import pandas as pd

from RCF.src import helper
from RCF.src.dataset import Dataset
from RCF.src.pure_attention import PureAttention
from RCF.src.attention import Attention
from RCF.src.pure_fia import PureFIA
from RCF.src.fia import FIA
from RCF.src.accent import Accent
import os
from commons.generate_counterfactual import append_result, init_all_results


def generate_cf(ks):
	"""
	generate counterfactual explanations for multiple k values
	Args:
		ks: values of k to consider

	Returns:

	"""
	data = Dataset()
	args = helper.parse_args()
	model = helper.get_pretrained_RCF_model(data, args, path=os.path.join(os.path.dirname(__file__), 'RCF/src/pretrain-rcf'))

	user_ids = list(range(data.num_users))
	n_samples = data.num_users

	all_results = init_all_results(ks)

	if args.algo == 'pure_att':
		explaner = PureAttention()
	elif args.algo == 'attention':
		explaner = Attention()
	elif args.algo == 'pure_fia':
		explaner = PureFIA()
	elif args.algo == 'fia':
		explaner = FIA()
	else:
		explaner = Accent()

	for i, user_id in enumerate(user_ids):
		print(f'testing user {i}/{n_samples}: {user_id}')
		res = explaner.find_counterfactual_multiple_k(user_id, ks, model, data, args)
		append_result(ks, all_results, user_id, res)

	for j in range(len(ks)):
		df = pd.DataFrame(all_results[j])
		df.to_csv(f'{args.algo}_{ks[j]}.csv', index=False)


if __name__ == "__main__":
	generate_cf([5, 10, 20])
