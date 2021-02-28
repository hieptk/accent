from time import time

import numpy as np

from dataset import Dataset
from helper import parse_args, get_pretrained_RCF_model, get_topk


def try_replace(repl, score_gap, gap_infl):
	print(f'try replace', repl, score_gap)
	sorted_infl = np.argsort(-gap_infl)

	removed_items = set()

	for idx in sorted_infl:
		if gap_infl[idx] < 0:  # cannot reduce the gap any more
			break
		removed_items.add(idx)
		score_gap -= gap_infl[idx]
		if score_gap < 0:  # the replacement passed the predicted
			break
	if score_gap < 0:
		print(f'replace {repl}: {removed_items}')
		return removed_items, score_gap
	else:
		print(f'cannot replace {repl}')
		return None, 1e9


def find_counterfactual(user_id, k, model, data, args):
	cur_scores = model.get_scores_per_user(user_id, data, args)
	visited = data.user_positive_list[user_id]
	_, topk = get_topk(cur_scores, set(visited), k)
	recommended_item = topk[0][0]
	influences = np.zeros((k, len(visited)))
	for i in range(k):
		influences[i] = model.get_influence3(user_id, topk[i][0], data, args)

	res = None
	best_repl = -1
	best_i = -1
	best_gap = 1e9
	for i in range(1, k):
		tmp_res, tmp_gap = try_replace(topk[i][0], topk[0][1] - topk[i][1], influences[0] - influences[i])
		if tmp_res is not None and (res is None or len(tmp_res) < len(res) or (len(tmp_res) == len(res) and tmp_gap < best_gap)):
			res, best_repl, best_i, best_gap = tmp_res, topk[i][0], i, tmp_gap

	predicted_scores = np.array([cur_scores[item] for item, _ in topk])
	for item in res:
		predicted_scores -= influences[:, item]
	assert predicted_scores[0] < predicted_scores[best_i]
	assert abs(predicted_scores[0] - predicted_scores[best_i] - best_gap) < 1e-6

	res = set(visited[idx] for idx in res)

	return res, recommended_item, [item for item, _ in topk], list(predicted_scores), best_repl


def find_counterfactual_multiple_k(user_id, ks, model, data, args):
	begin = time()
	for i in range(len(ks) - 1):
		assert ks[i] < ks[i + 1]
	cur_scores = model.get_scores_per_user(user_id, data, args)
	visited = data.user_positive_list[user_id]
	_, topk = get_topk(cur_scores, set(visited), ks[-1])
	recommended_item = topk[0][0]
	influences = np.zeros((ks[-1], len(visited)))
	for i in range(ks[-1]):
		influences[i] = model.get_influence3(user_id, topk[i][0], data, args)

	res = None
	best_repl = -1
	best_i = -1
	best_gap = 1e9

	ret = []
	for i in range(1, ks[-1]):
		tmp_res, tmp_gap = try_replace(topk[i][0], topk[0][1] - topk[i][1], influences[0] - influences[i])
		if tmp_res is not None and (res is None or len(tmp_res) < len(res) or (len(tmp_res) == len(res) and tmp_gap < best_gap)):
			res, best_repl, best_i, best_gap = tmp_res, topk[i][0], i, tmp_gap

		if i + 1 == ks[len(ret)]:
			predicted_scores = np.array([cur_scores[item] for item, _ in topk[:(i + 1)]])
			for item in res:
				predicted_scores -= influences[:(i + 1), item]
			assert predicted_scores[0] < predicted_scores[best_i]
			assert abs(predicted_scores[0] - predicted_scores[best_i] - best_gap) < 1e-6

			ret.append((set(visited[idx] for idx in res), recommended_item, [item for item, _ in topk[:(i + 1)]],
						list(predicted_scores), best_repl))

	print('counterfactual time', time() - begin)
	return ret


def main():
	data = Dataset()
	args = parse_args()
	model = get_pretrained_RCF_model(data, args, path='pretrain-rcf')

	user_id = 0
	find_counterfactual(user_id, 5, model, data, args)


if __name__ == "__main__":
	main()
