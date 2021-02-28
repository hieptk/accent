from time import time

from attention import init, try_remove


def find_counterfactual(user_id, k, model, data, args):
	cur_scores, recommended_item, topk, item_weights, cur_diff = init(user_id, k, model, data, args)

	removed_items = set()
	best_replacement = -1
	for item, _ in item_weights:
		removed_items.add(item)
		best_replacement, cur_diff, cur_scores = try_remove(user_id, recommended_item, removed_items, topk, model, data, args)
		if cur_diff < 0:  # the old recommended item is not the best any more
			break

	if cur_diff < 0:
		return removed_items, recommended_item, topk, [cur_scores[item] for item in topk], best_replacement
	else:
		return None, recommended_item, topk, None, -1


def find_counterfactual_multiple_k(user_id, ks, model, data, args):
	res = []
	for k in ks:
		begin = time()
		counterfactual, rec, topk, predicted_scores, repl = find_counterfactual(user_id, k, model, data, args)
		print('counterfactual time k =', k, time() - begin)
		res.append((counterfactual, rec, topk, predicted_scores, repl))
	return res
