from collections import Counter
from time import time

import helper
from Utilis import get_relational_data


def get_item_weights(user_id, item_id, model, data, args):
	"""
	get attention weights of all interactions made by user
	Args:
		user_id: ID of user
		item_id: ID of item whose score is being predicted
		model: the recommender model
		data: dataset used to trained model, see dataset.py
		args: extra arguments for model

	Returns:
		list of actions made by user, sorted by decreasing attention weights
	"""
	first_att = model.get_attention_type_user(user_id)[0]

	n_types = 4
	related_items = [None] * n_types
	values = [0] * n_types
	cnt = [0] * n_types

	related_items[0], related_items[1], related_items[2], related_items[3], values[1], values[2], values[3], cnt[0], \
		cnt[1], cnt[2], cnt[3] = get_relational_data(user_id, item_id, data)
	tmp = model.get_attention_user_item(user_id, item_id, related_items, values, cnt, args)
	second_att = [tmp[0][0][0], tmp[1][0][0], tmp[2][0][0], tmp[3][0][0]]

	item_att = {}
	for i in range(n_types):
		second_att[i] = second_att[i] * first_att[i]
		for j in range(cnt[i]):
			k = related_items[i][j]
			cur = item_att.get(k, 0)
			item_att[k] = cur + second_att[i][j]

	return sorted(item_att.items(), key=lambda x: -x[1])


def try_remove(user_id, item_id, removed_item_id, topk, model, data, args):
	"""
	try to remove some items and get the new scores while fixing all model parameters
	Args:
		user_id: ID of user
		item_id: ID of item to get score
		removed_item_id: a set of removed items
		topk: the original top k items
		model: the recommender model
		data: the dataset used to train the model, see dataset.py
		args: extra arguments for the model

	Returns:
		the new recommendation,
		the score gap between the new recommendation and the top-2 item
		the list of new scores of all items
	"""
	scores = model.get_scores_per_user(user_id, data, args, removed_item_id)
	scores_dict = Counter({item: scores[item] for item in topk if item != item_id})
	replacement, replacement_score = scores_dict.most_common(1)[0]
	return replacement, scores[item_id] - replacement_score, scores


def init(user_id, k, model, data, args):
	cur_scores = model.get_scores_per_user(user_id, data, args)
	visited = set(data.user_positive_list[user_id])  # get positive list for the userID
	_, topk = helper.get_topk(cur_scores, visited, k)
	recommended_item = topk[0][0]

	item_weights = get_item_weights(user_id, recommended_item, model, data, args)
	assert(len(visited) == len(item_weights))
	cur_diff = topk[0][1] - topk[1][1]
	return cur_scores, recommended_item, [item for item, _ in topk], item_weights, cur_diff


def find_counterfactual(user_id, k, model, data, args):
	"""
		given a user, find an explanation for that user using the "attention" algorithm
		Args:
			user_id: ID of user
			k: a single value of k to consider
			model: the recommender model, a Tensorflow Model object
			data: the dataset used to train the model, see dataset.py
			args: other args used for model

		Returns: a tuple consisting of:
					- a set of items in the counterfactual explanation
					- the originally recommended item
					- a list of items in the original top k
					- a list of predicted scores after the removal of the counterfactual explanation
					- the predicted replacement item
	"""
	cur_scores, recommended_item, topk, item_weights, cur_diff = init(user_id, k, model, data, args)

	removed_items = set()
	best_replacement = -1
	for item, _ in item_weights:
		removed_items.add(item)
		replacement, new_diff, new_scores = try_remove(user_id, recommended_item, removed_items, topk, model, data, args)
		if new_diff < cur_diff:
			best_replacement, cur_diff, cur_scores = replacement, new_diff, new_scores
			if new_diff < 0:  # the old recommended item is not the best any more
				break
		else:
			removed_items.remove(item)

	if cur_diff < 0:
		return removed_items, recommended_item, topk, [cur_scores[item] for item in topk], best_replacement
	else:
		return None, recommended_item, topk, None, -1


def find_counterfactual_multiple_k(user_id, ks, model, data, args):
	"""
	given a user, find an explanation for that user using the "attention" algorithm
	Args:
		user_id: ID of user
		ks: a list of values of k to consider
		model: the recommender model, a Tensorflow Model object
		data: the dataset used to train the model, see dataset.py
		args: other args used for model

	Returns: a list explanations, each correspond to one value of k. Each explanation is a tuple consisting of:
				- a set of items in the counterfactual explanation
				- the originally recommended item
				- a list of items in the original top k
				- a list of predicted scores after the removal of the counterfactual explanation
				- the predicted replacement item
	"""
	res = []
	for k in ks:
		begin = time()
		counterfactual, rec, topk, predicted_scores, repl = find_counterfactual(user_id, k, model, data, args)
		print('counterfactual time k =', k, time() - begin)
		res.append((counterfactual, rec, topk, predicted_scores, repl))
	return res
