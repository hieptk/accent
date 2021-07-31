import os

import pandas as pd

from RCF.src.Utilis import get_share_attributes
from RCF.src.moive_loader import movie_loader


def generate_interaction_data():
	"""
	generate interaction data for RCF
	Returns:
		write generated data to RCF/data/train.csv and RCF/data/test.csv
	"""
	path = os.path.join(os.path.dirname(__file__), '../data/')
	data = pd.read_table(os.path.join(path, 'u.data'), names=['user', 'item', 'rating', 'timestamp'])
	loader = movie_loader()
	train_data = {'user': [], 'pos_item': [], 'neg_item': []}
	test_data = {'user': [], 'pos_item': [], 'neg_item': []}
	for user in data['user'].unique():
		user_data = data[data['user'] == user].sort_values(by=['timestamp'])
		pos_items = user_data['item'][user_data['rating'] >= 3]
		neg_items = user_data['item'][user_data['rating'] < 3]
		pos_items = list(filter(lambda item: loader.movie_dict.get(item) is not None, pos_items))
		neg_items = list(filter(lambda item: loader.movie_dict.get(item) is not None, neg_items))
		if len(neg_items) < 10 or len(pos_items) < 10:
			continue
		n_pos_tests = len(pos_items) // 5
		n_neg_tests = len(neg_items) // 5
		used = dict()
		cnt = 0
		for pos_item in pos_items[:-n_pos_tests]:
			m1 = loader.movie_dict.get(pos_item)
			best_neg_item, best_shared_type, best_shared_tag, best_used = -1, -1, -1, -1
			for neg_item in neg_items[:-n_neg_tests]:
				m2 = loader.movie_dict.get(neg_item)
				shared_genre, shared_director, shared_actor = get_share_attributes(m1, m2)
				shared_tag = len(shared_genre) + len(shared_director) + len(shared_actor)
				shared_type = (len(shared_genre) > 0) + (len(shared_director) > 0) + (len(shared_actor) > 0)
				if (shared_type > best_shared_type) \
						or (shared_type == best_shared_type and shared_tag > best_shared_tag) \
						or (shared_type == best_shared_type and shared_tag == best_shared_tag and used.get(neg_item, 0) < best_used):
					best_neg_item, best_shared_type, best_shared_tag, best_used = neg_item, shared_type, shared_tag, used.get(neg_item, 0)
			if best_neg_item != -1:
				train_data['user'].append(user)
				train_data['pos_item'].append(pos_item)
				train_data['neg_item'].append(best_neg_item)
				used[best_neg_item] = used.get(best_neg_item, 0) + 1
				cnt += 1
			else:
				print(f'unmatched {user} {pos_item} not found negative')
		if len(used) > 0:
			for i in range(n_pos_tests):
				for j in range(n_neg_tests):
					test_data['user'].append(user)
					test_data['pos_item'].append(pos_items[-i - 1])
					test_data['neg_item'].append(neg_items[-j - 1])
	train_data = pd.DataFrame(train_data)
	test_data = pd.DataFrame(test_data)
	train_data.to_csv(os.path.join(path, 'train.csv'), index=False)
	test_data.to_csv(os.path.join(path, 'test.csv'), index=False)


def compress(series):
	values = set()
	for s in series:
		values.update(s.unique())
	raw_id = sorted(list(values))
	compressed_id = {raw_id[i]: i for i in range(len(raw_id))}

	res = [compressed_id, raw_id]
	for s in series:
		res.append(s.map(compressed_id))
	return res


def generate_interaction_data_ncf():
	"""
	generate interaction data for NCF
	Returns:
		write generated data to movielens_train.tsv
	"""
	data = pd.read_table('../data/u.data', names=['user', 'item', 'rating', 'timestamp'])
	loader = movie_loader()
	train_data = []
	for user in data['user'].unique():
		user_data = data[data['user'] == user].sort_values(by=['timestamp'])
		pos_items = user_data['item'][user_data['rating'] >= 3]
		neg_items = user_data['item'][user_data['rating'] < 3]
		pos_items = list(filter(lambda item: loader.movie_dict.get(item) is not None, pos_items))
		neg_items = list(filter(lambda item: loader.movie_dict.get(item) is not None, neg_items))
		if len(neg_items) < 10 or len(pos_items) < 10:
			continue
		n_pos_tests = len(pos_items) // 5
		n_neg_tests = len(neg_items) // 5
		all_items = set(pos_items[:-n_pos_tests]).union(set(neg_items[:-n_neg_tests]))
		user_data = user_data[user_data['item'].isin(all_items)]
		train_data.append(user_data)
	train_data = pd.concat(train_data)
	_, _, train_data['user'] = compress([train_data['user']])
	_, _, train_data['item'] = compress([train_data['item']])
	train_data.to_csv('movielens_train.tsv', index=False, sep='\t', header=False)


if __name__ == "__main__":
	generate_interaction_data()
	generate_interaction_data_ncf()
