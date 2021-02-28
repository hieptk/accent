import pandas as pd

from moive_loader import movie_loader
from Utilis import get_share_attributes


class Dataset:
	def __init__(self, path='./ML100K/', ignored_user=-1, ignored_items=None):
		self.train_data = pd.read_csv(path + 'train.csv')
		self.test_data = pd.read_csv(path + 'test.csv')
		self.rel_data = pd.read_csv(path + 'relational_data.csv')
		self.user_id, self.raw_user_id, self.train_data['user'], self.test_data['user'] = \
			self.compress([self.train_data['user'], self.test_data['user']])
		self.item_id, self.raw_item_id, self.train_data['pos_item'], self.train_data['neg_item'], \
		self.test_data['pos_item'], self.test_data['neg_item'], self.rel_data['head'], self.rel_data['tail_pos'], self.rel_data['tail_neg'] = \
			self.compress([self.train_data['pos_item'], self.train_data['neg_item'], self.test_data['pos_item'], self.test_data['neg_item'],
						  self.rel_data['head'], self.rel_data['tail_pos'], self.rel_data['tail_neg']])
		self.num_users = len(self.user_id)
		self.num_items = len(self.item_id)
		loader = movie_loader(self.item_id)
		self.movie_dict = loader.movie_dict
		self.all_genres = loader.genre_list
		self.all_directors = loader.director_list
		self.all_actors = loader.actor_list
		self.num_genres = len(self.all_genres)
		self.num_directors = len(self.all_directors)
		self.num_actors = len(self.all_actors)

		if ignored_user != -1:
			self.train_data = self.train_data[(self.train_data['user'] != ignored_user) |
											  (~self.train_data['pos_item'].isin(ignored_items))].reset_index(drop=True)
		self.user_positive_list = self.get_user_positive_list()

	def get_user_positive_list(self):
		res = [list(self.train_data['pos_item'][self.train_data['user'] == u]) for u in range(self.num_users)]
		return res

	@staticmethod
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


if __name__ == "__main__":
	data = Dataset()
	print(data.num_users, data.num_items)
	print(data.train_data)
	print(data.test_data)
	# items = [305, 566, 595, 494, 483]
	# items = [305, 14, 595, 117, 566, 483]
	items = [305, 14, 595, 150, 566, 185, 483]
	for item in items:
		print(data.raw_item_id[item])
	m1 = data.movie_dict[items[0]]
	for movie in items[1:]:
		m2 = data.movie_dict[movie]
		print(get_share_attributes(m1, m2))