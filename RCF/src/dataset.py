import os

import pandas as pd

from RCF.src.Utilis import get_share_attributes
from RCF.src.moive_loader import movie_loader


class Dataset:
	"""
		a dataset consists of:
		- train_data: a pandas dataframe with all user-pos-neg triples
		- test_data: similar to train_data
		- rel_data: a pandas dataframe with relation data
		- user_id: a map from raw user id to compressed user id
		- raw_user_id: reverse map of user_id
		- item_id: a map from raw item id to compressed item id
		- raw_item_id: reverse map of item_id
		- num_users: number of users
		- num_items: number of items
		- movie_dict: a dict from movie id to movie object
		- all_genres: list of all genres
		- all_directors: list of all directors
		- all_actors: list of all actors
		- num_genres: number of genres
		- num_directors: number of directors
		- num_actors: number of actors
		- user_positive_list: a 2d array where user_positive_list[i] is the list of all pos items of user i
	"""
	def __init__(self, ignored_user=-1, ignored_items=None):
		path = os.path.join(os.path.dirname(__file__), '../data/')
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
		"""
		given a list of some pandas series of integers, compress them to continous values.
		Example: [[1, 10, 3], [7, 9]] => [[0, 4, 1], [2, 3]]
		Args:
			series: a list of pandas series of integers to be compressed

		Returns: a list with two more elements than the input list:
					- the map from raw numbers to compressed values
					- the reverse map from compressed values to the raw values
					- compressed series
		"""
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