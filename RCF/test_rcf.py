import helper
from dataset import Dataset


def main():
	data = Dataset()
	args = helper.parse_args()
	model = helper.get_pretrained_RCF_model(data, args, path='pretrain-rcf')

	correct = 0
	for user in range(data.num_users):
		scores = model.get_scores_per_user(user, data, args)
		tmp = data.test_data[data.test_data['user'] == user]
		if tmp.shape[0] == 0:
			print(user, data.raw_user_id[user])
			continue
		scores_pos = scores[tmp['pos_item']]
		scores_neg = scores[tmp['neg_item']]
		u_correct = sum(scores_pos > scores_neg)
		print(user, u_correct, tmp.shape[0], u_correct / tmp.shape[0])
		correct += u_correct
	print(correct, data.test_data.shape[0], correct / data.test_data.shape[0])


def get_map():
	data = Dataset()
	args = helper.parse_args()
	model = helper.get_pretrained_RCF_model(data, args, path='pretrain-rcf')

	map = 0
	for user in range(data.num_users):
		print('begin user', user)
		scores = model.get_scores_per_user(user, data, args)

		tmp = data.test_data[data.test_data['user'] == user]
		if tmp.shape[0] == 0:
			print('empty user', user, data.raw_user_id[user])
			continue
		rank = [(scores[item], 1) for item in tmp['pos_item'].unique()]
		rank.extend([(scores[item], 0) for item in tmp['neg_item'].unique()])
		rank.sort(key=lambda item: -item[0])

		sum_pos = 0
		sum_precision = 0
		for i, (score, label) in enumerate(rank):
			sum_pos += label
			sum_precision += label * sum_pos / (i + 1)
		print(sum_precision / sum_pos)
		map += sum_precision / sum_pos

	print(map / data.num_users)


if __name__ == "__main__":
	main()
