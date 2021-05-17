import helper
from dataset import Dataset


def main():
	"""
	evaluate RCF models using percentage of correctly scored positive-negative pairs
	"""
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


if __name__ == "__main__":
	main()
