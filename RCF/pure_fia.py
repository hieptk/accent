from time import time

import numpy as np

from dataset import Dataset
from attention import init
from helper import parse_args, get_pretrained_RCF_model


def try_remove(removed_item, cur_scores, influences):
    new_scores = cur_scores - influences[:, removed_item]
    repl = np.argmax(new_scores[1:]) + 1
    return repl, new_scores[0] - new_scores[repl], new_scores


def find_counterfactual(cur_scores, recommended_item, topk, visited, influences):
    """
        given a user, find an explanation for that user using the "pure FIA" algorithm
        Args:
            cur_scores: current scores,
            recommended_item: current recommendation,
            topk: the original top k items,
            visited: list of interacted items,
            influences: list of influences of interactions on the recommendations

        Returns: a tuple consisting of:
                    - a set of items in the counterfactual explanation
                    - the originally recommended item
                    - a list of predicted scores after the removal of the counterfactual explanation
                    - the predicted replacement item
    """
    removed_items = set()
    cur_diff = cur_scores[0] - cur_scores[1]
    cur_repl = -1

    items = np.argsort(-influences[0])

    for item in items:
        if item in removed_items:
            continue
        cur_repl, cur_diff, cur_scores = try_remove(item, cur_scores, influences)
        removed_items.add(item)
        if cur_diff < 0:
            break

    res = set(visited[idx] for idx in removed_items)

    if cur_diff < 0:
        return res, recommended_item, list(cur_scores), topk[cur_repl]
    else:
        return None, recommended_item, None, -1


def find_counterfactual_multiple_k(user_id, ks, model, data, args):
    """
    given a user, find an explanation for that user using the "pure FIA" algorithm
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
    begin = time()
    for i in range(len(ks) - 1):
        assert ks[i] < ks[i + 1]
    cur_scores, recommended_item, topk, item_weights, cur_diff = init(user_id, ks[-1], model, data, args)
    cur_scores = np.array([cur_scores[item] for item in topk])
    influences = np.zeros((ks[-1], len(item_weights)))
    for i in range(ks[-1]):
        influences[i] = model.get_influence3(user_id, topk[i], data, args)

    visited = data.user_positive_list[user_id]

    res = []
    for k in ks:
        counterfactual, rec, predicted_scores, repl = find_counterfactual(cur_scores[:k], recommended_item,
                                                                          topk[:k], visited, influences[:k])
        res.append((counterfactual, rec, topk[:k], predicted_scores, repl))

    print('counterfactual time:', time() - begin)

    return res


def main():
    data = Dataset()
    args = parse_args()
    model = get_pretrained_RCF_model(data, args, path='pretrain-rcf')

    user_id = 1
    find_counterfactual_multiple_k(user_id, [5, 10, 15, 20], model, data, args)


if __name__ == "__main__":
    main()
