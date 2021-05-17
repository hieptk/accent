from time import time

import numpy as np

from helper import get_scores, get_model


def try_replace(repl, score_gap, gap_infl):
    """
    given a replacement item, try to swap the replacement and the recommendation
    Args:
        repl: the replacement item
        score_gap: the current score gap between repl and the recommendation
        gap_infl: a list of items and their influence on the score gap

    Returns: if possible, return the set of items that must be removed to swap and the new score gap
            else, None, 1e9
    """
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


def find_counterfactual_multiple_k(user, ks, model):
    """
        given a user, find an explanation for that user using ACCENT
        Args:
            user: ID of user
            ks: a list of values of k to consider
            model: the recommender model, a Tensorflow Model object

        Returns: a list explanations, each correspond to one value of k. Each explanation is a tuple consisting of:
                    - a set of items in the counterfactual explanation
                    - the originally recommended item
                    - a list of items in the original top k
                    - a list of predicted scores after the removal of the counterfactual explanation
                    - the predicted replacement item
    """
    begin = time()
    u_indices = np.where(model.data_sets.train.x[:, 0] == user)[0]
    visited = [int(model.data_sets.train.x[i, 1]) for i in u_indices]
    assert set(visited) == model.data_sets.train.visited[user]
    influences = np.zeros((ks[-1], len(u_indices)))
    scores, topk = get_scores(user, ks[-1], model)
    for i in range(ks[-1]):
        test_idx = user * ks[-1] + i
        assert int(model.data_sets.test.x[test_idx, 0]) == user
        assert int(model.data_sets.test.x[test_idx, 1]) == topk[i]
        train_idx = model.get_train_indices_of_test_case([test_idx])
        tmp, u_idx, _ = np.intersect1d(train_idx, u_indices, return_indices=True)
        assert np.all(tmp == u_indices)
        tmp = -model.get_influence_on_test_loss([test_idx], train_idx)
        influences[i] = tmp[u_idx]

    res = None
    best_repl = -1
    best_i = -1
    best_gap = 1e9

    ret = []
    for i in range(1, ks[-1]):
        tmp_res, tmp_gap = try_replace(topk[i], scores[topk[0]] - scores[topk[i]], influences[0] - influences[i])
        if tmp_res is not None and (
                res is None or len(tmp_res) < len(res) or (len(tmp_res) == len(res) and tmp_gap < best_gap)):
            res, best_repl, best_i, best_gap = tmp_res, topk[i], i, tmp_gap

        if i + 1 == ks[len(ret)]:
            if res is not None:
                predicted_scores = np.array([scores[item] for item in topk[:(i + 1)]])
                for item in res:
                    predicted_scores -= influences[:(i + 1), item]
                assert predicted_scores[0] < predicted_scores[best_i]
                assert abs(predicted_scores[0] - predicted_scores[best_i] - best_gap) < 1e-3
                ret.append((set(visited[idx] for idx in res), topk[0], topk[:(i + 1)], list(predicted_scores), best_repl))
            else:
                ret.append((None, topk[0], topk[:(i + 1)], None, -1))

    print('counterfactual time', time() - begin)
    return ret


def main():
    model = get_model(use_recs=True)
    user_id = 11
    print(find_counterfactual_multiple_k(user_id, [5, 10, 15, 20], model))


if __name__ == "__main__":
    main()
