from time import time

import numpy as np

from RCF.src.helper import get_topk
from commons.accent_template import AccentTemplate


class Accent(AccentTemplate):
    @classmethod
    def find_counterfactual_multiple_k(cls, user_id, ks, model, data, args):
        """
        given a user, find an explanation for that user using ACCENT
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
        cur_scores = model.get_scores_per_user(user_id, data, args)
        visited = data.user_positive_list[user_id]
        _, topk = get_topk(cur_scores, set(visited), ks[-1])
        recommended_item = topk[0][0]

        # init influence of actions on the top k items
        influences = np.zeros((ks[-1], len(visited)))
        for i in range(ks[-1]):
            influences[i] = model.get_influence3(user_id, topk[i][0], data, args)

        res = None
        best_repl = -1
        best_i = -1
        best_gap = 1e9

        ret = []
        for i in range(1, ks[-1]):  # for each item in the original top k
            # try to replace rec with this item
            tmp_res, tmp_gap = Accent.try_replace(topk[i][0], topk[0][1] - topk[i][1], influences[0] - influences[i])
            if tmp_res is not None and (
                    res is None or len(tmp_res) < len(res) or (len(tmp_res) == len(res) and tmp_gap < best_gap)):
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
