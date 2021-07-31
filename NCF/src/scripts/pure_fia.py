from time import time

import numpy as np

from NCF.src.scripts.helper import get_scores
from commons.fia_template import FIATemplate


class PureFIA(FIATemplate):
    @staticmethod
    def find_counterfactual_multiple_k(user, ks, model, data, args):
        """
            given a user, find an explanation for that user using the "pure FIA" algorithm
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

        cur_scores = np.array([scores[item] for item in topk])

        res = []
        for k in ks:
            counterfactual, rec, predicted_scores, repl = FIATemplate.find_counterfactual(cur_scores[:k], topk[0],
                                                                              topk[:k], visited, influences[:k])
            res.append((counterfactual, rec, topk[:k], predicted_scores, repl))

        print('counterfactual time:', time() - begin)

        return res
