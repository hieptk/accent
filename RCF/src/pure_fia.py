from time import time

import numpy as np

from RCF.src.helper import init_explanation
from commons.fia_template import FIATemplate


class PureFIA(FIATemplate):

    @classmethod
    def find_counterfactual_multiple_k(cls, user_id, ks, model, data, args):
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
        cur_scores, recommended_item, topk, item_weights, cur_diff = init_explanation(user_id, ks[-1], model, data, args)
        cur_scores = np.array([cur_scores[item] for item in topk])
        influences = np.zeros((ks[-1], len(item_weights)))
        for i in range(ks[-1]):
            influences[i] = model.get_influence3(user_id, topk[i], data, args)

        visited = data.user_positive_list[user_id]

        res = []
        for k in ks:
            counterfactual, rec, predicted_scores, repl = cls.find_counterfactual(cur_scores[:k], recommended_item,
                                                                                      topk[:k], visited, influences[:k])
            res.append((counterfactual, rec, topk[:k], predicted_scores, repl))

        print('counterfactual time:', time() - begin)

        return res
