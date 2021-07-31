from time import time
from explanation_algo import ExplanationAlgorithm
from collections import Counter


class PureAttention(ExplanationAlgorithm):
    @staticmethod
    def try_remove(user_id, item_id, removed_item_id, topk, model, data, args):
        """
        try to remove some items and get the new scores while fixing all model parameters
        Args:
            user_id: ID of user
            item_id: ID of item to get score
            removed_item_id: a set of removed items
            topk: the original top k items
            model: the recommender model
            data: the dataset used to train the model, see dataset.py
            args: extra arguments for the model

        Returns:
            the new recommendation,
            the score gap between the new recommendation and the top-2 item
            the list of new scores of all items
        """
        scores = model.get_scores_per_user(user_id, data, args, removed_item_id)
        scores_dict = Counter({item: scores[item] for item in topk if item != item_id})
        replacement, replacement_score = scores_dict.most_common(1)[0]
        return replacement, scores[item_id] - replacement_score, scores

    @staticmethod
    def find_counterfactual(user_id, k, model, data, args):
        """
        given a user, find an explanation for that user using the "pure attention" algorithm
        Args:
            user_id: ID of user
            k: a single value of k to consider
            model: the recommender model, a Tensorflow Model object
            data: the dataset used to train the model, see dataset.py
            args: other args used for model

        Returns: a tuple consisting of:
            - a set of items in the counterfactual explanation
            - the originally recommended item
            - a list of items in the original top k
            - a list of predicted scores after the removal of the counterfactual explanation
            - the predicted replacement item
        """
        cur_scores, recommended_item, topk, item_weights, cur_diff = ExplanationAlgorithm.init(user_id, k, model, data,
                                                                                               args)

        removed_items = set()
        best_replacement = -1
        for item, _ in item_weights:
            removed_items.add(item)
            best_replacement, cur_diff, cur_scores = PureAttention.try_remove(user_id, recommended_item, removed_items,
                                                                              topk, model, data, args)
            if cur_diff < 0:  # the old recommended item is not the best any more
                break

        if cur_diff < 0:
            return removed_items, recommended_item, topk, [cur_scores[item] for item in topk], best_replacement
        else:
            return None, recommended_item, topk, None, -1

    @staticmethod
    def find_counterfactual_multiple_k(user_id, ks, model, data, args):
        """
        given a user, find an explanation for that user using the "pure attention" algorithm
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
        res = []
        for k in ks:
            begin = time()
            counterfactual, rec, topk, predicted_scores, repl = PureAttention.find_counterfactual(user_id, k, model,
                                                                                                  data, args)
            print('counterfactual time k =', k, time() - begin)
            res.append((counterfactual, rec, topk, predicted_scores, repl))
        return res
