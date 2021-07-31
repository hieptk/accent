from pure_attention import PureAttention


class Attention(PureAttention):
    @staticmethod
    def find_counterfactual(user_id, k, model, data, args):
        """
        given a user, find an explanation for that user using the "attention" algorithm
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
        cur_scores, recommended_item, topk, item_weights, cur_diff = super().init(user_id, k, model, data, args)

        removed_items = set()
        best_replacement = -1
        for item, _ in item_weights:
            removed_items.add(item)
            replacement, new_diff, new_scores = super().try_remove(user_id, recommended_item, removed_items, topk,
                                                                   model, data, args)
            if new_diff < cur_diff:
                best_replacement, cur_diff, cur_scores = replacement, new_diff, new_scores
                if new_diff < 0:  # the old recommended item is not the best any more
                    break
            else:
                removed_items.remove(item)

        if cur_diff < 0:
            return removed_items, recommended_item, topk, [cur_scores[item] for item in topk], best_replacement
        else:
            return None, recommended_item, topk, None, -1
