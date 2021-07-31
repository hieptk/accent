import numpy as np

from commons.explanation_algorithm_template import ExplanationAlgorithmTemplate


class FIATemplate(ExplanationAlgorithmTemplate):
    @staticmethod
    def try_remove(removed_item, cur_scores, influences):
        """
        predict new scores after removing some items using FIA
        Args:
            removed_item: the set of removed items
            cur_scores: the current scores
            influences: the influences of interactions on recommendations

        Returns:
            the new top recommendation,
            the score gap between the top-1 and the top-2 recommendations,
            the new scores
        """
        new_scores = cur_scores - influences[:, removed_item]
        repl = np.argmax(new_scores[1:]) + 1
        return repl, new_scores[0] - new_scores[repl], new_scores

    @staticmethod
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
            cur_repl, cur_diff, cur_scores = FIATemplate.try_remove(item, cur_scores, influences)
            removed_items.add(item)
            if cur_diff < 0:
                break

        res = set(visited[idx] for idx in removed_items)

        if cur_diff < 0:
            return res, recommended_item, list(cur_scores), topk[cur_repl]
        else:
            return None, recommended_item, None, -1
