import numpy as np

from pure_fia import PureFIA


class FIA(PureFIA):
    @staticmethod
    def find_counterfactual(cur_scores, recommended_item, topk, visited, influences):
        """
            given a user, find an explanation for that user using FIA
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

        items = np.argsort(-influences[0])  # sort items based on the influence on the top recommendation

        for item in items:
            if item in removed_items:
                continue
            repl, new_diff, new_scores = super().try_remove(item, cur_scores, influences)

            if new_diff < cur_diff:  # if the score gap is reduced
                cur_repl, cur_diff, cur_scores = repl, new_diff, new_scores
                removed_items.add(item)
                if cur_diff < 0:
                    break

        res = set(visited[idx] for idx in removed_items)

        if cur_diff < 0:
            return res, recommended_item, list(cur_scores), topk[cur_repl]
        else:
            return None, recommended_item, None, -1
