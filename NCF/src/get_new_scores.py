from pathlib import Path

import tensorflow.compat.v1 as tf

from NCF.src.helper import get_model
from NCF.src.helper import get_scores as get_scores_per_user
from commons.helper import prepare_new_scores, counterfactual2path, get_new_scores_main


def get_scores(idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement, item2scores, home_dir):
    """
    get scores of all items after retrained
    Args:
        idx: test number
        user_id: ID of user
        item_id: ID of item
        topk: the top-k items
        counterfactual: the counterfactual set
        predicted_scores: the predicted scores
        replacement: the replacement item
        item2scores: a dict for caching
        home_dir: the directory where trained models are stored

    Returns:
        a 2d array where each row is the scores of all items in one retrain.
    """
    key = counterfactual2path(user_id, counterfactual)
    if key in item2scores:
        return item2scores[key]
    subfolders = prepare_new_scores(user_id, key, home_dir)
    if subfolders is None:
        return None

    new_scores = []

    for i in range(5):
        tf.reset_default_graph()
        model = get_model(use_recs=True)
        path = f'{home_dir}/{counterfactual2path(user_id, counterfactual)}/{i}/'
        model.saver.restore(model.sess, path + 'model')
        print('begin scoring', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement, i, path)
        scores, _ = get_scores_per_user(user_id, 1, model)
        new_scores.append(scores)

    item2scores[key] = new_scores
    return new_scores


def get_new_scores(algo, ks):
    """
        get new scores after retrained for the given values of k
        Args:
            algo: algorithm used to generate explanations
            ks: values of k to consider
    """
    input_files = [f"{algo}_{k}.csv" for k in ks]

    home_dir = str(Path.home()) + '/pretrain-ncf'
    get_new_scores_main(home_dir, input_files, get_scores)


if __name__ == "__main__":
    get_new_scores(algo='accent', ks=[5, 10, 20])
