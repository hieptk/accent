import hashlib
from ast import literal_eval
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

from helper import get_model, parse_args


def counterfactual2path(user, counterfactual_set):
    res = f'{user}-{"-".join(str(x) for x in sorted(counterfactual_set))}'
    if len(res) < 255:
        return res
    return hashlib.sha224(res.encode()).hexdigest()


def retrain(algo, ks):
    """
        retrain models without counterfactual sets for given values of k.
        Trained models are saved to user's home directory
        Args:
            algo: algorithms used to generate explanations
            ks:	values of k to consider
    """
    inputs = []
    input_files = [f"{algo}_{k}.csv" for k in ks]
    for file in input_files:
        inputs.append(pd.read_csv(file))
    inputs = pd.concat(inputs, ignore_index=True)
    print(inputs)

    home_dir = str(Path.home()) + '/pretrain-ncf-amazon'
    args = parse_args()
    np.random.seed(1802)

    tf.reset_default_graph()
    model = get_model(use_recs=True)

    for row in inputs.itertuples():
        idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement = row[:7]
        if not isinstance(counterfactual, str):
            print('skip', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement)
            continue
        topk = literal_eval(topk)
        counterfactual = literal_eval(counterfactual)
        if isinstance(predicted_scores, str):
            predicted_scores = literal_eval(predicted_scores)
        else:
            predicted_scores = None
        print('begin idx', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement)

        keep = [i for i in range(model.data_sets.train.x.shape[0]) if int(model.data_sets.train.x[i, 0]) != user_id or
                int(model.data_sets.train.x[i, 1]) not in counterfactual]
        for i in range(5):
            path = f'{home_dir}/{counterfactual2path(user_id, counterfactual)}/{i}/'
            if Path(path).exists():
                print('already done', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement, i)
                continue

            Path(path).mkdir(parents=True, exist_ok=True)
            tf.reset_default_graph()
            model = get_model(use_recs=True)
            np.random.seed(i)
            tf.set_random_seed(i)
            # model.sess.run(model.reset_optimizer_op)
            print('begin retraining', idx, user_id, item_id, topk, counterfactual, predicted_scores, replacement, i)
            begin = time()
            model.retrain(num_steps=args.num_steps_retrain, feed_dict=model.fill_feed_dict_with_some_ex(model.data_sets.train, keep))
            print(f"done retraining {time() - begin}")
            model.saver.save(model.sess, path + '/model')


if __name__ == "__main__":
    retrain(algo='accent', ks=[5, 10, 20])
