from ast import literal_eval

import pandas as pd

from helper import parse_args


def evaluate(ks):
    """
    given a list of k values, run the experiment for these values
    Args:
        ks: a list of k values
    """
    args = parse_args()
    input_files = [f"{args.algo}_{k}.csv" for k in ks]

    for file in input_files:
        print(file)
        data = pd.read_csv(file)

        swap = 0
        set_size = 0

        for id, row in data.iterrows():
            user_id, item_id, topk, counterfactual, predicted_scores, replacement = row[:6]
            if not isinstance(counterfactual, str) or not isinstance(row['actual_scores_avg'], str):
                continue
            topk = literal_eval(topk)
            counterfactual = literal_eval(counterfactual)
            assert item_id == topk[0]
            actual_scores = literal_eval(row['actual_scores_avg'])

            replacement_rank = topk.index(replacement)
            if actual_scores[replacement_rank] > actual_scores[0]:
                swap += 1
                set_size += len(counterfactual)

        print('swap', swap, swap / data.shape[0])
        print('size', set_size / swap)


if __name__ == "__main__":
    evaluate(ks=[5, 10, 20])
