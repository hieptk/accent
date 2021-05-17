import pandas as pd
import tensorflow.compat.v1 as tf

from helper import get_model


def generate_cf(algo, ks):
    """
    generate counterfactual explanations for multiple k values
    Args:
        algo: algorithm used to generate explanations
    	ks: values of k to consider

    Returns:

    """
    if algo == 'pure_fia':
        from pure_fia import find_counterfactual_multiple_k
    elif algo == 'fia':
        from fia import find_counterfactual_multiple_k
    else:
        from accent import find_counterfactual_multiple_k

    for i in range(len(ks) - 1):
        assert ks[i] < ks[i + 1]
    assert ks[-1] == 20

    all_results = []
    for _ in ks:
        all_results.append(
            {
                'user': [],
                'item': [],
                'topk': [],
                'counterfactual': [],
                'predicted_scores': [],
                'replacement': []
            }
        )

    model = get_model(use_recs=True)
    for user_id in range(model.num_users):
        print('testing user', user_id)
        tf.reset_default_graph()
        model = get_model(use_recs=True)
        res = find_counterfactual_multiple_k(user_id, ks, model)

        for j in range(len(ks)):
            all_results[j]['user'].append(user_id)
            counterfactual, rec, topk, predicted_scores, repl = res[j]
            all_results[j]['item'].append(rec)
            all_results[j]['topk'].append(topk)
            all_results[j]['counterfactual'].append(counterfactual)
            all_results[j]['predicted_scores'].append(predicted_scores)
            all_results[j]['replacement'].append(repl)

            print('k =', ks[j])
            if not counterfactual:
                print(f"Can't find counterfactual set for user {user_id}")
            else:
                print(f"Found a set of size {len(counterfactual)}: {counterfactual}")
                print("Old top k: ", topk)
                print("Replacement: ", repl, predicted_scores)

    for j in range(len(ks)):
        df = pd.DataFrame(all_results[j])
        df.to_csv(f'{algo}_{ks[j]}.csv', index=False)


if __name__ == "__main__":
    generate_cf(algo='accent', ks=[5, 10, 20])
