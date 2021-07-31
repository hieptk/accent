import pandas as pd
import tensorflow.compat.v1 as tf

from NCF.src.accent import Accent
from NCF.src.fia import FIA
from NCF.src.helper import get_model
from NCF.src.pure_fia import PureFIA
from commons.helper import init_all_results, append_result


def generate_cf(algo, ks):
    """
    generate counterfactual explanations for multiple k values
    Args:
        algo: algorithm used to generate explanations
    	ks: values of k to consider

    Returns:

    """
    if algo == 'pure_fia':
        explaner = PureFIA()
    elif algo == 'fia':
        explaner = FIA()
    else:
        explaner = Accent()

    for i in range(len(ks) - 1):
        assert ks[i] < ks[i + 1]
    assert ks[-1] == 20

    all_results = init_all_results(ks)

    model = get_model(use_recs=True)
    for user_id in range(model.num_users):
        print('testing user', user_id)
        tf.reset_default_graph()
        model = get_model(use_recs=True)
        res = explaner.find_counterfactual_multiple_k(user_id, ks, model, None, None)
        append_result(ks, all_results, user_id, res)

    for j in range(len(ks)):
        df = pd.DataFrame(all_results[j])
        df.to_csv(f'{algo}_{ks[j]}.csv', index=False)


if __name__ == "__main__":
    generate_cf(algo='accent', ks=[5, 10, 20])
