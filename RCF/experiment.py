from generate_counterfactual import generate_cf
from retrain_counterfactual import retrain
from get_new_scores import get_new_scores
from evaluate_counterfactual import evaluate


def main():
    ks = [5, 10, 20]
    generate_cf(ks)
    retrain(ks)
    get_new_scores(ks)
    evaluate(ks)


if __name__ == "__main__":
    main()
