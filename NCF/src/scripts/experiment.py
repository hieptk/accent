from evaluate_counterfactual import evaluate
from get_counterfactual import generate_cf
from get_new_scores import get_new_scores
from helper import parse_args
from retrain import retrain


def main():
    """
    run the full experiment for an algorithm passed via the command line argument --algo
    """
    args = parse_args()
    ks = [5, 10, 20]
    generate_cf(args.algo, ks)
    get_new_scores(args.algo, ks)
    retrain(args.algo, ks)
    get_new_scores(args.algo, ks)
    evaluate(args.algo, ks)


if __name__ == "__main__":
    main()
