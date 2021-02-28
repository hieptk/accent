from get_counterfactual import generate_cf
from retrain import retrain
from get_new_scores import get_new_scores
from evaluate_counterfactual import evaluate
from helper import parse_args
from get_rec import get_rec


def main():
    args = parse_args()
    ks = [5, 10, 20]
    get_rec()
    generate_cf(args.algo, ks)
    get_new_scores(args.algo, ks)
    retrain(args.algo, ks)
    get_new_scores(args.algo, ks)
    evaluate(args.algo, ks)


if __name__ == "__main__":
    main()
