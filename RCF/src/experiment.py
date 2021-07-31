from RCF.src.generate_counterfactual import generate_cf
from RCF.src.helper import parse_args
from commons.helper import evaluate_files
from get_new_scores import get_new_scores
from retrain_counterfactual import retrain


def main():
    """
    run the full experiment for an algorithm passed via the command line argument --algo
    """
    ks = [5, 10, 20]
    generate_cf(ks)
    retrain(ks)
    get_new_scores(ks)
    evaluate_files(parse_args, ks)


if __name__ == "__main__":
    main()
