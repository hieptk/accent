from get_rec import get_rec
from helper import get_model


def main():
    """
    train a new NCF model
    """
    get_model(use_recs=False)
    get_rec()


if __name__ == "__main__":
    main()
