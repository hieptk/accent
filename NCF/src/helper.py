import argparse
import os
from collections import Counter

import numpy as np

from NCF.src.influence.NCF import NCF
from NCF.src.scripts.load_movielens import load_movielens


def parse_args():
    """
        parse all args from the console
        Returns:
            the parsed args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--avextol', type=float, default=1e-3,
                        help='threshold for optimization in influence function')
    parser.add_argument('--damping', type=float, default=1e-6,
                        help='damping term in influence function')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='l2 regularization term for training MF or NCF model')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate for training MF or NCF model')
    parser.add_argument('--embed_size', type=int, default=16,
                        help='embedding size')
    parser.add_argument('--maxinf', type=int, default=1,
                        help='remove type of train indices')
    parser.add_argument('--dataset', type=str, default='movielens',
                        help='name of dataset: movielens or yelp')
    parser.add_argument('--model', type=str, default='NCF',
                        help='model type: MF or NCF')
    parser.add_argument('--num_test', type=int, default=5,
                        help='number of test points of retraining')
    parser.add_argument('--num_steps_train', type=int, default=120000,
                        help='training steps')
    parser.add_argument('--num_steps_retrain', type=int, default=120000,
                        help='retraining steps')
    parser.add_argument('--reset_adam', type=int, default=0)
    parser.add_argument('--load_checkpoint', type=int, default=1)
    parser.add_argument('--retrain_times', type=int, default=4)
    parser.add_argument('--sort_test_case', type=int, default=1)
    parser.add_argument('--algo', default='accent')
    return parser.parse_args()


def get_scores(user, k, model):
    """
    get scores for a user
    Args:
        user: user to score
        k: number of top recommendations
        model: recommender model

    Returns:
        a dictionary containing score of all items,
        top k recommendations
    """
    users = np.ones(model.num_items) * user
    items = np.arange(model.num_items, dtype=np.float32)
    x = np.vstack([users, items]).T
    feed_dict = {
        model.input_placeholder: x
    }
    scores = model.sess.run(model.logits, feed_dict=feed_dict)
    visited = model.data_sets.train.visited[user]
    score_dict = Counter({item: scores[item] for item in range(model.num_items) if item not in visited})
    topk = [item for item, _ in score_dict.most_common(k)]
    return score_dict, topk


def get_model(use_recs=False):
    """
    get a new NCF model or load pretrained if exists
    Args:
        use_recs: if true, load top recommendations as the test set. See get_rec.py

    Returns:
        an NCF model
    """
    args = parse_args()
    if args.dataset == 'movielens':
        batch_size = 1246
        path = os.path.join(os.path.dirname(__file__), '../data')
        data_sets = load_movielens(path, batch=batch_size, use_recs=use_recs)
    else:
        raise NotImplementedError

    print(data_sets.train._num_examples)
    print(data_sets.validation._num_examples)
    print(data_sets.test._num_examples)

    weight_decay = args.weight_decay
    initial_learning_rate = args.lr
    num_users = int(np.max(data_sets.train._x[:, 0]) + 1)
    num_items = int(np.max(data_sets.train._x[:, 1]) + 1)
    print("number of users: %d" % num_users)
    print("number of items: %d" % num_items)
    print("number of training examples: %d" % data_sets.train._x.shape[0])
    print("number of testing examples: %d" % data_sets.test._x.shape[0])
    avextol = args.avextol
    damping = args.damping
    print("Using avextol of %.0e" % avextol)
    print("Using damping of %.0e" % damping)
    print("Using embedding size of %d" % args.embed_size)
    Model = NCF

    model = Model(
        num_users=num_users,
        num_items=num_items,
        embedding_size=args.embed_size,
        weight_decay=weight_decay,
        num_classes=1,
        batch_size=batch_size,
        data_sets=data_sets,
        initial_learning_rate=initial_learning_rate,
        damping=damping,
        decay_epochs=[10000, 20000],
        mini_batch=True,
        train_dir='scripts/output',
        log_dir='log',
        avextol=avextol,
        model_name='%s_%s_explicit_damping%.0e_avextol%.0e_embed%d_maxinf%d_wd%.0e' % (
            args.dataset, args.model, damping, avextol, args.embed_size, args.maxinf, weight_decay))
    print(f'Model name is: {model.model_name}')

    num_steps = args.num_steps_train
    iter_to_load = num_steps - 1
    if os.path.isfile("%s-%s.index" % (model.checkpoint_file, iter_to_load)):
        print('Checkpoint found, loading...')
        model.load_checkpoint(iter_to_load=iter_to_load)
    else:
        print('Checkpoint not found, start training...')
        model.train(
            num_steps=num_steps)
        model.saver.save(model.sess, model.checkpoint_file, global_step=num_steps - 1)

    return model
