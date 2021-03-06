from time import time

from dataset import Dataset
from helper import parse_args, get_new_RCF_model
from test_rcf import main

if __name__ == '__main__':
    # Data loading
    args = parse_args()
    data = Dataset()

    save_file = 'pretrain-rcf/%s_%d' % ('ml1M', args.hidden_factor)
    # Training
    model = get_new_RCF_model(data, args)

    # model.evaluate()
    begin = time()
    print("begin train {}".format(begin))
    model.train(data, args, seed=2512)
    end = time()
    print("end train {} {}".format(end, end - begin))

    begin = time()
    print("begin test {}".format(begin))
    main()
    end = time()
    print("finish {} {}".format(end, end - begin))
