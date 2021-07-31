import os
from time import time

from RCF.src.dataset import Dataset
from RCF.src.helper import parse_args, get_new_RCF_model
from RCF.src.test_rcf import main

if __name__ == '__main__':
    # Data loading
    args = parse_args()
    args.pretrain = -1
    data = Dataset()

    path = os.path.dirname(__file__)
    save_file = os.path.join(path, 'pretrain-rcf/%s_%d' % ('ml1M', args.hidden_factor))
    # Training
    model = get_new_RCF_model(data, args, save_file)

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
