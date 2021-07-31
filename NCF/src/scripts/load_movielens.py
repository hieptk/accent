import numpy as np
import pandas as pd

from NCF.src.influence.dataset import DataSet
from NCF.src.influence.datasets import Datasets


def load_movielens(train_dir, batch, use_recs=False):

  train = np.loadtxt("%s/movielens_train.tsv"%train_dir, delimiter='\t')
  valid = np.loadtxt("%s/movielens_train.tsv"%train_dir, delimiter='\t')
  if use_recs:
    test = pd.read_csv('recs.csv').to_numpy()
    n_padding = 0 if test.shape[0] % batch == 0 else batch - test.shape[0] % batch
    test = np.vstack([test, np.zeros((n_padding, test.shape[1]))])
  else:
    test = np.loadtxt("%s/movielens_train.tsv" % train_dir, delimiter='\t')
  train_input = train[:,:2].astype(np.int32)
  train_output = train[:,2]
  valid_input = valid[:, :2].astype(np.int32)
  valid_output = valid[:, 2]
  # test_input = test[:-1, :2].astype(np.int32)
  # test_output = test[:-1, 2]
  test_input = test[:, :2].astype(np.int32)
  test_output = test[:, 2]

  train = DataSet(train_input, train_output)
  validation = DataSet(valid_input, valid_output)
  test = DataSet(test_input, test_output)

  return Datasets(train=train, validation=validation, test=test)


  
