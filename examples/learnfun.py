import numpy as np
from dask.multiprocessing import get

from torchy.utils import timeit

import torch
from torch.nn.functional import conv2d
from torch.autograd import Variable

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    # Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(200 * rng.rand(600, 1) - 100, axis=0)
    y = np.array([0 if x > X.mean() or np.random.choice([True, False, False, False]) else 1 for x in X]).T

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=400,
                                                        random_state=4)

    def train(rf, X_train, y_train):
        rf.fit(X_train, y_train)
        return rf

    def predict(rf, X_test):
        return rf.predict_proba(X_test)

    dsk = {'RF': RandomForestClassifier(max_depth=30, random_state=2),
           "X_train" : X_train,
           "X_test"  : X_test,
           "y_train" : y_train,
           "y_test"  : y_test,
           'trained-RF': (train, 'RF', "X_train", "y_train"),
           'predict': (predict, 'trained-RF', 'X_test')}

    with timeit() as time_info:
        output = get(dsk, 'predict')

    print("Time Elapsed: {}".format(time_info.elapsed_time))

