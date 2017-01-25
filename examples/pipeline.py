from __future__ import print_function
import numpy as np
from dask.threaded import get

import vigra

from torchy.utils import timeit
from sklearn.model_selection import train_test_split
from torchy.learning import learning
from torchy.filters import filters
import h5py
import os

import dask.array as da

def dummy_feature_prediciton(data, sigmas):
    shape = [len(sigmas), 1] + list(data.shape)
    output = np.empty(shape)
    for i,s in enumerate(sigmas):
        output[i, 0] = vigra.filters.gaussianGradientMagnitude(data, s)
    return output


if __name__ == '__main__':
    # Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(200 * rng.rand(600, 1) - 100, axis=0)
    y = np.array([0 if x > X.mean() or np.random.choice([True, False, False, False]) else 1 for x in X]).T

    f = h5py.File(os.path.join('datasets', 'smallFibStack.h5'), "r")
    dset = f['volume/data']
    x = da.from_array(dset, chunks=(10, 10, 1, 1))
    
    dsk = {"image": dset[0:30,0:30,0,0].astype(np.float32),
            "RF":"RandomForest",
            "RF_File": 'datasets/hackathon_flyem_forest.h5',
            "sigmas": np.arange(17)/10+2,
            "trained-RF": (learning.get_classifier, "RF", "RF_File"),
            "computed-features": (dummy_feature_prediciton, "image", "sigmas"),
            'predicted-image': (learning.image_prediction, "RF", "RF_File",'computed-features')}

    with timeit() as time_info:
        output = get(dsk, 'predicted-image')

    print("Time Elapsed: {}".format(time_info.elapsed_time))

