from __future__ import print_function
import numpy as np
from dask.threaded import get

import vigra
import json

from torchy.utils import timeit
from sklearn.model_selection import train_test_split
from torchy.learning import learning
from torchy.filters import filters
import h5py
import os
from glob import glob

import dask.array as da


def cutout_to_slice(cutout):
    c = cutout[0]
    # future parameter for halo size
    hs = 0
    if len(c)/2 == 2:
        return (slice(None), slice(None), 
                slice(c['xmin']-hs,c['xmax']+hs),
                slice(c['ymin']-hs,c['ymax']+hs), 0)
    if len(c)/2 == 3:
        return (slice(c['xmin']-hs,c['xmax']+hs),
                slice(c['ymin']-hs,c['ymax']+hs),
                slice(c['zmin']-hs,c['zmax']+hs), 0)

def process_request(base_name):
    requests = []
    for json_file_name in glob(base_name):
        with open(json_file_name) as data_file:    
            data = json.load(data_file)
            data["roi_with_halo"] = cutout_to_slice(data["cutout"])
            requests.append(data)
    return requests

def raw_data_loader(filename, slice_with_halo):
    with h5py.File(filename, "r") as f5:
        dset = f5['volume/data']
        out = np.array(dset[slice_with_halo])
    new_shape = [1, 1] + list(out.shape)
    return out.reshape(new_shape)

def reduce_request_size():
    pass

def get_filter_function(filter_name):
    if filter_name == 'Gaussian Smoothing':
        return vigra.gaussianSmoothing
    elif filter_name == 'Laplacian of Gaussian':
        return vigra.filters.laplacianOfGaussian
    elif filter_name == 'Hessian of Gaussian Eigenvalues':
        return vigra.filters.gaussianGradientMagnitude
        # should be ...
        return vigra.filters.hessianOfGaussianEigenvalues
        # but I can not handle the different size return yet :(
    elif filter_name == 'Gaussian Gradient Magnitude':
        return vigra.filters.gaussianGradientMagnitude
    else:
        raise NotImplementedError

def get_filter_size(filter_name):
    if filter_name == 'Gaussian Smoothing':
        return 1
    elif filter_name == 'Laplacian of Gaussian':
        return 1
    elif filter_name == 'Hessian of Gaussian Eigenvalues':
        return 3
    elif filter_name == 'Gaussian Gradient Magnitude':
        return 1
    else:
        raise NotImplementedError

def dummy_feature_prediciton(request):
    data = raw_data_loader(request["data_filename"], request["roi_with_halo"])
    features = request["features"]
    sigmas = np.unique([f['sigma'] for f in features])
    filters = set([f['name'] for f in features])
    shape = list(data.shape)
    shape[0] = len(sigmas)
    shape[1] = len(filters)
    output = np.empty(shape)
    for i, s in enumerate(sigmas):
        for j, f in enumerate(filters):
            output[i, j] = get_filter_function(f)(data[0, 0].astype(np.float32), s)
    return output


if __name__ == '__main__':

    req_list = process_request("datasets/sample_req*.json")
    dsk = {"RF":"VigraRF"}
    target = []
    for r in req_list:
        rid = r["req_id"]
        dsk["request-%i"%rid] = r
        dsk["RF-File-%i"%rid] = "datasets/hackathon_flyem_forest.h5"# should be r["classifier_filename"] when we get the correct json files
        dsk["computed-features-%i"%rid] = (dummy_feature_prediciton, "request-%i"%rid)
        dsk["predicted-image-%i"%rid] = (learning.image_prediction, "RF", "RF-File-%i"%rid, 'computed-features-%i'%rid)
        target.append("predicted-image-%i"%rid)

    with timeit() as time_info:
        output = get(dsk, target)
    print(output[0].shape, output[1].shape)

    print("Time Elapsed: {}".format(time_info.elapsed_time))

