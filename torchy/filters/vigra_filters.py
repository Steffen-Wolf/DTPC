"""This file contains all filters (kernels), i.e. icky-yucky image processing stuff."""

import numpy as np
import vigra
from torchy.utils import load_raw_data


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
    # load data slice in 
    data = load_raw_data(request["data_filename"], request["roi_with_halo"])
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