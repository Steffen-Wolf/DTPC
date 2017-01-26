from contextlib import contextmanager
from argparse import Namespace
import time
import h5py
import numpy as np
# from torchy.filters.vigra_filters import get_filter_size


@contextmanager
def timeit():
    """
    Context manager that times what happens under it. Maybe with a wee-bit overhead,
    but shouldn't matter much for relative measurements.

    Usage:
    ```
    with timeit() as timeit_info:
        output = ...

    print("Elapsed time: {}".format(timit_info.elapsed_time))
    ```
    """
    timeit_info = Namespace()
    now = time.time()
    yield timeit_info
    later = time.time()
    timeit_info.elapsed_time = later - now


def simulate_delay(duration):
    """Delay a given function by a certain `duration` (in seconds)."""
    def _decorator(function):
        def _function(*args, **kwargs):
            time.sleep(duration)
            return function(*args, **kwargs)
        return _function
    return _decorator


def load_raw_data(filename, slice_with_halo):
    """
    wrapper for h5py that slices the ROI and reshapes to the 
    format (batch, channel, x, y, z).
    For raw input data batch = channel =1
    """
    with h5py.File(filename, "r") as f5:
        dset = f5['volume/data']
        out = np.array(dset[slice_with_halo])
    new_shape = [1, 1] + list(out.shape)
    return out.reshape(new_shape)


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


def get_feature_index(r):
    """
    function that returns the slices that generate the feature file in 
    matching request r
    """

    sigmas = np.unique([f['sigma'] for f in r["features"]]).tolist()
    filters = list(set([f['name'] for f in r["features"]]))
    s_list = []
    f_list = []
    for f in r["features"]:
        for k in range(get_filter_size(f["name"])):
            s_list.append(sigmas.index(f["sigma"]))
            f_list.append(filters.index(f["name"]))
    return s_list, f_list


def reshape_volume_for_torch(volume):
    assert volume.ndim == 3 or volume.ndim == 2
    return volume[None, None, ...]