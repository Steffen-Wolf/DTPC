from contextlib import contextmanager
from argparse import Namespace
import time
import h5py
import numpy as np

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