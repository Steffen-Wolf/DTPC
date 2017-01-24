"""This file contains all filters (kernels), i.e. icky-yucky image processing stuff."""

import numpy as np


def gaussian(sigma, filter_size):
    pass


def laplacian(sigma):
    pass


def get_kernel(kernel_name, *args, **kwargs):
    """General interface to fetch 2D kernels."""
    _ALL_KERNELS = {'gaussian': gaussian,
                    'laplacian': laplacian}
    return _ALL_KERNELS.get(kernel_name.to_lower())(*args, **kwargs)
