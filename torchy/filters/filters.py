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
    _2D_kernel = _ALL_KERNELS.get(kernel_name.to_lower())(*args, **kwargs)
    # Torch expects a 3D kernel, where the leading dimension corresponds to channel,
    # and the trailing two dimensions are spatial.
    _3D_kernel = _2D_kernel[None, ...]
    return _3D_kernel
