"""Torch filters for accelerated feature computation."""

import numpy as np

import torch
from torch.nn.functional import conv2d, conv3d
from torch.autograd.variable import Variable

from torchy.utils import timeit

torch.set_num_threads(8)


def to_variable(tensor, device='cpu'):
    if isinstance(tensor, np.ndarray):
        if device == 'cpu':
            tensor = Variable(torch.from_numpy(tensor))
        elif device == 'gpu':
            tensor = Variable(torch.from_numpy(tensor).cuda())
    return tensor


class FeatureSuite(object):
    GAUSSIAN_KERNELS = {(2.5, 9): np.array([0.048297,
                                            0.08393,
                                            0.124548,
                                            0.157829,
                                            0.170793,
                                            0.157829,
                                            0.124548,
                                            0.08393,
                                            0.048297]),
                        (1.2, 9): np.array([0.001681,
                                            0.016844,
                                            0.087055,
                                            0.232853,
                                            0.323135,
                                            0.232853,
                                            0.087055,
                                            0.016844,
                                            0.001681]),
                        (0.6, 9): np.array([0,
                                            0.000015,
                                            0.006194,
                                            0.196119,
                                            0.595343,
                                            0.196119,
                                            0.006194,
                                            0.000015,
                                            0]),
                        (0.3, 9): np.array([0,
                                            0,
                                            0,
                                            0.04779,
                                            0.904419,
                                            0.04779,
                                            0,
                                            0,
                                            0]),
                        (0.3, 5): np.array([0,
                                            0.04779,
                                            0.904419,
                                            0.04779,
                                            0]),
                        (0.6, 5): np.array([0.006194,
                                            0.196125,
                                            0.595362,
                                            0.196125,
                                            0.006194]),
                        (5.0, 15): np.array([0.034619,
                                             0.044859,
                                             0.055857,
                                             0.066833,
                                             0.076841,
                                             0.084894,
                                             0.090126,
                                             0.09194,
                                             0.090126,
                                             0.084894,
                                             0.076841,
                                             0.066833,
                                             0.055857,
                                             0.044859,
                                             0.034619]),
                        (10.0, 15): np.array([0.057099,
                                              0.060931,
                                              0.064373,
                                              0.067333,
                                              0.06973,
                                              0.071493,
                                              0.072573,
                                              0.072936,
                                              0.072573,
                                              0.071493,
                                              0.06973,
                                              0.067333,
                                              0.064373,
                                              0.060931,
                                              0.057099]),
                        (1.0, 5): np.array([0.06136,
                                            0.24477,
                                            0.38774,
                                            0.24477,
                                            0.06136]),
                        (1.6, 9): np.array([0.011954,
                                            0.044953,
                                            0.115735,
                                            0.204083,
                                            0.246551,
                                            0.204083,
                                            0.115735,
                                            0.044953,
                                            0.011954]),
                        (3.5, 15): np.array([0.0161,
                                             0.027272,
                                             0.042598,
                                             0.061355,
                                             0.081488,
                                             0.099798,
                                             0.112705,
                                             0.117367,
                                             0.112705,
                                             0.099798,
                                             0.081488,
                                             0.061355,
                                             0.042598,
                                             0.027272,
                                             0.0161]),
                        (0.7, 5): np.array([0.01589,
                                            0.221542,
                                            0.525136,
                                            0.221542,
                                            0.01589])}

    DERIVATIVE_KERNEL = {None: np.array([-0.5, 0, 0.5])}

    SMALL_KERNEL_SIZE = 5
    MED_KERNEL_SIZE = 9
    BIG_KERNEL_SIZE = 15

    def __init__(self, ndim=2):
        assert ndim in [2, 3]

        self.cache = {}
        self.ndim = ndim

    @property
    def conv(self):
        return conv2d if self.ndim == 2 else conv3d

    def stack_filters(self, *filters, convert_to_variable=True):
        if self.ndim == 2:
            kernel_tensor = np.array([filter_.reshape(-1, 1)[None, ...] for filter_ in filters])
        else:
            kernel_tensor = np.array([filter_.reshape(-1, 1, 1)[None, ...] for filter_ in filters])
        kernel_tensor = to_variable(kernel_tensor) if convert_to_variable else kernel_tensor
        return kernel_tensor

    def pad_input(self, input_tensor, kernel_size):
        half_pad_size = kernel_size // 2
        pad_spec = [(0, 0), (0, 0), (half_pad_size, half_pad_size), (half_pad_size, half_pad_size)] + \
                   ([] if self.ndim == 2 else [(half_pad_size, half_pad_size)])
        padded_input_tensor = np.pad(input_tensor, pad_spec, 'reflect')
        return padded_input_tensor

    def sconv(self, input_tensor, kernel_tensor):
        # Separable convolutions
        # Get number of outputs
        num_outputs = kernel_tensor.size()[0]
        bias = to_variable(np.zeros(shape=(num_outputs,)))

        if self.ndim == 3:
            conved_012 = self.conv(input_tensor, kernel_tensor)
            conved_201 = self.conv(conved_012, kernel_tensor.permute(0, 1, 4, 2, 3))
            conved_120 = self.conv(conved_201, kernel_tensor.permute(0, 1, 3, 4, 2))
            output = conved_120
        else:
            conved_01 = self.conv(input_tensor, kernel_tensor)
            conved_10 = self.conv(conved_01, kernel_tensor.permute(0, 1, 3, 2), groups=num_outputs,
                                  bias=bias)
            output = conved_10
        return output

    def channel_to_batch(self, tensor):
        if self.ndim == 3:
            return tensor.permute(1, 0, 2, 3, 4)
        else:
            return tensor.permute(1, 0, 2, 3)

    def presmoothing(self, input_tensor):

        input_tensor_small = to_variable(self.pad_input(input_tensor, self.SMALL_KERNEL_SIZE))
        input_tensor_med = to_variable(self.pad_input(input_tensor, self.MED_KERNEL_SIZE))
        input_tensor_big = to_variable(self.pad_input(input_tensor, self.BIG_KERNEL_SIZE))

        small_kernel_sigmas = [0.3, 0.7, 1.0]
        med_kernel_sigmas = [1.6]
        big_kernel_sigmas = [3.5, 5.0, 10.0]

        # No need to launch the kernel at every run
        if 'kernel_tensor_small' not in self.cache.keys():
            # Stack filters
            kernel_tensor_small = self.stack_filters(*[self.GAUSSIAN_KERNELS[(sig, self.SMALL_KERNEL_SIZE)]
                                                       for sig in small_kernel_sigmas])
            self.cache.update({'kernel_tensor_small': kernel_tensor_small})
        else:
            kernel_tensor_small = self.cache['kernel_tensor_small']

        if 'kernel_tensor_med' not in self.cache.keys():
            kernel_tensor_med = self.stack_filters(*[self.GAUSSIAN_KERNELS[(sig, self.MED_KERNEL_SIZE)]
                                                     for sig in med_kernel_sigmas])
            self.cache.update({'kernel_tensor_med': kernel_tensor_med})
        else:
            kernel_tensor_med = self.cache['kernel_tensor_med']

        if 'kernel_tensor_big' not in self.cache.keys():
            kernel_tensor_big = self.stack_filters(*[self.GAUSSIAN_KERNELS[(sig, self.BIG_KERNEL_SIZE)]
                                                     for sig in big_kernel_sigmas])
            self.cache.update({'kernel_tensor_big': kernel_tensor_big})
        else:
            kernel_tensor_big = self.cache['kernel_tensor_big']

        # Compute convolutions
        conved_small = self.sconv(input_tensor_small, kernel_tensor_small).data
        conved_med = self.sconv(input_tensor_med, kernel_tensor_med).data
        conved_big = self.sconv(input_tensor_big, kernel_tensor_big).data
        # Concatenate results and move to batch axis
        all_conved = self.channel_to_batch(torch.cat((conved_small, conved_med, conved_big), 1))
        return all_conved

    def image_gradients(self, presmoothed):
        pass

    def _test_presmoothing(self, input_shape):
        input_array = np.random.uniform(size=input_shape)

        with timeit() as timestats:
            presmoothed = self.presmoothing(input_array)

        print("Input shape: {} || Output shape: {}".format(input_shape, presmoothed.size()))
        print("Elapsed time: {}".format(timestats.elapsed_time))


if __name__ == '__main__':
    fs = FeatureSuite()

    fs._test_presmoothing((1, 1, 1000, 1000))
    print("---------------------------------")

    fs._test_presmoothing((1, 1, 1000, 1000))
    print("---------------------------------")

    fs._test_presmoothing((1, 1, 1000, 1000))

