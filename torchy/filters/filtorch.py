"""Torch filters for accelerated feature computation."""

import numpy as np

import torch
from torch.nn.functional import conv2d, conv3d
from torch.autograd.variable import Variable

import dask.threaded as dth

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

    EPS = 0.0001

    def __init__(self, ndim=2, num_workers=4):
        assert ndim in [2, 3]
        self.num_workers = num_workers
        self.cache = {}
        self.ndim = ndim

    @property
    def conv(self):
        return conv2d if self.ndim == 2 else conv3d

    def stack_filters(self, *filters, **kwargs):
        convert_to_variable = kwargs.get('convert_to_variable', True)
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
        bias = to_variable(np.zeros(shape=(num_outputs,))) if num_outputs != 1 else None

        if self.ndim == 3:
            conved_012 = self.conv(input_tensor, kernel_tensor)
            conved_201 = self.conv(conved_012, kernel_tensor.permute(0, 1, 4, 2, 3),
                                   groups=num_outputs, bias=bias)
            conved_120 = self.conv(conved_201, kernel_tensor.permute(0, 1, 3, 4, 2),
                                   groups=num_outputs, bias=bias)
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

    def d0(self, input_tensor):
        # Gradient along axis 0
        kernel_tensor = self.stack_filters(self.DERIVATIVE_KERNEL[None])
        input_tensor = to_variable(input_tensor)
        return self.sconv(input_tensor, kernel_tensor).data

    def d1(self, input_tensor):
        # Gradient along axis 1
        kernel_tensor = self.stack_filters(self.DERIVATIVE_KERNEL[None])
        input_tensor = to_variable(input_tensor)
        if self.ndim == 2:
            return self.sconv(input_tensor, kernel_tensor.permute(0, 1, 3, 2)).data
        else:
            return self.sconv(input_tensor, kernel_tensor.permute(0, 1, 3, 2, 4)).data

    def d2(self, input_tensor):
        # Gradient along axis 2
        kernel_tensor = self.stack_filters(self.DERIVATIVE_KERNEL[None])
        input_tensor = to_variable(input_tensor)
        if self.ndim == 2:
            raise RuntimeError
        else:
            return self.sconv(input_tensor, kernel_tensor.permute(0, 1, 3, 4, 2)).data

    def dmag(self, *dns):
        if self.ndim == 3:
            d0, d1, d2 = dns[0], dns[1], dns[2]
            # No inplace ops, might cause threading bugs
            return torch.sqrt(d0 ** 2 + d1 ** 2 + d2 ** 2)
        else:
            d0, d1 = dns[0], dns[1]
            return torch.sqrt(d0 ** 2 + d1 ** 2)

    def laplacian(self, *dnns):
        if self.ndim == 3:
            d00, d11, d22 = dnns[0], dnns[1], dnns[2]
            return d00 ** 2 + d11 ** 2 + d22 ** 2
        else:
            d00, d11 = dnns[0], dnns[1]
            return d00 ** 2 + d11 ** 2

    def eighess(self, *dnms):
        if self.ndim == 3:
            d00, d01, d02, d11, d12, d22 = dnms[0], dnms[1], dnms[2], dnms[3], dnms[4], dnms[5]

            p1 = d01 ** 2 + d02 ** 2 + d12 ** 2

            T = (d00 + d11 + d22)/3

            p2 = (d00 - T) ** 2 + (d11 - T) ** 2 + (d22 - T) ** 2 + 2 * p1
            p = torch.sqrt(p2 / 6.)
            p_inv = (1./p)

            B00 = p_inv * (d00 - T)
            B01 = p_inv * d01
            B02 = p_inv * d02
            B11 = p_inv * (d11 - T)
            B12 = p_inv * d12
            B22 = p_inv * (d22 - T)

            detB = B00 * (B11 * B22 - B12 * B12) - \
                   B01 * (B01 * B22 - B12 * B02) + \
                   B02 * (B01 * B12 - B11 * B02)
            r = detB / 2.

            phi = torch.zeros(*detB.size())
            phi[r <= -1] = np.pi / 3.
            phi[r >= 1] = 0.
            phi[-1 < r < 1] = torch.acos(r[-1 < r < 1]) / 3.

            eig1 = T + 2 * p * torch.cos(phi)
            eig3 = T + 2 * p * torch.cos(phi + ((2. * np.pi) / 3.))
            eig2 = 3 * T - eig1 - eig3
            return torch.cat((eig1, eig2, eig3), 1)

        else:
            d00, d01, d11 = dnms[0], dnms[1], dnms[2]
            T = d00 + d11
            D = d00 * d11 - d01 * d01
            K = torch.sqrt(4. - D + self.EPS)
            L1 = T * (0.5 + 1/K)
            L2 = T * (0.5 - 1/K)
            return torch.cat((L1, L2), 1)

    @property
    def dsk(self):
        if self.ndim == 2:
            # 2D
            _dsk = {'input': None,
                    'smooth': (self.presmoothing, 'input'),
                    'd0': (self.d0, 'smooth'),
                    'd1': (self.d1, 'smooth'),
                    'dmag': (self.dmag, 'd0', 'd1'),
                    'd00': (self.d0, 'd0'),
                    'd01': (self.d1, 'd0'),
                    'd11': (self.d1, 'd1'),
                    'laplacian': (self.laplacian, 'd00', 'd11'),
                    'eighess': (self.eighess, 'd00', 'd01', 'd11')}
        else:
            # 3D
            _dsk = {'input': None,
                    'smooth': (self.presmoothing, 'input'),
                    'd0': (self.d0, 'smooth'),
                    'd1': (self.d1, 'smooth'),
                    'd2': (self.d2, 'smooth'),
                    'dmag': (self.dmag, 'd0', 'd1', 'd2'),
                    'd00': (self.d0, 'd0'),
                    'd01': (self.d1, 'd0'),
                    'd02': (self.d2, 'd0'),
                    'd11': (self.d1, 'd1'),
                    'd12': (self.d2, 'd1'),
                    'd22': (self.d2, 'd2'),
                    'laplacian': (self.laplacian, 'd00', 'd11', 'd22'),
                    'eighess': (self.eighess, 'd00', 'd01', 'd02', 'd11', 'd12', 'd22')}
        return _dsk

    def compute_features(self, input_tensor, *feature_names):
        _dsk = self.dsk
        _dsk.update({'input': input_tensor})
        return dth.get(_dsk, list(feature_names), num_workers=self.num_workers)

    def _test_presmoothing(self, input_shape):
        input_array = np.random.uniform(size=input_shape)

        with timeit() as timestats:
            presmoothed = self.presmoothing(input_array)

        print("Input shape: {} || Output shape: {}".format(input_shape, presmoothed.size()))
        print("Elapsed time: {}".format(timestats.elapsed_time))


if __name__ == '__main__':
    fs = FeatureSuite()

    fs._test_presmoothing((1, 1, 2000, 2000))
    print("---------------------------------")

    fs._test_presmoothing((1, 1, 2000, 2000))
    print("---------------------------------")

    fs._test_presmoothing((1, 1, 2000, 2000))

