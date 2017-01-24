import numpy as np

import torch
from torch.nn.functional import conv2d, conv3d
from torch.autograd.variable import Variable

from . import filters


# TODO: Set torch num OMP threads


class FeatureSpec(object):
    def __init__(self, features=None):
        """
        Specify features to use.
        :type features: tuple
        :param features: A tuple of (str, tuple, dict) containing:
                         ("feature_name",
                         (feature_param1, feature_param2,...),
                         {'feature_param3': feature_param3, ...})
        """
        # Private
        self._features = None
        self._kernel = None

        # Assignments
        self.features = features

    @property
    def features(self):
        if self._features is None:
            return []
        return self._features

    @features.setter
    def features(self, value):
        if value is None:
            return
        # Validate
        assert isinstance(value, tuple)
        assert isinstance(value[0], str)

        # Parse value
        feature_name = value[0]
        feature_args = value[1] if len(value) > 1 else ()
        feature_kwargs = value[2] if len(value) > 2 else {}

        self._features = [] if self._features is None else self._features
        self._features.append({'name': feature_name,
                               'args': feature_args,
                               'kwargs': feature_kwargs})

    @property
    def kernel(self):
        if self._kernel is not None:
            return self._kernel
        # Get 2D kernels
        _filters = np.array([filters.get_kernel(feature_spec.get('name'),
                                                *feature_spec.get('args'),
                                                **feature_spec.get('kwargs'))
                             for feature_spec in self._features])
        # Sanity check
        assert _filters.ndim == 4 or _filters.ndim == 5, "Was expecting 4D or 5D kernel tensors."
        self._kernel = _filters
        return self._kernel

    def reset_kernel(self):
        self._kernel = None

    def add_feature(self, feature_name, *feature_args, **feature_kwargs):
        self.features.append({'name': feature_name,
                              'args': feature_args,
                              'kwargs': feature_kwargs})


class FeatureComputer(object):
    def __init__(self, smoothing_feature_spec, feature_spec, device='cpu'):
        """

        :type feature_spec: FeatureSpec
        :param feature_spec: Feature specificications

        :type smoothing_feature_spec: FeatureSpec
        :param smoothing_feature_spec: Smoothing features.

        :type device: str
        :param device: Device to process on.
                       Supported now: 'cpu'.
                       Supported later: 'cpu', 'gpu0', 'gpu0+gpu1', 'gpu0+gpu1+gpu2', ...
        """
        # Validate
        assert isinstance(feature_spec, FeatureSpec)
        assert isinstance(smoothing_feature_spec, FeatureSpec)
        assert isinstance(device, str)
        assert device in ['cpu']
        # Private
        self._data_slice = None
        self._ndim = None
        # Assignments
        self.smoothing_feature_spec = smoothing_feature_spec
        self.feature_spec = feature_spec
        self.device = device

    def _validate_input(self, input_, kernel):
        """Ensures input is in a torchy format."""
        assert input_.ndim == 4 or input_.ndim == 5, "Must be a torch-like 4D or 5D tensor."
        assert len(input_.shape) == len(kernel), \
            "Kernel and input not compatible."
        assert input_.shape[1] == 1, "Must be a single channel input."

    def _call(self, input_, device):
        """Compute on the CPU."""
        # Get smoothing_kernel
        smoothing_kernel = self.smoothing_feature_spec.kernel
        # Validate input
        self._validate_input(input_, smoothing_kernel)

        # Get feature kernel
        feature_kernel = self.feature_spec.kernel
        # Validate input
        self._validate_input(input_, feature_kernel)

        # Get conv function
        conv = conv2d if input_.ndim == 4 else conv3d

        # Wrap smoothing kernel
        smoothing_kernel_variable = Variable((torch.from_numpy(smoothing_kernel)
                                              if device == 'cpu' else
                                              torch.from_numpy(smoothing_kernel).cuda()))
        # Wrap feature kernel
        feature_kernel_variable = Variable((torch.from_numpy(feature_kernel)
                                            if device == 'cpu' else
                                            torch.from_numpy(smoothing_kernel).cuda()))
        # Wrap input
        input_variable = Variable((torch.from_numpy(input_)
                                   if device == 'cpu' else
                                   torch.from_numpy(input_).cuda()))

        # 2D or 3D
        if input_.ndim == 5:
            # ----- 3D
            # Smooth with the smoothing kernel
            smoothed_tensor_t012 = conv(input_variable,
                                        smoothing_kernel_variable)
            smoothed_tensor_t201 = conv(smoothed_tensor_t012,
                                        smoothing_kernel_variable.permute(0, 1, 4, 2, 3))
            smoothed_tensor_t120 = conv(smoothed_tensor_t201,
                                        smoothing_kernel_variable.permute(0, 1, 3, 4, 2))
            # Move smoothed images to the batch axis
            smoothed_tensor = smoothed_tensor_t120.permute(1, 0, 2, 3, 4)
            # Compute features
            feature_tensor_t012 = conv(smoothed_tensor,
                                       feature_kernel_variable)
            feature_tensor_t201 = conv(feature_tensor_t012,
                                       feature_kernel_variable.permute(0, 1, 4, 2, 3))
            feature_tensor_t120 = conv(feature_tensor_t201,
                                       feature_kernel_variable.permute(0, 1, 3, 4, 2))
            feature_tensor = feature_tensor_t120

        else:
            # ----- 2D
            # Smooth with smoothing kernel
            smoothed_tensor_t01 = conv(input_variable,
                                       smoothing_kernel_variable)
            smoothed_tensor_t10 = conv(smoothed_tensor_t01,
                                       smoothing_kernel_variable.permute(0, 1, 3, 2))
            # Move smoothed images to the batch axis
            smoothed_tensor = smoothed_tensor_t10.permute(1, 0, 2, 3)
            # Compute features
            feature_tensor_t01 = conv(smoothed_tensor,
                                      feature_kernel_variable)
            feature_tensor_t10 = conv(feature_tensor_t01,
                                      feature_kernel_variable.permute(0, 1, 3, 2))
            feature_tensor = feature_tensor_t10

        # Convert to numpy and return
        return feature_tensor.data.numpy()

    def __call__(self, *args, **kwargs):
        pass


