import numpy as np

from . import filters


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
        assert _filters.ndim == 4
        self._kernel = _filters
        return self._kernel

    def reset_kernel(self):
        self._kernel = None

    def add_feature(self, feature_name, *feature_args, **feature_kwargs):
        self.features.append({'name': feature_name,
                              'args': feature_args,
                              'kwargs': feature_kwargs})


class FeatureComputer(object):
    def __init__(self, feature_spec, device='cpu'):
        # Validate
        assert isinstance(feature_spec, FeatureSpec)
        assert isinstance(device, str)
        # Assignments
        self.feature_spec = feature_spec
        self.device = device

    def call(self):
        # TODO
        pass

    def __call__(self, *args, **kwargs):
        pass

