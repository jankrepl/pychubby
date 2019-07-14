"""Base classes and functions."""

import pathlib

import numpy as np

CACHE_FOLDER = pathlib.Path.home() / '.pychubby/'

CACHE_FOLDER.mkdir(parents=True, exist_ok=True)


class DisplacementField:
    """Represents a coordinate transformation."""

    def __init__(self, delta_x, delta_y):
        """Construct."""
        if not (isinstance(delta_x, np.ndarray) and isinstance(delta_y, np.ndarray)):
            raise TypeError('The deltas need to be a numpy array.')

        if not (delta_x.ndim == delta_y.ndim == 2):
            raise ValueError('The dimensions of delta_x and delta_y need to be 2.')

        if delta_x.shape != delta_y.shape:
            raise ValueError('The shapes of deltas need to be equal')

        self.delta_x = delta_x.astype(np.float32)
        self.delta_y = delta_y.astype(np.float32)

    @property
    def is_valid(self):
        """Check whether both delta_x and delta_y finite."""
        return np.all(np.isfinite(self.delta_x)) and np.all(np.isfinite(self.delta_y))
