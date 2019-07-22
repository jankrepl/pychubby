"""Definition of actions."""

from abc import ABC, abstractmethod

import numpy as np

from pychubby.base import DisplacementField
from pychubby.detect import LandmarkFace


class Action(ABC):
    """General Action class to be subclassed."""

    @abstractmethod
    def perform(self, lf, **kwargs):
        """Perfom action on an instance of a LandmarkFace.

        Parameters
        ----------
        lf : LandmarkFace
            Instance of a ``LandmarkFace``.

        kwargs : dict
            Action specific parameters.

        Returns
        -------
        new_lf : LandmarkFace
            Instance of a ``LandmarkFace`` after a specified action was
            taken on the input `lf`.

        """

    @staticmethod
    def pts2inst(new_points, lf, **interpolation_kwargs):
        """Generate instance of LandmarkFace via interpolation.

        Parameters
        ----------
        new_points : np.ndarray
            Array of shape `(N, 2)` representing the x and y coordinates of the
            new landmark points.

        lf : LandmarkFace
            Instance of a ``LandmarkFace`` before taking any actions.

        interpolation_kwargs : dict
            Interpolation parameters passed onto scipy.

        Returns
        -------
        new_lf : LandmarkFace
            Instance of a ``LandmarkFace`` after taking an action.

        df : DisplacementField
            Displacement field representing per pixel displacements between the
            old and new image.

        """
        if not interpolation_kwargs:
            interpolation_kwargs = {'function': 'linear'}

        df = DisplacementField.generate(lf.img.shape,
                                        lf.points,
                                        new_points,
                                        anchor_edges=True,
                                        **interpolation_kwargs)

        new_img = df.warp(lf.img)

        return LandmarkFace(new_points, new_img), df


class AbsoluteMove(Action):
    """Absolute offsets of any landmark points.

    Parameters
    ----------
    x_shifts : dict or None
        Keys are integers from 0 to 67 representing a chosen landmark points. The
        values represent the shift in the x direction to be made. If a landmark
        not specified assumed shift is 0.

    y_shifts : dict or None
        Keys are integers from 0 to 67 representing a chosen landmark points. The
        values represent the shift in the y direction to be made. If a landmark
        not specified assumed shift is 0.

    """

    def __init__(self, x_shifts=None, y_shifts=None):
        """Construct."""
        self.x_shifts = x_shifts or {}
        self.y_shifts = y_shifts or {}

    def perform(self, lf):
        """Perform absolute move.

        Specified landmarks will be shifted in either the x or y direction.

        Parameters
        ----------
        lf : LandmarkFace
            Instance of a ``LandmarkFace``.

        Returns
        -------
        new_lf : LandmarkFace
            Instance of a ``LandmarkFace`` after taking the action.

        df : DisplacementField
            Displacement field representing the transformation between the old and
            new image.

        """
        offsets = np.zeros((68, 2))

        # x shifts
        for k, v in self.x_shifts.items():
            offsets[k, 0] = v
        # y shifts
        for k, v in self.y_shifts.items():
            offsets[k, 1] = v

        new_points = lf.points + offsets

        new_lf, df = self.pts2inst(new_points, lf)

        return new_lf, df
