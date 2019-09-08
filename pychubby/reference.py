"""Module focused on creation of reference spaces."""

from abc import ABC, abstractmethod

import numpy as np
from skimage.transform import AffineTransform


class ReferenceSpace(ABC):
    """Abstract class for reference spaces."""

    @abstractmethod
    def estimate(*args, **kwargs):
        """Fit parameters of the model."""

    @abstractmethod
    def ref2inp(*args, **kwargs):
        """Transform from reference to input."""

    @abstractmethod
    def inp2ref(*args, **kwargs):
        """Transform from input to reference."""


class DefaultRS(ReferenceSpace):
    """Default reference space.

    Attributes
    ----------
    tform : skimage.transform.GeometricTransform
        Affine transformation.

    keypoints : dict
        Defining landmarks used for estimating the parameters of the model.

    """

    def __init__(self):
        """Construct."""
        self.tform = AffineTransform()
        self.keypoints = {
                         'CHIN': (0, 1),
                         'UPPER_TEMPLE_L': (-1, -1),
                         'UPPER_TEMPLE_R': (1, -1),
                         'UPPERMOST_NOSE': (0, -1),
                         'MIDDLE_NOSTRIL': (0, 0)
                            }

    def estimate(self, lf):
        """Estimate parameters of the affine transformation.

        Parameters
        ----------
        lf : pychubby.detect.LandmarFace
            Instance of the ``LandmarkFace``.

        """
        src = []
        dst = []
        for name, ref_coordinate in self.keypoints.items():
            dst.append(ref_coordinate)
            src.append(lf[name])

        src = np.array(src)
        dst = np.array(dst)

        self.tform.estimate(src, dst)

    def ref2inp(self, coords):
        """Transform from reference to input space.

        Parameters
        ----------
        coords : np.ndarray
            Array of shape `(N, 2)` where the columns represent x and y reference coordinates.

        Returns
        -------
        tformed_coords : np.ndarray
            Array of shape `(N, 2)` where the columns represent x and y coordinates in the input image
            correspoding row-wise to `coords`.

        """
        return self.tform.inverse(coords)

    def inp2ref(self, coords):
        """Transform from input to reference space.

        Parameters
        ----------
        coords : np.ndarray
            Array of shape `(N, 2)` where the columns represent x and y coordinates in the input space.

        Returns
        -------
        tformed_coords : np.ndarray
            Array of shape `(N, 2)` where the columns represent x and y coordinates in the reference space
            correspoding row-wise to `coords`.

        """
        return self.tform(coords)
