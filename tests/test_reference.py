"""Tests focused on the reference module."""

import numpy as np

from pychubby.detect import LandmarkFace
from pychubby.reference import DefaultRS


class TestDefaultRS:
    """Tests focused on the ``DefaultRS``."""

    def test_all(self):
        lf = LandmarkFace(np.random.random((68, 2)), np.zeros((12, 13)))

        rs = DefaultRS()

        rs.estimate(lf)

        random_ref_points = np.random.random((10, 2))

        assert np.allclose(rs.inp2ref(rs.ref2inp(random_ref_points)), random_ref_points)
