"""Collection of tests focused on the `visualization` module."""

import numpy as np
from matplotlib.animation import ArtistAnimation

from pychubby.base import DisplacementField
from pychubby.visualization import create_animation


class TestCreateAnimation:
    """Collection of tests focused on the `create_animation` function."""

    def test_overall(self, face_img):
        shape = (10, 11)

        delta_x = np.random.random(shape)
        delta_y = np.random.random(shape)

        df = DisplacementField(delta_x, delta_y)
        ani = create_animation(df, face_img, fps=2, n_seconds=1)

        assert isinstance(ani, ArtistAnimation)
