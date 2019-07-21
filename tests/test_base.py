"""Collection of tests focused on the base module."""

import numpy as np
import pytest

from pychubby.base import DisplacementField


class TestConstructor:
    def test_incorrect_input_type(self):
        delta_x = "aa"
        delta_y = 12

        with pytest.raises(TypeError):
            DisplacementField(delta_x, delta_y)

    def test_incorrect_ndim(self):
        delta_x = np.ones((2, 3, 4))
        delta_y = np.ones((2, 3, 4))

        with pytest.raises(ValueError):
            DisplacementField(delta_x, delta_y)

    def test_different_shape(self):
        delta_x = np.ones((2, 3))
        delta_y = np.ones((2, 4))

        with pytest.raises(ValueError):
            DisplacementField(delta_x, delta_y)

    def test_dtype(self):

        delta_x = np.ones((2, 3))
        delta_y = np.ones((2, 3))

        df = DisplacementField(delta_x, delta_y)

        assert df.delta_x.dtype == np.float32
        assert df.delta_y.dtype == np.float32


class TestGenerate:
    """Tests focused on the `generate` class method."""

    def test_incorrect_input(self):
        new_points = [1, 12]
        old_points = "a"

        with pytest.raises(TypeError):
            DisplacementField.generate((4, 5), old_points, new_points)

    def test_incorrect_input_shape(self):
        new_points = np.array([[1, 1], [2, 2]])
        old_points = np.array([[1, 1], [2, 2], [3, 3]])

        with pytest.raises(ValueError):
            DisplacementField.generate((10, 11), old_points, new_points)

    @pytest.mark.parametrize(
        "interpolation_kwargs",
        [
            {"function": "cubic"},
            {"function": "gaussian"},
            {"function": "inverse"},
            {"function": "linear"},
            {"function": "multiquadric"},
            {"function": "quintic"},
            {"function": "thin_plate"},
            {},
        ],
    )
    @pytest.mark.parametrize(
        "anchor_corners", [True, False], ids=["do_anchor", "dont_anchor"]
    )
    def test_identity(self, anchor_corners, interpolation_kwargs):
        """Specifying identical old and new points leads to identity."""
        shape = (12, 13)
        old_points = np.array([[1, 8], [10, 10], [5, 2]])
        new_points = old_points
        df = DisplacementField.generate(
            shape,
            old_points,
            new_points,
            anchor_corners=anchor_corners,
            **interpolation_kwargs
        )
        print(df.delta_x.mean())
        assert np.all(df.delta_x == 0)
        assert np.all(df.delta_y == 0)

    @pytest.mark.parametrize(
        "interpolation_kwargs",
        [
            {"function": "cubic"},
            {"function": "gaussian"},
            {"function": "inverse"},
            {"function": "linear"},
            {"function": "multiquadric"},
            {"function": "quintic"},
            {"function": "thin_plate"},
            {},
        ],
    )
    @pytest.mark.parametrize(
        "anchor_corners", [True, False], ids=["do_anchor", "dont_anchor"]
    )
    def test_interpolation_on_nodes(self, anchor_corners, interpolation_kwargs):
        """Make sure that the transformation on the landmarks are precise."""
        shape = (20, 30)
        old_points = np.array([[7, 7], [7, 14], [14, 7], [14, 14]])
        new_points = old_points.copy() + np.random.randint(-3, 3, size=(4, 2))

        df = DisplacementField.generate(
            shape,
            old_points,
            new_points,
            anchor_corners=anchor_corners,
            **interpolation_kwargs
        )

        for new_p, old_p in zip(new_points, old_points):
            assert df.delta_x[new_p[1], new_p[0]] == pytest.approx(old_p[0] - new_p[0])
            assert df.delta_y[new_p[1], new_p[0]] == pytest.approx(old_p[1] - new_p[1])


class TestProperties:
    def test_is_valid(self):

        delta_x = np.ones((2, 3))
        delta_y = np.ones((2, 3))
        delta_y_inv_1 = np.ones((2, 3))
        delta_y_inv_1[0, 1] = np.inf
        delta_y_inv_2 = np.ones((2, 3))
        delta_y_inv_2[0, 1] = np.nan

        df_val = DisplacementField(delta_x, delta_y)
        df_inv_1 = DisplacementField(delta_x, delta_y_inv_1)
        df_inv_2 = DisplacementField(delta_x, delta_y_inv_2)

        assert df_val.is_valid
        assert not df_inv_1.is_valid
        assert not df_inv_2.is_valid

    def test_transformation(self):

        delta_x = np.zeros((2, 3))
        delta_y = np.zeros((2, 3))

        transformation_x = np.array([[0, 1, 2],
                                     [0, 1, 2]])
        transformation_y = np.array([[0, 0, 0],
                                     [1, 1, 1]])

        df = DisplacementField(delta_x, delta_y)

        tf_x, tf_y = df.transformation

        assert np.all(tf_x == transformation_x)
        assert np.all(tf_y == transformation_y)
