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


class TestCall:
    """Collection of tests focused on the `__call__` method."""

    def test_identity(self):
        delta = np.zeros((11, 12))

        df = DisplacementField(delta, delta)

        assert df == df(df)


class TestEquality:
    """Test that __eq__ works."""

    def test_itself(self):
        delta_x = np.ones((10, 12)) * 1.9
        delta_y = np.ones((10, 12)) * 1.2

        df = DisplacementField(delta_x, delta_y)

        assert df == df


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


class TestMulAndDiv:
    """Tests focus on the __mul__, __truediv__ and __rmul__ dunders."""

    @pytest.mark.parametrize('inp', ['a', [1, 2]])
    def test_incorrect_type(self, inp):
        delta_x = np.ones((10, 12)) * 3
        delta_y = np.ones((10, 12)) * 2
        df = DisplacementField(delta_x, delta_y)

        with pytest.raises(TypeError):
            df * inp

        with pytest.raises(TypeError):
            inp * df

        with pytest.raises(TypeError):
            df / inp

    def test_works(self):
        delta_x = np.ones((10, 12)) * 3
        delta_y = np.ones((10, 12)) * 2

        delta_x_true = np.ones((10, 12)) * 6
        delta_y_true = np.ones((10, 12)) * 4

        df = DisplacementField(delta_x, delta_y)
        df_true = DisplacementField(delta_x_true, delta_y_true)

        assert df * 2 == df_true
        assert 2 * df == df_true
        assert df_true / 2 == df


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

    def test_norm(self):

        shape = (2, 3)
        delta_x = np.ones(shape) * 3
        delta_y = np.ones(shape) * 4

        df = DisplacementField(delta_x, delta_y)

        assert np.allclose(df.norm, np.ones(shape) * 5)

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


class TestWarp:
    """Collection of tests focused on the "warp" method."""

    @pytest.mark.parametrize('order', [0, 1, 2])
    def test_identity_transformation(self, face_img, order):
        shape = face_img.shape[:2]
        delta_x = np.zeros(shape)
        delta_y = np.zeros(shape)

        df = DisplacementField(delta_x, delta_y)
        warped_img = df.warp(face_img, order)

        assert np.allclose(warped_img, face_img)
        assert warped_img.dtype == face_img.dtype
