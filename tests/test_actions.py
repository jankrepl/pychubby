"""Collection of tests focused on the `actions` module."""

import numpy as np
import pytest

from pychubby.actions import (Action, AbsoluteMove, Chubbify, Lambda, LinearTransform, Multiple,
                              Pipeline, Smile, OpenEyes)
from pychubby.base import DisplacementField
from pychubby.detect import LandmarkFace, LandmarkFaces

# FIXTURES
@pytest.fixture()
def random_lf():
    return LandmarkFace(np.random.random((68, 2)), np.zeros((10, 12)))


class TestAction:
    """Tests focused on the metaclass ``Action``."""

    def test_not_possible_to_inst(self):
        with pytest.raises(TypeError):
            Action()

        class SubAction(Action):
            pass

        with pytest.raises(TypeError):
            SubAction()

    def test_possible_to_inst(self):
        class SubAction(Action):
            def perform(*args, **kwargs):
                pass

        SubAction()

    def test_pts2inst(self, random_lf):
        new_points = np.random.random((68, 2))

        new_lf, df = Action.pts2inst(new_points, random_lf)

        assert isinstance(new_lf, LandmarkFace)
        assert isinstance(df, DisplacementField)


class TestAbsoluteMove:
    """Collection of tests focused on the `AbsoluteMove` action."""

    def test_attributes_dicts(self):
        a = AbsoluteMove()

        assert isinstance(a.x_shifts, dict)
        assert isinstance(a.y_shifts, dict)

    def test_default_constructor(self, random_lf):
        a = AbsoluteMove()

        new_lf, df = a.perform(random_lf)

        assert isinstance(new_lf, LandmarkFace)
        assert isinstance(df, DisplacementField)
        assert df.is_valid

    def test_interpolation_works(self, random_lf):
        a = AbsoluteMove(x_shifts={3: 4},
                         y_shifts={32: 5})

        new_lf, df = a.perform(random_lf)

        old_points = random_lf.points
        new_points = new_lf.points

        for i in range(68):
            if i == 3:
                assert np.allclose(new_points[i], old_points[i] + np.array([4, 0]))
            elif i == 32:
                assert np.allclose(new_points[i], old_points[i] + np.array([0, 5]))
            else:
                assert np.allclose(new_points[i], old_points[i])


class TestLambda:
    """Collection of tests focused on the ``Lambda`` action."""

    def test_noop(self, random_lf):

        a = Lambda(0.5, {})

        new_lf, df = a.perform(random_lf)

        assert np.allclose(df.delta_x, np.zeros_like(df.delta_x))
        assert np.allclose(df.delta_y, np.zeros_like(df.delta_y))

    def test_simple(self, random_lf):

        a = Lambda(0.5, {'CHIN': (90, 2)})

        new_lf, df = a.perform(random_lf)

        assert not np.allclose(df.delta_x, np.zeros_like(df.delta_x))
        assert not np.allclose(df.delta_y, np.zeros_like(df.delta_y))


class TestChubbify:
    """Collection of tests focused on the ``Chubbify`` action."""

    @pytest.mark.parametrize('scale', (0.2, 0.4, 1, 0))
    def test_simple(self, random_lf, scale):
        a = Chubbify(scale)
        new_lf, df = a.perform(random_lf)

        if scale == 0:
            assert np.allclose(df.delta_x, np.zeros_like(df.delta_x))
            assert np.allclose(df.delta_y, np.zeros_like(df.delta_y))
        else:
            assert not np.allclose(df.delta_x, np.zeros_like(df.delta_x))
            assert not np.allclose(df.delta_y, np.zeros_like(df.delta_y))


class TestLinearTransform:
    """Collection of tests focused on the ``LinearTransform`` action."""

    def test_noop(self, random_lf):
        a = LinearTransform()
        new_lf, df = a.perform(random_lf)

        assert np.allclose(df.delta_x, np.zeros_like(df.delta_x))
        assert np.allclose(df.delta_y, np.zeros_like(df.delta_y))

    def test_simple(self, random_lf):
        a = LinearTransform(translation_x=1)
        new_lf, df = a.perform(random_lf)

        assert not np.allclose(df.delta_x, np.zeros_like(df.delta_x))
        assert not np.allclose(df.delta_y, np.zeros_like(df.delta_y))


class TestOpenEyes:
    """Collection of tests focused on the ``OpenEyes`` action."""

    @pytest.mark.parametrize('scale', (0.2, 0.4, 1, 0))
    def test_simple(self, random_lf, scale):
        a = OpenEyes(scale)
        new_lf, df = a.perform(random_lf)

        if scale == 0:
            assert np.allclose(df.delta_x, np.zeros_like(df.delta_x))
            assert np.allclose(df.delta_y, np.zeros_like(df.delta_y))
        else:
            assert not np.allclose(df.delta_x, np.zeros_like(df.delta_x))
            assert not np.allclose(df.delta_y, np.zeros_like(df.delta_y))


class TestMultiple:
    """Collection of tests focused on the ``Multiple`` action."""

    @pytest.mark.parametrize('per_face_action', [Smile(), [Smile(), Smile()]],
                             ids=['single', 'many'])
    def test_overall(self, random_lf, per_face_action):
        lf_1 = random_lf
        lf_2 = LandmarkFace(random_lf.points + np.random.random((68, 2)), random_lf.img)

        lfs = LandmarkFaces(lf_1, lf_2)

        a = Multiple(per_face_action)

        new_lfs, df = a.perform(lfs)

        assert isinstance(new_lfs, LandmarkFaces)
        assert isinstance(df, DisplacementField)
        assert len(lfs) == len(new_lfs)


class TestPipeline:
    """Collection of tests focused on the ``Smile`` action."""

    def test_overall(self, random_lf):
        steps = [Smile(), Chubbify()]

        a = Pipeline(steps)

        new_lf, df = a.perform(random_lf)

        assert df.is_valid


class TestSmile:
    """Collection of tests focused on the ``Smile`` action."""

    @pytest.mark.parametrize('scale', (0.2, 0.4, 1, 0))
    def test_simple(self, random_lf, scale):
        a = Smile(scale)
        new_lf, df = a.perform(random_lf)

        if scale == 0:
            assert np.allclose(df.delta_x, np.zeros_like(df.delta_x))
            assert np.allclose(df.delta_y, np.zeros_like(df.delta_y))
        else:
            assert not np.allclose(df.delta_x, np.zeros_like(df.delta_x))
            assert not np.allclose(df.delta_y, np.zeros_like(df.delta_y))
