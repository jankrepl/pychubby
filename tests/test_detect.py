"""Collection of tests focused on the `detect` module."""
from collections import namedtuple
from unittest.mock import Mock
import math

import dlib
import numpy as np
import pytest
import scipy

import pychubby.detect
from pychubby.detect import LandmarkFace, face_rectangle, landmarks_68


class TestFaceRectangle:
    """Tests of the `face_rectangle` function."""

    def test_incorrect_input(self):
        with pytest.raises(TypeError):
            face_rectangle('123')

    def test_output_face(self, face_img):
        res, _ = face_rectangle(face_img)
        assert isinstance(res, list)
        assert len(res) == 1

    def test_output_blank(self, blank_img):
        res, _ = face_rectangle(blank_img)
        assert isinstance(res, list)
        assert len(res) == 0

    def test_output_faces(self, faces_img):
        res, _ = face_rectangle(faces_img)
        assert isinstance(res, list)
        assert len(res) == 2


class TestLandmarks68:
    """Tests of the `landmarks_68` function."""
    @pytest.fixture()
    def dlib_rectangle(self):
        return Mock(spec=dlib.rectangle)

    def test_incorrect_input_model(self, face_img, dlib_rectangle, tmp_path, monkeypatch):
        with pytest.raises(IOError):
            lm_points, original = landmarks_68(face_img, dlib_rectangle, tmp_path / 'fake_model.dat')
        # nasty hack
        monkeypatch.setattr('pychubby.detect.CACHE_FOLDER', tmp_path)
        with pytest.raises(IOError):
            lm_points, original = pychubby.detect.landmarks_68(face_img, dlib_rectangle)

    def test_correct_output(self, face_img, dlib_rectangle, tmp_path, monkeypatch):

        fake_model = tmp_path / 'fake_model.dat'
        fake_model.touch()

        def fake_shape_predictor(model_path):
            trained_model = Mock()
            Point = namedtuple('Point', ['x', 'y'])
            trained_model.parts = Mock(return_value=68 * [Point(0, 0)])
            dlib_predictor = Mock(return_value=trained_model)

            return dlib_predictor

        monkeypatch.setattr('dlib.shape_predictor', fake_shape_predictor)

        lm_points, original = landmarks_68(face_img, dlib_rectangle, fake_model)

        assert isinstance(lm_points, np.ndarray)
        assert lm_points.shape == (68, 2)


class TestLandmarkFaceAngle:
    """Collection of tests focused on the `angle` method."""

    def test_identical(self):
        lf = LandmarkFace(np.random.random((68, 2)), np.zeros((12, 13)))

        for i in range(68):
            for j in range(68):
                ref_vector = lf.points[j] - lf.points[i]
                assert lf.angle(i, j, reference_vector=ref_vector) == 0

    def test_rad2deg(self):
        lf = LandmarkFace(np.random.random((68, 2)), np.zeros((12, 13)))

        for i in range(68):
            for j in range(68):
                res_rad = lf.angle(i, j, use_radians=True)
                res_deg = lf.angle(i, j)
                assert math.degrees(res_rad) == res_deg


class TestLandmarkFaceEssentials:
    """Tests focused on attributes and properties of the LandmarkFace class."""

    def test_constructor_wrong_input(self):

        img = np.zeros((10, 11))
        points = np.random.random((12, 2))

        with pytest.raises(ValueError):
            LandmarkFace(points, img)

    def test_duplicate_landmarks(self):
        img = np.zeros((10, 11))
        points = np.random.random((67, 2))
        points = np.vstack([points, np.array([points[-1]])])

        with pytest.raises(ValueError):
            LandmarkFace(points, img)

    def test_onedimensional_points(self):
        img = np.zeros((10, 11))
        points = np.stack([np.arange(68), np.arange(68)], axis=1)

        with pytest.raises(scipy.spatial.qhull.QhullError):
            LandmarkFace(points, img)

    def test_area_and_volume(self):
        points = np.random.random((68, 2))
        points[0, :] = [0, 0]
        points[1, :] = [1, 1]
        points[2, :] = [0, 1]
        points[3, :] = [1, 0]

        img = np.zeros((10, 11))

        lf = LandmarkFace(points, img)

        assert lf.face_area == 4
        assert lf.face_volume == 1


class TestLandmarkFaceEstimate:
    """Tests focused on the class method `estimate` of the LandmarkFace."""

    def test_incorrect_input(self, monkeypatch):
        img = np.random.random((10, 11))

        monkeypatch.setattr('pychubby.detect.face_rectangle',
                            lambda *args, **kwargs: (2 * [None], 4 * [None]))
        with pytest.raises(ValueError):
            LandmarkFace.estimate(img)

    def test_overall(self, monkeypatch):
        img = np.random.random((10, 11))

        monkeypatch.setattr('pychubby.detect.face_rectangle',
                            lambda *args, **kwargs: ([None], [None]))

        monkeypatch.setattr('pychubby.detect.landmarks_68',
                            lambda *args: (np.random.random((68, 2)), None))

        lf = LandmarkFace.estimate(img)

        assert isinstance(lf, LandmarkFace)
        assert lf.points.shape == (68, 2)
        assert lf.img.shape == (10, 11)


class TestLandmakrFaceEuclideanDistance:
    """Collection of tests focused on the `euclidean_distance` method."""

    def test_identical(self):
        lf = LandmarkFace(np.random.random((68, 2)), np.zeros((12, 13)))

        for lix in range(68):
            assert lf.euclidean_distance(lix, lix) == 0

    def test_precise_value(self):
        points = np.random.random((68, 2))
        points[0] = [2, 3]
        points[1] = [5, 7]

        lf = LandmarkFace(points, np.zeros((12, 13)))

        assert lf.euclidean_distance(0, 1) == 5
        assert lf.euclidean_distance(1, 0) == 5


class TestLandmarkFacePlot:
    def test_plot(self, monkeypatch):
        mock = Mock()

        monkeypatch.setattr('pychubby.detect.plt', mock)

        lf = LandmarkFace(np.random.random((68, 2)), np.random.random((12, 13)))

        lf.plot()

        mock.figure.assert_called()
        mock.scatter.assert_called()
        mock.imshow.assert_called()
