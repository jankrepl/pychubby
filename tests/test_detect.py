"""Collection of tests focused on the `detect` module."""
from collections import namedtuple
from unittest.mock import Mock
import math

import dlib
import numpy as np
import pytest

import pychubby.detect
from pychubby.detect import LANDMARK_NAMES, LandmarkFace, LandmarkFaces, face_rectangle, landmarks_68


class TestLandmarkNames:
    """Tests focused on the `LANDMARK_NAMES` dictionary."""
    def test_unique(self):
        assert len(set(LANDMARK_NAMES.keys())) == 68
        assert len(set(LANDMARK_NAMES.values())) == 68


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


class TestLandmarkFaceEstimate:
    """Tests focused on the class method `estimate` of the LandmarkFace."""

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

    def test_no_faces(self, monkeypatch):
        monkeypatch.setattr('pychubby.detect.face_rectangle',
                            lambda *args, **kwargs: ([], []))
        img = np.random.random((10, 11))

        with pytest.raises(ValueError):
            LandmarkFace.estimate(img)

    def test_multiple_faces(self, monkeypatch):
        img = np.random.random((10, 11))

        monkeypatch.setattr('pychubby.detect.face_rectangle',
                            lambda *args, **kwargs: (2 * [None], 2 * [None]))

        monkeypatch.setattr('pychubby.detect.landmarks_68',
                            lambda *args: (np.random.random((68, 2)), None))

        with pytest.raises(ValueError):
            LandmarkFace.estimate(img, allow_multiple=False)

        lfs = LandmarkFace.estimate(img, allow_multiple=True)

        assert isinstance(lfs, LandmarkFaces)
        assert len(lfs) == 2

        # only feed invalid, empty entry for LandmarkFaces constructor
        monkeypatch.setattr('pychubby.detect.landmarks_68',
                            lambda *args: (np.zeros((68, 2)), None))

        with pytest.raises(ValueError):
            LandmarkFace.estimate(img, allow_multiple=True)


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


class TestLandmarkFaceGetItem:
    """Collection of tests focused on the `__get_item__` method."""

    def test_with_int(self):
        random_state = 2
        np.random.seed(random_state)

        points = np.random.random((68, 2))
        lf = LandmarkFace(points, np.zeros((12, 13)))

        # one by one
        for i in range(68):
            assert np.allclose(lf[i], points[i])
        # random list of indices
        ixs = np.random.randint(0, 68, size=10)

        assert np.allclose(lf[ixs], points[ixs])
        assert np.allclose(lf[[x.item() for x in ixs]], points[ixs])

    def test_with_str(self):
        random_state = 2
        np.random.seed(random_state)

        ix2name = {v: k for k, v in LANDMARK_NAMES.items()}

        points = np.random.random((68, 2))
        lf = LandmarkFace(points, np.zeros((12, 13)))

        # one by one
        for i in range(68):
            assert np.allclose(lf[ix2name[i]], points[i])
        # random list of indices
        ixs = np.random.randint(0, 68, size=10)
        strings = [ix2name[x] for x in ixs]

        assert np.allclose(lf[strings], points[ixs])

    def test_incorrect_input(self):
        points = np.random.random((68, 2))
        lf = LandmarkFace(points, np.zeros((12, 13)))

        with pytest.raises(TypeError):
            lf[(1, 2)]

        with pytest.raises(TypeError):
            lf[[32.1]]

        with pytest.raises(ValueError):
            lf[np.zeros((2, 2))]


class TestLandmarkFacePlot:
    def test_plot(self, monkeypatch):
        mock = Mock()

        monkeypatch.setattr('pychubby.detect.plt', mock)

        lf = LandmarkFace(np.random.random((68, 2)), np.random.random((12, 13)))

        lf.plot()

        mock.figure.assert_called()
        mock.scatter.assert_called()
        mock.imshow.assert_called()


@pytest.fixture()
def lf():
    points = np.random.random((68, 2))
    return LandmarkFace(points, np.zeros((12, 13)))


class TestLandmarkFacesAll:
    """Collection of tests focused on the ``LandmarkFaces`` class."""

    def test_constructor(self):
        with pytest.raises(ValueError):
            LandmarkFaces()

        with pytest.raises(TypeError):
            LandmarkFaces('a')

        with pytest.raises(ValueError):
            points = np.random.random((68, 2))
            lf_1 = LandmarkFace(points, np.zeros((12, 13)))
            lf_2 = LandmarkFace(points, np.ones((12, 13)))
            LandmarkFaces(lf_1, lf_2)

    def test_length(self, lf):
        assert len(LandmarkFaces(lf, lf, lf)) == 3
        assert len(LandmarkFaces(lf, lf, lf, lf, lf)) == 5

    def test_getitem(self, lf):
        lfs = LandmarkFaces(lf)

        assert np.allclose(lfs[0].points, lf.points)
        assert np.allclose(lfs[0].img, lf.img)

    def test_plot(self, lf, monkeypatch):
        mock = Mock()

        monkeypatch.setattr('pychubby.detect.plt', mock)

        lfs = LandmarkFaces(lf)

        lfs.plot(show_numbers=True, show_landmarks=True)

        mock.figure.assert_called()
        mock.scatter.assert_called()
        mock.annotate.assert_called()
        mock.imshow.assert_called()
