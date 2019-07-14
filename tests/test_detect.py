"""Collection of tests focused on the `detect` module."""
from collections import namedtuple
from unittest.mock import Mock

import dlib
import numpy as np
import pytest

import pychubby.detect
from pychubby.detect import face_rectangle, landmarks_68

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
