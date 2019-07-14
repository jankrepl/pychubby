"""Collection of tests focused on the `detect` module."""

import numpy as np
import pytest

from pychubby.detect import face_rectangle


class TestFaceRectangle:
    """Tests of the `face_rectangle` function."""

    def test_incorrect_input(self):
        with pytest.raises(TypeError):
            face_rectangle('123')

    def test_output_face(self, face_img):
        res = face_rectangle(face_img)
        assert isinstance(res, list)
        assert len(res) == 1

    def test_output_blank(self, blank_img):
        res = face_rectangle(blank_img)
        assert isinstance(res, list)
        assert len(res) == 0

    def test_output_faces(self, faces_img):
        res = face_rectangle(faces_img)
        assert isinstance(res, list)
        assert len(res) == 2
