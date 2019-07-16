"""Collection of detection algorithms."""
import pathlib

import dlib
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte

from pychubby.base import CACHE_FOLDER


def face_rectangle(img):
    """Find a face rectangle.

    Parameters
    ----------
    img : np.ndarray
        Image of any dtype and number of channels.

    Returns
    -------
    corners : list
        List of tuples where each tuple represents the top left and bottom right coordinates of
        the face rectangle. Note that these coordinates use the `(row, column)` convention. The
        length of the list is equal to the number of detected faces.

    faces : list
        Instance of ``dlib.rectagles`` that can be used in other algorithm.

    """
    if not isinstance(img, np.ndarray):
        raise TypeError('The input needs to be a np.ndarray')

    dlib_detector = dlib.get_frontal_face_detector()

    faces = dlib_detector(img_as_ubyte(img))

    corners = []
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        top_left = (y1, x1)
        bottom_right = (y2, x2)
        corners.append((top_left, bottom_right))

    return corners, faces


def landmarks_68(img, rectangle, model_path=None):
    """Predict 68 face landmarks.

    Parameters
    ----------
    img : np.ndarray
        Image of any dtype and number of channels.

    rectangle : dlib.rectangle
        Rectangle that represents the bounding box around a single face.

    model_path : str or pathlib.Path, default=None
        Path to where the pretrained model is located. If None then using the `CACHE_FOLDER` model.

    Returns
    -------
    lm_points : np.ndarray
        Array of shape `(68, 2)` where rows are different landmark points and the columns
        are x and y coordinates.

    original : dlib.full_object_detection
        Instance of ``dlib.full_object_detection``.

    """
    if model_path is None:
        model_path = CACHE_FOLDER / 'shape_predictor_68_face_landmarks.dat'
    else:
        model_path = pathlib.Path(str(model_path))

    if not model_path.is_file():
        raise IOError('Invalid landmark model, {}'.format(str(model_path)))

    lm_predictor = dlib.shape_predictor(str(model_path))

    original = lm_predictor(img_as_ubyte(img), rectangle)

    lm_points = np.array([[p.x, p.y] for p in original.parts()])

    return lm_points, original


class LandmarkFace:
    """Class representing a combination of a face image and its landmarks.

    Parameters
    ----------
    points : np.ndarray
        Array of shape `(68, 2)` where rows are different landmark points and the columns
        are x and y coordinates.

    img : np.ndarray
        Array representing an image of a face. Any dtype and any number of channels.

    """

    @classmethod
    def estimate(cls, img, model_path=None):
        """Estimate the 68 landmarks.

        Parameters
        ----------
        img : np.ndarray
            Array representing an image of a face. Any dtype and any number of channels.

        model_path : str or pathlib.Path, default=None
            Path to where the pretrained model is located. If None then using
            the `CACHE_FOLDER` model.

        Attributes
        ----------
        shape : tuple
            Two element tuple representing the height and width of the image.

        """
        corners, faces = face_rectangle(img)

        if len(corners) != 1:
            raise ValueError('Only possible to model one face, detected faces {}'.format(len(corners)))
        _, face = corners[0], faces[0]
        points, _ = landmarks_68(img, face)

        return cls(points, img)

    def __init__(self, points, img):
        """Construct."""
        self.points = points
        self.img = img
        self.img_shape = self.img.shape[:2]  # only first two dims matter - height and width

    def plot(self, figsize=(12, 12)):
        """Plot face together with landmarks."""
        plt.figure(figsize=figsize)
        plt.scatter(self.points[:, 0], self.points[:, 1], c='black')
        plt.imshow(self.img, cmap='gray')
