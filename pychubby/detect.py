"""Collection of detection algorithms."""
import math
import pathlib

import dlib
import numpy as np
import matplotlib.pyplot as plt
import scipy
from skimage.util import img_as_ubyte

from pychubby.base import CACHE_FOLDER

LANDMARK_NAMES = {
    "UPPER_TEMPLE_L": 0,
    "MIDDLE_TEMPLE_L": 1,
    "LOWER_TEMPLE_L": 2,
    "UPPERMOST_CHEEK_L": 3,
    "UPPER_CHEEK_L": 4,
    "LOWER_CHEEK_L": 5,
    "LOWERMOST_CHEEK_L": 6,
    "CHIN_L": 7,
    "CHIN": 8,
    "CHIN_R": 9,
    "LOWERMOST_CHEEK_R": 10,
    "LOWER_CHEEK_R": 11,
    "UPPER_CHEEK_R": 12,
    "UPPERMOST_CHEEK_R": 13,
    "LOWER_TEMPLE_R": 14,
    "MIDDLE_TEMPLE_R": 15,
    "UPPER_TEMPLE_R": 16,
    "OUTERMOST_EYEBROW_L": 17,
    "OUTER_EYEBROW_L": 18,
    "MIDDLE_EYEBROW_L": 19,
    "INNER_EYEBROW_L": 20,
    "INNERMOST_EYEBROW_L": 21,
    "INNERMOST_EYEBROW_R": 22,
    "INNER_EYEBROW_R": 23,
    "MIDDLE_EYEBROW_R": 24,
    "OUTER_EYEBROW_R": 25,
    "OUTERMOST_EYEBROW_R": 26,
    "UPPERMOST_NOSE": 27,
    "UPPER_NOSE": 28,
    "LOWER_NOSE": 29,
    "LOWERMOST_NOSE": 30,
    "OUTER_NOSTRIL_L": 31,
    "INNER_NOSTRIL_L": 32,
    "MIDDLE_NOSTRIL": 33,
    "INNER_NOSTRIL_R": 34,
    "OUTER_NOSTRIL_R": 35,
    "OUTER_EYE_CORNER_L": 36,
    "OUTER_EYE_LID_L": 37,
    "INNER_EYE_LID_L": 38,
    "INNER_EYE_CORNER_L": 39,
    "INNER_EYE_BOTTOM_L": 40,
    "OUTER_EYE_BOTTOM_L": 41,
    "INNER_EYE_CORNER_R": 42,
    "INNER_EYE_LID_R": 43,
    "OUTER_EYE_LID_R": 44,
    "OUTER_EYE_CORNER_R": 45,
    "OUTER_EYE_BOTTOM_R": 46,
    "INNER_EYE_BOTTOM_R": 47,
    "OUTSIDE_MOUTH_CORNER_L": 48,
    "OUTER_OUTSIDE_UPPER_LIP_L": 49,
    "INNER_OUTSIDE_UPPER_LIP_L": 50,
    "MIDDLE_OUTSIDE_UPPER_LIP": 51,
    "INNER_OUTSIDE_UPPER_LIP_R": 52,
    "OUTER_OUTSIDE_UPPER_LIP_R": 53,
    "OUTSIDE_MOUTH_CORNER_R": 54,
    "OUTER_OUTSIDE_LOWER_LIP_R": 55,
    "INNER_OUTSIDE_LOWER_LIP_R": 56,
    "MIDDLE_OUTSIDE_LOWER_LIP": 57,
    "INNER_OUTSIDE_LOWER_LIP_L": 58,
    "OUTER_OUTSIDE_LOWER_LIP_L": 59,
    "INSIDE_MOUTH_CORNER_L": 60,
    "INSIDE_UPPER_LIP_L": 61,
    "MIDDLE_INSIDE_UPPER_LIP": 62,
    "INSIDE_UPPER_LIP_R": 63,
    "INSIDE_MOUTH_CORNER_R": 64,
    "INSIDE_LOWER_LIP_R": 65,
    "MIDDLE_INSIDE_LOWER_LIP": 66,
    "INSIDE_LOWER_LIP_L": 67,
}


def face_rectangle(img, n_upsamples=1):
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

    n_upsamples : int
        Upsample factor to apply to the image before detection. Allows to recognize
        more faces.

    """
    if not isinstance(img, np.ndarray):
        raise TypeError("The input needs to be a np.ndarray")

    dlib_detector = dlib.get_frontal_face_detector()

    faces = dlib_detector(img_as_ubyte(img), n_upsamples)

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
        model_path = CACHE_FOLDER / "shape_predictor_68_face_landmarks.dat"
    else:
        model_path = pathlib.Path(str(model_path))

    if not model_path.is_file():
        raise IOError("Invalid landmark model, {}".format(str(model_path)))

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
        Representing an image of a face. Any dtype and any number of channels.

    rectangle : tuple
        Two containing two tuples where the first one represents the top left corner
        of a rectangle and the second one the bottom right corner of a rectangle.

    Attributes
    ----------
    shape : tuple
        Tuple representing the height and width of the image.

    _convex_hull : scipy.spatial.ConvexHull
        Convex hull of the input landmark points.

    """

    @classmethod
    def estimate(cls, img, model_path=None, n_upsamples=1):
        """Estimate the 68 landmarks.

        Parameters
        ----------
        img : np.ndarray
            Array representing an image of a face. Any dtype and any number of channels.

        model_path : str or pathlib.Path, default=None
            Path to where the pretrained model is located. If None then using
            the `CACHE_FOLDER` model.

        n_upsamples : int
            Upsample factor to apply to the image before detection. Allows to recognize
            more faces.

        """
        corners, faces = face_rectangle(img, n_upsamples=n_upsamples)

        if len(corners) != 1:
            raise ValueError(
                "Only possible to model one face, detected faces {}".format(
                    len(corners)
                )
            )
        _, face = corners[0], faces[0]
        points, _ = landmarks_68(img, face)

        return cls(points, img)

    def __init__(self, points, img, rectangle=None):
        """Construct."""
        # Checks
        if points.shape != (68, 2):
            raise ValueError("There needs to be 68 2D landmarks.")

        if np.unique(points, axis=0).shape != (68, 2):
            raise ValueError("There are some duplicates.")

        self.points = points
        self.img = img
        self.rectangle = rectangle
        self.img_shape = self.img.shape[
            :2
        ]  # only first two dims matter - height and width

        # Inner attributes
        self._convex_hull = scipy.spatial.ConvexHull(self.points)

    @property
    def face_area(self):
        """Area of the face."""
        return self._convex_hull.area

    @property
    def face_volume(self):
        """Volume  of the face in the image."""
        return self._convex_hull.volume

    def angle(self, landmark_1, landmark_2, reference_vector=None, use_radians=False):
        """Angle between two landmarks and positive part of the x axis.

        The possible values range from (-180, 180) in degrees.

        Parameters
        ----------
        landmark_1 : int
            An integer from [0,57] representing a landmark point. The start
            of the vector.

        landmark_2 : int
            An integer from [0,57] representing a landmark point. The end
            of the vector.

        reference_vector : None or tuple
            If None, then positive part of the x axis used (1, 0). Otherwise
            specified by the user.

        use_radians : bool
            If True, then radians used. Otherwise degrees.

        Returns
        -------
        angle : float
            The angle between the two landmarks and positive part of the x axis.

        """
        v_1 = (
            np.array([1, 0]) if reference_vector is None else np.array(reference_vector)
        )
        v_2 = self.points[landmark_2] - self.points[landmark_1]

        res_radians = math.atan2(
            v_1[0] * v_2[1] - v_1[1] * v_2[0], v_1[0] * v_2[0] + v_1[1] * v_2[1]
        )

        if use_radians:
            return res_radians
        else:
            return math.degrees(res_radians)

    def euclidean_distance(self, landmark_1, landmark_2):
        """Euclidean distance between 2 landmarks.

        Parameters
        ----------
        landmark_1 : int
            An integer from [0,57] representing a landmark point.

        landmark_2 : int
            An integer from [0,57] representing a landmark point.


        Returns
        -------
        dist : float
            Euclidean distance between `landmark_1` and `landmark_2`.

        """
        return np.linalg.norm(self.points[landmark_1] - self.points[landmark_2])

    def plot(self, figsize=(12, 12)):
        """Plot face together with landmarks."""
        plt.figure(figsize=figsize)
        plt.scatter(self.points[:, 0], self.points[:, 1], c="black")
        plt.imshow(self.img, cmap="gray")
