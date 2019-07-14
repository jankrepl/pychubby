"""Collection of detection algorithms."""

import dlib
import numpy as np
from skimage.util import img_as_ubyte


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
