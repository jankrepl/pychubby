import pathlib

import cv2
import numpy as np
import pytest



TEST_DATA_PATH = pathlib.Path(__file__).parent / 'data'
TEST_IMAGE_PATH = TEST_DATA_PATH / 'brad.jpg'


@pytest.fixture()
def face_img_gs_uint():
   """Load a grayscale image of dtype uint8.""" 
   img = cv2.imread(str(TEST_IMAGE_PATH), 0)

   assert img.ndim == 2
   assert img.dtype == np.uint8

   return img

@pytest.fixture()
def face_img_rgb_uint():
   """Load a RGB image of dtype uint8.""" 
   img = cv2.imread(str(TEST_IMAGE_PATH), 1)
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   
   assert img.ndim == 3
   assert img.dtype == np.uint8

   return img

@pytest.fixture()
def face_img_gs_float():
   """Load a grayscale image of dtype float32.""" 
   img = cv2.imread(str(TEST_IMAGE_PATH), 0)
   img = img.astype(np.float32) / 255
   
   assert img.ndim == 2
   assert img.dtype == np.float32

   return img

@pytest.fixture()
def face_img_rgb_float():
   """Load a rgb image of dtype float32.""" 
   img = cv2.imread(str(TEST_IMAGE_PATH), 1)
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
   
   assert img.ndim == 3
   assert img.dtype == np.float32

   return img

@pytest.fixture(params=['gs_uint', 'rgb_uint', 'gs_float', 'rgb_float'])
def face_img(request, face_img_gs_uint, face_img_rgb_uint, face_img_gs_float, face_img_rgb_float):
    """Load an image of a face."""
    img_type = request.param
    
    if img_type == 'gs_uint':
        return face_img_gs_uint
    elif img_type == 'rgb_uint':
        return face_img_rgb_uint
    elif img_type == 'gs_float':
        return face_img_gs_float
    elif img_type == 'rgb_float':
        return face_img_rgb_float
    else:
        raise ValueError('Invalid img_type {}'.format(img_type))

    print(TEST_DATA_PATH)
    pass

