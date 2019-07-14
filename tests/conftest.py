import pathlib

import cv2
import numpy as np
import pytest



TEST_DATA_PATH = pathlib.Path(__file__).parent / 'data'
TEST_FACE_IMAGE_PATH = TEST_DATA_PATH / 'brad.jpg'
TEST_FACES_IMAGE_PATH = TEST_DATA_PATH / 'kids.jpg'

@pytest.fixture()
def face_img_gs_uint():
   """Load a grayscale image of dtype uint8.""" 
   img = cv2.imread(str(TEST_FACE_IMAGE_PATH), 0)

   assert img.ndim == 2
   assert img.dtype == np.uint8

   return img

@pytest.fixture()
def face_img_rgb_uint():
   """Load a RGB image of dtype uint8.""" 
   img = cv2.imread(str(TEST_FACE_IMAGE_PATH), 1)
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   
   assert img.ndim == 3
   assert img.dtype == np.uint8

   return img

@pytest.fixture()
def face_img_gs_float():
   """Load a grayscale image of dtype float32.""" 
   img = cv2.imread(str(TEST_FACE_IMAGE_PATH), 0)
   img = img.astype(np.float32) / 255
   
   assert img.ndim == 2
   assert img.dtype == np.float32

   return img

@pytest.fixture()
def face_img_rgb_float():
   """Load a rgb image of dtype float32.""" 
   img = cv2.imread(str(TEST_FACE_IMAGE_PATH), 1)
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

@pytest.fixture(params=['gs_uint', 'rgb_uint', 'gs_float', 'rgb_float'])
def blank_img(request):
    """Create a black image."""
    img_type = request.param
    shape = (50, 60)
    
    if img_type == 'gs_uint':
        return np.zeros(shape, dtype=np.uint8)
    elif img_type == 'rgb_uint':
        return np.zeros((*shape, 3), dtype=np.uint8)
    elif img_type == 'gs_float':
        return np.zeros(shape, dtype=np.float32)
    elif img_type == 'rgb_float':
        return np.zeros((*shape, 3), dtype=np.float32)
    else:
        raise ValueError('Invalid img_type {}'.format(img_type))


@pytest.fixture(params=['gs_uint', 'rgb_uint', 'gs_float', 'rgb_float'])
def faces_img(request):
    """Load an image of multiple faces."""
    img_type = request.param
    
    if img_type == 'gs_uint':
        return cv2.imread(str(TEST_FACES_IMAGE_PATH), 0)
    elif img_type == 'rgb_uint':
        return cv2.cvtColor(cv2.imread(str(TEST_FACES_IMAGE_PATH)), cv2.COLOR_BGR2RGB) 
    elif img_type == 'gs_float':
        return cv2.imread(str(TEST_FACES_IMAGE_PATH), 0).astype(np.float32) / 255
    elif img_type == 'rgb_float':
        return cv2.cvtColor(cv2.imread(str(TEST_FACES_IMAGE_PATH)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    else:
        raise ValueError('Invalid img_type {}'.format(img_type))

