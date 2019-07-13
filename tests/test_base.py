"""Collection of tests focused on the base module."""

import numpy as np
import pytest

from pychubby.base import DisplacementField

class TestConstructor:
    def test_incorrect_input_type(self):
        delta_x = 'aa'
        delta_y = 12

        with pytest.raises(TypeError):
            DisplacementField(delta_x, delta_y)
    
    def test_incorrect_ndim(self):
        delta_x = np.ones((2, 3, 4))
        delta_y = np.ones((2, 3, 4))

        with pytest.raises(ValueError):
            DisplacementField(delta_x, delta_y)

    def test_different_shape(self):
        delta_x = np.ones((2, 3))
        delta_y = np.ones((2, 4))
        
        with pytest.raises(ValueError):
            DisplacementField(delta_x, delta_y)

    def test_dtype(self):
        
        delta_x = np.ones((2, 3))
        delta_y = np.ones((2, 3))

        df = DisplacementField(delta_x, delta_y)
        
        assert df.delta_x.dtype == np.float32
        assert df.delta_y.dtype == np.float32


class TestProperties:
    def test_is_valid(self):

        delta_x = np.ones((2, 3))
        delta_y = np.ones((2, 3))
        delta_y_inv_1 = np.ones((2, 3))
        delta_y_inv_1[0, 1] = np.inf
        delta_y_inv_2 = np.ones((2, 3))
        delta_y_inv_2[0, 1] = np.nan


        df_val = DisplacementField(delta_x, delta_y)
        df_inv_1 = DisplacementField(delta_x, delta_y_inv_1)
        df_inv_2 = DisplacementField(delta_x, delta_y_inv_2)

        assert df_val.is_valid
        assert not df_inv_1.is_valid
        assert not df_inv_2.is_valid
