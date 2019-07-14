"""Collections of tests focused on the utils.py module"""

import pytest

from pychubby.utils import points_to_rectangle_mask


class TestPointsToRectangleMask:
    """Tests focused on the `points_to_rectangle_mask` method."""
    
    def test_wrong_input(self):
        shape = (10, 12, 3)
        top_left = (2, 3)
        bottom_right = (4, 8)
        
        with pytest.raises(ValueError):
            points_to_rectangle_mask(shape, top_left, bottom_right)

    @pytest.mark.parametrize('coords', [((2, 3), (4, 9)),
                                        ((4, 4), (9, 10))])
    def test_output_shape_and_count(self, coords):   
        
        shape = (13, 15)
        top_left, bottom_right = coords
        
        out = points_to_rectangle_mask(shape, top_left, bottom_right, width=1)
        
        assert out.shape == shape
        assert out.sum() == -4 + 2 * (3 + bottom_right[0] - top_left[0]) + 2 * (3 + bottom_right[1] - top_left[1]) 
