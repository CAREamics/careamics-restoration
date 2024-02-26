import pytest

import numpy as np
from albumentations import Compose

from careamics.transforms import ManipulateN2V
from careamics.config.support import SupportedPixelManipulation



@pytest.mark.parametrize("strategy",
    [
        SupportedPixelManipulation.UNIFORM,
        SupportedPixelManipulation.MEDIAN
    ]
)

def test_manipulate_n2v(strategy):
    """Test the N2V augmentation."""
    # create array
    array = np.arange(16*16).reshape((16, 16))

    # create augmentation
    aug = Compose([
        ManipulateN2V(
            roi_size=5,
            masked_pixel_percentage=5,
            strategy=strategy
        )
    ])

    # apply augmentation
    augmented = aug(image=array)
    assert "image" in augmented
    assert len(augmented["image"]) == 3 # transformed_patch, original_patch, mask

    # assert that the difference between the original and transformed patch are the 
    # same pixels that are selected by the mask
    tr_path, orig_patch, mask = augmented["image"]
    diff_coords = np.array(np.where(tr_path != orig_patch))
    mask_coords = np.array(np.where(mask == 1))
    assert np.array_equal(diff_coords, mask_coords)