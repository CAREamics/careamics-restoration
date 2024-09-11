import numpy as np
import pytest

from careamics.transforms import XYRandomRotate90


@pytest.mark.parametrize(
    "shape",
    [
        # 2D
        (1, 2, 2),
        (2, 2, 2),
        # 3D
        (1, 2, 2, 2),
        (2, 2, 2, 2),
    ],
)
def test_xy_rotate(ordered_array, shape):
    """Test rotation for 2D and 3D arrays with a fixed seed."""
    # create array
    array = ordered_array(shape)

    # create augmentation
    aug = XYRandomRotate90(p=1, seed=42)
    r = np.random.default_rng(seed=42)

    axes = (2, 3) if len(array.shape) == 4 else (1, 2)
    rots = [
        np.rot90(array, k=1, axes=axes),
        np.rot90(array, k=2, axes=axes),
        np.rot90(array, k=3, axes=axes),
    ]

    # check rotations
    for _ in range(5):
        r.random()  # consume random number
        augmented, *_ = aug(array)

        assert np.array_equal(augmented, rots[r.integers(1, 4) - 1])


@pytest.mark.parametrize(
    "shape",
    [
        # 2D
        (1, 2, 2),
        (2, 2, 2),
        # 3D
        (1, 2, 2, 2),
        (2, 2, 2, 2),
    ],
)
def test_additional_arrays_xy_rotate(ordered_array, shape):
    """Test rotation for 2D and 3D arrays with a fixed seed."""
    # create array
    array: np.ndarray = ordered_array(shape)
    additional_arrays = {"arr": ordered_array(shape)}

    # create augmentation
    aug = XYRandomRotate90(p=1, seed=42)
    r = np.random.default_rng(seed=42)

    axes = (2, 3) if len(array.shape) == 4 else (1, 2)
    rots = [
        np.rot90(array, k=1, axes=axes),
        np.rot90(array, k=2, axes=axes),
        np.rot90(array, k=3, axes=axes),
    ]

    # check rotations
    for _ in range(5):
        r.random()  # consume random number
        augmented, _, additional_augmented = aug(array, **additional_arrays)

        assert np.array_equal(augmented, rots[r.integers(1, 4) - 1])
        assert np.array_equal(augmented, additional_augmented["arr"])


def test_mask_rotate(ordered_array):
    """Test rotating masks in 3D."""
    aug = XYRandomRotate90(p=1, seed=42)
    r = np.random.default_rng(seed=42)

    # create array
    array = ordered_array((4, 2, 2, 2))
    mask = array[2:, ...]
    array = array[:2, ...]

    # potential rotations
    axes = (2, 3)  # 3D
    array_rots = [
        np.rot90(array, k=1, axes=axes),
        np.rot90(array, k=2, axes=axes),
        np.rot90(array, k=3, axes=axes),
    ]
    mask_rots = [
        np.rot90(mask, k=1, axes=axes),
        np.rot90(mask, k=2, axes=axes),
        np.rot90(mask, k=3, axes=axes),
    ]

    # apply augmentation 10 times
    for _ in range(5):
        r.random()  # consume random number
        augmented_p, augmented_m, _ = aug(array, mask)
        n_rot = r.integers(1, 4)

        assert np.array_equal(augmented_p, array_rots[n_rot - 1])
        assert np.array_equal(augmented_m, mask_rots[n_rot - 1])


def test_p(ordered_array):
    """Test that the probability of rotation is respected."""
    # create array
    array = ordered_array((2, 2, 2))

    # create augmentation
    aug = XYRandomRotate90(p=0.0, seed=42)

    # apply augmentation 5 times
    for _ in range(5):
        augmented, *_ = aug(array)

        assert np.array_equal(augmented, array)

    # create augmentation that always applies
    aug = XYRandomRotate90(p=1.0, seed=42)

    # apply augmentation 5 times
    for _ in range(5):
        augmented, *_ = aug(array)

        assert not np.array_equal(augmented, array)
