import numpy as np
import pytest

from careamics.transforms import XYFlip


def test_p_wrong_values():
    """Test that an error is raised for wrong probability values."""
    with pytest.raises(ValueError):
        XYFlip(p=-1)

    with pytest.raises(ValueError):
        XYFlip(p=2)


def test_no_flip_error():
    """Test that an error is raised if no axis is flippable."""
    with pytest.raises(ValueError):
        XYFlip(flip_x=False, flip_y=False)


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
def test_flip_xy(ordered_array, shape):
    """Test flipping for 2D and 3D arrays."""
    # create array
    array: np.ndarray = ordered_array(shape)

    # create augmentation
    aug = XYFlip(p=1, seed=42)
    r = np.random.default_rng(seed=42)

    # potential flips
    axes = [-2, -1]
    flips = [np.flip(array, axis=axis) for axis in axes]

    # apply augmentation 5 times
    for _ in range(4):
        r.random()  # consume random number
        augmented, *_ = aug(array)

        # draw axis
        axis = r.choice(axes)

        assert np.array_equal(augmented, flips[axis])


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
def test_additional_arrays_flip_xy(ordered_array, shape):
    """Test flipping for 2D and 3D arrays."""
    # create array
    array: np.ndarray = ordered_array(shape)
    additional_arrays = {"arr": ordered_array(shape)}

    # create augmentation
    aug = XYFlip(p=1, seed=42)
    r = np.random.default_rng(seed=42)

    # potential flips
    axes = [-2, -1]
    flips = [np.flip(array, axis=axis) for axis in axes]

    # apply augmentation 5 times
    for _ in range(4):
        r.random()  # consume random number
        augmented, _, additional_augmented = aug(array, **additional_arrays)

        # draw axis
        axis = r.choice(axes)

        assert np.array_equal(augmented, flips[axis])
        assert np.array_equal(augmented, additional_augmented["arr"])


@pytest.mark.parametrize(
    "shape, flip_x",
    [
        # 2D
        ((1, 2, 2), True),
        ((2, 2, 2), True),
        ((1, 2, 2), False),
        ((2, 2, 2), False),
        # 3D
        ((1, 2, 2), True),
        ((2, 2, 2), True),
        ((1, 2, 2), False),
        ((2, 2, 2), False),
    ],
)
def test_flip_single_axis(ordered_array, shape, flip_x):
    """Test flipping for 2D and 3D arrays."""
    # create array
    array: np.ndarray = ordered_array(shape)

    # create augmentation
    aug = XYFlip(flip_x=flip_x, flip_y=not flip_x, p=1)

    # potential flips
    axis = -1 if flip_x else -2

    # apply augmentation 5 times
    for _ in range(5):
        augmented, *_ = aug(array)

        assert np.array_equal(augmented, np.flip(array, axis=axis))


def test_flip_mask(ordered_array):
    """Test flipping masks in 3D."""
    # create array
    array = ordered_array((4, 2, 2, 2))
    mask = array[2:, ...]
    array = array[:2, ...]

    # create augmentation
    aug = XYFlip(p=1, seed=42)
    r = np.random.default_rng(seed=42)

    # potential flips on Y and X axes
    axes = [-2, -1]
    array_flips = [np.flip(array, axis=axis) for axis in axes]
    mask_flips = [np.flip(mask, axis=axis) for axis in axes]

    # apply augmentation 5 times
    for _ in range(5):
        aug_array, aug_mask, _ = aug(patch=array, target=mask)
        r.random()  # consume random number
        axis = r.choice(axes)

        assert np.array_equal(aug_array, array_flips[axis])
        assert np.array_equal(aug_mask, mask_flips[axis])


def test_p(ordered_array):
    """Test that the probability of flipping is respected."""
    # create array
    array = ordered_array((2, 2, 2))

    # create augmentation that never applies
    aug = XYFlip(p=0.0, seed=42)

    # apply augmentation 5 times
    for _ in range(5):
        augmented, *_ = aug(array)

        assert np.array_equal(augmented, array)

    # create augmentation that always applies
    aug = XYFlip(p=1.0, seed=42)

    # apply augmentation 5 times
    for _ in range(5):
        augmented, *_ = aug(array)

        assert not np.array_equal(augmented, array)
