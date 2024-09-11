import numpy as np
import pytest

from careamics.dataset.dataset_utils.running_stats import compute_normalization_stats
from careamics.transforms import Denormalize, Normalize
from careamics.transforms.normalize import _reshape_stats


@pytest.mark.parametrize("ndim", [3, 4])
def test_reshape_stats(ndim):
    """Test that reshape stats allows a list of float to be broadcasted to a given
    number of dimensions."""
    shape = (4,) + (8,) * (ndim - 1)
    assert len(shape) == ndim

    # create stats
    stats = [(i + 1) for i in range(shape[0])]

    # create test array
    array = np.ones(shape)

    # reshape and check that you can perform simple operations without error
    reshaped = _reshape_stats(stats, ndim)
    assert reshaped.shape == (shape[0],) + (1,) * (ndim - 1)

    mult = array * reshaped
    for i in range(shape[0]):
        assert (mult[i] == 1 * stats[i]).all()

    add = array + reshaped
    for i in range(shape[0]):
        assert (add[i] == 1 + stats[i]).all()

    sub = array - reshaped
    for i in range(shape[0]):
        assert (sub[i] == 1 - stats[i]).all()

    div = array / reshaped
    for i in range(shape[0]):
        assert (div[i] == 1 / stats[i]).all()


@pytest.mark.parametrize("channels", [1, 2])
def test_normalize_denormalize(channels):
    """Test the Normalize transform."""
    # Create data, adding sample dimension for stats computation
    array = np.arange(100 * channels).reshape((1, channels, 10, 10))

    # Compute mean and std
    means, stds = compute_normalization_stats(image=array)

    # Create the transform
    norm = Normalize(
        image_means=means,
        image_stds=stds,
    )

    # Apply the transform, removing the sample dimension
    normalized, *_ = norm(patch=array[0])
    assert np.abs(normalized.mean()) < 0.02
    assert np.abs(normalized.std() - 1) < 0.2

    # Create the denormalize transform
    denorm = Denormalize(
        image_means=means,
        image_stds=stds,
    )

    # Apply the denormalize transform
    denormalized = denorm(patch=normalized[np.newaxis, ...])  # need to add batch dim
    assert np.isclose(denormalized, array, atol=1e-6).all()


# long name sorry
def test_transform_additional_arrays_not_implemented(ordered_array):
    """Test normalize raises not implemented if additional arrays are used"""
    # create inputs
    shape = (2, 2, 5, 5)
    array = ordered_array(shape)
    additional_arrays = {"arr": ordered_array(shape)}

    # Compute mean and std
    means, stds = compute_normalization_stats(image=array)

    # Create the transform
    norm = Normalize(
        image_means=means,
        image_stds=stds,
    )

    with pytest.raises(NotImplementedError):
        norm(array, **additional_arrays)
