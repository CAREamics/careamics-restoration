import numpy as np
import pytest

from careamics.transforms.pixel_manipulation import (
    _apply_struct_mask,
    _get_stratified_coords,
    median_manipulate,
    uniform_manipulate,
)
from careamics.transforms.struct_mask_parameters import StructMaskParameters


@pytest.mark.parametrize(
    "mask_pixel_perc, shape, num_iterations",
    [(0.4, (32, 32), 1000), (0.4, (10, 10, 10), 1000)],
)
def test_get_stratified_coords(mask_pixel_perc, shape, num_iterations):
    """Test the get_stratified_coords function.

    Ensure that the array of coordinates is randomly distributed across the
    image and that most pixels get selected.
    """
    rng = np.random.default_rng(42)

    # Define the dummy array
    array = np.zeros(shape)

    # Iterate over the number of iterations and add the coordinates. This is an MC
    # simulation to ensure that the coordinates are randomly distributed and not
    # biased towards any particular region.
    for _ in range(num_iterations):
        # Get the coordinates of the pixels to be masked
        coords = _get_stratified_coords(mask_pixel_perc, shape, rng)

        # Check that there is at least one coordinate chosen
        assert len(coords) > 0

        # Check every pair in the array of coordinates
        for coord_pair in coords:
            # Check that the coordinates are of the same shape as the patch dims
            assert len(coord_pair) == len(shape)

            # Check that the coordinates are positive values
            assert all(coord_pair) >= 0

            # Check that the coordinates are within the shape of the array
            assert [c <= s for c, s in zip(coord_pair, shape)]

        # Add the 1 to the every coordinate location.
        array[tuple(np.array(coords).T.tolist())] += 1

    # Ensure that there's no strong pattern in the array and sufficient number of
    # pixels is masked.
    assert np.sum(array == 0) < np.sum(shape)


@pytest.mark.parametrize("shape", [(8, 8), (3, 8, 8), (8, 8, 8)])
def test_uniform_manipulate(ordered_array, shape):
    """Test the uniform_manipulate function.

    Ensures that the mask corresponds to the manipulated pixels, and that the
    manipulated pixels have a value taken from a ROI surrounding them.
    """
    rng = np.random.default_rng(42)

    # create the array
    patch = ordered_array(shape)

    # manipulate the array
    transform_patch, mask = uniform_manipulate(
        patch, mask_pixel_percentage=10, subpatch_size=5, rng=rng
    )

    # find pixels that have different values between patch and transformed patch
    diff_coords = np.array(np.where(patch != transform_patch))

    # find non-zero pixels in the mask
    mask_coords = np.array(np.where(mask == 1))

    # check that the transformed pixels correspond to the masked pixels
    assert np.array_equal(diff_coords, mask_coords)

    # for each pixel masked, check that the manipulated pixel value is within the roi
    for i in range(mask_coords.shape[-1]):
        # get coordinates
        coords = mask_coords[..., i]

        # get roi using slice in each dimension
        slices = tuple(
            [
                slice(max(0, coords[i] - 2), min(shape[i], coords[i] + 3))
                for i in range(-coords.shape[0] + 1, 0)  # range -4, -3, -2, -1
            ]
        )
        roi = patch[
            (...,) + slices
        ]  # TODO ellipsis needed bc singleton dim, might need to go away

        # TODO needs to be revisited !
        # check that the pixel value comes from the actual roi
        assert transform_patch[tuple(coords)] in roi


@pytest.mark.parametrize("shape", [(8, 8), (3, 8, 8), (8, 8, 8)])
def test_median_manipulate(ordered_array, shape):
    """Test the uniform_manipulate function.

    Ensures that the mask corresponds to the manipulated pixels, and that the
    manipulated pixels have a value taken from a ROI surrounding them.
    """
    rng = np.random.default_rng(42)

    # create the array
    patch = ordered_array(shape).astype(np.float32)

    # manipulate the array
    transform_patch, mask = median_manipulate(
        patch, subpatch_size=5, mask_pixel_percentage=10, rng=rng
    )

    # find pixels that have different values between patch and transformed patch
    diff_coords = np.array(np.where(patch != transform_patch))

    # find non-zero pixels in the mask
    mask_coords = np.array(np.where(mask == 1))

    # check that the transformed pixels correspond to the masked pixels
    assert np.array_equal(diff_coords, mask_coords)

    # for each pixel masked, check that the manipulated pixel value is within the roi
    for i in range(mask_coords.shape[-1]):
        # get coordinates
        coords = mask_coords[..., i]

        # get roi using slice in each dimension
        slices = tuple(
            [
                slice(max(0, coords[i] - 2), min(shape[i], coords[i] + 3))
                for i in range(coords.shape[0])  # range -4, -3, -2, -1
            ]
        )
        roi = patch[tuple(slices)]

        # remove value of roi center from roi
        roi = roi[roi != patch[tuple(coords)]]

        # check that the pixel value comes from the actual roi
        assert transform_patch[tuple(coords)] == np.median(roi)


@pytest.mark.parametrize(
    "coords, struct_axis, struct_span",
    [((2, 2), 1, 5), ((3, 4), 0, 5), ((9, 0), 0, 5), (((1, 2), (3, 4)), 1, 5)],
)
def test_apply_struct_mask(coords, struct_axis, struct_span):
    """Test the uniform_manipulate function.

    Ensures that the mask corresponds to the manipulated pixels, and that the
    manipulated pixels have a value taken from a ROI surrounding them.
    """
    rng = np.random.default_rng(42)

    struct_params = StructMaskParameters(axis=struct_axis, span=struct_span)

    # create array
    patch = np.arange(
        100,
    ).reshape((10, 10))

    # make a copy of the original patch for comparison
    original_patch = patch.copy()
    coords = np.array(coords)

    # expand the coords if only one roi is given
    if coords.ndim == 1:
        coords = coords[None, :]

    # manipulate the array
    transform_patch = _apply_struct_mask(
        patch,
        coords=coords,
        struct_params=struct_params,
        rng=rng,
    )
    changed_values = patch[np.where(original_patch != transform_patch)]

    # check that the transformed pixels correspond to the masked pixels
    transformed = []
    axis = 1 - struct_axis
    for i in range(coords.shape[0]):
        # get indices to mask
        indices_to_mask = [
            c
            for c in range(
                max(0, coords[i, axis] - struct_span // 2),
                min(transform_patch.shape[1], coords[i, axis] + struct_span // 2) + 1,
            )
            if c != coords[i, axis]
        ]

        # add to transform
        if struct_axis == 0:
            transformed.append(transform_patch[coords[i, 0]][indices_to_mask])
        else:
            transformed.append(transform_patch[:, coords[i, 1]][indices_to_mask])

    assert np.array_equal(
        np.sort(changed_values), np.sort(np.concatenate(transformed, axis=0))
    )


@pytest.mark.parametrize(
    "coords, struct_axis, struct_span",
    [
        ((1, 2, 2), 1, 5),
        ((2, 3, 4), 0, 5),
        ((0, 9, 0), 0, 5),
        (((2, 1, 2), (1, 9, 0), (0, 3, 4)), 1, 5),
    ],
)
def test_apply_struct_mask_3D(coords, struct_axis, struct_span):
    """Test the uniform_manipulate function.

    Ensures that the mask corresponds to the manipulated pixels, and that the
    manipulated pixels have a value taken from a ROI surrounding them.
    """
    rng = np.random.default_rng(42)

    struct_params = StructMaskParameters(axis=struct_axis, span=struct_span)

    # create array
    patch = np.arange(
        100 * 3,
    ).reshape((3, 10, 10))

    # make a copy of the original patch for comparison
    original_patch = patch.copy()
    coords = np.array(coords)

    # expand the coords if only one roi is given
    if coords.ndim == 1:
        coords = coords[None, :]

    # manipulate the array
    transform_patch = _apply_struct_mask(
        patch,
        coords=coords,
        struct_params=struct_params,
        rng=rng,
    )
    changed_values = patch[np.where(original_patch != transform_patch)]

    # check that the transformed pixels correspond to the masked pixels
    transformed = []
    axis = -2 + 1 - struct_axis
    for i in range(coords.shape[0]):
        # get indices to mask
        indices_to_mask = [
            c
            for c in range(
                max(0, coords[i, axis] - struct_span // 2),
                min(transform_patch.shape[1] - 1, coords[i, axis] + struct_span // 2)
                + 1,
            )
            if c != coords[i, axis]
        ]

        # add to transform
        if struct_axis == 0:
            transformed.append(
                transform_patch[coords[i, 0], coords[i, 1]][indices_to_mask]
            )
        else:
            transformed.append(
                transform_patch[coords[i, 0], :, coords[i, 2]][indices_to_mask]
            )

    assert np.array_equal(
        np.sort(changed_values), np.sort(np.concatenate(transformed, axis=0))
    )
