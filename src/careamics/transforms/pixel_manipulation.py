"""
Pixel manipulation methods.

Pixel manipulation is used in N2V and similar algorithm to replace the value of
masked pixels.
"""

from typing import Optional, Tuple

import numpy as np

from .struct_mask_parameters import StructMaskParameters


def _apply_struct_mask(
    patch: np.ndarray,
    coords: np.ndarray,
    struct_params: StructMaskParameters,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply structN2V masks to patch.

    Each point in `coords` corresponds to the center of a mask, masks are paremeterized
    by `struct_params` and pixels in the mask (with respect to `coords`) are replaced by
    a random value.

    Note that the structN2V mask is applied in 2D at the coordinates given by `coords`.

    Parameters
    ----------
    patch : np.ndarray
        Patch to be manipulated, 2D or 3D.
    coords : np.ndarray
        Coordinates of the ROI(subpatch) centers.
    struct_params : StructMaskParameters
        Parameters for the structN2V mask (axis and span).
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    np.ndarray
        Patch with the structN2V mask applied.
    """
    if rng is None:
        rng = np.random.default_rng()

    # relative axis
    moving_axis = -1 - struct_params.axis

    # Create a mask array
    mask = np.expand_dims(
        np.ones(struct_params.span), axis=list(range(len(patch.shape) - 1))
    )  # (1, 1, span) or (1, span)

    # Move the moving axis to the correct position
    # i.e. the axis along which the coordinates should change
    mask = np.moveaxis(mask, -1, moving_axis)
    center = np.array(mask.shape) // 2

    # Mark the center
    mask[tuple(center.T)] = 0

    # displacements from center
    dx = np.indices(mask.shape)[:, mask == 1] - center[:, None]

    # combine all coords (ndim, npts,) with all displacements (ncoords,ndim,)
    mix = dx.T[..., None] + coords.T[None]
    mix = mix.transpose([1, 0, 2]).reshape([mask.ndim, -1]).T

    # delete entries that are out of bounds
    mix = np.delete(mix, mix[:, moving_axis] < 0, axis=0)

    max_bound = patch.shape[moving_axis] - 1
    mix = np.delete(mix, mix[:, moving_axis] > max_bound, axis=0)

    # replace neighbouring pixels with random values from flat dist
    patch[tuple(mix.T)] = rng.uniform(patch.min(), patch.max(), size=mix.shape[0])

    return patch


def _odd_jitter_func(step: float, rng: np.random.Generator) -> np.ndarray:
    """
    Randomly sample a jitter to be applied to the masking grid.

    This is done to account for cases where the step size is not an integer.

    Parameters
    ----------
    step : float
        Step size of the grid, output of np.linspace.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of random jitter to be added to the grid.
    """
    # Define the random jitter to be added to the grid
    odd_jitter = np.where(np.floor(step) == step, 0, rng.integers(0, 2))

    # Round the step size to the nearest integer depending on the jitter
    return np.floor(step) if odd_jitter == 0 else np.ceil(step)


def _get_stratified_coords(
    mask_pixel_perc: float,
    shape: Tuple[int, ...],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate coordinates of the pixels to mask.

    Randomly selects the coordinates of the pixels to mask in a stratified way, i.e.
    the distance between masked pixels is approximately the same.

    Parameters
    ----------
    mask_pixel_perc : float
        Actual (quasi) percentage of masked pixels across the whole image. Used in
        calculating the distance between masked pixels across each axis.
    shape : Tuple[int, ...]
        Shape of the input patch.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of coordinates of the masked pixels.
    """
    if len(shape) < 2 or len(shape) > 3:
        raise ValueError(
            "Calculating coordinates is only possible for 2D and 3D patches"
        )

    if rng is None:
        rng = np.random.default_rng()

    mask_pixel_distance = np.round((100 / mask_pixel_perc) ** (1 / len(shape))).astype(
        np.int32
    )

    # Define a grid of coordinates for each axis in the input patch and the step size
    pixel_coords = []
    steps = []
    for axis_size in shape:
        # make sure axis size is evenly divisible by box size
        num_pixels = int(np.ceil(axis_size / mask_pixel_distance))
        axis_pixel_coords, step = np.linspace(
            0, axis_size, num_pixels, dtype=np.int32, endpoint=False, retstep=True
        )
        # explain
        pixel_coords.append(axis_pixel_coords.T)
        steps.append(step)

    # Create a meshgrid of coordinates for each axis in the input patch
    coordinate_grid_list = np.meshgrid(*pixel_coords)
    coordinate_grid = np.array(coordinate_grid_list).reshape(len(shape), -1).T

    grid_random_increment = rng.integers(
        _odd_jitter_func(float(max(steps)), rng)  # type: ignore
        * np.ones_like(coordinate_grid).astype(np.int32)
        - 1,
        size=coordinate_grid.shape,
        endpoint=True,
    )
    coordinate_grid += grid_random_increment
    coordinate_grid = np.clip(coordinate_grid, 0, np.array(shape) - 1)
    return coordinate_grid


def _create_subpatch_center_mask(
    subpatch: np.ndarray, center_coords: np.ndarray
) -> np.ndarray:
    """Create a mask with the center of the subpatch masked.

    Parameters
    ----------
    subpatch : np.ndarray
        Subpatch to be manipulated.
    center_coords : np.ndarray
        Coordinates of the original center before possible crop.

    Returns
    -------
    np.ndarray
        Mask with the center of the subpatch masked.
    """
    mask = np.ones(subpatch.shape)
    mask[tuple(center_coords)] = 0
    return np.ma.make_mask(mask)  # type: ignore


def _create_subpatch_struct_mask(
    subpatch: np.ndarray, center_coords: np.ndarray, struct_params: StructMaskParameters
) -> np.ndarray:
    """Create a structN2V mask for the subpatch.

    Parameters
    ----------
    subpatch : np.ndarray
        Subpatch to be manipulated.
    center_coords : np.ndarray
        Coordinates of the original center before possible crop.
    struct_params : StructMaskParameters
        Parameters for the structN2V mask (axis and span).

    Returns
    -------
    np.ndarray
        StructN2V mask for the subpatch.
    """
    # Create a mask with the center of the subpatch masked
    mask_placeholder = np.ones(subpatch.shape)

    # reshape to move the struct axis to the first position
    mask_reshaped = np.moveaxis(mask_placeholder, struct_params.axis, 0)

    # create the mask index for the struct axis
    mask_index = slice(
        max(0, center_coords.take(struct_params.axis) - (struct_params.span - 1) // 2),
        min(
            1 + center_coords.take(struct_params.axis) + (struct_params.span - 1) // 2,
            subpatch.shape[struct_params.axis],
        ),
    )
    mask_reshaped[struct_params.axis][mask_index] = 0

    # reshape back to the original shape
    mask = np.moveaxis(mask_reshaped, 0, struct_params.axis)

    return np.ma.make_mask(mask)  # type: ignore


def uniform_manipulate(
    patch: np.ndarray,
    mask_pixel_percentage: float,
    subpatch_size: int = 11,
    remove_center: bool = True,
    struct_params: Optional[StructMaskParameters] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Manipulate pixels by replacing them with a neighbor values.

    Manipulated pixels are selected unformly selected in a subpatch, away from a grid
    with an approximate uniform probability to be selected across the whole patch.
    If `struct_params` is not None, an additional structN2V mask is applied to the
    data, replacing the pixels in the mask with random values (excluding the pixel
    already manipulated).

    Parameters
    ----------
    patch : np.ndarray
        Image patch, 2D or 3D, shape (y, x) or (z, y, x).
    mask_pixel_percentage : float
        Approximate percentage of pixels to be masked.
    subpatch_size : int
        Size of the subpatch the new pixel value is sampled from, by default 11.
    remove_center : bool
        Whether to remove the center pixel from the subpatch, by default False.
    struct_params : StructMaskParameters or None
        Parameters for the structN2V mask (axis and span).
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    Tuple[np.ndarray]
        Tuple containing the manipulated patch and the corresponding mask.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Get the coordinates of the pixels to be replaced
    transformed_patch = patch.copy()

    subpatch_centers = _get_stratified_coords(mask_pixel_percentage, patch.shape, rng)

    # Generate coordinate grid for subpatch
    roi_span_full = np.arange(
        -np.floor(subpatch_size / 2), np.ceil(subpatch_size / 2)
    ).astype(np.int32)

    # Remove the center pixel from the grid if needed
    roi_span = roi_span_full[roi_span_full != 0] if remove_center else roi_span_full

    # Randomly select coordinates from the grid
    random_increment = rng.choice(roi_span, size=subpatch_centers.shape)

    # Clip the coordinates to the patch size
    replacement_coords = np.clip(
        subpatch_centers + random_increment,
        0,
        [patch.shape[i] - 1 for i in range(len(patch.shape))],
    )

    # Get the replacement pixels from all subpatchs
    replacement_pixels = patch[tuple(replacement_coords.T.tolist())]

    # Replace the original pixels with the replacement pixels
    transformed_patch[tuple(subpatch_centers.T.tolist())] = replacement_pixels
    mask = np.where(transformed_patch != patch, 1, 0).astype(np.uint8)

    if struct_params is not None:
        transformed_patch = _apply_struct_mask(
            transformed_patch, subpatch_centers, struct_params
        )

    return (
        transformed_patch,
        mask,
    )


def median_manipulate(
    patch: np.ndarray,
    mask_pixel_percentage: float,
    subpatch_size: int = 11,
    struct_params: Optional[StructMaskParameters] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Manipulate pixels by replacing them with the median of their surrounding subpatch.

    N2V2 version, manipulated pixels are selected randomly away from a grid with an
    approximate uniform probability to be selected across the whole patch.

    If `struct_params` is not None, an additional structN2V mask is applied to the data,
    replacing the pixels in the mask with random values (excluding the pixel already
    manipulated).

    Parameters
    ----------
    patch : np.ndarray
        Image patch, 2D or 3D, shape (y, x) or (z, y, x).
    mask_pixel_percentage : floar
        Approximate percentage of pixels to be masked.
    subpatch_size : int
        Size of the subpatch the new pixel value is sampled from, by default 11.
    struct_params : StructMaskParameters or None, optional
        Parameters for the structN2V mask (axis and span).
    rng : np.random.Generator or None, optional
        Random number generato, by default None.

    Returns
    -------
    Tuple[np.ndarray]
           Tuple containing the manipulated patch, the original patch and the mask.
    """
    if rng is None:
        rng = np.random.default_rng()

    transformed_patch = patch.copy()

    # Get the coordinates of the pixels to be replaced
    subpatch_centers = _get_stratified_coords(mask_pixel_percentage, patch.shape, rng)

    # Generate coordinate grid for subpatch
    roi_span = np.array(
        [-np.floor(subpatch_size / 2), np.ceil(subpatch_size / 2)]
    ).astype(np.int32)

    subpatch_crops_span_full = subpatch_centers[np.newaxis, ...].T + roi_span

    # Dimensions n dims, n centers, (min, max)
    subpatch_crops_span_clipped = np.clip(
        subpatch_crops_span_full,
        a_min=np.zeros_like(patch.shape)[:, np.newaxis, np.newaxis],
        a_max=np.array(patch.shape)[:, np.newaxis, np.newaxis],
    )

    for idx in range(subpatch_crops_span_clipped.shape[1]):
        subpatch_coords = subpatch_crops_span_clipped[:, idx, ...]
        idxs = [
            slice(x[0], x[1]) if x[1] - x[0] > 0 else slice(0, 1)
            for x in subpatch_coords
        ]
        subpatch = patch[tuple(idxs)]
        subpatch_center_adjusted = subpatch_centers[idx] - subpatch_coords[:, 0]

        if struct_params is None:
            subpatch_mask = _create_subpatch_center_mask(
                subpatch, subpatch_center_adjusted
            )
        else:
            subpatch_mask = _create_subpatch_struct_mask(
                subpatch, subpatch_center_adjusted, struct_params
            )
        transformed_patch[tuple(subpatch_centers[idx])] = np.median(
            subpatch[subpatch_mask]
        )

    mask = np.where(transformed_patch != patch, 1, 0).astype(np.uint8)

    if struct_params is not None:
        transformed_patch = _apply_struct_mask(
            transformed_patch, subpatch_centers, struct_params
        )

    return (
        transformed_patch,
        mask,
    )
