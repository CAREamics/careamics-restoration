"""Functions to reimplement the tiling in the Disentangle repository."""

import builtins
import itertools
from typing import Any, Generator, Optional, Union

import numpy as np
from numpy.typing import NDArray

from careamics.config.tile_information import TileInformation
from careamics.lvae_training.dataset.data_utils import GridIndexManager


def extract_tiles(
    arr: NDArray,
    tile_size: NDArray[np.int_],
    overlaps: NDArray[np.int_],
    padding_kwargs: Optional[dict[str, Any]] = None,
) -> Generator[tuple[NDArray, TileInformation], None, None]:
    """Generate tiles from the input array with specified overlap.

    The tiles cover the whole array; which will be additionally padded, to ensure that
    the section of the tile that contributes to the final image comes from the center
    of the tile.

    The method returns a generator that yields tuples of array and tile information,
    the latter includes whether the tile is the last one, the coordinates of the
    overlap crop, and the coordinates of the stitched tile.

    Input array should have shape SC(Z)YX, while the returned tiles have shape C(Z)YX,
    where C can be a singleton.

    Parameters
    ----------
    arr : np.ndarray
        Array of shape (S, C, (Z), Y, X).
    tile_size : 1D numpy.ndarray of tuple
        Tile sizes in each dimension, of length 2 or 3.
    overlaps : 1D numpy.ndarray of tuple
        Overlap values in each dimension, of length 2 or 3.
    padding_kwargs : dict, optional
        The arguments of `np.pad` after the first two arguments, `array` and
        `pad_width`. If not specified the default will be `{"mode": "reflect"}`. See
        `numpy.pad` docs:
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html.

    Yields
    ------
    Generator[Tuple[np.ndarray, TileInformation], None, None]
        Tile generator, yields the tile and additional information.
    """
    if padding_kwargs is None:
        padding_kwargs = {"mode": "reflect"}

    # Iterate over num samples (S)
    for sample_idx in range(arr.shape[0]):
        sample = arr[sample_idx, ...]
        data_shape = np.array(sample.shape)

        # add padding to ensure evenly spaced & overlapping tiles.
        spatial_padding = compute_padding(data_shape, tile_size, overlaps)
        padding = ((0, 0), *spatial_padding)
        sample = np.pad(sample, padding, **padding_kwargs)

        # The number of tiles in each dimension, should be of length 2 or 3
        tile_grid_shape = compute_tile_grid_shape(data_shape, tile_size, overlaps)
        # itertools.product is equivalent of nested loops

        stitch_size = tile_size - overlaps
        for tile_grid_indices in itertools.product(
            *[range(n) for n in tile_grid_shape]
        ):

            # calculate crop coordinates
            crop_coords_start = np.array(tile_grid_indices) * stitch_size
            crop_slices: tuple[Union[builtins.ellipsis, slice], ...] = (
                ...,
                *[
                    slice(coords, coords + extent)
                    for coords, extent in zip(crop_coords_start, tile_size)
                ],
            )
            tile = sample[crop_slices]

            tile_info = compute_tile_info(
                np.array(tile_grid_indices),
                np.array(data_shape),
                np.array(tile_size),
                np.array(overlaps),
                sample_idx,
            )
            # TODO: kinda weird this is a generator,
            #   -> doesn't really save memory ? Don't think there are any places the
            #    tiles are not exracted all at the same time.
            #   Although I guess it would make sense for a zarr tile extractor.
            yield tile, tile_info


def compute_tile_info_legacy(
    grid_index_manager: GridIndexManager, index: int
) -> TileInformation:
    """
    Compute the tile information for a tile at a given dataset index.

    Parameters
    ----------
    grid_index_manager : GridIndexManager
        The grid index manager that keeps track of tile locations.
    index : int
        The dataset index.

    Returns
    -------
    TileInformation
        Information that describes how to crop and stitch a tile to create a full image.

    Raises
    ------
    ValueError
        If `grid_index_manager.data_shape` does not have 4 or 5 dimensions.
    """
    data_shape = np.array(grid_index_manager.data_shape)
    if len(data_shape) == 5:
        n_spatial_dims = 3
    elif len(data_shape) == 4:
        n_spatial_dims = 2
    else:
        raise ValueError("Data shape must have 4 or 5 dimensions, equating to SC(Z)YX.")

    stitch_coords_start = np.array(
        grid_index_manager.get_location_from_dataset_idx(index)
    )
    stitch_coords_end = stitch_coords_start + np.array(grid_index_manager.grid_shape)

    tile_coords_start = stitch_coords_start - grid_index_manager.patch_offset()

    # --- replace out of bounds indices
    out_of_lower_bound = stitch_coords_start < 0
    out_of_upper_bound = stitch_coords_end > data_shape
    stitch_coords_start[out_of_lower_bound] = 0
    stitch_coords_end[out_of_upper_bound] = data_shape[out_of_upper_bound]

    # TODO: TilingMode not in current version
    # if grid_index_manager.tiling_mode == TilingMode.ShiftBoundary:
    #     for dim in range(len(stitch_coords_start)):
    #         if tile_coords_start[dim] == 0:
    #             stitch_coords_start[dim] = 0
    #         if tile_coords_end[dim] == grid_index_manager.data_shape[dim]:
    #             tile_coords_end [dim]= grid_index_manager.data_shape[dim]

    # --- calculate overlap crop coords
    overlap_crop_coords_start = stitch_coords_start - tile_coords_start
    overlap_crop_coords_end = overlap_crop_coords_start + (
        stitch_coords_end - stitch_coords_start
    )

    last_tile = index == grid_index_manager.total_grid_count() - 1

    # --- combine start and end
    stitch_coords = tuple(
        (start, end) for start, end in zip(stitch_coords_start, stitch_coords_end)
    )
    overlap_crop_coords = tuple(
        (start, end)
        for start, end in zip(overlap_crop_coords_start, overlap_crop_coords_end)
    )

    tile_info = TileInformation(
        array_shape=data_shape[1:],  # remove S dim
        last_tile=last_tile,
        overlap_crop_coords=overlap_crop_coords[-n_spatial_dims:],
        stitch_coords=stitch_coords[-n_spatial_dims:],
        sample_id=0,
    )
    return tile_info


def compute_tile_info(
    tile_grid_indices: NDArray[np.int_],
    data_shape: NDArray[np.int_],
    tile_size: NDArray[np.int_],
    overlaps: NDArray[np.int_],
    sample_id: int = 0,
) -> TileInformation:
    """
    Compute the tile information for a tile with the coordinates `tile_grid_indices`.

    Parameters
    ----------
    tile_grid_indices : 1D np.array of int
        The coordinates of the tile within the tile grid, ((Z), Y, X), i.e. for 2D
        tiling the coordinates for the second tile in the first row of tiles would be
        (0, 1).
    data_shape : 1D np.array of int
        The shape of the data, should be (C, (Z), Y, X) where Z is optional.
    tile_size : 1D np.array of int
        Tile sizes in each dimension, of length 2 or 3.
    overlaps : 1D np.array of int
        Overlap values in each dimension, of length 2 or 3.
    sample_id : int, default=0
        An ID to identify which sample a tile belongs to.

    Returns
    -------
    TileInformation
        Information that describes how to crop and stitch a tile to create a full image.
    """
    spatial_dims_shape = data_shape[-len(tile_size) :]

    # The extent of the tile which will make up part of the stitched image.
    stitch_size = tile_size - overlaps
    stitch_coords_start = tile_grid_indices * stitch_size
    stitch_coords_end = stitch_coords_start + stitch_size

    tile_coords_start = stitch_coords_start - overlaps // 2

    # --- replace out of bounds indices
    out_of_lower_bound = stitch_coords_start < 0
    out_of_upper_bound = stitch_coords_end > spatial_dims_shape
    stitch_coords_start[out_of_lower_bound] = 0
    stitch_coords_end[out_of_upper_bound] = spatial_dims_shape[out_of_upper_bound]

    # --- calculate overlap crop coords
    overlap_crop_coords_start = stitch_coords_start - tile_coords_start
    overlap_crop_coords_end = overlap_crop_coords_start + (
        stitch_coords_end - stitch_coords_start
    )

    # --- combine start and end
    stitch_coords = tuple(
        (start, end) for start, end in zip(stitch_coords_start, stitch_coords_end)
    )
    overlap_crop_coords = tuple(
        (start, end)
        for start, end in zip(overlap_crop_coords_start, overlap_crop_coords_end)
    )

    # --- Check if last tile
    tile_grid_shape = np.array(compute_tile_grid_shape(data_shape, tile_size, overlaps))
    last_tile = (tile_grid_indices == (tile_grid_shape - 1)).all()

    tile_info = TileInformation(
        array_shape=data_shape,
        last_tile=last_tile,
        overlap_crop_coords=overlap_crop_coords,
        stitch_coords=stitch_coords,
        sample_id=sample_id,
    )
    return tile_info


def compute_padding(
    data_shape: NDArray[np.int_],
    tile_size: NDArray[np.int_],
    overlaps: NDArray[np.int_],
) -> tuple[tuple[int, int], ...]:
    """
    Calculate padding to ensure stitched data comes from the center of a tile.

    Padding is added to an array with shape `data_shape` so that when tiles are
    stitched together, the data used always comes from the center of a tile, even for
    tiles at the boundaries of the array.

    Parameters
    ----------
    data_shape : 1D numpy.array of int
        The shape of the data to be tiled and stitched together, (S, C, (Z), Y, X).
    tile_size : 1D numpy.array of int
        The tile size in each dimension, ((Z), Y, X).
    overlaps : 1D numpy.array of int
        The tile overlap in each dimension, ((Z), Y, X).

    Returns
    -------
    tuple of (int, int)
        A tuple specifying the padding to add in each dimension, each element is a two
        element tuple specifying the padding to add before and after the data. This
        can be used as the `pad_width` argument to `numpy.pad`.
    """
    tile_grid_shape = np.array(compute_tile_grid_shape(data_shape, tile_size, overlaps))
    covered_shape = (tile_size - overlaps) * tile_grid_shape + overlaps

    pad_before = overlaps // 2
    pad_after = covered_shape - data_shape[-len(tile_size) :] - pad_before

    return tuple((before, after) for before, after in zip(pad_before, pad_after))


def n_tiles_1d(axis_size: int, tile_size: int, overlap: int) -> int:
    """Calculate the number of tiles in a specific dimension.

    Parameters
    ----------
    axis_size : int
        The length of the data for in a specific dimension.
    tile_size : int
        The length of the tiles in a specific dimension.
    overlap : int
        The tile overlap in a specific dimension.

    Returns
    -------
    int
        The number of tiles that fit in one dimension given the arguments.
    """
    return int(np.ceil(axis_size / (tile_size - overlap)))


def total_n_tiles(
    data_shape: tuple[int, ...], tile_size: tuple[int, ...], overlaps: tuple[int, ...]
) -> int:
    """Calculate The total number of tiles over all dimensions.

    Parameters
    ----------
    data_shape : 1D numpy.array of int
        The shape of the data to be tiled and stitched together, (S, C, (Z), Y, X).
    tile_size : 1D numpy.array of int
        The tile size in each dimension, ((Z), Y, X).
    overlaps : 1D numpy.array of int
        The tile overlap in each dimension, ((Z), Y, X).


    Returns
    -------
    int
        The total number of tiles over all dimensions.
    """
    result = 1
    # assume spatial dimension are the last dimensions so iterate backwards
    for i in range(-1, -len(tile_size) - 1, -1):
        result = result * n_tiles_1d(data_shape[i], tile_size[i], overlaps[i])

    return result


def compute_tile_grid_shape(
    data_shape: NDArray[np.int_],
    tile_size: NDArray[np.int_],
    overlaps: NDArray[np.int_],
) -> tuple[int, ...]:
    """Calculate the number of tiles in each dimension.

    This can be thought of as a grid of tiles.

    Parameters
    ----------
    data_shape : 1D numpy.array of int
        The shape of the data to be tiled and stitched together, (S, C, (Z), Y, X).
    tile_size : 1D numpy.array of int
        The tile size in each dimension, ((Z), Y, X).
    overlaps : 1D numpy.array of int
        The tile overlap in each dimension, ((Z), Y, X).

    Returns
    -------
    tuple of int
        The number of tiles in each direction, ((Z, Y, X)).
    """
    shape = [0 for _ in range(len(tile_size))]
    # assume spatial dimension are the last dimensions so iterate backwards
    for i in range(-1, -len(tile_size) - 1, -1):
        shape[i] = n_tiles_1d(data_shape[i], tile_size[i], overlaps[i])
    return tuple(shape)
