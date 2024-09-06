import numpy as np
import pytest

from careamics.config.tile_information import TileInformation
from careamics.dataset.tiling.lvae_tiled_patching import (
    compute_padding,
    compute_tile_grid_shape,
    compute_tile_info,
    extract_tiles,
    n_tiles_1d,
    total_n_tiles,
)
from careamics.prediction_utils.stitch_prediction import stitch_prediction


@pytest.mark.parametrize(
    "data_shape, tile_size, overlaps",
    [
        # 2D
        ((1, 3, 10, 9), (4, 4), (2, 2)),
        ((1, 3, 10, 9), (8, 8), (4, 4)),
        # 3D
        ((1, 3, 8, 16, 17), (4, 4, 4), (2, 2, 2)),
        ((1, 3, 8, 16, 17), (8, 8, 8), (4, 4, 4)),
    ],
)
def test_extract_tiles(data_shape, tile_size, overlaps):
    """Test extracted tiles are all the same size and can reconstruct the image."""

    arr = np.random.random_sample(data_shape).astype(np.float32)

    tile_data_generator = extract_tiles(
        arr=arr, tile_size=np.array(tile_size), overlaps=np.array(overlaps)
    )

    tiles = []
    tile_infos = []

    # Assemble all tiles and their respective coordinates
    for tile, tile_info in tile_data_generator:

        overlap_crop_coords = tile_info.overlap_crop_coords
        stitch_coords = tile_info.stitch_coords

        # add data to lists
        tiles.append(tile)
        tile_infos.append(tile_info)

        # check tile shape, ignore channel dimension
        assert tile.shape[1:] == tile_size
        assert len(overlap_crop_coords) == len(stitch_coords) == len(tile_size)

    # stitch_prediction returns list
    stitched_arr = stitch_prediction(tiles, tile_infos)[0]

    np.testing.assert_array_equal(arr, stitched_arr)


def test_compute_tile_info():
    """Test `compute_tile_info` for a selection of known results."""

    # TODO: improve this test ?

    data_shape = np.array([1, 3, 10, 9])
    tile_size = np.array([4, 4])
    overlaps = np.array([2, 2])

    # first example
    tile_info = compute_tile_info((0, 0), data_shape[1:], tile_size, overlaps)
    assert tile_info == TileInformation(
        array_shape=tuple(data_shape[1:]),
        last_tile=False,
        overlap_crop_coords=((1, 3), (1, 3)),
        stitch_coords=((0, 2), (0, 2)),
        sample_id=0,
    )

    # second example
    tile_info = compute_tile_info((2, 2), data_shape[1:], tile_size, overlaps)
    assert tile_info == TileInformation(
        array_shape=tuple(data_shape[1:]),
        last_tile=False,
        overlap_crop_coords=((1, 3), (1, 3)),
        stitch_coords=((4, 6), (4, 6)),
        sample_id=0,
    )

    # third example
    tile_info = compute_tile_info((2, 4), data_shape[1:], tile_size, overlaps)
    assert tile_info == TileInformation(
        array_shape=tuple(data_shape[1:]),
        last_tile=False,
        overlap_crop_coords=((1, 3), (1, 2)),
        stitch_coords=((4, 6), (8, 9)),
        sample_id=0,
    )

    # fourth example
    tile_info = compute_tile_info((4, 4), data_shape[1:], tile_size, overlaps)
    assert tile_info == TileInformation(
        array_shape=tuple(data_shape[1:]),
        last_tile=True,
        overlap_crop_coords=((1, 3), (1, 2)),
        stitch_coords=((8, 10), (8, 9)),
        sample_id=0,
    )


@pytest.mark.parametrize(
    "data_shape, tile_size, overlaps",
    [
        # 2D
        ((1, 3, 10, 9), (4, 4), (2, 2)),
        ((1, 3, 10, 9), (8, 8), (4, 4)),
        # 3D
        ((1, 3, 8, 16, 17), (4, 4, 4), (2, 2, 2)),
        ((1, 3, 8, 16, 17), (8, 8, 8), (4, 4, 4)),
    ],
)
def test_compute_padding(data_shape, tile_size, overlaps):

    padding = compute_padding(
        np.array(data_shape), np.array(tile_size), np.array(overlaps)
    )

    for axis, (before, after) in enumerate(padding):
        # padded array should be divisible by the stitch size
        stitch_size = tile_size[axis] - overlaps[axis]
        axis_size = data_shape[axis + 2]  # + 2 for sample and channel dims
        assert (before + axis_size + after) % stitch_size == 0

        assert before == overlaps[axis] // 2


@pytest.mark.parametrize(
    "axis_size, tile_size, overlap",
    [(9, 4, 2), (10, 8, 4), (17, 8, 4)],
)
def test_n_tiles_1d(axis_size, tile_size, overlap):
    """Test calculating the number of tiles in a specific dimension."""
    result = n_tiles_1d(axis_size, tile_size, overlap)
    assert result == int(np.ceil(axis_size / (tile_size - overlap)))


@pytest.mark.parametrize(
    "data_shape, tile_size, overlaps",
    [
        # 2D
        ((1, 3, 10, 9), (4, 4), (2, 2)),
        ((1, 3, 10, 9), (8, 8), (4, 4)),
        # 3D
        ((1, 3, 8, 16, 17), (4, 4, 4), (2, 2, 2)),
        ((1, 3, 8, 16, 17), (8, 8, 8), (4, 4, 4)),
    ],
)
def test_total_n_tiles(data_shape, tile_size, overlaps):
    """Test calculating the total number of tiles."""

    result = total_n_tiles(data_shape, tile_size, overlaps)
    n_tiles = 1
    for i in range(-1, -len(tile_size) - 1, -1):
        n_tiles = n_tiles * int(np.ceil(data_shape[i] / (tile_size[i] - overlaps[i])))

    assert result == n_tiles


@pytest.mark.parametrize(
    "data_shape, tile_size, overlaps",
    [
        # 2D
        ((1, 3, 10, 9), (4, 4), (2, 2)),
        ((1, 3, 10, 9), (8, 8), (4, 4)),
        # 3D
        ((1, 3, 8, 16, 17), (4, 4, 4), (2, 2, 2)),
        ((1, 3, 8, 16, 17), (8, 8, 8), (4, 4, 4)),
    ],
)
def test_compute_tile_grid_shape(data_shape, tile_size, overlaps):
    """Test computing tile grid shape."""

    result = compute_tile_grid_shape(data_shape, tile_size, overlaps)

    tile_grid_shape = tuple(
        int(np.ceil(data_shape[i] / (tile_size[i] - overlaps[i])))
        for i in range(-len(tile_size), 0, 1)
    )

    assert result == tile_grid_shape
