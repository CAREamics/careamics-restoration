"""Prediction utility functions."""

import builtins
from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from careamics.config.tile_information import TileInformation


# TODO: why not allow input and output of torch.tensor ?
def stitch_prediction(
    tiles: List[np.ndarray],
    tile_infos: List[TileInformation],
) -> List[np.ndarray]:
    """
    Stitch tiles back together to form a full image(s).

    Tiles are of dimensions SC(Z)YX, where C is the number of channels and can be a
    singleton dimension.

    Parameters
    ----------
    tiles : list of numpy.ndarray
        Cropped tiles and their respective stitching coordinates. Can contain tiles
        from multiple images.
    tile_infos : list of TileInformation
        List of information and coordinates obtained from
        `dataset.tiled_patching.extract_tiles`.

    Returns
    -------
    list of numpy.ndarray
        Full image(s).
    """
    # Find where to split the lists so that only info from one image is contained.
    # Do this by locating the last tiles of each image.
    last_tiles = [tile_info.last_tile for tile_info in tile_infos]
    last_tile_position = np.where(last_tiles)[0]
    image_slices = [
        slice(
            None if i == 0 else last_tile_position[i - 1] + 1, last_tile_position[i] + 1
        )
        for i in range(len(last_tile_position))
    ]
    image_predictions = []
    # slice the lists and apply stitch_prediction_single to each in turn.
    for image_slice in image_slices:
        image_predictions.append(
            stitch_prediction_single(tiles[image_slice], tile_infos[image_slice])
        )
    return image_predictions


def stitch_prediction_single(
    tiles: List[NDArray],
    tile_infos: List[TileInformation],
) -> NDArray:
    """
    Stitch tiles back together to form a full image.

    Tiles are of dimensions SC(Z)YX, where C is the number of channels and can be a
    singleton dimension.

    Parameters
    ----------
    tiles : list of numpy.ndarray
        Cropped tiles and their respective stitching coordinates.
    tile_infos : list of TileInformation
        List of information and coordinates obtained from
        `dataset.tiled_patching.extract_tiles`.

    Returns
    -------
    numpy.ndarray
        Full image, with dimensions SC(Z)YX.
    """
    # TODO: this is hacky... need a better way to deal with when input channels and
    #   target channels do not match
    if len(tile_infos[0].array_shape) == 4:
        # 4 dimensions => 3 spatial dimensions so -4 is channel dimension
        tile_channels = tiles[0].shape[-4]
    elif len(tile_infos[0].array_shape) == 3:
        # 3 dimensions => 2 spatial dimensions so -3 is channel dimension
        tile_channels = tiles[0].shape[-3]
    else:
        # Note pretty sure this is unreachable because array shape is already
        #   validated by TileInformation
        raise ValueError(
            f"Unsupported number of output dimension {len(tile_infos[0].array_shape)}"
        )
    # retrieve whole array size, add S dim and use number of channels in tile
    input_shape = (1, tile_channels, *tile_infos[0].array_shape[1:])
    predicted_image = np.zeros(input_shape, dtype=np.float32)

    for tile, tile_info in zip(tiles, tile_infos):

        # Compute coordinates for cropping predicted tile
        crop_slices: tuple[Union[builtins.ellipsis, slice], ...] = (
            ...,
            *[slice(c[0], c[1]) for c in tile_info.overlap_crop_coords],
        )

        # Crop predited tile according to overlap coordinates
        cropped_tile = tile[crop_slices]

        # Insert cropped tile into predicted image using stitch coordinates
        image_slices = (..., *[slice(c[0], c[1]) for c in tile_info.stitch_coords])
        predicted_image[image_slices] = cropped_tile.astype(np.float32)

    return predicted_image
