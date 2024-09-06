"""Iterable tiled prediction dataset used to load data file by file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Generator

from numpy.typing import NDArray
from torch.utils.data import IterableDataset

from careamics.file_io.read import read_tiff
from careamics.transforms import Compose

from ..config import InferenceConfig
from ..config.tile_information import TileInformation
from ..config.transformations import NormalizeModel
from .dataset_utils import iterate_over_files
from .tiling import extract_tiles


class IterableTiledPredDataset(IterableDataset):
    """Tiled prediction dataset.

    Parameters
    ----------
    prediction_config : InferenceConfig
        Inference configuration.
    src_files : list of pathlib.Path
        List of data files.
    read_source_func : Callable, optional
        Read source function for custom types, by default read_tiff.
    **kwargs : Any
        Additional keyword arguments, unused.

    Attributes
    ----------
    data_path : str or pathlib.Path
        Path to the data, must be a directory.
    axes : str
        Description of axes in format STCZYX.
    mean : float, optional
        Expected mean of the dataset, by default None.
    std : float, optional
        Expected standard deviation of the dataset, by default None.
    patch_transform : Callable, optional
        Patch transform callable, by default None.
    """

    def __init__(
        self,
        prediction_config: InferenceConfig,
        src_files: list[Path],
        read_source_func: Callable = read_tiff,
        **kwargs: Any,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        prediction_config : InferenceConfig
            Inference configuration.
        src_files : List[Path]
            List of data files.
        read_source_func : Callable, optional
            Read source function for custom types, by default read_tiff.
        **kwargs : Any
            Additional keyword arguments, unused.

        Raises
        ------
        ValueError
            If mean and std are not provided in the inference configuration.
        """
        if (
            prediction_config.tile_size is None
            or prediction_config.tile_overlap is None
        ):
            raise ValueError(
                "Tile size and overlap must be provided for tiled prediction."
            )

        self.prediction_config = prediction_config
        self.data_files = src_files
        self.axes = prediction_config.axes
        self.tile_size = prediction_config.tile_size
        self.tile_overlap = prediction_config.tile_overlap
        self.read_source_func = read_source_func

        # check mean and std and create normalize transform
        if (
            self.prediction_config.image_means is None
            or self.prediction_config.image_stds is None
        ):
            raise ValueError("Mean and std must be provided for prediction.")
        else:
            self.image_means = self.prediction_config.image_means
            self.image_stds = self.prediction_config.image_stds

            # instantiate normalize transform
            self.patch_transform = Compose(
                transform_list=[
                    NormalizeModel(
                        image_means=self.image_means,
                        image_stds=self.image_stds,
                    )
                ],
            )

    def __iter__(
        self,
    ) -> Generator[tuple[NDArray, TileInformation], None, None]:
        """
        Iterate over data source and yield single patch.

        Yields
        ------
        Generator of NDArray and TileInformation tuple
            Generator of single tiles.
        """
        assert (
            self.image_means is not None and self.image_stds is not None
        ), "Mean and std must be provided"

        for sample, _ in iterate_over_files(
            self.prediction_config,
            self.data_files,
            read_source_func=self.read_source_func,
        ):
            # generate patches, return a generator of single tiles
            patch_gen = extract_tiles(
                arr=sample,
                tile_size=self.tile_size,
                overlaps=self.tile_overlap,
            )

            # apply transform to patches
            for patch_array, tile_info in patch_gen:
                transformed_patch, _ = self.patch_transform(patch=patch_array)

                yield transformed_patch, tile_info
