"""Iterable prediction dataset used to load data file by file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Generator

from numpy.typing import NDArray
from torch.utils.data import IterableDataset

from careamics.file_io.read import read_tiff
from careamics.transforms import Compose

from ..config import InferenceConfig
from ..config.transformations import NormalizeModel
from .dataset_utils import iterate_over_files


class IterablePredDataset(IterableDataset):
    """Simple iterable prediction dataset.

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

    Attributes
    ----------
    data_path : Union[str, Path]
        Path to the data, must be a directory.
    axes : str
        Description of axes in format STCZYX.
    mean : Optional[float], optional
        Expected mean of the dataset, by default None.
    std : Optional[float], optional
        Expected standard deviation of the dataset, by default None.
    patch_transform : Optional[Callable], optional
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
        src_files : list of pathlib.Path
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
        self.prediction_config = prediction_config
        self.data_files = src_files
        self.axes = prediction_config.axes
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
    ) -> Generator[NDArray, None, None]:
        """
        Iterate over data source and yield single patch.

        Yields
        ------
        NDArray
            Single patch.
        """
        assert (
            self.image_means is not None and self.image_stds is not None
        ), "Mean and std must be provided"

        for sample, _ in iterate_over_files(
            self.prediction_config,
            self.data_files,
            read_source_func=self.read_source_func,
        ):
            # sample has S dimension
            for i in range(sample.shape[0]):

                transformed_sample, _ = self.patch_transform(patch=sample[i])

                yield transformed_sample
