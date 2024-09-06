"""Prediction Lightning data modules."""

from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import pytorch_lightning as L
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from careamics.config import InferenceConfig
from careamics.config.support import SupportedData
from careamics.dataset import (
    InMemoryPredDataset,
    InMemoryTiledPredDataset,
    IterablePredDataset,
    IterableTiledPredDataset,
)
from careamics.dataset.dataset_utils import list_files
from careamics.dataset.tiling.collate_tiles import collate_tiles
from careamics.file_io.read import get_read_func
from careamics.utils import get_logger

PredictDatasetType = Union[
    InMemoryPredDataset,
    InMemoryTiledPredDataset,
    IterablePredDataset,
    IterableTiledPredDataset,
]

logger = get_logger(__name__)


class PredictDataModule(L.LightningDataModule):
    """
    CAREamics Lightning prediction data module.

    The data module can be used with Path, str or numpy arrays. The data can be either
    a folder containing images or a single file.

    To read custom data types, you can set `data_type` to `custom` in `data_config`
    and provide a function that returns a numpy array from a path as
    `read_source_func` parameter. The function will receive a Path object and
    an axies string as arguments, the axes being derived from the `data_config`.

    You can also provide a `fnmatch` and `Path.rglob` compatible expression (e.g.
    "*.czi") to filter the files extension using `extension_filter`.

    Parameters
    ----------
    pred_config : InferenceModel
        Pydantic model for CAREamics prediction configuration.
    pred_data : pathlib.Path or str or numpy.ndarray
        Prediction data, can be a path to a folder, a file or a numpy array.
    read_source_func : Callable, optional
        Function to read custom types, by default None.
    extension_filter : str, optional
        Filter to filter file extensions for custom types, by default "".
    dataloader_params : dict, optional
        Dataloader parameters, by default {}.
    """

    def __init__(
        self,
        pred_config: InferenceConfig,
        pred_data: Union[Path, str, NDArray],
        read_source_func: Optional[Callable] = None,
        extension_filter: str = "",
        dataloader_params: Optional[dict] = None,
    ) -> None:
        """
        Constructor.

        The data module can be used with Path, str or numpy arrays. The data can be
        either a folder containing images or a single file.

        To read custom data types, you can set `data_type` to `custom` in `data_config`
        and provide a function that returns a numpy array from a path as
        `read_source_func` parameter. The function will receive a Path object and
        an axies string as arguments, the axes being derived from the `data_config`.

        You can also provide a `fnmatch` and `Path.rglob` compatible expression (e.g.
        "*.czi") to filter the files extension using `extension_filter`.

        Parameters
        ----------
        pred_config : InferenceModel
            Pydantic model for CAREamics prediction configuration.
        pred_data : pathlib.Path or str or numpy.ndarray
            Prediction data, can be a path to a folder, a file or a numpy array.
        read_source_func : Callable, optional
            Function to read custom types, by default None.
        extension_filter : str, optional
            Filter to filter file extensions for custom types, by default "".
        dataloader_params : dict, optional
            Dataloader parameters, by default {}.

        Raises
        ------
        ValueError
            If the data type is `custom` and no `read_source_func` is provided.
        ValueError
            If the data type is `array` and the input is not a numpy array.
        ValueError
            If the data type is `tiff` and the input is neither a Path nor a str.
        """
        if dataloader_params is None:
            dataloader_params = {}
        if dataloader_params is None:
            dataloader_params = {}
        super().__init__()

        # check that a read source function is provided for custom types
        if pred_config.data_type == SupportedData.CUSTOM and read_source_func is None:
            raise ValueError(
                f"Data type {SupportedData.CUSTOM} is not allowed without "
                f"specifying a `read_source_func` and an `extension_filer`."
            )

        # check correct input type
        if (
            isinstance(pred_data, np.ndarray)
            and pred_config.data_type != SupportedData.ARRAY
        ):
            raise ValueError(
                f"Received a numpy array as input, but the data type was set to "
                f"{pred_config.data_type}. Set the data type "
                f"to {SupportedData.ARRAY} to predict on numpy arrays."
            )

        # and that Path or str are passed, if tiff file type specified
        elif (isinstance(pred_data, Path) or isinstance(pred_config, str)) and (
            pred_config.data_type != SupportedData.TIFF
            and pred_config.data_type != SupportedData.CUSTOM
        ):
            raise ValueError(
                f"Received a path as input, but the data type was neither set to "
                f"{SupportedData.TIFF} nor {SupportedData.CUSTOM}. Set the data type "
                f" to {SupportedData.TIFF} or "
                f"{SupportedData.CUSTOM} to predict on files."
            )

        # configuration data
        self.prediction_config = pred_config
        self.data_type = pred_config.data_type
        self.batch_size = pred_config.batch_size
        self.dataloader_params = dataloader_params

        self.pred_data = pred_data
        self.tile_size = pred_config.tile_size
        self.tile_overlap = pred_config.tile_overlap

        # check if it is tiled
        self.tiled = self.tile_size is not None and self.tile_overlap is not None

        # read source function
        if pred_config.data_type == SupportedData.CUSTOM:
            # mypy check
            assert read_source_func is not None

            self.read_source_func: Callable = read_source_func
        elif pred_config.data_type != SupportedData.ARRAY:
            self.read_source_func = get_read_func(pred_config.data_type)

        self.extension_filter = extension_filter

    def prepare_data(self) -> None:
        """Hook used to prepare the data before calling `setup`."""
        # if the data is a Path or a str
        if not isinstance(self.pred_data, np.ndarray):
            self.pred_files = list_files(
                self.pred_data, self.data_type, self.extension_filter
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Hook called at the beginning of predict.

        Parameters
        ----------
        stage : Optional[str], optional
            Stage, by default None.
        """
        # if numpy array
        if self.data_type == SupportedData.ARRAY:
            if self.tiled:
                self.predict_dataset: PredictDatasetType = InMemoryTiledPredDataset(
                    prediction_config=self.prediction_config,
                    inputs=self.pred_data,
                )
            else:
                self.predict_dataset = InMemoryPredDataset(
                    prediction_config=self.prediction_config,
                    inputs=self.pred_data,
                )
        else:
            if self.tiled:
                self.predict_dataset = IterableTiledPredDataset(
                    prediction_config=self.prediction_config,
                    src_files=self.pred_files,
                    read_source_func=self.read_source_func,
                )
            else:
                self.predict_dataset = IterablePredDataset(
                    prediction_config=self.prediction_config,
                    src_files=self.pred_files,
                    read_source_func=self.read_source_func,
                )

    def predict_dataloader(self) -> DataLoader:
        """
        Create a dataloader for prediction.

        Returns
        -------
        DataLoader
            Prediction dataloader.
        """
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_tiles if self.tiled else None,
            **self.dataloader_params,
        )


def create_predict_datamodule(
    pred_data: Union[str, Path, NDArray],
    data_type: Union[Literal["array", "tiff", "custom"], SupportedData],
    axes: str,
    image_means: list[float],
    image_stds: list[float],
    tile_size: Optional[tuple[int, ...]] = None,
    tile_overlap: Optional[tuple[int, ...]] = None,
    batch_size: int = 1,
    tta_transforms: bool = True,
    read_source_func: Optional[Callable] = None,
    extension_filter: str = "",
    dataloader_params: Optional[dict] = None,
) -> PredictDataModule:
    """Create a CAREamics prediction Lightning datamodule.

    This function is used to explicitly pass the parameters usually contained in an
    `inference_model` configuration.

    Since the lightning datamodule has no access to the model, make sure that the
    parameters passed to the datamodule are consistent with the model's requirements
    and are coherent. This can be done by creating a `Configuration` object beforehand
    and passing its parameters to the different Lightning modules.

    The data module can be used with Path, str or numpy arrays. To use array data, set
    `data_type` to `array` and pass a numpy array to `train_data`.

    By default, CAREamics only supports types defined in
    `careamics.config.support.SupportedData`. To read custom data types, you can set
    `data_type` to `custom` and provide a function that returns a numpy array from a
    path. Additionally, pass a `fnmatch` and `Path.rglob` compatible expression
    (e.g. "*.jpeg") to filter the files extension using `extension_filter`.

    In `dataloader_params`, you can pass any parameter accepted by PyTorch
    dataloaders, except for `batch_size`, which is set by the `batch_size`
    parameter.

    Parameters
    ----------
    pred_data : str or pathlib.Path or numpy.ndarray
        Prediction data.
    data_type : {"array", "tiff", "custom"}
        Data type, see `SupportedData` for available options.
    axes : str
        Axes of the data, chosen among SCZYX.
    image_means : list of float
        Mean values for normalization, only used if Normalization is defined.
    image_stds : list of float
        Std values for normalization, only used if Normalization is defined.
    tile_size : tuple of int, optional
        Tile size, 2D or 3D tile size.
    tile_overlap : tuple of int, optional
        Tile overlap, 2D or 3D tile overlap.
    batch_size : int
        Batch size.
    tta_transforms : bool, optional
        Use test time augmentation, by default True.
    read_source_func : Callable, optional
        Function to read the source data, used if `data_type` is `custom`, by
        default None.
    extension_filter : str, optional
        Filter for file extensions, used if `data_type` is `custom`, by default "".
    dataloader_params : dict, optional
        Pytorch dataloader parameters, by default {}.

    Returns
    -------
    PredictDataModule
        CAREamics prediction datamodule.

    Notes
    -----
    If you are using a UNet model and tiling, the tile size must be
    divisible in every dimension by 2**d, where d is the depth of the model. This
    avoids artefacts arising from the broken shift invariance induced by the
    pooling layers of the UNet. If your image has less dimensions, as it may
    happen in the Z dimension, consider padding your image.
    """
    if dataloader_params is None:
        dataloader_params = {}

    prediction_dict: dict[str, Any] = {
        "data_type": data_type,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "axes": axes,
        "image_means": image_means,
        "image_stds": image_stds,
        "tta_transforms": tta_transforms,
        "batch_size": batch_size,
    }

    # validate configuration
    prediction_config = InferenceConfig(**prediction_dict)

    # sanity check on the dataloader parameters
    if "batch_size" in dataloader_params:
        # remove it
        del dataloader_params["batch_size"]

    return PredictDataModule(
        pred_config=prediction_config,
        pred_data=pred_data,
        read_source_func=read_source_func,
        extension_filter=extension_filter,
        dataloader_params=dataloader_params,
    )
