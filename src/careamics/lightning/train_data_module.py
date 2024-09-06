"""Training and validation Lightning data modules."""

from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import pytorch_lightning as L
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from careamics.config import DataConfig
from careamics.config.data_model import TRANSFORMS_UNION
from careamics.config.support import SupportedData
from careamics.dataset.dataset_utils import (
    get_files_size,
    list_files,
    validate_source_target_files,
)
from careamics.dataset.in_memory_dataset import (
    InMemoryDataset,
)
from careamics.dataset.iterable_dataset import (
    PathIterableDataset,
)
from careamics.file_io.read import get_read_func
from careamics.utils import get_logger, get_ram_size

DatasetType = Union[InMemoryDataset, PathIterableDataset]

logger = get_logger(__name__)


class TrainDataModule(L.LightningDataModule):
    """
    CAREamics Ligthning training and validation data module.

    The data module can be used with Path, str or numpy arrays. In the case of
    numpy arrays, it loads and computes all the patches in memory. For Path and str
    inputs, it calculates the total file size and estimate whether it can fit in
    memory. If it does not, it iterates through the files. This behaviour can be
    deactivated by setting `use_in_memory` to False, in which case it will
    always use the iterating dataset to train on a Path or str.

    The data can be either a folder containing images or a single file.

    Validation can be omitted, in which case the validation data is extracted from
    the training data. The percentage of the training data to use for validation,
    as well as the minimum number of patches or files to split from the training
    data can be set using `val_percentage` and `val_minimum_split`, respectively.

    To read custom data types, you can set `data_type` to `custom` in `data_config`
    and provide a function that returns a numpy array from a path as
    `read_source_func` parameter. The function will receive a Path object and
    an axies string as arguments, the axes being derived from the `data_config`.

    You can also provide a `fnmatch` and `Path.rglob` compatible expression (e.g.
    "*.czi") to filter the files extension using `extension_filter`.

    Parameters
    ----------
    data_config : DataModel
        Pydantic model for CAREamics data configuration.
    train_data : pathlib.Path or str or numpy.ndarray
        Training data, can be a path to a folder, a file or a numpy array.
    val_data : pathlib.Path or str or numpy.ndarray, optional
        Validation data, can be a path to a folder, a file or a numpy array, by
        default None.
    train_data_target : pathlib.Path or str or numpy.ndarray, optional
        Training target data, can be a path to a folder, a file or a numpy array, by
        default None.
    val_data_target : pathlib.Path or str or numpy.ndarray, optional
        Validation target data, can be a path to a folder, a file or a numpy array,
        by default None.
    read_source_func : Callable, optional
        Function to read the source data, by default None. Only used for `custom`
        data type (see DataModel).
    extension_filter : str, optional
        Filter for file extensions, by default "". Only used for `custom` data types
        (see DataModel).
    val_percentage : float, optional
        Percentage of the training data to use for validation, by default 0.1. Only
        used if `val_data` is None.
    val_minimum_split : int, optional
        Minimum number of patches or files to split from the training data for
        validation, by default 5. Only used if `val_data` is None.
    use_in_memory : bool, optional
        Use in memory dataset if possible, by default True.

    Attributes
    ----------
    data_config : DataModel
        CAREamics data configuration.
    data_type : SupportedData
        Expected data type, one of "tiff", "array" or "custom".
    batch_size : int
        Batch size.
    use_in_memory : bool
        Whether to use in memory dataset if possible.
    train_data : pathlib.Path or numpy.ndarray
        Training data.
    val_data : pathlib.Path or numpy.ndarray
        Validation data.
    train_data_target : pathlib.Path or numpy.ndarray
        Training target data.
    val_data_target : pathlib.Path or numpy.ndarray
        Validation target data.
    val_percentage : float
        Percentage of the training data to use for validation, if no validation data is
        provided.
    val_minimum_split : int
        Minimum number of patches or files to split from the training data for
        validation, if no validation data is provided.
    read_source_func : Optional[Callable]
        Function to read the source data, used if `data_type` is `custom`.
    extension_filter : str
        Filter for file extensions, used if `data_type` is `custom`.
    """

    def __init__(
        self,
        data_config: DataConfig,
        train_data: Union[Path, str, NDArray],
        val_data: Optional[Union[Path, str, NDArray]] = None,
        train_data_target: Optional[Union[Path, str, NDArray]] = None,
        val_data_target: Optional[Union[Path, str, NDArray]] = None,
        read_source_func: Optional[Callable] = None,
        extension_filter: str = "",
        val_percentage: float = 0.1,
        val_minimum_split: int = 5,
        use_in_memory: bool = True,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        data_config : DataModel
            Pydantic model for CAREamics data configuration.
        train_data : pathlib.Path or str or numpy.ndarray
            Training data, can be a path to a folder, a file or a numpy array.
        val_data : pathlib.Path or str or numpy.ndarray, optional
            Validation data, can be a path to a folder, a file or a numpy array, by
            default None.
        train_data_target : pathlib.Path or str or numpy.ndarray, optional
            Training target data, can be a path to a folder, a file or a numpy array, by
            default None.
        val_data_target : pathlib.Path or str or numpy.ndarray, optional
            Validation target data, can be a path to a folder, a file or a numpy array,
            by default None.
        read_source_func : Callable, optional
            Function to read the source data, by default None. Only used for `custom`
            data type (see DataModel).
        extension_filter : str, optional
            Filter for file extensions, by default "". Only used for `custom` data types
            (see DataModel).
        val_percentage : float, optional
            Percentage of the training data to use for validation, by default 0.1. Only
            used if `val_data` is None.
        val_minimum_split : int, optional
            Minimum number of patches or files to split from the training data for
            validation, by default 5. Only used if `val_data` is None.
        use_in_memory : bool, optional
            Use in memory dataset if possible, by default True.

        Raises
        ------
        NotImplementedError
            Raised if target data is provided.
        ValueError
            If the input types are mixed (e.g. Path and numpy.ndarray).
        ValueError
            If the data type is `custom` and no `read_source_func` is provided.
        ValueError
            If the data type is `array` and the input is not a numpy array.
        ValueError
            If the data type is `tiff` and the input is neither a Path nor a str.
        """
        super().__init__()

        # check input types coherence (no mixed types)
        inputs = [train_data, val_data, train_data_target, val_data_target]
        types_set = {type(i) for i in inputs}
        if len(types_set) > 2:  # None + expected type
            raise ValueError(
                f"Inputs for `train_data`, `val_data`, `train_data_target` and "
                f"`val_data_target` must be of the same type or None. Got "
                f"{types_set}."
            )

        # check that a read source function is provided for custom types
        if data_config.data_type == SupportedData.CUSTOM and read_source_func is None:
            raise ValueError(
                f"Data type {SupportedData.CUSTOM} is not allowed without "
                f"specifying a `read_source_func` and an `extension_filer`."
            )

        # check correct input type
        if (
            isinstance(train_data, np.ndarray)
            and data_config.data_type != SupportedData.ARRAY
        ):
            raise ValueError(
                f"Received a numpy array as input, but the data type was set to "
                f"{data_config.data_type}. Set the data type in the configuration "
                f"to {SupportedData.ARRAY} to train on numpy arrays."
            )

        # and that Path or str are passed, if tiff file type specified
        elif (isinstance(train_data, Path) or isinstance(train_data, str)) and (
            data_config.data_type != SupportedData.TIFF
            and data_config.data_type != SupportedData.CUSTOM
        ):
            raise ValueError(
                f"Received a path as input, but the data type was neither set to "
                f"{SupportedData.TIFF} nor {SupportedData.CUSTOM}. Set the data type "
                f"in the configuration to {SupportedData.TIFF} or "
                f"{SupportedData.CUSTOM} to train on files."
            )

        # configuration
        self.data_config: DataConfig = data_config
        self.data_type: str = data_config.data_type
        self.batch_size: int = data_config.batch_size
        self.use_in_memory: bool = use_in_memory

        # data: make data Path or np.ndarray, use type annotations for mypy
        self.train_data: Union[Path, NDArray] = (
            Path(train_data) if isinstance(train_data, str) else train_data
        )

        self.val_data: Union[Path, NDArray] = (
            Path(val_data) if isinstance(val_data, str) else val_data
        )

        self.train_data_target: Union[Path, NDArray] = (
            Path(train_data_target)
            if isinstance(train_data_target, str)
            else train_data_target
        )

        self.val_data_target: Union[Path, NDArray] = (
            Path(val_data_target)
            if isinstance(val_data_target, str)
            else val_data_target
        )

        # validation split
        self.val_percentage = val_percentage
        self.val_minimum_split = val_minimum_split

        # read source function corresponding to the requested type
        if data_config.data_type == SupportedData.CUSTOM.value:
            # mypy check
            assert read_source_func is not None

            self.read_source_func: Callable = read_source_func

        elif data_config.data_type != SupportedData.ARRAY:
            self.read_source_func = get_read_func(data_config.data_type)

        self.extension_filter: str = extension_filter

        # Pytorch dataloader parameters
        self.dataloader_params: dict[str, Any] = (
            data_config.dataloader_params if data_config.dataloader_params else {}
        )

    def prepare_data(self) -> None:
        """
        Hook used to prepare the data before calling `setup`.

        Here, we only need to examine the data if it was provided as a str or a Path.

        TODO: from lightning doc:
        prepare_data is called from the main process. It is not recommended to assign
        state here (e.g. self.x = y) since it is called on a single process and if you
        assign states here then they won't be available for other processes.

        https://lightning.ai/docs/pytorch/stable/data/datamodule.html
        """
        # if the data is a Path or a str
        if (
            not isinstance(self.train_data, np.ndarray)
            and not isinstance(self.val_data, np.ndarray)
            and not isinstance(self.train_data_target, np.ndarray)
            and not isinstance(self.val_data_target, np.ndarray)
        ):
            # list training files
            self.train_files = list_files(
                self.train_data, self.data_type, self.extension_filter
            )
            self.train_files_size = get_files_size(self.train_files)

            # list validation files
            if self.val_data is not None:
                self.val_files = list_files(
                    self.val_data, self.data_type, self.extension_filter
                )

            # same for target data
            if self.train_data_target is not None:
                self.train_target_files: list[Path] = list_files(
                    self.train_data_target, self.data_type, self.extension_filter
                )

                # verify that they match the training data
                validate_source_target_files(self.train_files, self.train_target_files)

            if self.val_data_target is not None:
                self.val_target_files = list_files(
                    self.val_data_target, self.data_type, self.extension_filter
                )

                # verify that they match the validation data
                validate_source_target_files(self.val_files, self.val_target_files)

    def setup(self, *args: Any, **kwargs: Any) -> None:
        """Hook called at the beginning of fit, validate, or predict.

        Parameters
        ----------
        *args : Any
            Unused.
        **kwargs : Any
            Unused.
        """
        # if numpy array
        if self.data_type == SupportedData.ARRAY:
            # mypy checks
            assert isinstance(self.train_data, np.ndarray)
            if self.train_data_target is not None:
                assert isinstance(self.train_data_target, np.ndarray)

            # train dataset
            self.train_dataset: DatasetType = InMemoryDataset(
                data_config=self.data_config,
                inputs=self.train_data,
                input_target=self.train_data_target,
            )

            # validation dataset
            if self.val_data is not None:
                # mypy checks
                assert isinstance(self.val_data, np.ndarray)
                if self.val_data_target is not None:
                    assert isinstance(self.val_data_target, np.ndarray)

                # create its own dataset
                self.val_dataset: DatasetType = InMemoryDataset(
                    data_config=self.data_config,
                    inputs=self.val_data,
                    input_target=self.val_data_target,
                )
            else:
                # extract validation from the training patches
                self.val_dataset = self.train_dataset.split_dataset(
                    percentage=self.val_percentage,
                    minimum_patches=self.val_minimum_split,
                )

        # else we read files
        else:
            # Heuristics, if the file size is smaller than 80% of the RAM,
            # we run the training in memory, otherwise we switch to iterable dataset
            # The switch is deactivated if use_in_memory is False
            if self.use_in_memory and self.train_files_size < get_ram_size() * 0.8:
                # train dataset
                self.train_dataset = InMemoryDataset(
                    data_config=self.data_config,
                    inputs=self.train_files,
                    input_target=(
                        self.train_target_files if self.train_data_target else None
                    ),
                    read_source_func=self.read_source_func,
                )

                # validation dataset
                if self.val_data is not None:
                    self.val_dataset = InMemoryDataset(
                        data_config=self.data_config,
                        inputs=self.val_files,
                        input_target=(
                            self.val_target_files if self.val_data_target else None
                        ),
                        read_source_func=self.read_source_func,
                    )
                else:
                    # split dataset
                    self.val_dataset = self.train_dataset.split_dataset(
                        percentage=self.val_percentage,
                        minimum_patches=self.val_minimum_split,
                    )

            # else if the data is too large, load file by file during training
            else:
                # create training dataset
                self.train_dataset = PathIterableDataset(
                    data_config=self.data_config,
                    src_files=self.train_files,
                    target_files=(
                        self.train_target_files if self.train_data_target else None
                    ),
                    read_source_func=self.read_source_func,
                )

                # create validation dataset
                if self.val_data is not None:
                    # create its own dataset
                    self.val_dataset = PathIterableDataset(
                        data_config=self.data_config,
                        src_files=self.val_files,
                        target_files=(
                            self.val_target_files if self.val_data_target else None
                        ),
                        read_source_func=self.read_source_func,
                    )
                elif len(self.train_files) <= self.val_minimum_split:
                    raise ValueError(
                        f"Not enough files to split a minimum of "
                        f"{self.val_minimum_split} files, got {len(self.train_files)} "
                        f"files."
                    )
                else:
                    # extract validation from the training patches
                    self.val_dataset = self.train_dataset.split_dataset(
                        percentage=self.val_percentage,
                        minimum_number=self.val_minimum_split,
                    )

    def get_data_statistics(self) -> tuple[list[float], list[float]]:
        """Return training data statistics.

        Returns
        -------
        tuple of list
            Means and standard deviations across channels of the training data.
        """
        return self.train_dataset.get_data_statistics()

    def train_dataloader(self) -> Any:
        """
        Create a dataloader for training.

        Returns
        -------
        Any
            Training dataloader.
        """
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, **self.dataloader_params
        )

    def val_dataloader(self) -> Any:
        """
        Create a dataloader for validation.

        Returns
        -------
        Any
            Validation dataloader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
        )


def create_train_datamodule(
    train_data: Union[str, Path, NDArray],
    data_type: Union[Literal["array", "tiff", "custom"], SupportedData],
    patch_size: list[int],
    axes: str,
    batch_size: int,
    val_data: Optional[Union[str, Path, NDArray]] = None,
    transforms: Optional[list[TRANSFORMS_UNION]] = None,
    train_target_data: Optional[Union[str, Path, NDArray]] = None,
    val_target_data: Optional[Union[str, Path, NDArray]] = None,
    read_source_func: Optional[Callable] = None,
    extension_filter: str = "",
    val_percentage: float = 0.1,
    val_minimum_patches: int = 5,
    dataloader_params: Optional[dict] = None,
    use_in_memory: bool = True,
    use_n2v2: bool = False,
    struct_n2v_axis: Literal["horizontal", "vertical", "none"] = "none",
    struct_n2v_span: int = 5,
) -> TrainDataModule:
    """Create a TrainDataModule.

    This function is used to explicitly pass the parameters usually contained in a
    `data_model` configuration to a TrainDataModule.

    Since the lightning datamodule has no access to the model, make sure that the
    parameters passed to the datamodule are consistent with the model's requirements and
    are coherent.

    The data module can be used with Path, str or numpy arrays. In the case of
    numpy arrays, it loads and computes all the patches in memory. For Path and str
    inputs, it calculates the total file size and estimate whether it can fit in
    memory. If it does not, it iterates through the files. This behaviour can be
    deactivated by setting `use_in_memory` to False, in which case it will
    always use the iterating dataset to train on a Path or str.

    To use array data, set `data_type` to `array` and pass a numpy array to
    `train_data`.

    In particular, N2V requires a specific transformation (N2V manipulates), which is
    not compatible with supervised training. The default transformations applied to the
    training patches are defined in `careamics.config.data_model`. To use different
    transformations, pass a list of transforms. See examples for more details.

    By default, CAREamics only supports types defined in
    `careamics.config.support.SupportedData`. To read custom data types, you can set
    `data_type` to `custom` and provide a function that returns a numpy array from a
    path. Additionally, pass a `fnmatch` and `Path.rglob` compatible expression (e.g.
    "*.jpeg") to filter the files extension using `extension_filter`.

    In the absence of validation data, the validation data is extracted from the
    training data. The percentage of the training data to use for validation, as well as
    the minimum number of patches to split from the training data for validation can be
    set using `val_percentage` and `val_minimum_patches`, respectively.

    In `dataloader_params`, you can pass any parameter accepted by PyTorch dataloaders,
    except for `batch_size`, which is set by the `batch_size` parameter.

    Finally, if you intend to use N2V family of algorithms, you can set `use_n2v2` to
    use N2V2, and set the `struct_n2v_axis` and `struct_n2v_span` parameters to define
    the axis and span of the structN2V mask. These parameters are without effect if
    a `train_target_data` or if `transforms` are provided.

    Parameters
    ----------
    train_data : pathlib.Path or str or numpy.ndarray
        Training data.
    data_type : {"array", "tiff", "custom"}
        Data type, see `SupportedData` for available options.
    patch_size : list of int
        Patch size, 2D or 3D patch size.
    axes : str
        Axes of the data, chosen amongst SCZYX.
    batch_size : int
        Batch size.
    val_data : pathlib.Path or str or numpy.ndarray, optional
        Validation data, by default None.
    transforms : list of Transforms, optional
        List of transforms to apply to training patches. If None, default transforms
        are applied.
    train_target_data : pathlib.Path or str or numpy.ndarray, optional
        Training target data, by default None.
    val_target_data : pathlib.Path or str or numpy.ndarray, optional
        Validation target data, by default None.
    read_source_func : Callable, optional
        Function to read the source data, used if `data_type` is `custom`, by
        default None.
    extension_filter : str, optional
        Filter for file extensions, used if `data_type` is `custom`, by default "".
    val_percentage : float, optional
        Percentage of the training data to use for validation if no validation data
        is given, by default 0.1.
    val_minimum_patches : int, optional
        Minimum number of patches to split from the training data for validation if
        no validation data is given, by default 5.
    dataloader_params : dict, optional
        Pytorch dataloader parameters, by default {}.
    use_in_memory : bool, optional
        Use in memory dataset if possible, by default True.
    use_n2v2 : bool, optional
        Use N2V2 transformation during training, by default False.
    struct_n2v_axis : {"horizontal", "vertical", "none"}, optional
        Axis for the structN2V mask, only applied if `struct_n2v_axis` is `none`, by
        default "none".
    struct_n2v_span : int, optional
        Span for the structN2V mask, by default 5.

    Returns
    -------
    TrainDataModule
        CAREamics training Lightning data module.

    Examples
    --------
    Create a TrainingDataModule with default transforms with a numpy array:
    >>> import numpy as np
    >>> from careamics.lightning import create_train_datamodule
    >>> my_array = np.arange(256).reshape(16, 16)
    >>> data_module = create_train_datamodule(
    ...     train_data=my_array,
    ...     data_type="array",
    ...     patch_size=(8, 8),
    ...     axes='YX',
    ...     batch_size=2,
    ... )

    For custom data types (those not supported by CAREamics), then one can pass a read
    function and a filter for the files extension:
    >>> import numpy as np
    >>> from careamics.lightning import create_train_datamodule
    >>>
    >>> def read_npy(path):
    ...     return np.load(path)
    >>>
    >>> data_module = create_train_datamodule(
    ...     train_data="path/to/data",
    ...     data_type="custom",
    ...     patch_size=(8, 8),
    ...     axes='YX',
    ...     batch_size=2,
    ...     read_source_func=read_npy,
    ...     extension_filter="*.npy",
    ... )

    If you want to use a different set of transformations, you can pass a list of
    transforms:
    >>> import numpy as np
    >>> from careamics.lightning import create_train_datamodule
    >>> from careamics.config.support import SupportedTransform
    >>> my_array = np.arange(256).reshape(16, 16)
    >>> my_transforms = [
    ...     {
    ...         "name": SupportedTransform.XY_FLIP.value,
    ...     }
    ... ]
    >>> data_module = create_train_datamodule(
    ...     train_data=my_array,
    ...     data_type="array",
    ...     patch_size=(8, 8),
    ...     axes='YX',
    ...     batch_size=2,
    ...     transforms=my_transforms,
    ... )
    """
    if dataloader_params is None:
        dataloader_params = {}

    data_dict: dict[str, Any] = {
        "mode": "train",
        "data_type": data_type,
        "patch_size": patch_size,
        "axes": axes,
        "batch_size": batch_size,
        "dataloader_params": dataloader_params,
    }

    # if transforms are passed (otherwise it will use the default ones)
    if transforms is not None:
        data_dict["transforms"] = transforms

    # validate configuration
    data_config = DataConfig(**data_dict)

    # N2V specific checks, N2V, structN2V, and transforms
    if data_config.has_n2v_manipulate():
        # there is not target, n2v2 and structN2V can be changed
        if train_target_data is None:
            data_config.set_N2V2(use_n2v2)
            data_config.set_structN2V_mask(struct_n2v_axis, struct_n2v_span)
        else:
            raise ValueError(
                "Cannot have both supervised training (target data) and "
                "N2V manipulation in the transforms. Pass a list of transforms "
                "that is compatible with your supervised training."
            )

    # sanity check on the dataloader parameters
    if "batch_size" in dataloader_params:
        # remove it
        del dataloader_params["batch_size"]

    return TrainDataModule(
        data_config=data_config,
        train_data=train_data,
        val_data=val_data,
        train_data_target=train_target_data,
        val_data_target=val_target_data,
        read_source_func=read_source_func,
        extension_filter=extension_filter,
        val_percentage=val_percentage,
        val_minimum_split=val_minimum_patches,
        use_in_memory=use_in_memory,
    )
