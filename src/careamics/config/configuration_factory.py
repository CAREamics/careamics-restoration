"""Convenience functions to create configurations for training and inference."""

from typing import Any, Dict, List, Literal, Optional

from .architectures import UNetModel
from .configuration_model import Configuration
from .data_model import DataConfig
from .fcn_algorithm_model import FCNAlgorithmConfig
from .support import (
    SupportedAlgorithm,
    SupportedArchitecture,
    SupportedLoss,
    SupportedPixelManipulation,
    SupportedTransform,
)
from .training_model import TrainingConfig


# TODO rename ?
def _create_supervised_configuration(
    algorithm_type: Literal["fcn"],
    algorithm: Literal["care", "n2n"],
    experiment_name: str,
    data_type: Literal["array", "tiff", "custom"],
    axes: str,
    patch_size: List[int],
    batch_size: int,
    num_epochs: int,
    use_augmentations: bool = True,
    independent_channels: bool = False,
    loss: Literal["mae", "mse"] = "mae",
    n_channels_in: int = 1,
    n_channels_out: int = 1,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    model_kwargs: Optional[dict] = None,
) -> Configuration:
    """
    Create a configuration for training CARE or Noise2Noise.

    Parameters
    ----------
    algorithm_type : Literal["fcn"]
        Type of the algorithm.
    algorithm : Literal["care", "n2n"]
        Algorithm to use.
    experiment_name : str
        Name of the experiment.
    data_type : Literal["array", "tiff", "custom"]
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : List[int]
        Size of the patches along the spatial dimensions (e.g. [64, 64]).
    batch_size : int
        Batch size.
    num_epochs : int
        Number of epochs.
    use_augmentations : bool, optional
        Whether to use augmentations, by default True.
    independent_channels : bool, optional
        Whether to train all channels independently, by default False.
    loss : Literal["mae", "mse"], optional
        Loss function to use, by default "mae".
    n_channels_in : int, optional
        Number of channels in, by default 1.
    n_channels_out : int, optional
        Number of channels out, by default 1.
    logger : Literal["wandb", "tensorboard", "none"], optional
        Logger to use, by default "none".
    model_kwargs : dict, optional
        UNetModel parameters, by default {}.

    Returns
    -------
    Configuration
        Configuration for training CARE or Noise2Noise.
    """
    # if there are channels, we need to specify their number
    if "C" in axes and n_channels_in == 1:
        raise ValueError(
            f"Number of channels in must be specified when using channels "
            f"(got {n_channels_in} channel)."
        )
    elif "C" not in axes and n_channels_in > 1:
        raise ValueError(
            f"C is not present in the axes, but number of channels is specified "
            f"(got {n_channels_in} channels)."
        )

    # model
    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs["conv_dims"] = 3 if "Z" in axes else 2
    model_kwargs["in_channels"] = n_channels_in
    model_kwargs["num_classes"] = n_channels_out
    model_kwargs["independent_channels"] = independent_channels

    unet_model = UNetModel(
        architecture=SupportedArchitecture.UNET.value,
        **model_kwargs,
    )

    # algorithm model
    algorithm = FCNAlgorithmConfig(
        algorithm_type=algorithm_type,
        algorithm=algorithm,
        loss=loss,
        model=unet_model,
    )

    # augmentations
    if use_augmentations:
        transforms: List[Dict[str, Any]] = [
            {
                "name": SupportedTransform.XY_FLIP.value,
            },
            {
                "name": SupportedTransform.XY_RANDOM_ROTATE90.value,
            },
        ]
    else:
        transforms = []

    # data model
    data = DataConfig(
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        transforms=transforms,
    )

    # training model
    training = TrainingConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        logger=None if logger == "none" else logger,
    )

    # create configuration
    configuration = Configuration(
        experiment_name=experiment_name,
        algorithm_config=algorithm,
        data_config=data,
        training_config=training,
    )

    return configuration


def create_care_configuration(
    experiment_name: str,
    data_type: Literal["array", "tiff", "custom"],
    axes: str,
    patch_size: List[int],
    batch_size: int,
    num_epochs: int,
    use_augmentations: bool = True,
    independent_channels: bool = False,
    loss: Literal["mae", "mse"] = "mae",
    n_channels_in: int = 1,
    n_channels_out: int = -1,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    model_kwargs: Optional[dict] = None,
) -> Configuration:
    """
    Create a configuration for training CARE.

    If "Z" is present in `axes`, then `path_size` must be a list of length 3, otherwise
    2.

    If "C" is present in `axes`, then you need to set `n_channels_in` to the number of
    channels. Likewise, if you set the number of channels, then "C" must be present in
    `axes`.

    To set the number of output channels, use the `n_channels_out` parameter. If it is
    not specified, it will be assumed to be equal to `n_channels_in`.

    By default, all channels are trained together. To train all channels independently,
    set `independent_channels` to True.

    By setting `use_augmentations` to False, the only transformation applied will be
    normalization.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    data_type : Literal["array", "tiff", "custom"]
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : List[int]
        Size of the patches along the spatial dimensions (e.g. [64, 64]).
    batch_size : int
        Batch size.
    num_epochs : int
        Number of epochs.
    use_augmentations : bool, optional
        Whether to use augmentations, by default True.
    independent_channels : bool, optional
        Whether to train all channels independently, by default False.
    loss : Literal["mae", "mse"], optional
        Loss function to use, by default "mae".
    n_channels_in : int, optional
        Number of channels in, by default 1.
    n_channels_out : int, optional
        Number of channels out, by default -1.
    logger : Literal["wandb", "tensorboard", "none"], optional
        Logger to use, by default "none".
    model_kwargs : dict, optional
        UNetModel parameters, by default {}.

    Returns
    -------
    Configuration
        Configuration for training CARE.
    """
    if n_channels_out == -1:
        n_channels_out = n_channels_in

    return _create_supervised_configuration(
        algorithm_type="fcn",
        algorithm="care",
        experiment_name=experiment_name,
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        use_augmentations=use_augmentations,
        independent_channels=independent_channels,
        loss=loss,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        logger=logger,
        model_kwargs=model_kwargs,
    )


def create_n2n_configuration(
    experiment_name: str,
    data_type: Literal["array", "tiff", "custom"],
    axes: str,
    patch_size: List[int],
    batch_size: int,
    num_epochs: int,
    use_augmentations: bool = True,
    independent_channels: bool = False,
    loss: Literal["mae", "mse"] = "mae",
    n_channels_in: int = 1,
    n_channels_out: int = -1,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    model_kwargs: Optional[dict] = None,
) -> Configuration:
    """
    Create a configuration for training Noise2Noise.

    If "Z" is present in `axes`, then `path_size` must be a list of length 3, otherwise
    2.

    If "C" is present in `axes`, then you need to set `n_channels_in` to the number of
    channels. Likewise, if you set the number of channels, then "C" must be present in
    `axes`.

    To set the number of output channels, use the `n_channels_out` parameter. If it is
    not specified, it will be assumed to be equal to `n_channels_in`.

    By default, all channels are trained together. To train all channels independently,
    set `independent_channels` to True.

    By setting `use_augmentations` to False, the only transformation applied will be
    normalization.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    data_type : Literal["array", "tiff", "custom"]
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : List[int]
        Size of the patches along the spatial dimensions (e.g. [64, 64]).
    batch_size : int
        Batch size.
    num_epochs : int
        Number of epochs.
    use_augmentations : bool, optional
        Whether to use augmentations, by default True.
    independent_channels : bool, optional
        Whether to train all channels independently, by default False.
    loss : Literal["mae", "mse"], optional
        Loss function to use, by default "mae".
    n_channels_in : int, optional
        Number of channels in, by default 1.
    n_channels_out : int, optional
        Number of channels out, by default -1.
    logger : Literal["wandb", "tensorboard", "none"], optional
        Logger to use, by default "none".
    model_kwargs : dict, optional
        UNetModel parameters, by default {}.

    Returns
    -------
    Configuration
        Configuration for training Noise2Noise.
    """
    if n_channels_out == -1:
        n_channels_out = n_channels_in

    return _create_supervised_configuration(
        algorithm_type="fcn",
        algorithm="n2n",
        experiment_name=experiment_name,
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        use_augmentations=use_augmentations,
        independent_channels=independent_channels,
        loss=loss,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        logger=logger,
        model_kwargs=model_kwargs,
    )


def create_n2v_configuration(
    experiment_name: str,
    data_type: Literal["array", "tiff", "custom"],
    axes: str,
    patch_size: List[int],
    batch_size: int,
    num_epochs: int,
    use_augmentations: bool = True,
    independent_channels: bool = True,
    use_n2v2: bool = False,
    n_channels: int = 1,
    roi_size: int = 11,
    masked_pixel_percentage: float = 0.2,
    struct_n2v_axis: Literal["horizontal", "vertical", "none"] = "none",
    struct_n2v_span: int = 5,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    model_kwargs: Optional[dict] = None,
) -> Configuration:
    """
    Create a configuration for training Noise2Void.

    N2V uses a UNet model to denoise images in a self-supervised manner. To use its
    variants structN2V and N2V2, set the `struct_n2v_axis` and `struct_n2v_span`
    (structN2V) parameters, or set `use_n2v2` to True (N2V2).

    N2V2 modifies the UNet architecture by adding blur pool layers and removes the skip
    connections, thus removing checkboard artefacts. StructN2V is used when vertical
    or horizontal correlations are present in the noise; it applies an additional mask
    to the manipulated pixel neighbors.

    If "Z" is present in `axes`, then `path_size` must be a list of length 3, otherwise
    2.

    If "C" is present in `axes`, then you need to set `n_channels` to the number of
    channels.

    By default, all channels are trained independently. To train all channels together,
    set `independent_channels` to False.

    By setting `use_augmentations` to False, the only transformations applied will be
    normalization and N2V manipulation.

    The `roi_size` parameter specifies the size of the area around each pixel that will
    be manipulated by N2V. The `masked_pixel_percentage` parameter specifies how many
    pixels per patch will be manipulated.

    The parameters of the UNet can be specified in the `model_kwargs` (passed as a
    parameter-value dictionary). Note that `use_n2v2` and 'n_channels' override the
    corresponding parameters passed in `model_kwargs`.

    If you pass "horizontal" or "vertical" to `struct_n2v_axis`, then structN2V mask
    will be applied to each manipulated pixel.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    data_type : Literal["array", "tiff", "custom"]
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : List[int]
        Size of the patches along the spatial dimensions (e.g. [64, 64]).
    batch_size : int
        Batch size.
    num_epochs : int
        Number of epochs.
    use_augmentations : bool, optional
        Whether to use augmentations, by default True.
    independent_channels : bool, optional
        Whether to train all channels together, by default True.
    use_n2v2 : bool, optional
        Whether to use N2V2, by default False.
    n_channels : int, optional
        Number of channels (in and out), by default 1.
    roi_size : int, optional
        N2V pixel manipulation area, by default 11.
    masked_pixel_percentage : float, optional
        Percentage of pixels masked in each patch, by default 0.2.
    struct_n2v_axis : Literal["horizontal", "vertical", "none"], optional
        Axis along which to apply structN2V mask, by default "none".
    struct_n2v_span : int, optional
        Span of the structN2V mask, by default 5.
    logger : Literal["wandb", "tensorboard", "none"], optional
        Logger to use, by default "none".
    model_kwargs : dict, optional
        UNetModel parameters, by default {}.

    Returns
    -------
    Configuration
        Configuration for training N2V.

    Examples
    --------
    Minimum example:
    >>> config = create_n2v_configuration(
    ...     experiment_name="n2v_experiment",
    ...     data_type="array",
    ...     axes="YX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100
    ... )

    To use N2V2, simply pass the `use_n2v2` parameter:
    >>> config = create_n2v_configuration(
    ...     experiment_name="n2v2_experiment",
    ...     data_type="tiff",
    ...     axes="YX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     use_n2v2=True
    ... )

    For structN2V, there are two parameters to set, `struct_n2v_axis` and
    `struct_n2v_span`:
    >>> config = create_n2v_configuration(
    ...     experiment_name="structn2v_experiment",
    ...     data_type="tiff",
    ...     axes="YX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     struct_n2v_axis="horizontal",
    ...     struct_n2v_span=7
    ... )

    If you are training multiple channels independently, then you need to specify the
    number of channels:
    >>> config = create_n2v_configuration(
    ...     experiment_name="n2v_experiment",
    ...     data_type="array",
    ...     axes="YXC",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     n_channels=3
    ... )

    If instead you want to train multiple channels together, you need to turn off the
    `independent_channels` parameter:
    >>> config = create_n2v_configuration(
    ...     experiment_name="n2v_experiment",
    ...     data_type="array",
    ...     axes="YXC",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     independent_channels=False,
    ...     n_channels=3
    ... )

    To turn off the augmentations, except normalization and N2V manipulation, use the
    relevant keyword argument:
    >>> config = create_n2v_configuration(
    ...     experiment_name="n2v_experiment",
    ...     data_type="array",
    ...     axes="YX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     use_augmentations=False
    ... )
    """
    # if there are channels, we need to specify their number
    if "C" in axes and n_channels == 1:
        raise ValueError(
            f"Number of channels must be specified when using channels "
            f"(got {n_channels} channel)."
        )
    elif "C" not in axes and n_channels > 1:
        raise ValueError(
            f"C is not present in the axes, but number of channels is specified "
            f"(got {n_channels} channel)."
        )

    # model
    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs["n2v2"] = use_n2v2
    model_kwargs["conv_dims"] = 3 if "Z" in axes else 2
    model_kwargs["in_channels"] = n_channels
    model_kwargs["num_classes"] = n_channels
    model_kwargs["independent_channels"] = independent_channels

    unet_model = UNetModel(
        architecture=SupportedArchitecture.UNET.value,
        **model_kwargs,
    )

    # algorithm model
    algorithm = FCNAlgorithmConfig(
        algorithm_type="fcn",
        algorithm=SupportedAlgorithm.N2V.value,
        loss=SupportedLoss.N2V.value,
        model=unet_model,
    )

    # augmentations
    if use_augmentations:
        transforms: List[Dict[str, Any]] = [
            {
                "name": SupportedTransform.XY_FLIP.value,
            },
            {
                "name": SupportedTransform.XY_RANDOM_ROTATE90.value,
            },
        ]
    else:
        transforms = []

    # n2v2 and structn2v
    nv2_transform = {
        "name": SupportedTransform.N2V_MANIPULATE.value,
        "strategy": (
            SupportedPixelManipulation.MEDIAN.value
            if use_n2v2
            else SupportedPixelManipulation.UNIFORM.value
        ),
        "roi_size": roi_size,
        "masked_pixel_percentage": masked_pixel_percentage,
        "struct_mask_axis": struct_n2v_axis,
        "struct_mask_span": struct_n2v_span,
    }
    transforms.append(nv2_transform)

    # data model
    data = DataConfig(
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        transforms=transforms,
    )

    # training model
    training = TrainingConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        logger=None if logger == "none" else logger,
    )

    # create configuration
    configuration = Configuration(
        experiment_name=experiment_name,
        algorithm_config=algorithm,
        data_config=data,
        training_config=training,
    )

    return configuration
