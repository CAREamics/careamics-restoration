"""Module use to build BMZ model description."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from bioimageio.spec.model.v0_5 import (
    ArchitectureFromLibraryDescr,
    Author,
    AxisBase,
    AxisId,
    BatchAxis,
    ChannelAxis,
    EnvironmentFileDescr,
    FileDescr,
    FixedZeroMeanUnitVarianceAlongAxisKwargs,
    FixedZeroMeanUnitVarianceDescr,
    Identifier,
    InputTensorDescr,
    ModelDescr,
    OutputTensorDescr,
    PytorchStateDictWeightsDescr,
    SpaceInputAxis,
    SpaceOutputAxis,
    TensorId,
    Version,
    WeightsDescr,
)

from careamics.config import Configuration, DataConfig

from ._readme_factory import readme_factory


def _create_axes(
    array: np.ndarray,
    data_config: DataConfig,
    channel_names: Optional[List[str]] = None,
    is_input: bool = True,
) -> List[AxisBase]:
    """Create axes description.

    Array shape is expected to be SC(Z)YX.

    Parameters
    ----------
    array : np.ndarray
        Array.
    data_config : DataModel
        CAREamics data configuration.
    channel_names : Optional[List[str]], optional
        Channel names, by default None.
    is_input : bool, optional
        Whether the axes are input axes, by default True.

    Returns
    -------
    List[AxisBase]
        List of axes description.

    Raises
    ------
    ValueError
        If channel names are not provided when channel axis is present.
    """
    # axes have to be SC(Z)YX
    spatial_axes = data_config.axes.replace("S", "").replace("C", "")

    # batch is always present
    axes_model = [BatchAxis()]

    if "C" in data_config.axes:
        if channel_names is not None:
            axes_model.append(
                ChannelAxis(channel_names=[Identifier(name) for name in channel_names])
            )
        else:
            raise ValueError(
                f"Channel names must be provided if channel axis is present, axes: "
                f"{data_config.axes}."
            )
    else:
        # singleton channel
        axes_model.append(ChannelAxis(channel_names=[Identifier("channel")]))

    # spatial axes
    for ind, axes in enumerate(spatial_axes):
        if axes in ["X", "Y", "Z"]:
            if is_input:
                axes_model.append(
                    SpaceInputAxis(id=AxisId(axes.lower()), size=array.shape[2 + ind])
                )
            else:
                axes_model.append(
                    SpaceOutputAxis(id=AxisId(axes.lower()), size=array.shape[2 + ind])
                )

    return axes_model


def _create_inputs_ouputs(
    input_array: np.ndarray,
    output_array: np.ndarray,
    data_config: DataConfig,
    input_path: Union[Path, str],
    output_path: Union[Path, str],
    channel_names: Optional[List[str]] = None,
) -> Tuple[InputTensorDescr, OutputTensorDescr]:
    """Create input and output tensor description.

    Input and output paths must point to a `.npy` file.

    Parameters
    ----------
    input_array : np.ndarray
        Input array.
    output_array : np.ndarray
        Output array.
    data_config : DataModel
        CAREamics data configuration.
    input_path : Union[Path, str]
        Path to input .npy file.
    output_path : Union[Path, str]
        Path to output .npy file.
    channel_names : Optional[List[str]], optional
        Channel names, by default None.

    Returns
    -------
    Tuple[InputTensorDescr, OutputTensorDescr]
        Input and output tensor descriptions.
    """
    input_axes = _create_axes(input_array, data_config, channel_names)
    output_axes = _create_axes(output_array, data_config, channel_names, False)

    # mean and std
    assert data_config.image_means is not None, "Mean cannot be None."
    assert data_config.image_means is not None, "Std cannot be None."
    means = data_config.image_means
    stds = data_config.image_stds

    # and the mean and std required to invert the normalization
    # CAREamics denormalization: x = y * (std + eps) + mean
    # BMZ normalization : x = (y - mean') / (std' + eps)
    # to apply the BMZ normalization as a denormalization step, we need:
    eps = 1e-6
    inv_means = []
    inv_stds = []
    if means and stds:
        for mean, std in zip(means, stds):
            inv_means.append(-mean / (std + eps))
            inv_stds.append(1 / (std + eps) - eps)

        # create input/output descriptions
        input_descr = InputTensorDescr(
            id=TensorId("input"),
            axes=input_axes,
            test_tensor=FileDescr(source=input_path),
            preprocessing=[
                FixedZeroMeanUnitVarianceDescr(
                    kwargs=FixedZeroMeanUnitVarianceAlongAxisKwargs(
                        mean=means, std=stds, axis="channel"
                    )
                )
            ],
        )
        output_descr = OutputTensorDescr(
            id=TensorId("prediction"),
            axes=output_axes,
            test_tensor=FileDescr(source=output_path),
            postprocessing=[
                FixedZeroMeanUnitVarianceDescr(
                    kwargs=FixedZeroMeanUnitVarianceAlongAxisKwargs(  # invert norm
                        mean=inv_means, std=inv_stds, axis="channel"
                    )
                )
            ],
        )

        return input_descr, output_descr
    else:
        raise ValueError("Mean and std cannot be None.")


def create_model_description(
    config: Configuration,
    name: str,
    general_description: str,
    authors: List[Author],
    inputs: Union[Path, str],
    outputs: Union[Path, str],
    weights_path: Union[Path, str],
    torch_version: str,
    careamics_version: str,
    config_path: Union[Path, str],
    env_path: Union[Path, str],
    channel_names: Optional[List[str]] = None,
    data_description: Optional[str] = None,
) -> ModelDescr:
    """Create model description.

    Parameters
    ----------
    config : Configuration
        CAREamics configuration.
    name : str
        Name of the model.
    general_description : str
        General description of the model.
    authors : List[Author]
        Authors of the model.
    inputs : Union[Path, str]
        Path to input .npy file.
    outputs : Union[Path, str]
        Path to output .npy file.
    weights_path : Union[Path, str]
        Path to model weights.
    torch_version : str
        Pytorch version.
    careamics_version : str
        CAREamics version.
    config_path : Union[Path, str]
        Path to model configuration.
    env_path : Union[Path, str]
        Path to environment file.
    channel_names : Optional[List[str]], optional
        Channel names, by default None.
    data_description : Optional[str], optional
        Description of the data, by default None.

    Returns
    -------
    ModelDescr
        Model description.
    """
    # documentation
    doc = readme_factory(
        config,
        careamics_version=careamics_version,
        data_description=data_description,
    )

    # inputs, outputs
    input_descr, output_descr = _create_inputs_ouputs(
        input_array=np.load(inputs),
        output_array=np.load(outputs),
        data_config=config.data_config,
        input_path=inputs,
        output_path=outputs,
        channel_names=channel_names,
    )

    # weights description
    architecture_descr = ArchitectureFromLibraryDescr(
        import_from="careamics.models.unet",
        callable=f"{config.algorithm_config.model.architecture}",
        kwargs=config.algorithm_config.model.model_dump(),
    )

    weights_descr = WeightsDescr(
        pytorch_state_dict=PytorchStateDictWeightsDescr(
            source=weights_path,
            architecture=architecture_descr,
            pytorch_version=Version(torch_version),
            dependencies=EnvironmentFileDescr(source=env_path),
        ),
    )

    # overall model description
    model = ModelDescr(
        name=name,
        authors=authors,
        description=general_description,
        documentation=doc,
        inputs=[input_descr],
        outputs=[output_descr],
        tags=config.get_algorithm_keywords(),
        links=[
            "https://github.com/CAREamics/careamics",
            "https://careamics.github.io/latest/",
        ],
        license="BSD-3-Clause",
        version="0.1.0",
        weights=weights_descr,
        attachments=[FileDescr(source=config_path)],
        cite=config.get_algorithm_citations(),
        config={  # conversion from float32 to float64 creates small differences...
            "bioimageio": {
                "test_kwargs": {
                    "pytorch_state_dict": {
                        "decimals": 0,  # ...so we relax the constraints on the decimals
                    }
                }
            }
        },
    )

    return model


def extract_model_path(model_desc: ModelDescr) -> Tuple[Path, Path]:
    """Return the relative path to the weights and configuration files.

    Parameters
    ----------
    model_desc : ModelDescr
        Model description.

    Returns
    -------
    Tuple[Path, Path]
        Weights and configuration paths.
    """
    weights_path = model_desc.weights.pytorch_state_dict.source.path

    if len(model_desc.attachments) == 1:
        config_path = model_desc.attachments[0].source.path
    else:
        for file in model_desc.attachments:
            if file.source.path.suffix == ".yml":
                config_path = file.source.path
                break

        if config_path is None:
            raise ValueError("Configuration file not found.")

    return weights_path, config_path
