"""Function to export to the BioImage Model Zoo format."""

import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pkg_resources
from bioimageio.core import load_description, test_model
from bioimageio.spec import ValidationSummary, save_bioimageio_package
from torch import __version__, load, save

from careamics.config import Configuration, load_configuration, save_configuration
from careamics.config.support import SupportedArchitecture
from careamics.lightning.lightning_module import FCNModule, VAEModule

from .bioimage import (
    create_env_text,
    create_model_description,
    extract_model_path,
    get_unzip_path,
)


def _export_state_dict(
    model: Union[FCNModule, VAEModule], path: Union[Path, str]
) -> Path:
    """
    Export the model state dictionary to a file.

    Parameters
    ----------
    model : CAREamicsKiln
        CAREamics model to export.
    path : Union[Path, str]
        Path to the file where to save the model state dictionary.

    Returns
    -------
    Path
        Path to the saved model state dictionary.
    """
    path = Path(path)

    # make sure it has the correct suffix
    if path.suffix not in ".pth":
        path = path.with_suffix(".pth")

    # save model state dictionary
    # we save through the torch model itself to avoid the initial "model." in the
    # layers naming, which is incompatible with the way the BMZ load torch state dicts
    save(model.model.state_dict(), path)

    return path


def _load_state_dict(
    model: Union[FCNModule, VAEModule], path: Union[Path, str]
) -> None:
    """
    Load a model from a state dictionary.

    Parameters
    ----------
    model : CAREamicsKiln
        CAREamics model to be updated with the weights.
    path : Union[Path, str]
        Path to the model state dictionary.
    """
    path = Path(path)

    # load model state dictionary
    # same as in _export_state_dict, we load through the torch model to be compatible
    # witht bioimageio.core expectations for a torch state dict
    state_dict = load(path)
    model.model.load_state_dict(state_dict)


# TODO break down in subfunctions
def export_to_bmz(
    model: Union[FCNModule, VAEModule],
    config: Configuration,
    path_to_archive: Union[Path, str],
    model_name: str,
    general_description: str,
    authors: List[dict],
    input_array: np.ndarray,
    output_array: np.ndarray,
    channel_names: Optional[List[str]] = None,
    data_description: Optional[str] = None,
) -> None:
    """Export the model to BioImage Model Zoo format.

    Arrays are expected to be SC(Z)YX with singleton dimensions allowed for S and C.

    `model_name` should consist of letters, numbers, dashes, underscores and parentheses
    only.

    Parameters
    ----------
    model : CAREamicsModule
        CAREamics model to export.
    config : Configuration
        Model configuration.
    path_to_archive : Union[Path, str]
        Path to the output file.
    model_name : str
        Model name.
    general_description : str
        General description of the model.
    authors : List[dict]
        Authors of the model.
    input_array : np.ndarray
        Input array, should not have been normalized.
    output_array : np.ndarray
        Output array, should have been denormalized.
    channel_names : Optional[List[str]], optional
        Channel names, by default None.
    data_description : Optional[str], optional
        Description of the data, by default None.

    Raises
    ------
    ValueError
        If the model is a Custom model.
    """
    path_to_archive = Path(path_to_archive)

    # method is not compatible with Custom models
    if config.algorithm_config.model.architecture == SupportedArchitecture.CUSTOM:
        raise ValueError(
            "Exporting Custom models to BioImage Model Zoo format is not supported."
        )

    if path_to_archive.suffix != ".zip":
        raise ValueError(
            f"Path to archive must point to a zip file, got {path_to_archive}."
        )

    if not path_to_archive.parent.exists():
        path_to_archive.parent.mkdir(parents=True, exist_ok=True)

    # versions
    pytorch_version = __version__
    careamics_version = pkg_resources.get_distribution("careamics").version

    # save files in temporary folder
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_path = Path(tmpdirname)

        # create environment file
        # TODO move in bioimage module
        env_path = temp_path / "environment.yml"
        env_path.write_text(create_env_text(pytorch_version))

        # export input and ouputs
        inputs = temp_path / "inputs.npy"
        np.save(inputs, input_array)
        outputs = temp_path / "outputs.npy"
        np.save(outputs, output_array)

        # export configuration
        config_path = save_configuration(config, temp_path)

        # export model state dictionary
        weight_path = _export_state_dict(model, temp_path / "weights.pth")

        # create model description
        model_description = create_model_description(
            config=config,
            name=model_name,
            general_description=general_description,
            authors=authors,
            inputs=inputs,
            outputs=outputs,
            weights_path=weight_path,
            torch_version=pytorch_version,
            careamics_version=careamics_version,
            config_path=config_path,
            env_path=env_path,
            channel_names=channel_names,
            data_description=data_description,
        )

        # test model description
        summary: ValidationSummary = test_model(model_description, decimal=1)
        if summary.status == "failed":
            raise ValueError(f"Model description test failed: {summary}")

        # save bmz model
        save_bioimageio_package(model_description, output_path=path_to_archive)


def load_from_bmz(
    path: Union[Path, str]
) -> Tuple[Union[FCNModule, VAEModule], Configuration]:
    """Load a model from a BioImage Model Zoo archive.

    Parameters
    ----------
    path : Union[Path, str]
        Path to the BioImage Model Zoo archive.

    Returns
    -------
    Tuple[CAREamicsKiln, Configuration]
        CAREamics model and configuration.

    Raises
    ------
    ValueError
        If the path is not a zip file.
    """
    path = Path(path)

    if path.suffix != ".zip":
        raise ValueError(f"Path must be a bioimage.io zip file, got {path}.")

    # load description, this creates an unzipped folder next to the archive
    model_desc = load_description(path)

    # extract relative paths
    weights_path, config_path = extract_model_path(model_desc)

    # create folder path and absolute paths
    unzip_path = get_unzip_path(path)
    weights_path = unzip_path / weights_path
    config_path = unzip_path / config_path

    # load configuration
    config = load_configuration(config_path)

    # create careamics lightning module
    if config.algorithm_config.model.architecture == SupportedArchitecture.UNET:
        model = FCNModule(algorithm_config=config.algorithm_config)
    elif config.algorithm_config.model.architecture == SupportedArchitecture.LVAE:
        model = VAEModule(algorithm_config=config.algorithm_config)
    else:
        raise ValueError(
            f"Unsupported architecture {config.algorithm_config.model.architecture}"
        )  # TODO ugly ?

    # load model state dictionary
    _load_state_dict(model, weights_path)

    return model, config
