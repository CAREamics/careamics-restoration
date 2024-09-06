"""Utility functions to load pretrained models."""

from pathlib import Path
from typing import Tuple, Union

import torch

from careamics.config import Configuration
from careamics.lightning.lightning_module import FCNModule, VAEModule
from careamics.model_io.bmz_io import load_from_bmz
from careamics.utils import check_path_exists


def load_pretrained(
    path: Union[Path, str]
) -> Tuple[Union[FCNModule, VAEModule], Configuration]:
    """
    Load a pretrained model from a checkpoint or a BioImage Model Zoo model.

    Expected formats are .ckpt or .zip files.

    Parameters
    ----------
    path : Union[Path, str]
        Path to the pretrained model.

    Returns
    -------
    Tuple[CAREamicsKiln, Configuration]
        Tuple of CAREamics model and its configuration.

    Raises
    ------
    ValueError
        If the model format is not supported.
    """
    path = check_path_exists(path)

    if path.suffix == ".ckpt":
        return _load_checkpoint(path)
    elif path.suffix == ".zip":
        return load_from_bmz(path)
    else:
        raise ValueError(
            f"Invalid model format. Expected .ckpt or .zip, got {path.suffix}."
        )


def _load_checkpoint(
    path: Union[Path, str]
) -> Tuple[Union[FCNModule, VAEModule], Configuration]:
    """
    Load a model from a checkpoint and return both model and configuration.

    Parameters
    ----------
    path : Union[Path, str]
        Path to the checkpoint.

    Returns
    -------
    Tuple[CAREamicsKiln, Configuration]
        Tuple of CAREamics model and its configuration.

    Raises
    ------
    ValueError
        If the checkpoint file does not contain hyper parameters (configuration).
    """
    # load checkpoint
    # here we might run into issues between devices
    # see https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint: dict = torch.load(path, map_location=device)

    # attempt to load configuration
    try:
        cfg_dict = checkpoint["hyper_parameters"]
    except KeyError as e:
        raise ValueError(
            f"Invalid checkpoint file. No `hyper_parameters` found in the "
            f"checkpoint: {checkpoint.keys()}"
        ) from e

    if cfg_dict["algorithm_config"]["model"]["architecture"] == "UNet":
        model = FCNModule.load_from_checkpoint(path)
    elif cfg_dict["algorithm_config"]["model"]["architecture"] == "LVAE":
        model = VAEModule.load_from_checkpoint(path)
    else:
        raise ValueError(
            "Invalid model architecture: "
            f"{cfg_dict['algorithm_config']['model']['architecture']}"
        )

    return model, Configuration(**cfg_dict)
