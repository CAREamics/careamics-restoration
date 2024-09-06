"""Functions to read tiff images."""

import logging
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import tifffile

from careamics.config.support import SupportedData
from careamics.utils.logging import get_logger

logger = get_logger(__name__)


def read_tiff(file_path: Path, *args: list, **kwargs: dict) -> np.ndarray:
    """
    Read a tiff file and return a numpy array.

    Parameters
    ----------
    file_path : Path
        Path to a file.
    *args : list
        Additional arguments.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    np.ndarray
        Resulting array.

    Raises
    ------
    ValueError
        If the file failed to open.
    OSError
        If the file failed to open.
    ValueError
        If the file is not a valid tiff.
    ValueError
        If the data dimensions are incorrect.
    ValueError
        If the axes length is incorrect.
    """
    if fnmatch(
        file_path.suffix, SupportedData.get_extension_pattern(SupportedData.TIFF)
    ):
        try:
            array = tifffile.imread(file_path)
        except (ValueError, OSError) as e:
            logging.exception(f"Exception in file {file_path}: {e}, skipping it.")
            raise e
    else:
        raise ValueError(f"File {file_path} is not a valid tiff.")

    return array
