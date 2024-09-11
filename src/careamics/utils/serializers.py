"""A script for serializers in the careamics package."""

import ast
import json
from typing import Union

import numpy as np
import torch


def _array_to_json(arr: Union[np.ndarray, torch.Tensor]) -> str:
    """Convert an array to a list and then to a JSON string.

    Parameters
    ----------
    arr : Union[np.ndarray, torch.Tensor]
        Array to be serialized.

    Returns
    -------
    str
        JSON string representing the array.
    """
    return json.dumps(arr.tolist())


def _to_numpy(lst: Union[str, list]) -> np.ndarray:
    """Deserialize a list or string representing a list into `np.ndarray`.

    Parameters
    ----------
    lst : list
        List or string representing a list with the array content to be deserialized.

    Returns
    -------
    np.ndarray
        The deserialized array.
    """
    if isinstance(lst, str):
        lst = ast.literal_eval(lst)
    return np.asarray(lst)


def _to_torch(lst: Union[str, list]) -> torch.Tensor:
    """Deserialize list or string representing a list into `torch.Tensor`.

    Parameters
    ----------
    lst : Union[str, list]
        List or string representing a list swith the array content to be deserialized.

    Returns
    -------
    torch.Tensor
        The deserialized tensor.
    """
    if isinstance(lst, str):
        lst = ast.literal_eval(lst)
    return torch.tensor(lst)
