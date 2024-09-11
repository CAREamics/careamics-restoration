import json
from pathlib import Path
from typing import Union

import numpy as np
import pytest
import torch
from pydantic import BaseModel, ConfigDict

from careamics.config.likelihood_model import Tensor
from careamics.config.nm_model import Array


class MyArray(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    arr: Array


class MyTensor(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    arr: Tensor


@pytest.mark.parametrize("arr", [np.array([1, 2]), torch.tensor([1, 2])])
def test_serialize_array(arr: Union[np.ndarray, torch.Tensor]):
    """Test array_to_json function."""
    arr_model = MyArray(arr=arr)
    assert arr_model.model_dump() == {"arr": "[1, 2]"}


@pytest.mark.parametrize("arr", [np.array([1, 2]), torch.tensor([1, 2])])
def test_serialize_tensor(arr: Union[np.ndarray, torch.Tensor]):
    """Test array_to_json function."""
    arr_model = MyTensor(arr=arr)
    assert arr_model.model_dump() == {"arr": "[1, 2]"}


def test_deserialize_array(tmp_path: Path):
    """Test list_to_numpy function."""
    arr_model = MyArray(arr=np.array([1, 2]))
    # save to JSON
    with open(tmp_path / "array_config.json", "w") as f:
        f.write(arr_model.model_dump_json())
    # load from JSON
    with open(tmp_path / "array_config.json") as f:
        config = json.load(f)
    new_arr_model = MyArray(**config)
    assert np.array_equal(new_arr_model.arr, np.array([1, 2]))


def test_deserialize_tensor(tmp_path: Path):
    """Test list_to_tensor function."""
    arr_model = MyTensor(arr=torch.tensor([1, 2]))
    # save to JSON
    with open(tmp_path / "tensor_config.json", "w") as f:
        f.write(arr_model.model_dump_json())
    # load from JSON
    with open(tmp_path / "tensor_config.json") as f:
        config = json.load(f)
    new_arr_model = MyTensor(**config)
    assert torch.equal(new_arr_model.arr, torch.tensor([1, 2]))
