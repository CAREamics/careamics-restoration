import os

import numpy as np
import pytest
import tifffile

from careamics.config import DataConfig
from careamics.config.support import SupportedData
from careamics.dataset import PathIterableDataset
from careamics.file_io.read import read_tiff


@pytest.mark.parametrize(
    "shape, axes, patch_size",
    [
        ((32, 32), "YX", (8, 8)),
        ((2, 32, 32), "CYX", (8, 8)),
        ((32, 32, 32), "ZYX", (8, 8, 8)),
    ],
)
def test_number_of_files(tmp_path, ordered_array, shape, axes, patch_size):
    """Test number of files in PathIterableDataset."""
    # create array
    array_size = 32
    n_files = 3
    factor = 3 if axes == "ZYX" else 2
    array = ordered_array(shape)

    # save three files
    files = []
    for i in range(n_files):
        file = tmp_path / f"array{i}.tif"
        tifffile.imwrite(file, array)
        files.append(file)

    # create config
    config_dict = {
        "data_type": SupportedData.TIFF.value,
        "patch_size": patch_size,
        "axes": axes,
    }
    config = DataConfig(**config_dict)

    # create dataset
    dataset = PathIterableDataset(
        data_config=config, src_files=files, read_source_func=read_tiff
    )

    # check number of files
    assert dataset.data_files == files

    # iterate over dataset
    patches = list(dataset)
    assert len(patches) == n_files * (array_size / patch_size[0]) ** factor


def test_read_function(tmp_path, ordered_array):
    """Test reading files in PathIterableDataset using a custom read function."""

    # read function for .npy files
    def read_npy(file_path, *args, **kwargs):
        return np.load(file_path)

    array_size = 32
    patch_size = 8
    n_files = 3
    patch_sizes = [patch_size] * 2

    # create array
    array: np.ndarray = ordered_array((n_files, array_size, array_size))

    # save each plane in a single .npy file
    files = []
    for i in range(array.shape[0]):
        file_path = tmp_path / f"array{i}.npy"
        np.save(file_path, array[i])
        files.append(file_path)

    # create config
    config_dict = {
        "data_type": SupportedData.CUSTOM.value,
        "patch_size": patch_sizes,
        "axes": "YX",
    }
    config = DataConfig(**config_dict)

    # create dataset
    dataset = PathIterableDataset(
        data_config=config,
        src_files=files,
        read_source_func=read_npy,
    )
    assert dataset.data_files == files

    # iterate over dataset
    patches = list(dataset)
    assert len(patches) == n_files * (array_size / patch_size) ** 2


@pytest.mark.parametrize("percentage", [0.1, 0.6])
def test_extracting_val_files(tmp_path, ordered_array, percentage):
    """Test extracting a validation set patches from PathIterableDataset."""
    # create array
    array = ordered_array((20, 20))

    # save array to 25 files
    files = []
    for i in range(25):
        file_path = tmp_path / f"array{i}.tif"
        tifffile.imwrite(file_path, array)
        files.append(file_path)

    # create config
    config_dict = {
        "data_type": SupportedData.TIFF.value,
        "patch_size": [8, 8],
        "axes": "YX",
    }
    config = DataConfig(**config_dict)

    # create dataset
    dataset = PathIterableDataset(
        data_config=config, src_files=files, read_source_func=read_tiff
    )

    # compute number of patches
    total_n_files = dataset.get_number_of_files()
    minimum_files = 5
    n_files = max(round(percentage * total_n_files), minimum_files)

    # extract datset
    valset = dataset.split_dataset(percentage, minimum_files)

    # check number of patches
    assert valset.get_number_of_files() == n_files
    assert dataset.get_number_of_files() == total_n_files - n_files

    # check that none of the validation files are in the original dataset
    assert set(valset.data_files).isdisjoint(set(dataset.data_files))


@pytest.mark.parametrize(
    "shape, axes, patch_size",
    [
        ((32, 32), "YX", (8, 8)),
        ((2, 32, 32), "CYX", (8, 8)),
        ((32, 32, 32), "ZYX", (8, 8, 8)),
    ],
)
def test_compute_mean_std_transform_welford(tmp_path, shape, axes, patch_size):
    """Test that mean and std are computed and correctly added to the configuration
    and transform."""
    n_files = 10
    files = []
    array = np.random.randint(0, np.iinfo(np.uint16).max, (n_files, *shape))

    for i in range(n_files):
        file = tmp_path / f"array{i}.tif"
        tifffile.imwrite(file, array[i])
        files.append(file)

    array = array[:, np.newaxis, ...] if "C" not in axes else array

    # create config
    config_dict = {
        "data_type": SupportedData.TIFF.value,
        "patch_size": patch_size,
        "axes": axes,
    }
    config = DataConfig(**config_dict)

    # create dataset
    dataset = PathIterableDataset(
        data_config=config, src_files=files, read_source_func=read_tiff
    )

    # define axes for mean and std computation
    stats_axes = tuple(np.delete(np.arange(array.ndim), 1))

    assert np.allclose(array.mean(axis=stats_axes), dataset.data_config.image_means)
    assert np.allclose(array.std(axis=stats_axes), dataset.data_config.image_stds)


@pytest.mark.parametrize(
    "shape, axes, patch_size",
    [
        ((32, 32), "YX", (8, 8)),
        ((2, 32, 32), "CYX", (8, 8)),
        ((32, 32, 32), "ZYX", (8, 8, 8)),
    ],
)
def test_compute_mean_std_transform_welford_with_targets(
    tmp_path, shape, axes, patch_size
):
    """Test that mean and std are computed and correctly added to the configuration
    and transform."""
    n_files = 10
    files = []
    target_files = []
    array = np.random.randint(0, np.iinfo(np.uint16).max, (n_files, *shape))
    target_array = np.random.randint(0, np.iinfo(np.uint16).max, (n_files, *shape))

    for i in range(n_files):
        file = tmp_path / "images" / f"array{i}.tif"
        target_file = tmp_path / "targets" / f"array{i}.tif"
        os.makedirs(file.parent, exist_ok=True)
        os.makedirs(target_file.parent, exist_ok=True)
        tifffile.imwrite(file, array[i])
        tifffile.imwrite(target_file, target_array[i])
        files.append(file)
        target_files.append(target_file)

    array = array[:, np.newaxis, ...] if "C" not in axes else array
    target_array = target_array[:, np.newaxis, ...] if "C" not in axes else target_array

    # create config
    config_dict = {
        "data_type": SupportedData.TIFF.value,
        "patch_size": patch_size,
        "axes": axes,
    }
    config = DataConfig(**config_dict)

    # create dataset
    dataset = PathIterableDataset(
        data_config=config,
        src_files=files,
        target_files=target_files,
        read_source_func=read_tiff,
    )

    # define axes for mean and std computation
    stats_axes = tuple(np.delete(np.arange(array.ndim), 1))

    assert np.allclose(
        target_array.mean(axis=stats_axes), dataset.data_config.target_means
    )
    assert np.allclose(
        target_array.std(axis=stats_axes), dataset.data_config.target_stds
    )
