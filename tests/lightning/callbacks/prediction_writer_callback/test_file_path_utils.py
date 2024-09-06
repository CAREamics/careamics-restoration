from pathlib import Path

import numpy as np
import pytest
from tifffile import imwrite

from careamics.config import InferenceConfig
from careamics.config.tile_information import TileInformation
from careamics.dataset import IterablePredDataset, IterableTiledPredDataset
from careamics.lightning.callbacks.prediction_writer_callback.file_path_utils import (
    create_write_file_path,
    get_sample_file_path,
)


@pytest.mark.parametrize("axes", ["YX", "SYX"])
def test_get_sample_file_path_tiled(tmp_path, axes):
    """Test file name generation for tiled prediction dataset."""
    input_shape = (2, 16, 16) if "S" in axes else (16, 16)
    tile_size = (8, 8)
    tile_overlap = (4, 4)
    # create files
    src_files = [tmp_path / f"{i}.tiff" for i in range(2)]
    for file_path in src_files:
        arr = np.random.rand(*input_shape)
        imwrite(file_path, arr)

    pred_config = InferenceConfig(
        data_type="tiff",
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        axes=axes,
        image_means=[0],
        image_stds=[0],
    )
    ds = IterableTiledPredDataset(pred_config, src_files=src_files)

    for sample in ds:
        sample: tuple[np.ndarray, TileInformation]
        _, tile_info = sample
        file_path = get_sample_file_path(ds, sample_id=tile_info.sample_id)
        file_index = ds.current_file_index
        # if samples axis for each sample, the index is added to the filename.
        if "S" in axes:
            save_path = tmp_path / f"{file_index}_{tile_info.sample_id}.tiff"
        else:
            save_path = tmp_path / f"{file_index}.tiff"
        assert file_path == save_path


@pytest.mark.parametrize("axes", ["YX", "SYX"])
def test_get_sample_file_path_untiled(tmp_path, axes):
    """Test file name generation for untiled prediction dataset."""
    input_shape = (2, 16, 16) if "S" in axes else (16, 16)
    # create files
    src_files = [tmp_path / f"{i}.tiff" for i in range(2)]
    for file_path in src_files:
        arr = np.random.rand(*input_shape)
        imwrite(file_path, arr)

    pred_config = InferenceConfig(
        data_type="tiff",
        axes=axes,
        image_means=[0],
        image_stds=[0],
    )
    ds = IterablePredDataset(pred_config, src_files=src_files)

    for i, _ in enumerate(ds):
        file_path = get_sample_file_path(ds, sample_id=i)
        file_index = ds.current_file_index
        # if samples axis for each sample, the index is added to the filename.
        if "S" in axes:
            save_path = tmp_path / f"{file_index}_{i}.tiff"
        else:
            save_path = tmp_path / f"{file_index}.tiff"
        assert file_path == save_path


def test_get_sample_file_path_error():
    """Test error is raised if dataset iteration has not commenced."""
    src_files = [f"{i}.ext" for i in range(2)]
    pred_config = InferenceConfig(
        data_type="tiff",
        tile_size=(8, 8),
        tile_overlap=(4, 4),
        axes="YX",
        image_means=[0],
        image_stds=[0],
    )
    ds = IterablePredDataset(pred_config, src_files=src_files)
    with pytest.raises(ValueError):
        get_sample_file_path(ds, sample_id=0)


def test_create_write_file_path():
    dirpath = Path("output_directory")
    file_path = Path("input_directory/file_name.in_ext")
    write_extension = ".out_ext"

    write_file_path = create_write_file_path(
        dirpath=dirpath, file_path=file_path, write_extension=write_extension
    )
    assert write_file_path == Path("output_directory/file_name.out_ext")
