import numpy as np
import pytest
import tifffile

from careamics.config import InferenceConfig
from careamics.dataset import IterablePredDataset


@pytest.mark.parametrize(
    "n_files, shape, axes, expected_shape",
    [
        (1, (16, 16), "YX", (1, 16, 16)),
        (1, (3, 16, 16), "CYX", (3, 16, 16)),
        (1, (8, 16, 16), "ZYX", (1, 8, 16, 16)),
        (1, (3, 8, 16, 16), "CZYX", (3, 8, 16, 16)),
        (1, (4, 16, 16), "SYX", (1, 16, 16)),
        (1, (4, 3, 16, 16), "SCYX", (3, 16, 16)),
        (1, (4, 3, 8, 16, 16), "SCZYX", (3, 8, 16, 16)),
        (3, (16, 16), "YX", (1, 16, 16)),
        (3, (3, 16, 16), "CYX", (3, 16, 16)),
        (3, (8, 16, 16), "ZYX", (1, 8, 16, 16)),
        (3, (3, 8, 16, 16), "CZYX", (3, 8, 16, 16)),
        (3, (4, 16, 16), "SYX", (1, 16, 16)),
        (3, (4, 3, 16, 16), "SCYX", (3, 16, 16)),
        (3, (4, 3, 8, 16, 16), "SCZYX", (3, 8, 16, 16)),
    ],
)
def test_correct_normalized_outputs(tmp_path, n_files, shape, axes, expected_shape):
    """Test that the dataset returns normalized images with singleton
    sample dimension."""
    rng = np.random.default_rng(42)

    # check expected length
    if "S" in axes:
        # find index of S and check shape
        idx = axes.index("S")
        n_patches = shape[idx]
    else:
        n_patches = 1

    # check number of channels
    if "C" in axes:
        # get index
        idx = axes.index("C")
        n_channels = shape[idx]
    else:
        n_channels = 1

    # create array
    new_shape = (n_files,) + shape
    array = 255 * rng.random(new_shape)

    # create config
    config = InferenceConfig(
        data_type="tiff",
        axes=axes,
        image_means=[np.mean(array)] * n_channels,
        image_stds=[np.std(array)] * n_channels,
    )

    files = []
    for i in range(n_files):
        file = tmp_path / f"file_{i}.tif"
        tifffile.imwrite(file, array[i])
        files.append(file)

    # create dataset
    dataset = IterablePredDataset(config, files)

    # get all images
    dataset = list(dataset)

    # check length
    assert len(dataset) == n_files * n_patches

    # check that the dataset returns normalized images
    for i in range(len(dataset)):
        img = dataset[i]

        # check that it has the correct shape
        assert img.shape == expected_shape

        # check that the image is normalized
        assert np.isclose(np.mean(img), 0, atol=0.1)
        assert np.isclose(np.std(img), 1, atol=0.1)

        # check that they are independent slices
        for j in range(i + 1, len(dataset)):
            img2 = dataset[j]
            assert not np.allclose(img, img2)


def test_file_index_update(tmp_path):
    """Test interal dataset attribute `current_file_index` updates during iteration."""
    axes = "YX"
    input_shape = (16, 16)
    # create files
    src_files = [tmp_path / f"{i}.tiff" for i in range(2)]
    for file_path in src_files:
        arr = np.random.rand(*input_shape)
        tifffile.imwrite(file_path, arr)

    pred_config = InferenceConfig(
        data_type="tiff",
        axes=axes,
        image_means=[0],
        image_stds=[0],
    )
    ds = IterablePredDataset(pred_config, src_files=src_files)

    for i, _ in enumerate(ds):
        assert ds.current_file_index == i
