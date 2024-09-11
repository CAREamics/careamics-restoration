from pathlib import Path

import numpy as np
import pytest

from careamics.lvae_training.dataset.lc_dataset import LCMultiChDloader
from careamics.lvae_training.dataset.lc_dataset_config import LCVaeDatasetConfig
from careamics.lvae_training.dataset.vae_data_config import (
    DataSplitType,
    DataType,
    VaeDatasetConfig,
)
from careamics.lvae_training.dataset.vae_dataset import MultiChDloader

pytestmark = pytest.mark.lvae


@pytest.fixture
def dummy_data_path_biorc(tmp_path: Path) -> str:
    import SimpleITK as sitk

    (tmp_path / "ER").mkdir(exist_ok=True)
    (tmp_path / "Microtubules").mkdir(exist_ok=True)

    max_val = 65535

    er_data = np.random.rand(68, 1004, 1004)
    er_data = er_data * max_val
    er_data = er_data.astype(np.uint16)
    er_image = sitk.GetImageFromArray(er_data)

    microtubules_data = np.random.rand(55, 1004, 1004)
    microtubules_data = microtubules_data * max_val
    microtubules_data = microtubules_data.astype(np.uint16)
    microtubules_image = sitk.GetImageFromArray(microtubules_data)

    sitk.WriteImage(er_image, tmp_path / "ER/GT_all.mrc")
    sitk.WriteImage(microtubules_image, tmp_path / "Microtubules/GT_all.mrc")

    return str(tmp_path)


@pytest.fixture
def default_config() -> VaeDatasetConfig:
    return VaeDatasetConfig(
        ch1_fname="ER/GT_all.mrc",
        ch2_fname="Microtubules/GT_all.mrc",
        # TODO: something breaks when set to ALL
        datasplit_type=DataSplitType.Train,
        data_type=DataType.BioSR_MRC,
        enable_gaussian_noise=False,
        image_size=128,
        input_has_dependant_noise=True,
        multiscale_lowres_count=None,
        num_channels=2,
        enable_random_cropping=False,
        enable_rotation_aug=False,
    )


@pytest.mark.skip(reason="SimpleITK is not in the list of dependencies now")
def test_create_biosr_vae_dataset(default_config, dummy_data_path_biorc):
    dataset = MultiChDloader(
        default_config,
        dummy_data_path_biorc,
        val_fraction=0.1,
        test_fraction=0.1,
    )

    max_val = dataset.get_max_val()
    assert max_val is not None, max_val > 0

    mean_val, std_val = dataset.compute_mean_std()
    dataset.set_mean_std(mean_val, std_val)

    sample = dataset[0]
    assert len(sample) == 2

    inputs, targets = sample
    assert inputs.shape == (1, 128, 128)
    assert len(targets) == 2

    for channel in targets:
        assert channel.shape == (128, 128)

    # input is normalized
    assert inputs.mean() < 1
    assert inputs.std() < 1

    # TODO: check the outputs sum
    # output is not normalized
    assert targets[0].mean() > 1
    assert targets[0].std() > 1


@pytest.mark.skip(reason="SimpleITK is not in the list of dependencies now")
@pytest.mark.parametrize("num_scales", [1, 2, 3])
def test_create_biosr_lc_dataset(
    default_config, dummy_data_path_biorc, num_scales: int
):
    lc_config = LCVaeDatasetConfig(**default_config.model_dump(exclude_none=True))
    lc_config.num_scales = num_scales
    lc_config.multiscale_lowres_count = num_scales
    lc_config.overlapping_padding_kwargs = lc_config.padding_kwargs

    dataset = LCMultiChDloader(
        lc_config, dummy_data_path_biorc, val_fraction=0.1, test_fraction=0.1
    )

    max_val = dataset.get_max_val()
    assert max_val is not None, max_val > 0

    mean_val, std_val = dataset.compute_mean_std()

    dataset.set_mean_std(mean_val, std_val)

    sample = dataset[0]
    assert len(sample) == 2

    inputs, targets = sample
    assert inputs.shape == (num_scales, 128, 128)
    assert len(targets) == 2

    for channel in targets:
        assert channel.shape == (128, 128)

    # input is normalized
    assert inputs.mean() < 1
    assert inputs.std() < 1

    # output is not normalized
    assert targets[0].mean() > 1
    assert targets[0].std() > 1
