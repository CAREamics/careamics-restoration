from pathlib import Path

import numpy as np
import pytest
import torch

from careamics.config import GaussianMixtureNMConfig, MultiChannelNMConfig
from careamics.models.lvae.noise_models import (
    GaussianMixtureNoiseModel,
    MultiChannelNoiseModel,
    noise_model_factory,
)


def test_factory_no_noise_model():
    noise_model = noise_model_factory(None)
    assert noise_model is None


def test_instantiate_noise_model(tmp_path: Path, create_dummy_noise_model) -> None:
    # Create a dummy noise model
    np.savez(tmp_path / "dummy_noise_model.npz", **create_dummy_noise_model)

    # Instantiate the noise model
    gmm = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        path=tmp_path / "dummy_noise_model.npz",
        # all other params are default
    )
    noise_model_config = MultiChannelNMConfig(noise_models=[gmm])
    noise_model = noise_model_factory(noise_model_config)
    assert noise_model is not None
    assert noise_model.nmodel_0.weight.shape == (9, 3)
    assert noise_model.nmodel_0.min_signal == 0
    assert noise_model.nmodel_0.max_signal == 2**16 - 1
    assert noise_model.nmodel_0.min_sigma == 0.125


def test_instantiate_multiple_noise_models(
    tmp_path: Path, create_dummy_noise_model
) -> None:
    # Create a dummy noise model
    np.savez(tmp_path / "dummy_noise_model.npz", **create_dummy_noise_model)

    # Instantiate the noise model
    gmm = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        path=tmp_path / "dummy_noise_model.npz",
        # all other params are default
    )
    noise_model_config = MultiChannelNMConfig(noise_models=[gmm, gmm, gmm])
    noise_model = noise_model_factory(noise_model_config)
    assert noise_model is not None
    assert noise_model.nmodel_0 is not None
    assert noise_model.nmodel_1 is not None
    assert noise_model.nmodel_2 is not None
    assert noise_model.nmodel_0.weight.shape == (9, 3)
    assert noise_model.nmodel_0.min_signal == 0
    assert noise_model.nmodel_0.max_signal == 2**16 - 1
    assert noise_model.nmodel_0.min_sigma == 0.125
    assert noise_model.nmodel_1.weight.shape == (9, 3)
    assert noise_model.nmodel_1.min_signal == 0
    assert noise_model.nmodel_1.max_signal == 2**16 - 1
    assert noise_model.nmodel_1.min_sigma == 0.125
    assert noise_model.nmodel_2.weight.shape == (9, 3)
    assert noise_model.nmodel_2.min_signal == 0
    assert noise_model.nmodel_2.max_signal == 2**16 - 1
    assert noise_model.nmodel_2.min_sigma == 0.125


@pytest.mark.parametrize("img_size", [64, 128])
def test_noise_model_likelihood(
    tmp_path: Path,
    img_size: int,
    create_dummy_noise_model,
) -> None:
    np.savez(tmp_path / "dummy_noise_model.npz", **create_dummy_noise_model)

    gmm_config = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        path=tmp_path / "dummy_noise_model.npz",
        # all other params are default
    )
    nm = GaussianMixtureNoiseModel(gmm_config)
    assert nm is not None
    assert isinstance(nm, GaussianMixtureNoiseModel)

    inp_shape = (1, 1, img_size, img_size)
    signal = torch.ones(inp_shape)
    obs = signal + torch.randn(inp_shape) * 0.1
    likelihood = nm.likelihood(obs, signal)
    assert likelihood.shape == inp_shape


@pytest.mark.parametrize("img_size", [64, 128])
@pytest.mark.parametrize("target_ch", [1, 3, 5])
def test_multi_channel_noise_model_likelihood(
    tmp_path: Path,
    img_size: int,
    target_ch: int,
    create_dummy_noise_model,
) -> None:
    np.savez(tmp_path / "dummy_noise_model.npz", **create_dummy_noise_model)

    gmm = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        path=tmp_path / "dummy_noise_model.npz",
        # all other params are default
    )
    noise_model_config = MultiChannelNMConfig(noise_models=[gmm] * target_ch)
    nm = noise_model_factory(noise_model_config)
    assert nm is not None
    assert isinstance(nm, MultiChannelNoiseModel)
    assert nm._nm_cnt == target_ch
    assert all(
        isinstance(getattr(nm, f"nmodel_{i}"), GaussianMixtureNoiseModel)
        for i in range(nm._nm_cnt)
    )

    inp_shape = (1, target_ch, img_size, img_size)
    signal = torch.ones(inp_shape)
    obs = signal + torch.randn(inp_shape) * 0.1
    likelihood = nm.likelihood(obs, signal)
    assert likelihood.shape == inp_shape


@pytest.mark.skip(reason="Need to refac noise model to be able to train on CPU")
def test_gm_noise_model_training(tmp_path):
    x = np.random.rand(3)
    y = np.random.rand(3)

    nm_config = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel", signal=x, observation=y
    )

    noise_model = GaussianMixtureNoiseModel(nm_config)

    # Test training
    output = noise_model.train_noise_model(x, y, n_epochs=2)
    assert output is not None
    # TODO do something with output ?
