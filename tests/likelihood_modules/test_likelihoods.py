from pathlib import Path
from typing import Union

import numpy as np
import pytest
import torch

from careamics.config.likelihood_model import (
    GaussianLikelihoodConfig,
    NMLikelihoodConfig,
)
from careamics.config.nm_model import GaussianMixtureNMConfig, MultiChannelNMConfig
from careamics.models.lvae.likelihoods import likelihood_factory
from careamics.models.lvae.noise_models import noise_model_factory


# TODO: move to conftest.py as pytest.fixture
def create_dummy_noise_model(
    tmp_path: Path,
    n_gaussians: int = 3,
    n_coeffs: int = 3,
) -> None:
    weights = np.random.rand(3 * n_gaussians, n_coeffs)
    nm_dict = {
        "trained_weight": weights,
        "min_signal": np.array([0]),
        "max_signal": np.array([2**16 - 1]),
        "min_sigma": 0.125,
    }
    np.savez(tmp_path / "dummy_noise_model.npz", **nm_dict)


# TODO: move it under models/lvae/ ??


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("target_ch", [1, 3])
@pytest.mark.parametrize("predict_logvar", [None, "pixelwise"])
@pytest.mark.parametrize("logvar_lowerbound", [None, 0.1])
def test_gaussian_likelihood(
    batch_size: int,
    target_ch: int,
    predict_logvar: Union[str, None],
    logvar_lowerbound: Union[float, None],
) -> None:
    config = GaussianLikelihoodConfig(
        predict_logvar=predict_logvar, logvar_lowerbound=logvar_lowerbound
    )
    likelihood = likelihood_factory(config)

    img_size = 64
    inp_ch = target_ch * (1 + int(predict_logvar is not None))
    reconstruction = torch.rand((batch_size, inp_ch, img_size, img_size))
    target = torch.rand((batch_size, target_ch, img_size, img_size))
    out, data = likelihood(reconstruction, target)

    exp_out_shape = (batch_size, target_ch, img_size, img_size)
    assert out.shape == exp_out_shape
    assert out[0].mean() is not None
    assert data["mean"].shape == exp_out_shape
    if predict_logvar == "pixelwise":
        assert data["logvar"].shape == exp_out_shape
    else:
        assert data["logvar"] is None


@pytest.mark.parametrize("img_size", [64, 128])
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("target_ch", [1, 3, 5])
def test_noise_model_likelihood(
    tmp_path: Path, batch_size: int, img_size: int, target_ch: int
) -> None:
    # Instantiate the noise model
    create_dummy_noise_model(tmp_path, n_gaussians=4, n_coeffs=3)
    gmm = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        path=tmp_path / "dummy_noise_model.npz",
        # all other params are default
    )
    noise_model_config = MultiChannelNMConfig(noise_models=[gmm] * target_ch)
    nm = noise_model_factory(noise_model_config)

    # Instantiate the likelihood
    inp_shape = (batch_size, target_ch, img_size, img_size)
    reconstruction = target = torch.rand(inp_shape)
    # NOTE: `input_` is actually the output of LVAE decoder
    data_mean = target.mean(dim=(0, 2, 3), keepdim=True)
    data_std = target.std(dim=(0, 2, 3), keepdim=True)
    config = NMLikelihoodConfig(data_mean=data_mean, data_std=data_std, noise_model=nm)
    likelihood = likelihood_factory(config)

    out, data = likelihood(reconstruction, target)
    exp_out_shape = inp_shape
    assert out.shape == exp_out_shape
    assert out[0].mean() is not None
    assert data["mean"].shape == exp_out_shape
    assert data["logvar"] is None
