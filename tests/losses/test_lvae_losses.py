from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Union

import numpy as np
import pytest
import torch

from careamics.config.likelihood_model import (
    GaussianLikelihoodConfig,
    NMLikelihoodConfig,
)
from careamics.config.nm_model import GaussianMixtureNMConfig, MultiChannelNMConfig
from careamics.losses.loss_factory import (
    LVAELossParameters,
    SupportedLoss,
    loss_factory,
)
from careamics.losses.lvae.losses import (
    denoisplit_loss,
    denoisplit_musplit_loss,
    get_reconstruction_loss,
    musplit_loss,
    reconstruction_loss_musplit_denoisplit,
)
from careamics.models.lvae.likelihoods import likelihood_factory
from careamics.models.lvae.noise_models import noise_model_factory

if TYPE_CHECKING:
    from careamics.models.lvae.noise_models import MultiChannelNoiseModel
#     from careamics.models.lvae.likelihoods import (
#         LikelihoodModule,
#         GaussianLikelihood,
#         NoiseModelLikelihood
#     )
#     Likelihood = Union[LikelihoodModule, GaussianLikelihood, NoiseModelLikelihood]

pytestmark = pytest.mark.lvae


# TODO: move to conftest.py as pytest.fixture
def create_dummy_noise_model(
    tmp_path: Path,
    n_gaussians: int = 3,
    n_coeffs: int = 3,
) -> None:
    """Create a dummy noise model and save it in a `.npz` file."""
    weights = np.random.rand(3 * n_gaussians, n_coeffs)
    nm_dict = {
        "trained_weight": weights,
        "min_signal": np.array([0]),
        "max_signal": np.array([2**16 - 1]),
        "min_sigma": 0.125,
    }
    np.savez(tmp_path / "dummy_noise_model.npz", **nm_dict)


def init_noise_model(
    tmp_path: Path,
    target_ch: int,
    n_gaussians: int = 3,
    n_coeffs: int = 3,
) -> MultiChannelNoiseModel:
    """Instantiate a dummy noise model."""
    create_dummy_noise_model(tmp_path, n_gaussians, n_coeffs)
    gmm = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        path=tmp_path / "dummy_noise_model.npz",
        # all other params are default
    )
    noise_model_config = MultiChannelNMConfig(noise_models=[gmm] * target_ch)
    return noise_model_factory(noise_model_config)


@pytest.mark.parametrize(
    "loss_type, exp_loss_func, exp_error",
    [
        (SupportedLoss.MUSPLIT, musplit_loss, does_not_raise()),
        (SupportedLoss.DENOISPLIT, denoisplit_loss, does_not_raise()),
        (SupportedLoss.DENOISPLIT_MUSPLIT, denoisplit_musplit_loss, does_not_raise()),
        ("musplit", musplit_loss, does_not_raise()),
        ("denoisplit", denoisplit_loss, does_not_raise()),
        ("denoisplit_musplit", denoisplit_musplit_loss, does_not_raise()),
        ("made_up_loss", None, pytest.raises(NotImplementedError)),
    ],
)
def test_lvae_loss_factory(
    loss_type: Union[SupportedLoss, str], exp_loss_func: Callable, exp_error: Callable
):
    with exp_error:
        loss_func = loss_factory(loss_type)
        assert loss_func is not None
        assert callable(loss_func)
        assert loss_func == exp_loss_func


@pytest.mark.parametrize(
    "batch_size, target_ch, predict_logvar, likelihood_type",
    [
        (1, 1, None, "noise_model"),
        (1, 1, None, "gaussian"),
        (1, 1, "pixelwise", "gaussian"),
        (8, 1, None, "noise_model"),
        (8, 1, None, "gaussian"),
        (8, 1, "pixelwise", "gaussian"),
        (1, 3, None, "noise_model"),
        (1, 3, None, "gaussian"),
        (1, 3, "pixelwise", "gaussian"),
        (8, 3, None, "noise_model"),
        (8, 3, None, "gaussian"),
        (8, 3, "pixelwise", "gaussian"),
    ],
)
def test_reconstruction_loss(
    tmp_path: Path,
    batch_size: int,
    target_ch: int,
    predict_logvar: str,
    likelihood_type: str,
):
    # create test data
    img_size = 64
    inp_ch = target_ch * (1 + int(predict_logvar is not None))
    reconstruction = torch.rand((batch_size, inp_ch, img_size, img_size))
    target = torch.rand((batch_size, target_ch, img_size, img_size))

    # create likelihood object
    if likelihood_type == "noise_model":
        nm = init_noise_model(tmp_path, target_ch)
        data_mean = target.mean(dim=(0, 2, 3), keepdim=True)
        data_std = target.std(dim=(0, 2, 3), keepdim=True)
        config = NMLikelihoodConfig(
            data_mean=data_mean, data_std=data_std, noise_model=nm
        )
    else:
        config = GaussianLikelihoodConfig(predict_logvar=predict_logvar)
    likelihood = likelihood_factory(config)

    # compute the loss
    rec_loss = get_reconstruction_loss(
        reconstruction=reconstruction, target=target, likelihood_obj=likelihood
    )

    # check outputs
    assert rec_loss is not None
    for i in range(target_ch):
        assert rec_loss[f"ch{i+1}_loss"] is not None


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("target_ch", [1, 3])
@pytest.mark.parametrize("predict_logvar", [None, "pixelwise"])
@pytest.mark.parametrize("nm_weight", [0.0, 0.5, 1.0])
def test_reconstruction_loss_musplit_denoisplit(
    tmp_path: Path,
    batch_size: int,
    target_ch: int,
    predict_logvar: str,
    nm_weight: float,
):
    # create test data
    img_size = 64
    inp_ch = target_ch * (1 + int(predict_logvar is not None))
    reconstruction = torch.rand((batch_size, inp_ch, img_size, img_size))
    target = torch.rand((batch_size, target_ch, img_size, img_size))

    # create likelihood objects
    nm = init_noise_model(tmp_path, target_ch)
    data_mean = target.mean(dim=(0, 2, 3), keepdim=True)
    data_std = target.std(dim=(0, 2, 3), keepdim=True)
    nm_config = NMLikelihoodConfig(
        data_mean=data_mean, data_std=data_std, noise_model=nm
    )
    nm_likelihood = likelihood_factory(nm_config)
    gaussian_config = GaussianLikelihoodConfig(predict_logvar=predict_logvar)
    gaussian_likelihood = likelihood_factory(gaussian_config)

    # compute the loss
    rec_loss = reconstruction_loss_musplit_denoisplit(
        predictions=reconstruction,
        targets=target,
        nm_likelihood=nm_likelihood,
        gaussian_likelihood=gaussian_likelihood,
        nm_weight=nm_weight,
        gaussian_weight=1.0 - nm_weight,
    )

    # check outputs
    assert rec_loss is not None
    assert isinstance(rec_loss, torch.Tensor)


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("target_ch", [1, 3])
@pytest.mark.parametrize("predict_logvar", [None, "pixelwise"])
@pytest.mark.parametrize("n_layers", [1, 4])
@pytest.mark.parametrize("enable_LC", [False, True])
def test_musplit_loss(
    batch_size: int,
    target_ch: int,
    predict_logvar: str,
    n_layers: int,
    enable_LC: bool,
):
    # create test data
    img_size = 64
    inp_ch = target_ch * (1 + int(predict_logvar is not None))
    reconstruction = torch.rand((batch_size, inp_ch, img_size, img_size))
    target = torch.rand((batch_size, target_ch, img_size, img_size))
    if enable_LC:
        z = [torch.rand(batch_size, 128, img_size, img_size) for _ in range(n_layers)]
    else:
        sizes = [img_size // 2 ** (i + 1) for i in range(n_layers)]
        sizes = sizes[::-1]
        z = [torch.rand(batch_size, 128, sz, sz) for sz in sizes]
    td_data = {
        "z": z,
        "kl": [torch.rand(batch_size) for _ in range(n_layers)],
    }

    # create likelihood
    config = GaussianLikelihoodConfig(predict_logvar=predict_logvar)
    likelihood = likelihood_factory(config)

    # compute the loss
    loss_parameters = LVAELossParameters(
        gaussian_likelihood=likelihood,
    )
    output = musplit_loss((reconstruction, td_data), target, loss_parameters)

    # check outputs
    # NOTE: output should not be None in these test cases
    assert output is not None
    assert isinstance(output, dict)
    assert "loss" in output
    assert "reconstruction_loss" in output
    assert "kl_loss" in output


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("target_ch", [1, 3])
@pytest.mark.parametrize("n_layers", [1, 4])
def test_denoisplit_loss(
    tmp_path: Path,
    batch_size: int,
    target_ch: int,
    n_layers: int,
):
    # create test data
    img_size = 64
    reconstruction = torch.rand((batch_size, target_ch, img_size, img_size))
    target = torch.rand((batch_size, target_ch, img_size, img_size))
    td_data = {
        "z": [torch.rand(batch_size, 128, img_size, img_size) for _ in range(n_layers)],
        "kl": [torch.rand(batch_size) for _ in range(n_layers)],
    }

    # create likelihood
    nm = init_noise_model(tmp_path, target_ch)
    data_mean = target.mean(dim=(0, 2, 3), keepdim=True)
    data_std = target.std(dim=(0, 2, 3), keepdim=True)
    nm_config = NMLikelihoodConfig(
        data_mean=data_mean, data_std=data_std, noise_model=nm
    )
    likelihood = likelihood_factory(nm_config)

    # compute the loss
    loss_parameters = LVAELossParameters(
        noise_model_likelihood=likelihood,
    )
    output = denoisplit_loss((reconstruction, td_data), target, loss_parameters)

    # check outputs
    # NOTE: output should not be None in these test cases
    assert output is not None
    assert isinstance(output, dict)
    assert "loss" in output
    assert "reconstruction_loss" in output
    assert "kl_loss" in output


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("target_ch", [1, 3])
@pytest.mark.parametrize("predict_logvar", [None, "pixelwise"])
@pytest.mark.parametrize("n_layers", [1, 4])
@pytest.mark.parametrize("enable_LC", [False, True])
def test_denoisplit_musplit_loss(
    tmp_path: Path,
    batch_size: int,
    target_ch: int,
    predict_logvar: str,
    n_layers: int,
    enable_LC: bool,
):
    # create test data
    img_size = 64
    inp_ch = target_ch * (1 + int(predict_logvar is not None))
    reconstruction = torch.rand((batch_size, inp_ch, img_size, img_size))
    target = torch.rand((batch_size, target_ch, img_size, img_size))
    if enable_LC:
        z = [torch.rand(batch_size, 128, img_size, img_size) for _ in range(n_layers)]
    else:
        sizes = [img_size // 2 ** (i + 1) for i in range(n_layers)]
        sizes = sizes[::-1]
        z = [torch.rand(batch_size, 128, sz, sz) for sz in sizes]
    td_data = {
        "z": z,
        "kl": [torch.rand(batch_size) for _ in range(n_layers)],
    }

    # create likelihood objects
    nm = init_noise_model(tmp_path, target_ch)
    data_mean = target.mean(dim=(0, 2, 3), keepdim=True)
    data_std = target.std(dim=(0, 2, 3), keepdim=True)
    nm_config = NMLikelihoodConfig(
        data_mean=data_mean, data_std=data_std, noise_model=nm
    )
    nm_likelihood = likelihood_factory(nm_config)
    gaussian_config = GaussianLikelihoodConfig(predict_logvar=predict_logvar)
    gaussian_likelihood = likelihood_factory(gaussian_config)

    # compute the loss
    loss_parameters = LVAELossParameters(
        gaussian_likelihood=gaussian_likelihood,
        noise_model_likelihood=nm_likelihood,
    )
    output = denoisplit_musplit_loss((reconstruction, td_data), target, loss_parameters)

    # check outputs
    # NOTE: output should not be None in these test cases
    assert output is not None
    assert isinstance(output, dict)
    assert "loss" in output
    assert "reconstruction_loss" in output
    assert "kl_loss" in output
