from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset

from careamics.config import VAEAlgorithmConfig
from careamics.config.architectures import LVAEModel
from careamics.config.likelihood_model import (
    GaussianLikelihoodConfig,
    NMLikelihoodConfig,
)
from careamics.config.nm_model import GaussianMixtureNMConfig, MultiChannelNMConfig
from careamics.lightning import VAEModule
from careamics.losses import denoisplit_loss, denoisplit_musplit_loss, musplit_loss
from careamics.models.lvae.likelihoods import GaussianLikelihood, NoiseModelLikelihood
from careamics.models.lvae.noise_models import (
    MultiChannelNoiseModel,
    noise_model_factory,
)
from careamics.utils.metrics import RunningPSNR

pytestmark = pytest.mark.lvae


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


# TODO: move to conftest.py as pytest.fixture
# it can be split into modules for more clarity (?)
def create_split_lightning_model(
    tmp_path: Path,
    algorithm: str,
    loss_type: str,
    multiscale_count: int = 1,
    predict_logvar: Optional[Literal["pixelwise"]] = None,
    target_ch: int = 1,
) -> VAEModule:
    """Instantiate the muSplit lightining model."""
    lvae_config = LVAEModel(
        architecture="LVAE",
        input_shape=64,
        multiscale_count=multiscale_count,
        z_dims=[128, 128, 128, 128],
        output_channels=target_ch,
        predict_logvar=predict_logvar,
    )

    # gaussian likelihood
    if loss_type in ["musplit", "denoisplit_musplit"]:
        gaussian_lik_config = GaussianLikelihoodConfig(
            predict_logvar=predict_logvar,
            logvar_lowerbound=0.0,
        )
    else:
        gaussian_lik_config = None
    # noise model likelihood
    if loss_type in ["denoisplit", "denoisplit_musplit"]:
        create_dummy_noise_model(tmp_path, 3, 3)
        gmm = GaussianMixtureNMConfig(
            model_type="GaussianMixtureNoiseModel",
            path=tmp_path / "dummy_noise_model.npz",
        )
        noise_model_config = MultiChannelNMConfig(noise_models=[gmm] * target_ch)
        nm = noise_model_factory(noise_model_config)
        nm_lik_config = NMLikelihoodConfig(noise_model=nm)
    else:
        noise_model_config = None
        nm_lik_config = None

    vae_config = VAEAlgorithmConfig(
        algorithm=algorithm,
        loss=loss_type,
        model=lvae_config,
        gaussian_likelihood_model=gaussian_lik_config,
        noise_model=noise_model_config,
        noise_model_likelihood_model=nm_lik_config,
    )

    return VAEModule(
        algorithm_config=vae_config,
    )


class DummyDataset(Dataset):
    def __init__(
        self,
        img_size: int = 64,
        target_ch: int = 1,
        multiscale_count: int = 1,
    ):
        self.num_samples = 3
        self.img_size = img_size
        self.target_ch = target_ch
        self.multiscale_count = multiscale_count

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        input_ = torch.randn(self.multiscale_count, self.img_size, self.img_size)
        target = torch.randn(self.target_ch, self.img_size, self.img_size)
        return input_, target


def create_dummy_dloader(
    batch_size: int = 1,
    img_size: int = 64,
    target_ch: int = 1,
    multiscale_count: int = 1,
):
    dataset = DummyDataset(
        img_size=img_size,
        target_ch=target_ch,
        multiscale_count=multiscale_count,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@pytest.mark.parametrize(
    "multiscale_count, predict_logvar, loss_type, exp_error",
    [
        (1, None, "musplit", does_not_raise()),
        (1, "pixelwise", "musplit", does_not_raise()),
        (5, None, "musplit", does_not_raise()),
        (5, "pixelwise", "musplit", does_not_raise()),
        (1, None, "denoisplit", pytest.raises(ValueError)),
    ],
)
def test_musplit_lightining_init(
    multiscale_count: int,
    predict_logvar: str,
    loss_type: str,
    exp_error: Callable,
):
    lvae_config = LVAEModel(
        architecture="LVAE",
        input_shape=64,
        multiscale_count=multiscale_count,
        z_dims=[128, 128, 128, 128],
        output_channels=3,
        predict_logvar=predict_logvar,
    )

    likelihood_config = GaussianLikelihoodConfig(
        predict_logvar=predict_logvar,
        logvar_lowerbound=0.0,
    )

    with exp_error:
        vae_config = VAEAlgorithmConfig(
            algorithm="musplit",
            loss=loss_type,
            model=lvae_config,
            gaussian_likelihood_model=likelihood_config,
        )
        lightning_model = VAEModule(
            algorithm_config=vae_config,
        )
        assert lightning_model is not None
        assert isinstance(lightning_model.model, torch.nn.Module)
        assert lightning_model.noise_model is None
        assert lightning_model.noise_model_likelihood is None
        assert isinstance(lightning_model.gaussian_likelihood, GaussianLikelihood)
        assert lightning_model.loss_func == musplit_loss


@pytest.mark.parametrize(
    "multiscale_count, predict_logvar, target_ch, nm_cnt, loss_type, exp_error",
    [
        (1, None, 1, 1, "denoisplit", does_not_raise()),
        (5, None, 1, 1, "denoisplit", does_not_raise()),
        (1, None, 3, 3, "denoisplit", does_not_raise()),
        (5, None, 3, 3, "denoisplit", does_not_raise()),
        (1, None, 2, 4, "denoisplit", pytest.raises(ValidationError)),
        (1, None, 1, 1, "denoisplit_musplit", does_not_raise()),
        (1, None, 3, 3, "denoisplit_musplit", does_not_raise()),
        (1, "pixelwise", 1, 1, "denoisplit_musplit", does_not_raise()),
        (1, "pixelwise", 3, 3, "denoisplit_musplit", does_not_raise()),
        (5, None, 1, 1, "denoisplit_musplit", does_not_raise()),
        (5, None, 3, 3, "denoisplit_musplit", does_not_raise()),
        (5, "pixelwise", 1, 1, "denoisplit_musplit", does_not_raise()),
        (5, "pixelwise", 3, 3, "denoisplit_musplit", does_not_raise()),
        (5, "pixelwise", 2, 4, "denoisplit_musplit", pytest.raises(ValidationError)),
        (1, "pixelwise", 1, 1, "denoisplit", pytest.raises(ValidationError)),
        (1, None, 1, 1, "musplit", pytest.raises(ValueError)),
    ],
)
def test_denoisplit_lightining_init(
    tmp_path: Path,
    multiscale_count: int,
    predict_logvar: str,
    target_ch: int,
    nm_cnt: int,
    loss_type: str,
    exp_error: Callable,
):
    # Create the model config
    lvae_config = LVAEModel(
        architecture="LVAE",
        input_shape=64,
        multiscale_count=multiscale_count,
        z_dims=[128, 128, 128, 128],
        output_channels=target_ch,
        predict_logvar=predict_logvar,
    )

    # Create the likelihood config(s)
    # gaussian
    if loss_type == "denoisplit_musplit":
        gaussian_lik_config = GaussianLikelihoodConfig(
            predict_logvar=predict_logvar,
            logvar_lowerbound=0.0,
        )
    else:
        gaussian_lik_config = None
    # noise model
    create_dummy_noise_model(tmp_path, 3, 3)
    gmm = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        path=tmp_path / "dummy_noise_model.npz",
    )
    noise_model_config = MultiChannelNMConfig(noise_models=[gmm] * nm_cnt)
    nm = noise_model_factory(noise_model_config)
    nm_lik_config = NMLikelihoodConfig(noise_model=nm)

    with exp_error:
        vae_config = VAEAlgorithmConfig(
            algorithm="denoisplit",
            loss=loss_type,
            model=lvae_config,
            gaussian_likelihood_model=gaussian_lik_config,
            noise_model=noise_model_config,
            noise_model_likelihood_model=nm_lik_config,
        )
        lightning_model = VAEModule(
            algorithm_config=vae_config,
        )
        assert lightning_model is not None
        assert isinstance(lightning_model.model, torch.nn.Module)
        assert isinstance(lightning_model.noise_model, MultiChannelNoiseModel)
        assert isinstance(lightning_model.noise_model_likelihood, NoiseModelLikelihood)
        if loss_type == "denoisplit_musplit":
            assert isinstance(lightning_model.gaussian_likelihood, GaussianLikelihood)
            assert lightning_model.loss_func == denoisplit_musplit_loss
        else:
            assert lightning_model.gaussian_likelihood is None
            assert lightning_model.loss_func == denoisplit_loss


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("multiscale_count", [1, 5])
@pytest.mark.parametrize("predict_logvar", [None, "pixelwise"])
@pytest.mark.parametrize("target_ch", [1, 3])
def test_musplit_training_step(
    batch_size: int,
    multiscale_count: int,
    predict_logvar: str,
    target_ch: int,
):
    lightning_model = create_split_lightning_model(
        tmp_path=None,
        algorithm="musplit",
        loss_type="musplit",
        multiscale_count=multiscale_count,
        predict_logvar=predict_logvar,
        target_ch=target_ch,
    )
    dloader = create_dummy_dloader(
        batch_size=batch_size,
        img_size=64,
        multiscale_count=multiscale_count,
        target_ch=target_ch,
    )
    batch = next(iter(dloader))
    train_loss = lightning_model.training_step(batch=batch, batch_idx=0)

    # check outputs
    assert train_loss is not None
    assert isinstance(train_loss, dict)
    assert "loss" in train_loss
    assert "reconstruction_loss" in train_loss
    assert "kl_loss" in train_loss


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("multiscale_count", [1, 5])
@pytest.mark.parametrize("predict_logvar", [None, "pixelwise"])
@pytest.mark.parametrize("target_ch", [1, 3])
def test_musplit_validation_step(
    batch_size: int,
    multiscale_count: int,
    predict_logvar: str,
    target_ch: int,
):
    lightning_model = create_split_lightning_model(
        tmp_path=None,
        algorithm="musplit",
        loss_type="musplit",
        multiscale_count=multiscale_count,
        predict_logvar=predict_logvar,
        target_ch=target_ch,
    )
    dloader = create_dummy_dloader(
        batch_size=batch_size,
        img_size=64,
        multiscale_count=multiscale_count,
        target_ch=target_ch,
    )
    batch = next(iter(dloader))
    lightning_model.validation_step(batch=batch, batch_idx=0)
    # NOTE: `validation_step` does not return anything...


@pytest.mark.parametrize(
    "multiscale_count, predict_logvar, target_ch, loss_type",
    [
        (1, None, 1, "denoisplit"),
        (5, None, 1, "denoisplit"),
        (1, None, 3, "denoisplit"),
        (5, None, 3, "denoisplit"),
        (1, None, 1, "denoisplit_musplit"),
        (1, None, 3, "denoisplit_musplit"),
        (1, "pixelwise", 1, "denoisplit_musplit"),
        (1, "pixelwise", 3, "denoisplit_musplit"),
        (5, None, 1, "denoisplit_musplit"),
        (5, None, 3, "denoisplit_musplit"),
        (5, "pixelwise", 1, "denoisplit_musplit"),
        (5, "pixelwise", 3, "denoisplit_musplit"),
    ],
)
def test_denoisplit_training_step(
    tmp_path: Path,
    multiscale_count: int,
    predict_logvar: str,
    target_ch: int,
    loss_type: str,
):
    lightning_model = create_split_lightning_model(
        tmp_path=tmp_path,
        algorithm="denoisplit",
        loss_type=loss_type,
        multiscale_count=multiscale_count,
        predict_logvar=predict_logvar,
        target_ch=target_ch,
    )
    dloader = create_dummy_dloader(
        batch_size=8,
        img_size=64,
        multiscale_count=multiscale_count,
        target_ch=target_ch,
    )
    batch = next(iter(dloader))
    train_loss = lightning_model.training_step(batch=batch, batch_idx=0)

    # check outputs
    assert train_loss is not None
    assert isinstance(train_loss, dict)
    assert "loss" in train_loss
    assert "reconstruction_loss" in train_loss
    assert "kl_loss" in train_loss


@pytest.mark.parametrize(
    "multiscale_count, predict_logvar, target_ch, loss_type",
    [
        (1, None, 1, "denoisplit"),
        (5, None, 1, "denoisplit"),
        (1, None, 3, "denoisplit"),
        (5, None, 3, "denoisplit"),
        (1, None, 1, "denoisplit_musplit"),
        (1, None, 3, "denoisplit_musplit"),
        (1, "pixelwise", 1, "denoisplit_musplit"),
        (1, "pixelwise", 3, "denoisplit_musplit"),
        (5, None, 1, "denoisplit_musplit"),
        (5, None, 3, "denoisplit_musplit"),
        (5, "pixelwise", 1, "denoisplit_musplit"),
        (5, "pixelwise", 3, "denoisplit_musplit"),
    ],
)
def test_denoisplit_validation_step(
    tmp_path: Path,
    multiscale_count: int,
    predict_logvar: str,
    target_ch: int,
    loss_type: str,
):
    lightning_model = create_split_lightning_model(
        tmp_path=tmp_path,
        algorithm="denoisplit",
        loss_type=loss_type,
        multiscale_count=multiscale_count,
        predict_logvar=predict_logvar,
        target_ch=target_ch,
    )
    dloader = create_dummy_dloader(
        batch_size=8,
        img_size=64,
        multiscale_count=multiscale_count,
        target_ch=target_ch,
    )
    batch = next(iter(dloader))
    lightning_model.validation_step(batch=batch, batch_idx=0)


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("multiscale_count", [1, 5])
@pytest.mark.parametrize("predict_logvar", [None, "pixelwise"])
@pytest.mark.parametrize("target_ch", [1, 3])
def test_training_loop_musplit(
    batch_size: int,
    multiscale_count: int,
    predict_logvar: str,
    target_ch: int,
):
    lightning_model = create_split_lightning_model(
        tmp_path=None,
        algorithm="musplit",
        loss_type="musplit",
        multiscale_count=multiscale_count,
        predict_logvar=predict_logvar,
        target_ch=target_ch,
    )
    dloader = create_dummy_dloader(
        batch_size=batch_size,
        img_size=64,
        multiscale_count=multiscale_count,
        target_ch=target_ch,
    )
    trainer = Trainer(accelerator="cpu", max_epochs=2, logger=False, callbacks=[])

    try:
        trainer.fit(
            model=lightning_model,
            train_dataloaders=dloader,
            val_dataloaders=dloader,
        )
    except Exception as e:
        pytest.fail(f"Training routine failed with exception: {e}")


@pytest.mark.parametrize(
    "batch_size, predict_logvar, target_ch, loss_type",
    [
        (1, None, 1, "denoisplit"),
        (4, None, 1, "denoisplit"),
        (1, None, 3, "denoisplit"),
        (4, None, 3, "denoisplit"),
        (1, None, 1, "denoisplit_musplit"),
        (1, None, 3, "denoisplit_musplit"),
        (1, "pixelwise", 1, "denoisplit_musplit"),
        (1, "pixelwise", 3, "denoisplit_musplit"),
        (4, None, 1, "denoisplit_musplit"),
        (4, None, 3, "denoisplit_musplit"),
        (4, "pixelwise", 1, "denoisplit_musplit"),
        (4, "pixelwise", 3, "denoisplit_musplit"),
    ],
)
def test_training_loop_denoisplit(
    tmp_path: Path,
    batch_size: int,
    predict_logvar: str,
    target_ch: int,
    loss_type: str,
):
    lightning_model = create_split_lightning_model(
        tmp_path=tmp_path,
        algorithm="denoisplit",
        loss_type=loss_type,
        multiscale_count=1,
        predict_logvar=predict_logvar,
        target_ch=target_ch,
    )
    dloader = create_dummy_dloader(
        batch_size=batch_size,
        img_size=64,
        multiscale_count=1,
        target_ch=target_ch,
    )
    trainer = Trainer(accelerator="cpu", max_epochs=2, logger=False, callbacks=[])

    try:
        trainer.fit(
            model=lightning_model,
            train_dataloaders=dloader,
            val_dataloaders=dloader,
        )
    except Exception as e:
        pytest.fail(f"Training routine failed with exception: {e}")


@pytest.mark.parametrize("predict_logvar", [None, "pixelwise"])
@pytest.mark.parametrize("target_ch", [1, 3])
def test_get_reconstructed_tensor(
    predict_logvar: str,
    target_ch: int,
):
    lightning_model = create_split_lightning_model(
        tmp_path=None,
        algorithm="musplit",
        loss_type="musplit",
        multiscale_count=1,
        predict_logvar=predict_logvar,
        target_ch=target_ch,
    )
    dloader = create_dummy_dloader(
        batch_size=1,
        img_size=64,
        multiscale_count=1,
        target_ch=target_ch,
    )
    input_, target = next(iter(dloader))
    output = lightning_model(input_)
    rec_img = lightning_model.get_reconstructed_tensor(output)
    assert rec_img.shape == target.shape  # same shape as target


@pytest.mark.parametrize("predict_logvar", [None, "pixelwise"])
@pytest.mark.parametrize("target_ch", [1, 3])
def test_val_PSNR_computation(
    predict_logvar: str,
    target_ch: int,
):
    lightning_model = create_split_lightning_model(
        tmp_path=None,
        algorithm="musplit",
        loss_type="musplit",
        multiscale_count=1,
        predict_logvar=predict_logvar,
        target_ch=target_ch,
    )
    assert lightning_model.running_psnr is not None
    assert len(lightning_model.running_psnr) == target_ch
    for item in lightning_model.running_psnr:
        assert isinstance(item, RunningPSNR)

    dloader = create_dummy_dloader(
        batch_size=1,
        img_size=64,
        multiscale_count=1,
        target_ch=target_ch,
    )
    input_, target = next(iter(dloader))
    output = lightning_model(input_)

    curr_psnr = lightning_model.compute_val_psnr(output, target)
    assert curr_psnr is not None
    assert len(curr_psnr) == target_ch
    for i in range(target_ch):
        assert lightning_model.running_psnr[i].mse_sum != 0
        assert lightning_model.running_psnr[i].N == 1
        assert lightning_model.running_psnr[i].min is not None
        assert lightning_model.running_psnr[i].max is not None
        assert lightning_model.running_psnr[i].get() is not None
        lightning_model.running_psnr[i].reset()
        assert lightning_model.running_psnr[i].mse_sum == 0
        assert lightning_model.running_psnr[i].N == 0
        assert lightning_model.running_psnr[i].min is None
        assert lightning_model.running_psnr[i].max is None
        assert lightning_model.running_psnr[i].get() is None
