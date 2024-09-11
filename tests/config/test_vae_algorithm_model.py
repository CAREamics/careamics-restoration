from pathlib import Path

import numpy as np
import pytest

from careamics.config import VAEAlgorithmConfig
from careamics.config.architectures import LVAEModel
from careamics.config.nm_model import (
    GaussianMixtureNMConfig,
    MultiChannelNMConfig,
)
from careamics.config.support import SupportedLoss


@pytest.mark.skip(
    reason="VAEAlgorithmConfig model is not currently serializable.\n"
    "The line `schema = VAEAlgorithmConfig.model_json_schema()` currently results "
    "in the following error:\n"
    "PydanticInvalidForJsonSchema: Cannot generate a JsonSchema for "
    "core_schema.IsInstanceSchema (<class 'torch.nn.modules.module.Module'>)"
)
def test_all_losses_are_supported():
    """Test that all losses defined in the Literal are supported."""
    # list of supported losses
    losses = list(SupportedLoss)

    # Algorithm json schema
    schema = VAEAlgorithmConfig.model_json_schema()

    # check that all losses are supported
    for loss in schema["properties"]["loss"]["enum"]:
        assert loss in losses


def test_noise_model_usplit(minimum_algorithm_musplit):
    """Test that the noise model is correctly provided."""
    config = VAEAlgorithmConfig(**minimum_algorithm_musplit)
    assert config.noise_model is None


def test_noise_model_denoisplit(tmp_path: Path, create_dummy_noise_model):
    """Test that the noise model is correctly provided."""
    # TODO this construct with the minimum_config dicts is increasingly annoying

    # Create a dummy noise model
    np.savez(tmp_path / "dummy_noise_model.npz", **create_dummy_noise_model)

    # Instantiate the noise model
    gmm = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        path=tmp_path / "dummy_noise_model.npz",
        # all other params are default
    )
    config = VAEAlgorithmConfig(
        algorithm="denoisplit",
        loss="denoisplit",
        model=LVAEModel(architecture="LVAE"),
        noise_model=MultiChannelNMConfig(noise_models=[gmm]),
    )
    assert config.noise_model is not None


def test_no_noise_model_error_denoisplit(minimum_algorithm_denoisplit):
    """Test that the noise model is correctly provided."""
    minimum_algorithm_denoisplit["noise_model"] = None
    with pytest.raises(ValueError):
        VAEAlgorithmConfig(**minimum_algorithm_denoisplit)
