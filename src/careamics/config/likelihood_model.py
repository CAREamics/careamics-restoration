"""Likelihood model."""

from typing import Literal, Optional, Union

import torch
from pydantic import BaseModel, ConfigDict

from careamics.models.lvae.noise_models import (
    GaussianMixtureNoiseModel,
    MultiChannelNoiseModel,
)

NoiseModel = Union[GaussianMixtureNoiseModel, MultiChannelNoiseModel]


class GaussianLikelihoodConfig(BaseModel):
    """Gaussian likelihood configuration."""

    model_config = ConfigDict(validate_assignment=True)

    predict_logvar: Optional[Literal["pixelwise"]] = None
    """If `pixelwise`, log-variance is computed for each pixel, else log-variance
    is not computed."""

    logvar_lowerbound: Union[float, None] = None
    """The lowerbound value for log-variance."""


class NMLikelihoodConfig(BaseModel):
    """Noise model likelihood configuration."""

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    data_mean: Union[torch.Tensor] = torch.zeros(1)
    """The mean of the data, used to unnormalize data for noise model evaluation.
    Shape is (target_ch,) (or (1, target_ch, [1], 1, 1))."""

    data_std: Union[torch.Tensor] = torch.ones(1)
    """The standard deviation of the data, used to unnormalize data for noise
    model evaluation. Shape is (target_ch,) (or (1, target_ch, [1], 1, 1))."""

    noise_model: Union[NoiseModel, None] = None
    """The noise model instance used to compute the likelihood."""
