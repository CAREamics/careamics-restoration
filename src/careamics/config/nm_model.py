"""Noise models config."""

from pathlib import Path
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class GaussianMixtureNmModel(BaseModel):
    """Gaussian mixture noise model."""

    model_config = ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True, extra="allow"
    )
    model_type: Literal["GaussianMixtureNoiseModel"] # TODO: why do we need this? 
    path: Union[Path, str, None] = None
    weight: Any = None
    n_gaussian: int = Field(default=1, ge=1)
    n_coeff: int = Field(default=2, ge=2)
    min_signal: float = Field(default=0.0, ge=0.0)
    max_signal: float = Field(default=1.0, ge=0.0)
    min_sigma: Any = None
    tol: float = Field(default=1e-10)

# The noise model is given by a set of GMMs, one for each target
# e.g., 2 target channels, 2 noise models
class NMModel(BaseModel):
    """Noise Model that aggregates the noise models for single channels"""
    
    # TODO: check that this model config is OK
    model_config = ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True, extra="allow"
    )
    noise_models: list[GaussianMixtureNmModel]