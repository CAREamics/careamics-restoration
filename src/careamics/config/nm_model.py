"""Noise models config."""

from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PlainValidator,
    model_validator,
)
from typing_extensions import Annotated, Self

from careamics.utils.serializers import _array_to_json, _to_numpy

# TODO: this is a temporary solution to serialize and deserialize array fields
# in pydantic models. Specifically, the aim is to enable saving and loading configs
# with such arrays to/from JSON files during, resp., training and evaluation.
Array = Annotated[
    Union[np.ndarray, torch.Tensor],
    PlainSerializer(_array_to_json, return_type=str),
    PlainValidator(_to_numpy),
]
"""Annotated array type, used to serialize arrays or tensors to JSON strings
and deserialize them back to arrays."""


# TODO: add histogram-based noise model


class GaussianMixtureNMConfig(BaseModel):
    """Gaussian mixture noise model."""

    model_config = ConfigDict(
        protected_namespaces=(),
        validate_assignment=True,
        arbitrary_types_allowed=True,
        extra="allow",
    )
    # model type
    model_type: Literal["GaussianMixtureNoiseModel"]

    path: Optional[Union[Path, str]] = None
    """Path to the directory where the trained noise model (*.npz) is saved in the
    `train` method."""

    # TODO remove and use as parameters to the NM functions?
    signal: Optional[Union[str, Path, np.ndarray]] = Field(default=None, exclude=True)
    """Path to the file containing signal or respective numpy array."""

    # TODO remove and use as parameters to the NM functions?
    observation: Optional[Union[str, Path, np.ndarray]] = Field(
        default=None, exclude=True
    )
    """Path to the file containing observation or respective numpy array."""

    weight: Optional[Array] = None
    """A [3*n_gaussian, n_coeff] sized array containing the values of the weights
    describing the GMM noise model, with each row corresponding to one
    parameter of each gaussian, namely [mean, standard deviation and weight].
    Specifically, rows are organized as follows:
    - first n_gaussian rows correspond to the means
    - next n_gaussian rows correspond to the weights
    - last n_gaussian rows correspond to the standard deviations
    If `weight=None`, the weight array is initialized using the `min_signal`
    and `max_signal` parameters."""

    n_gaussian: int = Field(default=1, ge=1)
    """Number of gaussians used for the GMM."""

    n_coeff: int = Field(default=2, ge=2)
    """Number of coefficients to describe the functional relationship between gaussian
    parameters and the signal. 2 implies a linear relationship, 3 implies a quadratic
    relationship and so on."""

    min_signal: float = Field(default=0.0, ge=0.0)
    """Minimum signal intensity expected in the image."""

    max_signal: float = Field(default=1.0, ge=0.0)
    """Maximum signal intensity expected in the image."""

    min_sigma: float = Field(default=200.0, ge=0.0)  # TODO took from nb in pn2v
    """Minimum value of `standard deviation` allowed in the GMM.
    All values of `standard deviation` below this are clamped to this value."""

    tol: float = Field(default=1e-10)
    """Tolerance used in the computation of the noise model likelihood."""

    @model_validator(mode="after")
    def validate_path_to_pretrained_vs_training_data(self: Self) -> Self:
        """Validate paths provided in the config.

        Returns
        -------
        Self
            Returns itself.
        """
        if self.path and (self.signal is not None or self.observation is not None):
            raise ValueError(
                "Either only 'path' to pre-trained noise model should be"
                "provided or only signal and observation in form of paths"
                "or numpy arrays."
            )
        if not self.path and (self.signal is None or self.observation is None):
            raise ValueError(
                "Either only 'path' to pre-trained noise model should be"
                "provided or only signal and observation in form of paths"
                "or numpy arrays."
            )
        return self


# The noise model is given by a set of GMMs, one for each target
# e.g., 2 target channels, 2 noise models
class MultiChannelNMConfig(BaseModel):
    """Noise Model config aggregating noise models for single output channels."""

    # TODO: check that this model config is OK
    model_config = ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True, extra="allow"
    )
    noise_models: list[GaussianMixtureNMConfig]
    """List of noise models, one for each target channel."""
