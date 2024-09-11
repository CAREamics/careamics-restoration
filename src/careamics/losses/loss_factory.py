"""
Loss factory module.

This module contains a factory function for creating loss functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, Optional, Union

from torch import Tensor as tensor

from ..config.support import SupportedLoss
from .fcn.losses import mae_loss, mse_loss, n2v_loss
from .lvae.losses import denoisplit_loss, denoisplit_musplit_loss, musplit_loss

if TYPE_CHECKING:
    from careamics.models.lvae.likelihoods import (
        GaussianLikelihood,
        NoiseModelLikelihood,
    )
    from careamics.models.lvae.noise_models import (
        GaussianMixtureNoiseModel,
        MultiChannelNoiseModel,
    )

    NoiseModel = Union[GaussianMixtureNoiseModel, MultiChannelNoiseModel]


@dataclass
class FCNLossParameters:
    """Dataclass for FCN loss."""

    # TODO check
    prediction: tensor
    targets: tensor
    mask: tensor
    current_epoch: int
    loss_weight: float


@dataclass  # TODO why not pydantic?
class LVAELossParameters:
    """Dataclass for LVAE loss."""

    # TODO: refactor in more modular blocks (otherwise it gets messy very easily)
    # e.g., - weights, - kl_params, ...

    noise_model_likelihood: Optional[NoiseModelLikelihood] = None
    """Noise model likelihood instance."""
    gaussian_likelihood: Optional[GaussianLikelihood] = None
    """Gaussian likelihood instance."""
    current_epoch: int = 0
    """Current epoch in the training loop."""
    reconstruction_weight: float = 1.0
    """Weight for the reconstruction loss in the total net loss
    (i.e., `net_loss = reconstruction_weight * rec_loss + kl_weight * kl_loss`)."""
    musplit_weight: float = 0.1
    """Weight for the muSplit loss (used in the muSplit-denoiSplit loss)."""
    denoisplit_weight: float = 0.9
    """Weight for the denoiSplit loss (used in the muSplit-deonoiSplit loss)."""
    kl_type: Literal["kl", "kl_restricted", "kl_spatial", "kl_channelwise"] = "kl"
    """Type of KL divergence used as KL loss."""
    kl_weight: float = 1.0
    """Weight for the KL loss in the total net loss.
    (i.e., `net_loss = reconstruction_weight * rec_loss + kl_weight * kl_loss`)."""
    kl_annealing: bool = False
    """Whether to apply KL loss annealing."""
    kl_start: int = -1
    """Epoch at which KL loss annealing starts."""
    kl_annealtime: int = 10
    """Number of epochs for which KL loss annealing is applied."""
    non_stochastic: bool = False
    """Whether to sample latents and compute KL."""


# TODO: really needed?
# like it is now, it is difficult to use, we need a way to specify the
# loss parameters in a more user-friendly way.
def loss_parameters_factory(
    type: SupportedLoss,
) -> Union[FCNLossParameters, LVAELossParameters]:
    """Return loss parameters.

    Parameters
    ----------
    type : SupportedLoss
        Requested loss.

    Returns
    -------
    Union[FCNLossParameters, LVAELossParameters]
        Loss parameters.

    Raises
    ------
    NotImplementedError
        If the loss is unknown.
    """
    if type in [SupportedLoss.N2V, SupportedLoss.MSE, SupportedLoss.MAE]:
        return FCNLossParameters

    elif type in [
        SupportedLoss.MUSPLIT,
        SupportedLoss.DENOISPLIT,
        SupportedLoss.DENOISPLIT_MUSPLIT,
    ]:
        return LVAELossParameters  # it returns the class, not an instance

    else:
        raise NotImplementedError(f"Loss {type} is not yet supported.")


def loss_factory(loss: Union[SupportedLoss, str]) -> Callable:
    """Return loss function.

    Parameters
    ----------
    loss : Union[SupportedLoss, str]
        Requested loss.

    Returns
    -------
    Callable
        Loss function.

    Raises
    ------
    NotImplementedError
        If the loss is unknown.
    """
    if loss == SupportedLoss.N2V:
        return n2v_loss

    # elif loss_type == SupportedLoss.PN2V:
    #     return pn2v_loss

    elif loss == SupportedLoss.MAE:
        return mae_loss

    elif loss == SupportedLoss.MSE:
        return mse_loss

    elif loss == SupportedLoss.MUSPLIT:
        return musplit_loss

    elif loss == SupportedLoss.DENOISPLIT:
        return denoisplit_loss

    elif loss == SupportedLoss.DENOISPLIT_MUSPLIT:
        return denoisplit_musplit_loss

    else:
        raise NotImplementedError(f"Loss {loss} is not yet supported.")
