"""
Loss submodule.

This submodule contains the various losses used in CAREamics.
"""

import torch
from torch.nn import L1Loss, MSELoss


def mse_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error loss.

    Parameters
    ----------
    source : torch.Tensor
        Source patches.
    target : torch.Tensor
        Target patches.

    Returns
    -------
    torch.Tensor
        Loss value.
    """
    loss = MSELoss()
    return loss(source, target)


def n2v_loss(
    manipulated_patches: torch.Tensor,
    original_patches: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """
    N2V Loss function described in A Krull et al 2018.

    Parameters
    ----------
    manipulated_patches : torch.Tensor
        Patches with manipulated pixels.
    original_patches : torch.Tensor
        Noisy patches.
    masks : torch.Tensor
        Array containing masked pixel locations.

    Returns
    -------
    torch.Tensor
        Loss value.
    """
    errors = (original_patches - manipulated_patches) ** 2
    # Average over pixels and batch
    loss = torch.sum(errors * masks) / torch.sum(masks)
    return loss  # TODO change output to dict ?


def mae_loss(samples: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    N2N Loss function described in to J Lehtinen et al 2018.

    Parameters
    ----------
    samples : torch.Tensor
        Raw patches.
    labels : torch.Tensor
        Different subset of noisy patches.

    Returns
    -------
    torch.Tensor
        Loss value.
    """
    loss = L1Loss()
    return loss(samples, labels)


# def pn2v_loss(
#     samples: torch.Tensor,
#     labels: torch.Tensor,
#     masks: torch.Tensor,
#     noise_model: HistogramNoiseModel,
# ) -> torch.Tensor:
#     """Probabilistic N2V loss function described in A Krull et al., CVF (2019)."""
#     likelihoods = noise_model.likelihood(labels, samples)
#     likelihoods_avg = torch.log(torch.mean(likelihoods, dim=0, keepdim=True)[0, ...])

#     # Average over pixels and batch
#     loss = -torch.sum(likelihoods_avg * masks) / torch.sum(masks)
#     return loss


# def dice_loss(
#     samples: torch.Tensor, labels: torch.Tensor, mode: str = "multiclass"
# ) -> torch.Tensor:
#     """Dice loss function."""
#     return DiceLoss(mode=mode)(samples, labels.long())
