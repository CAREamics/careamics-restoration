"""Methods for Loss Computation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch

from careamics.losses.lvae.loss_utils import free_bits_kl, get_kl_weight
from careamics.models.lvae.likelihoods import (
    GaussianLikelihood,
    LikelihoodModule,
    NoiseModelLikelihood,
)
from careamics.models.lvae.utils import compute_batch_mean

if TYPE_CHECKING:
    from careamics.losses.loss_factory import LVAELossParameters

Likelihood = Union[LikelihoodModule, GaussianLikelihood, NoiseModelLikelihood]


def get_reconstruction_loss(
    reconstruction: torch.Tensor,  # TODO: naming -> predictions?
    target: torch.Tensor,
    likelihood_obj: Likelihood,
) -> dict[str, torch.Tensor]:
    """Compute the reconstruction loss.

    Parameters
    ----------
    reconstruction: torch.Tensor
        The output of the LVAE decoder. Shape is (B, C, [Z], Y, X), where C is the
        number of output channels (e.g., 1 in HDN, >1 in muSplit/denoiSplit).
    target: torch.Tensor
        The target image used to compute the reconstruction loss. Shape is
        (B, C, [Z], Y, X), where C is the number of output channels
        (e.g., 1 in HDN, >1 in muSplit/denoiSplit).
    likelihood_obj: Likelihood
        The likelihood object used to compute the reconstruction loss.

    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary containing the overall loss `["loss"]` and the loss for
        individual output channels `["ch{i}_loss"]`.
    """
    loss_dict = _get_reconstruction_loss_vector(
        reconstruction=reconstruction,
        target=target,
        likelihood_obj=likelihood_obj,
    )

    loss_dict["loss"] = loss_dict["loss"].sum() / len(reconstruction)
    for i in range(1, 1 + target.shape[1]):
        key = f"ch{i}_loss"
        loss_dict[key] = loss_dict[key].sum() / len(reconstruction)

    return loss_dict


def _get_reconstruction_loss_vector(
    reconstruction: torch.Tensor,  # TODO: naming -> predictions?
    target: torch.Tensor,
    likelihood_obj: LikelihoodModule,
) -> dict[str, torch.Tensor]:
    """Compute the reconstruction loss.

    Parameters
    ----------
    return_predicted_img: bool
        If set to `True`, the besides the loss, the reconstructed image is returned.
        Default is `False`.

    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary containing the overall loss `["loss"]` and the loss for
        individual output channels `["ch{i}_loss"]`. Shape of individual
        tensors is (B, ).
    """
    output = {"loss": None}
    for i in range(1, 1 + target.shape[1]):
        output[f"ch{i}_loss"] = None

    # Compute Log likelihood
    ll, _ = likelihood_obj(reconstruction, target)  # shape: (B, C, [Z], Y, X)

    output = {"loss": compute_batch_mean(-1 * ll)}  # shape: (B, )
    if ll.shape[1] > 1:  # target_ch > 1
        for i in range(1, 1 + target.shape[1]):
            output[f"ch{i}_loss"] = compute_batch_mean(-ll[:, i - 1])  # shape: (B, )
    else:  # target_ch == 1
        # TODO: hacky!!! Refactor this
        assert ll.shape[1] == 1
        output["ch1_loss"] = output["loss"]
        output["ch2_loss"] = output["loss"]

    return output


def reconstruction_loss_musplit_denoisplit(
    predictions: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    targets: torch.Tensor,
    nm_likelihood: NoiseModelLikelihood,
    gaussian_likelihood: GaussianLikelihood,
    nm_weight: float,
    gaussian_weight: float,
) -> torch.Tensor:
    """Compute the reconstruction loss for muSplit-denoiSplit loss.

    The resulting loss is a weighted mean of the noise model likelihood and the
    Gaussian likelihood.

    Parameters
    ----------
    predictions : torch.Tensor
        The output of the LVAE decoder. Shape is (B, C, [Z], Y, X), or
        (B, 2*C, [Z], Y, X), where C is the number of output channels,
        and the factor of 2 is for the case of predicted log-variance.
    targets : torch.Tensor
        The target image used to compute the reconstruction loss. Shape is
        (B, C, [Z], Y, X), where C is the number of output channels
        (e.g., 1 in HDN, >1 in muSplit/denoiSplit).
    nm_likelihood : NoiseModelLikelihood
        A `NoiseModelLikelihood` object used to compute the noise model likelihood.
    gaussian_likelihood : GaussianLikelihood
        A `GaussianLikelihood` object used to compute the Gaussian likelihood.
    nm_weight : float
        The weight for the noise model likelihood.
    gaussian_weight : float
        The weight for the Gaussian likelihood.

    Returns
    -------
    recons_loss : torch.Tensor
        The reconstruction loss. Shape is (1, ).
    """
    # TODO: is this safe to check for predict_logvar value?
    # otherwise use `gaussian_likelihood.predict_logvar` (or both)
    if predictions.shape[1] == 2 * targets.shape[1]:
        # predictions contain both mean and log-variance
        out_mean, _ = predictions.chunk(2, dim=1)
    else:
        out_mean = predictions

    recons_loss_nm = -1 * nm_likelihood(out_mean, targets)[0].mean()
    recons_loss_gm = -1 * gaussian_likelihood(predictions, targets)[0].mean()
    recons_loss = nm_weight * recons_loss_nm + gaussian_weight * recons_loss_gm
    return recons_loss


def get_kl_divergence_loss_usplit(
    topdown_data: dict[str, list[torch.Tensor]], kl_key: str = "kl"
) -> torch.Tensor:
    """Compute the KL divergence loss for muSplit.

    Parameters
    ----------
    topdown_data : dict[str, list[torch.Tensor]]
        A dictionary containing information computed for each layer during the top-down
        pass. The dictionary must include the following keys:
        - "kl": The KL-loss values for each layer. Shape of each tensor is (B,).
        - "z": The sampled latents for each layer. Shape of each tensor is
        (B, layers, `z_dims[i]`, H, W).
    kl_key : str
        The key for the KL-loss values in the top-down layer data dictionary.
        To choose among ["kl", "kl_restricted", "kl_spatial", "kl_channelwise"]
        Default is "kl".
    """
    kl = torch.cat(
        [kl_layer.unsqueeze(1) for kl_layer in topdown_data[kl_key]], dim=1
    )  # shape: (B, n_layers)
    # NOTE: Values are sum() and so are of the order 30000

    nlayers = kl.shape[1]
    for i in range(nlayers):
        # NOTE: we want to normalize the KL-loss w.r.t. the latent space dimensions,
        # i.e., the number of entries in the latent space tensors (C, [Z], Y, X).
        # We assume z has shape (B, C, [Z], Y, X), where `C = z_dims[i]`.
        norm_factor = np.prod(topdown_data["z"][i].shape[1:])
        kl[:, i] = kl[:, i] / norm_factor

    kl_loss = free_bits_kl(kl, 0.0).mean()  # shape: (1, )
    # NOTE: free_bits disabled!
    return kl_loss


def get_kl_divergence_loss_denoisplit(
    topdown_data: dict[str, torch.Tensor],
    img_shape: tuple[int],
    kl_key: str = "kl",
) -> torch.Tensor:
    """Compute the KL divergence loss for denoiSplit.

    Parameters
    ----------
    topdown_data : dict[str, torch.Tensor]
        A dictionary containing information computed for each layer during the top-down
        pass. The dictionary must include the following keys:
        - "kl": The KL-loss values for each layer. Shape of each tensor is (B,).
        - "z": The sampled latents for each layer. Shape of each tensor is
        (B, layers, `z_dims[i]`, H, W).
    img_shape : tuple[int]
        The shape of the input image to the LVAE model. Shape is ([Z], Y, X).
    kl_key : str
        The key for the KL-loss values in the top-down layer data dictionary.
        To choose among ["kl", "kl_restricted", "kl_spatial", "kl_channelwise"]
        Default is "kl"

    kl[i] for each i has length batch_size resulting kl shape: (bs, layers).
    """
    kl = torch.cat(
        [kl_layer.unsqueeze(1) for kl_layer in topdown_data[kl_key]],
        dim=1,
    )

    kl_loss = free_bits_kl(kl, 1.0).sum()
    # NOTE: as compared to uSplit kl divergence, this KL loss is larger by a factor of
    # `n_layers` since we sum KL contributions from different layers instead of taking
    # the mean.

    # NOTE: at each hierarchy, the KL loss is larger by a factor of (128/i**2).
    # 128/(2*2) = 32 (bottommost layer)
    # 128/(4*4) = 8
    # 128/(8*8) = 2
    # 128/(16*16) = 0.5 (topmost layer)

    # Normalize the KL-loss w.r.t. the input image spatial dimensions (e.g., 64x64)
    kl_loss = kl_loss / np.prod(img_shape)
    return kl_loss


# TODO: @melisande-c suggested to refactor this as a class (see PR #208)
# - loss computation happens by calling the `__call__` method
# - `__init__` method initializes the loss parameters now contained in
# the `LVAELossParameters` class
# NOTE: same for the other loss functions
def musplit_loss(
    model_outputs: tuple[torch.Tensor, dict[str, Any]],
    targets: torch.Tensor,
    loss_parameters: LVAELossParameters,
) -> Optional[dict[str, torch.Tensor]]:
    """Loss function for muSplit.

    Parameters
    ----------
    model_outputs : tuple[torch.Tensor, dict[str, Any]]
        Tuple containing the model predictions (shape is (B, `target_ch`, [Z], Y, X))
        and the top-down layer data (e.g., sampled latents, KL-loss values, etc.).
    targets : torch.Tensor
        The target image used to compute the reconstruction loss. Shape is
        (B, `target_ch`, [Z], Y, X).
    loss_parameters : LVAELossParameters
        The loss parameters for muSplit (e.g., KL hyperparameters, likelihood module,
        noise model, etc.).

    Returns
    -------
    output : Optional[dict[str, torch.Tensor]]
        A dictionary containing the overall loss `["loss"]`, the reconstruction loss
        `["reconstruction_loss"]`, and the KL divergence loss `["kl_loss"]`.
    """
    predictions, td_data = model_outputs

    # Reconstruction loss computation
    recons_loss_dict = get_reconstruction_loss(
        reconstruction=predictions,
        target=targets,
        likelihood_obj=loss_parameters.gaussian_likelihood,
    )
    recons_loss = recons_loss_dict["loss"] * loss_parameters.reconstruction_weight
    if torch.isnan(recons_loss).any():
        recons_loss = 0.0

    # KL loss computation
    kl_weight = get_kl_weight(
        loss_parameters.kl_annealing,
        loss_parameters.kl_start,
        loss_parameters.kl_annealtime,
        loss_parameters.kl_weight,
        loss_parameters.current_epoch,
    )
    kl_loss = kl_weight * get_kl_divergence_loss_usplit(td_data)

    net_loss = recons_loss + kl_loss
    output = {
        "loss": net_loss,
        "reconstruction_loss": (
            recons_loss.detach()
            if isinstance(recons_loss, torch.Tensor)
            else recons_loss
        ),
        "kl_loss": kl_loss.detach(),
    }
    # https://github.com/openai/vdvae/blob/main/train.py#L26
    if torch.isnan(net_loss).any():
        return None

    return output


def denoisplit_loss(
    model_outputs: tuple[torch.Tensor, dict[str, Any]],
    targets: torch.Tensor,
    loss_parameters: LVAELossParameters,
) -> Optional[dict[str, torch.Tensor]]:
    """Loss function for DenoiSplit.

    Parameters
    ----------
    model_outputs : tuple[torch.Tensor, dict[str, Any]]
        Tuple containing the model predictions (shape is (B, `target_ch`, [Z], Y, X))
        and the top-down layer data (e.g., sampled latents, KL-loss values, etc.).
    targets : torch.Tensor
        The target image used to compute the reconstruction loss. Shape is
        (B, `target_ch`, [Z], Y, X).
    loss_parameters : LVAELossParameters
        The loss parameters for muSplit (e.g., KL hyperparameters, likelihood module,
        noise model, etc.).

    Returns
    -------
    output : Optional[dict[str, torch.Tensor]]
        A dictionary containing the overall loss `["loss"]`, the reconstruction loss
        `["reconstruction_loss"]`, and the KL divergence loss `["kl_loss"]`.
    """
    predictions, td_data = model_outputs

    # Reconstruction loss computation
    recons_loss_dict = get_reconstruction_loss(
        reconstruction=predictions,
        target=targets,
        likelihood_obj=loss_parameters.noise_model_likelihood,
    )
    recons_loss = recons_loss_dict["loss"] * loss_parameters.reconstruction_weight
    if torch.isnan(recons_loss).any():
        recons_loss = 0.0

    # KL loss computation
    if loss_parameters.non_stochastic:  # TODO always false ?
        kl_loss = torch.Tensor([0.0]).cuda()
    else:
        kl_weight = get_kl_weight(
            loss_parameters.kl_annealing,
            loss_parameters.kl_start,
            loss_parameters.kl_annealtime,
            loss_parameters.kl_weight,
            loss_parameters.current_epoch,
        )
        kl_loss = kl_weight * get_kl_divergence_loss_denoisplit(
            topdown_data=td_data,
            img_shape=targets.shape[2:],  # input img spatial dims
        )

    net_loss = recons_loss + kl_loss
    output = {
        "loss": net_loss,
        "reconstruction_loss": (
            recons_loss.detach()
            if isinstance(recons_loss, torch.Tensor)
            else recons_loss
        ),
        "kl_loss": kl_loss.detach(),
    }
    # https://github.com/openai/vdvae/blob/main/train.py#L26
    if torch.isnan(net_loss).any():
        return None

    return output


def denoisplit_musplit_loss(
    model_outputs: tuple[torch.Tensor, dict[str, Any]],
    targets: torch.Tensor,
    loss_parameters: LVAELossParameters,
) -> Optional[dict[str, torch.Tensor]]:
    """Loss function for DenoiSplit.

    Parameters
    ----------
    model_outputs : tuple[torch.Tensor, dict[str, Any]]
        Tuple containing the model predictions (shape is (B, `target_ch`, [Z], Y, X))
        and the top-down layer data (e.g., sampled latents, KL-loss values, etc.).
    targets : torch.Tensor
        The target image used to compute the reconstruction loss. Shape is
        (B, `target_ch`, [Z], Y, X).
    loss_parameters : LVAELossParameters
        The loss parameters for muSplit (e.g., KL hyperparameters, likelihood module,
        noise model, etc.).

    Returns
    -------
    output : Optional[dict[str, torch.Tensor]]
        A dictionary containing the overall loss `["loss"]`, the reconstruction loss
        `["reconstruction_loss"]`, and the KL divergence loss `["kl_loss"]`.
    """
    predictions, td_data = model_outputs

    # Reconstruction loss computation
    recons_loss = reconstruction_loss_musplit_denoisplit(
        predictions=predictions,
        targets=targets,
        nm_likelihood=loss_parameters.noise_model_likelihood,
        gaussian_likelihood=loss_parameters.gaussian_likelihood,
        nm_weight=loss_parameters.denoisplit_weight,
        gaussian_weight=loss_parameters.musplit_weight,
    )
    if torch.isnan(recons_loss).any():
        recons_loss = 0.0

    # KL loss computation
    if loss_parameters.non_stochastic:  # TODO always false ?
        kl_loss = torch.Tensor([0.0]).cuda()
    else:
        # NOTE: 'kl' key stands for the 'kl_samplewise' key in the TopDownLayer class.
        # The different naming comes from `top_down_pass()` method in the LadderVAE.
        denoisplit_kl = get_kl_divergence_loss_denoisplit(
            topdown_data=td_data,
            img_shape=targets.shape[2:],  # input img spatial dims
        )
        musplit_kl = get_kl_divergence_loss_usplit(td_data)
        kl_loss = (
            loss_parameters.denoisplit_weight * denoisplit_kl
            + loss_parameters.musplit_weight * musplit_kl
        )
        # TODO `kl_weight` is hardcoded (???)
        kl_loss = loss_parameters.kl_weight * kl_loss

    net_loss = recons_loss + kl_loss
    output = {
        "loss": net_loss,
        "reconstruction_loss": (
            recons_loss.detach()
            if isinstance(recons_loss, torch.Tensor)
            else recons_loss
        ),
        "kl_loss": kl_loss.detach(),
    }
    # https://github.com/openai/vdvae/blob/main/train.py#L26
    if torch.isnan(net_loss).any():
        return None

    return output
