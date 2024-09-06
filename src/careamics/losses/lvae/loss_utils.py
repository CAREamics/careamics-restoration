import torch


def free_bits_kl(
    kl: torch.Tensor, free_bits: float, batch_average: bool = False, eps: float = 1e-6
) -> torch.Tensor:
    """Compute free-bits version of KL divergence.

    This function ensures that the KL doesn't go to zero for any latent dimension.
    Hence, it contributes to use latent variables more efficiently, leading to
    better representation learning.

    NOTE:
    Takes in the KL with shape (batch size, layers), returns the KL with
    free bits (for optimization) with shape (layers,), which is the average
    free-bits KL per layer in the current batch.
    If batch_average is False (default), the free bits are per layer and
    per batch element. Otherwise, the free bits are still per layer, but
    are assigned on average to the whole batch. In both cases, the batch
    average is returned, so it's simply a matter of doing mean(clamp(KL))
    or clamp(mean(KL)).

    Parameters
    ----------
    kl : torch.Tensor
        The KL divergence tensor with shape (batch size, layers).
    free_bits : float
        The free bits value. Set to 0.0 to disable free bits.
    batch_average : bool
        Whether to average over the batch before clamping to `free_bits`.
    eps : float
        A small value to avoid numerical instability.

    Returns
    -------
    torch.Tensor
        The free-bits version of the KL divergence with shape (layers,).
    """
    assert kl.dim() == 2
    if free_bits < eps:
        return kl.mean(0)
    if batch_average:
        return kl.mean(0).clamp(min=free_bits)
    return kl.clamp(min=free_bits).mean(0)


def get_kl_weight(
    kl_annealing: bool,
    kl_start: int,
    kl_annealtime: int,
    kl_weight: float,
    current_epoch: int,
) -> float:
    """Compute the weight of the KL loss in case of annealing.

    Parameters
    ----------
    kl_annealing : bool
        Whether to use KL annealing.
    kl_start : int
        The epoch at which to start
    kl_annealtime : int
        The number of epochs for which annealing is applied.
    kl_weight : float
        The weight for the KL loss. If `None`, the weight is computed
        using annealing, else it is set to a default of 1.
    current_epoch : int
        The current epoch.
    """
    if kl_annealing:
        # calculate relative weight
        kl_weight = (current_epoch - kl_start) * (1.0 / kl_annealtime)
        # clamp to [0,1]
        kl_weight = min(max(0.0, kl_weight), 1.0)

        # if the final weight is given, then apply that weight on top of it
        if kl_weight is not None:
            kl_weight = kl_weight * kl_weight
    elif kl_weight is not None:
        return kl_weight
    else:
        kl_weight = 1.0
    return kl_weight
