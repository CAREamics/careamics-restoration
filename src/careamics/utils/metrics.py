"""
Metrics submodule.

This module contains various metrics and a metrics tracking class.
"""

# NOTE: this doesn't work with torch tensors, since `torch` refuses to
# compute the `mean()` or `std()` of a tensor whose dtype is not float.

from typing import Union
from warnings import warn

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio

Array = Union[np.ndarray, torch.Tensor]


def psnr(gt: Array, pred: Array, range: float = 255.0) -> float:
    """
    Peak Signal to Noise Ratio.

    This method calls skimage.metrics.peak_signal_noise_ratio. See:
    https://scikit-image.org/docs/dev/api/skimage.metrics.html.

    Parameters
    ----------
    gt : Array
        Ground truth image.
    pred : Array
        Predicted image.
    range : float, optional
        The images pixel range, by default 255.0.

    Returns
    -------
    float
        PSNR value.
    """
    # TODO: replace with explicit formula (?) it'd be a couple lines of code
    # and won't impact performance. On the contrary it would make the code
    # more explicit and easier to test.
    return peak_signal_noise_ratio(
        np.asarray(gt),
        np.asarray(pred),
        data_range=range,
    )


def _zero_mean(x: Array) -> Array:
    """
    Zero the mean of an array.

    NOTE: `torch` does not support the `mean()` method for tensors whose
    `dtype` is not `float`. Hence, this function will raise a warning and
    automatically cast the input tensor to `float` if it is a `torch.Tensor`.

    Parameters
    ----------
    x : Array
        Input array.

    Returns
    -------
    Array
        Zero-mean array.
    """
    x = _torch_cast_to_double(x)
    return x - x.mean()


def _fix_range(gt: Array, x: Array) -> Array:
    """
    Adjust the range of an array based on a reference ground-truth array.

    Parameters
    ----------
    gt : Array
        Ground truth image.
    x : Array
        Input array.

    Returns
    -------
    Array
        Range-adjusted array.
    """
    a = (gt * x).sum() / (x * x).sum()
    return x * a


def _fix(gt: Array, x: Array) -> Array:
    """
    Zero mean a groud truth array and adjust the range of the array.

    Parameters
    ----------
    gt : Array
        Ground truth image.
    x : Array
        Input array.

    Returns
    -------
    Array
        Zero-mean and range-adjusted array.
    """
    gt_ = _zero_mean(gt)
    return _fix_range(gt_, _zero_mean(x))


def scale_invariant_psnr(gt: Array, pred: Array) -> Union[float, torch.tensor]:
    """
    Scale invariant PSNR.

    NOTE: `torch` does not support the `mean()` method for tensors whose
    `dtype` is not `float`. Hence, this function will raise a warning and
    automatically cast the input tensor to `float` if it is a `torch.Tensor`.

    NOTE: results may vary slightly between `numpy` and `torch` due to the way
    `var()` is computed. In `torch`, the unbiased estimator is used (i.e., SSE/n-1),
    while in `numpy` the biased estimator is used (i.e., SSE/n).

    Parameters
    ----------
    gt : Array
        Ground truth image.
    pred : Array
        Predicted image.

    Returns
    -------
    Union[float, torch.tensor]
        Scale invariant PSNR value.
    """
    # cast tensors to double dtype
    gt = _torch_cast_to_double(gt)
    pred = _torch_cast_to_double(pred)
    # compute scale-invariant PSNR
    range_parameter = (gt.max() - gt.min()) / gt.std()
    gt_ = _zero_mean(gt) / gt.std()
    return psnr(_zero_mean(gt_), _fix(gt_, pred), range_parameter)


def _torch_cast_to_double(x: Array) -> Array:
    """
    Cast a tensor to float.

    Parameters
    ----------
    x : Array
        Input tensor.

    Returns
    -------
    Array
        Float tensor.
    """
    if isinstance(x, torch.Tensor) and x.dtype != torch.float64:
        warn(
            f"Casting tensor  of type `{x.dtype}` to double (`torch.float64`).",
            UserWarning,
        )
        return x.double()
    return x
