"""
Metrics submodule.

This module contains various metrics and a metrics tracking class.
"""

from typing import Optional, Union

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio


def psnr(gt: np.ndarray, pred: np.ndarray, data_range: float) -> float:
    """Peak Signal to Noise Ratio.

    This method calls skimage.metrics.peak_signal_noise_ratio. See:
    https://scikit-image.org/docs/dev/api/skimage.metrics.html.

    NOTE: to avoid unwanted behaviors (e.g., data_range inferred from array dtype),
    the data_range parameter is mandatory.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth array.
    pred : np.ndarray
        Predicted array.
    data_range : float
        The images pixel range.

    Returns
    -------
    float
        PSNR value.
    """
    if data_range is None:
        raise ValueError("data_range must be provided")
    return peak_signal_noise_ratio(gt, pred, data_range=data_range)


def _zero_mean(x: np.ndarray) -> np.ndarray:
    """
    Zero the mean of an array.

    Parameters
    ----------
    x : NumPy array
        Input array.

    Returns
    -------
    NumPy array
        Zero-mean array.
    """
    return x - np.mean(x)


def _fix_range(gt: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Adjust the range of an array based on a reference ground-truth array.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth image.
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Range-adjusted array.
    """
    a = np.sum(gt * x) / (np.sum(x * x))
    return x * a


def _fix(gt: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Zero mean a groud truth array and adjust the range of the array.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth image.
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Zero-mean and range-adjusted array.
    """
    gt_ = _zero_mean(gt)
    return _fix_range(gt_, _zero_mean(x))


def scale_invariant_psnr(
    gt: np.ndarray, pred: np.ndarray
) -> Union[float, torch.tensor]:
    """
    Scale invariant PSNR.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth image.
    pred : np.ndarray
        Predicted image.

    Returns
    -------
    Union[float, torch.tensor]
        Scale invariant PSNR value.
    """
    range_parameter = (np.max(gt) - np.min(gt)) / np.std(gt)
    gt_ = _zero_mean(gt) / np.std(gt)
    return psnr(_zero_mean(gt_), _fix(gt_, pred), range_parameter)


class RunningPSNR:
    """Compute the running PSNR during validation step in training.

    This class allows to compute the PSNR on the entire validation set
    one batch at the time.

    Attributes
    ----------
    N : int
        Number of elements seen so far during the epoch.
    mse_sum : float
        Running sum of the MSE over the N elements seen so far.
    max : float
        Running max value of the N target images seen so far.
    min : float
        Running min value of the N target images seen so far.
    """

    def __init__(self):
        """Constructor."""
        self.N = None
        self.mse_sum = None
        self.max = self.min = None
        self.reset()

    def reset(self):
        """Reset the running PSNR computation.

        Usually called at the end of each epoch.
        """
        self.mse_sum = 0
        self.N = 0
        self.max = self.min = None

    def update(self, rec: torch.Tensor, tar: torch.Tensor) -> None:
        """Update the running PSNR statistics given a new batch.

        Parameters
        ----------
        rec : torch.Tensor
            Reconstructed batch.
        tar : torch.Tensor
            Target batch.
        """
        ins_max = torch.max(tar).item()
        ins_min = torch.min(tar).item()
        if self.max is None:
            assert self.min is None
            self.max = ins_max
            self.min = ins_min
        else:
            self.max = max(self.max, ins_max)
            self.min = min(self.min, ins_min)

        mse = (rec - tar) ** 2
        elementwise_mse = torch.mean(mse.view(len(mse), -1), dim=1)
        self.mse_sum += torch.nansum(elementwise_mse)
        self.N += len(elementwise_mse) - torch.sum(torch.isnan(elementwise_mse))

    def get(self) -> Optional[torch.Tensor]:
        """Get the actual PSNR value given the running statistics.

        Returns
        -------
        Optional[torch.Tensor]
            PSNR value.
        """
        if self.N == 0 or self.N is None:
            return None
        rmse = torch.sqrt(self.mse_sum / self.N)
        return 20 * torch.log10((self.max - self.min) / rmse)
