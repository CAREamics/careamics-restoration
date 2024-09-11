"""
Metrics submodule.

This module contains various metrics and a metrics tracking class.
"""

from typing import Callable, Optional, Union

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

# TODO: does this add additional dependency?


def psnr(gt: np.ndarray, pred: np.ndarray, data_range: float) -> float:
    """
    Peak Signal to Noise Ratio.

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
    return peak_signal_noise_ratio(gt, pred, data_range=data_range)


def _zero_mean(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Zero the mean of an array.

    Parameters
    ----------
    x : numpy.ndarray or torch.Tensor
        Input array.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Zero-mean array.
    """
    return x - x.mean()


def _fix_range(
    gt: Union[np.ndarray, torch.Tensor], x: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Adjust the range of an array based on a reference ground-truth array.

    Parameters
    ----------
    gt : Union[np.ndarray, torch.Tensor]
        Ground truth array.
    x : Union[np.ndarray, torch.Tensor]
        Input array.

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Range-adjusted array.
    """
    a = (gt * x).sum() / (x * x).sum()
    return x * a


def _fix(
    gt: Union[np.ndarray, torch.Tensor], x: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Zero mean a groud truth array and adjust the range of the array.

    Parameters
    ----------
    gt : Union[np.ndarray, torch.Tensor]
        Ground truth image.
    x : Union[np.ndarray, torch.Tensor]
        Input array.

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
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


def _range_invariant_multiscale_ssim(
    gt_: Union[np.ndarray, torch.Tensor], pred_: Union[np.ndarray, torch.Tensor]
) -> float:
    """Compute range invariant multiscale SSIM for a single channel.

    The advantage of this metric in comparison to commonly used SSIM is that
    it is invariant to scalar multiplications in the prediction.
    # TODO: Add reference to the paper.

    NOTE: images fed to this function should have channels dimension as the last one.

    Parameters
    ----------
    gt_ : Union[np.ndarray, torch.Tensor]
        Ground truth image with shape (N, H, W).
    pred_ : Union[np.ndarray, torch.Tensor]
        Predicted image with shape (N, H, W).

    Returns
    -------
    float
        Range invariant multiscale SSIM value.
    """
    shape = gt_.shape
    gt_ = torch.Tensor(gt_.reshape((shape[0], -1)))
    pred_ = torch.Tensor(pred_.reshape((shape[0], -1)))
    gt_ = _zero_mean(gt_)
    pred_ = _zero_mean(pred_)
    pred_ = _fix(gt_, pred_)
    pred_ = pred_.reshape(shape)
    gt_ = gt_.reshape(shape)

    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=gt_.max() - gt_.min()
    )
    return ms_ssim(torch.Tensor(pred_[:, None]), torch.Tensor(gt_[:, None])).item()


def multiscale_ssim(
    gt_: Union[np.ndarray, torch.Tensor],
    pred_: Union[np.ndarray, torch.Tensor],
    range_invariant: bool = True,
) -> list[Union[float, None]]:
    """Compute channel-wise multiscale SSIM for each channel.

    It allows to use either standard multiscale SSIM or its range-invariant version.

    NOTE: images fed to this function should have channels dimension as the last one.
    # TODO: do we want to allow this behavior? or we want the usual (N, C, H, W)?

    Parameters
    ----------
    gt_ : Union[np.ndarray, torch.Tensor]
        Ground truth image with shape (N, H, W, C).
    pred_ : Union[np.ndarray, torch.Tensor]
        Predicted image with shape (N, H, W, C).
    range_invariant : bool
        Whether to use standard or range invariant multiscale SSIM.

    Returns
    -------
    list[float]
        List of SSIM values for each channel.
    """
    ms_ssim_values = {}
    for ch_idx in range(gt_.shape[-1]):
        tar_tmp = gt_[..., ch_idx]
        pred_tmp = pred_[..., ch_idx]
        if range_invariant:
            ms_ssim_values[ch_idx] = _range_invariant_multiscale_ssim(
                gt_=tar_tmp, pred_=pred_tmp
            )
        else:
            ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
                data_range=tar_tmp.max() - tar_tmp.min()
            )
            ms_ssim_values[ch_idx] = ms_ssim(
                torch.Tensor(pred_tmp[:, None]), torch.Tensor(tar_tmp[:, None])
            ).item()

    return [ms_ssim_values[i] for i in range(gt_.shape[-1])]  # type: ignore


def _avg_psnr(target: np.ndarray, prediction: np.ndarray, psnr_fn: Callable) -> float:
    """Compute the average PSNR over a batch of images.

    Parameters
    ----------
    target : np.ndarray
        Array of ground truth images, shape is (N, C, H, W).
    prediction : np.ndarray
        Array of predicted images, shape is (N, C, H, W).
    psnr_fn : Callable
        PSNR function to use.

    Returns
    -------
    float
        Average PSNR value over the batch.
    """
    return np.mean(
        [
            psnr_fn(target[i : i + 1], prediction[i : i + 1]).item()
            for i in range(len(prediction))
        ]
    )


def avg_range_inv_psnr(target: np.ndarray, prediction: np.ndarray) -> float:
    """Compute the average range-invariant PSNR over a batch of images.

    Parameters
    ----------
    target : np.ndarray
        Array of ground truth images, shape is (N, C, H, W).
    prediction : np.ndarray
        Array of predicted images, shape is (N, C, H, W).

    Returns
    -------
    float
        Average range-invariant PSNR value over the batch.
    """
    return _avg_psnr(target, prediction, scale_invariant_psnr)


def avg_psnr(target: np.ndarray, prediction: np.ndarray) -> float:
    """Compute the average PSNR over a batch of images.

    Parameters
    ----------
    target : np.ndarray
        Array of ground truth images, shape is (N, C, H, W).
    prediction : np.ndarray
        Array of predicted images, shape is (N, C, H, W).

    Returns
    -------
    float
        Average PSNR value over the batch.
    """
    return _avg_psnr(target, prediction, psnr)


def avg_ssim(
    target: Union[np.ndarray, torch.Tensor], prediction: Union[np.ndarray, torch.Tensor]
) -> tuple[float, float]:
    """Compute the average Structural Similarity (SSIM) over a batch of images.

    Parameters
    ----------
    target : np.ndarray
        Array of ground truth images, shape is (N, C, H, W).
    prediction : np.ndarray
        Array of predicted images, shape is (N, C, H, W).

    Returns
    -------
    tuple[float, float]
        Mean and standard deviation of SSIM values over the batch.
    """
    ssim = [
        structural_similarity(
            target[i], prediction[i], data_range=(target[i].max() - target[i].min())
        )
        for i in range(len(target))
    ]
    return np.mean(ssim), np.std(ssim)
