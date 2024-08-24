import numpy as np
import pytest
import torch

from careamics.utils.metrics import (
    _zero_mean,
    scale_invariant_psnr,
)

# TODO: add tests for cudaTensors


@pytest.mark.parametrize(
    "x",
    [
        np.array([1, 2, 3, 4, 5]),
        np.array([[1, 2, 3], [4, 5, 6]]),
        torch.tensor([1, 2, 3, 4, 5]),
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
    ],
)
def test_zero_mean(x):
    x = np.asarray(x)
    assert np.allclose(_zero_mean(x), x - np.mean(x))


# NOTE: the behavior of the PSNR function for np.arrays is weird. Indeed, PSNR computed over
# identical vectors should be infinite, but the function returns a finite value.
# Using torch it gives instead `inf`.
@pytest.mark.parametrize(
    "gt, pred, result",
    [
        (np.array([1, 2, 3, 4, 5, 6]), np.array([1, 2, 3, 4, 5, 6]), 332.22),
        (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2, 3], [4, 5, 6]]), 332.22),
        (torch.tensor([1, 2, 3, 4, 5, 6]), torch.tensor([1, 2, 3, 4, 5, 6]), 332.22),
        (
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            332.22,
        ),
    ],
)
def test_scale_invariant_psnr(gt, pred, result):
    assert scale_invariant_psnr(gt, pred) == pytest.approx(result, rel=5e-3)


# TODO: add tests for RunningPSNR
