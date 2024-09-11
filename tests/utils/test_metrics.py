import numpy as np
import pytest
import torch

from careamics.utils.metrics import (
    _zero_mean,
    multiscale_ssim,
    psnr,
    scale_invariant_psnr,
)

# TODO: add missing test cases
# - `psnr` result
# - `_fix` and `fix_range`
# - `RunningPSNR`
# - `multiscale_ssim` result


@pytest.mark.parametrize(
    "x",
    [
        np.array([1, 2, 3, 4, 5]),
        np.array([[1, 2, 3], [4, 5, 6]]),
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    ],
)
def test_zero_mean(x: np.ndarray):
    assert np.allclose(_zero_mean(x), x - x.mean())


# TODO: with 2 identical arrays, shouldn't the result be `inf`?
@pytest.mark.parametrize(
    "gt, pred, result",
    [
        (np.array([1, 2, 3, 4, 5, 6]), np.array([1, 2, 3, 4, 5, 6]), 332.22),
        (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2, 3], [4, 5, 6]]), 332.22),
    ],
)
def test_scale_invariant_psnr(gt: np.ndarray, pred: np.ndarray, result: float):
    assert scale_invariant_psnr(gt, pred) == pytest.approx(result, rel=5e-3)


@pytest.mark.parametrize(
    "data_type",
    [np.uint8, np.uint16, np.float32, np.int64],
)
def test_psnr(data_type: np.dtype):
    gt_ = np.random.randint(0, 255, size=(16, 16)).astype(data_type)
    pred_ = np.random.randint(0, 255, size=(16, 16)).astype(data_type)

    assert psnr(gt_, pred_, 255) is not None

    range_ = gt_.max() - gt_.min()
    assert psnr(gt_, pred_, range_) is not None
    # add check on the result


@pytest.mark.parametrize("type_", ["torch", "numpy"])
@pytest.mark.parametrize("num_ch", [1, 4])
def test_multiscale_ssim(type_: str, num_ch: int):
    if type_ == "torch":
        gt = torch.rand(4, 256, 256, num_ch)
        pred = torch.rand(4, 256, 256, num_ch)
    elif type_ == "numpy":
        gt = np.random.rand(4, 256, 256, num_ch)
        pred = np.random.rand(4, 256, 256, num_ch)

    rinv_mssim = multiscale_ssim(gt, pred, range_invariant=True)
    mssim = multiscale_ssim(gt, pred, range_invariant=False)
    assert len(rinv_mssim) == num_ch
    assert len(mssim) == num_ch
