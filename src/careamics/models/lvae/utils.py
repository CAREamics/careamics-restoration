"""
Script for utility functions needed by the LVAE model.
"""

from collections.abc import Iterable
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.distributions.normal import Normal


def torch_nanmean(inp):
    return torch.mean(inp[~inp.isnan()])


def compute_batch_mean(x):
    N = len(x)
    return x.view(N, -1).mean(dim=1)


def power_of_2(self, x):
    assert isinstance(x, int)
    if x == 1:
        return True
    if x == 0:
        # happens with validation
        return False
    if x % 2 == 1:
        return False
    return self.power_of_2(x // 2)


class Enum:
    @classmethod
    def name(cls, enum_type):
        for key, value in cls.__dict__.items():
            if enum_type == value:
                return key

    @classmethod
    def contains(cls, enum_type):
        for key, value in cls.__dict__.items():
            if enum_type == value:
                return True
        return False

    @classmethod
    def from_name(cls, enum_type_str):
        for key, value in cls.__dict__.items():
            if key == enum_type_str:
                return value
        assert f"{cls.__name__}:{enum_type_str} doesnot exist."


class LossType(Enum):
    Elbo = 0
    ElboWithCritic = 1
    ElboMixedReconstruction = 2
    MSE = 3
    ElboWithNbrConsistency = 4
    ElboSemiSupMixedReconstruction = 5
    ElboCL = 6
    ElboRestrictedReconstruction = 7
    DenoiSplitMuSplit = 8


class ModelType(Enum):
    LadderVae = 3
    LadderVaeTwinDecoder = 4
    LadderVAECritic = 5
    # Separate vampprior: two optimizers
    LadderVaeSepVampprior = 6
    # one encoder for mixed input, two for separate inputs.
    LadderVaeSepEncoder = 7
    LadderVAEMultiTarget = 8
    LadderVaeSepEncoderSingleOptim = 9
    UNet = 10
    BraveNet = 11
    LadderVaeStitch = 12
    LadderVaeSemiSupervised = 13
    LadderVaeStitch2Stage = 14  # Note that previously trained models will have issue.
    # since earlier, LadderVaeStitch2Stage = 13, LadderVaeSemiSupervised = 14
    LadderVaeMixedRecons = 15
    LadderVaeCL = 16
    LadderVaeTwoDataSet = (
        17  # on one subdset, apply disentanglement, on other apply reconstruction
    )
    LadderVaeTwoDatasetMultiBranch = 18
    LadderVaeTwoDatasetMultiOptim = 19
    LVaeDeepEncoderIntensityAug = 20
    AutoRegresiveLadderVAE = 21
    LadderVAEInterleavedOptimization = 22
    Denoiser = 23
    DenoiserSplitter = 24
    SplitterDenoiser = 25
    LadderVAERestrictedReconstruction = 26
    LadderVAETwoDataSetRestRecon = 27
    LadderVAETwoDataSetFinetuning = 28


def _pad_crop_img(
    x: torch.Tensor, size: Iterable[int], mode: Literal["crop", "pad"]
) -> torch.Tensor:
    """Pads or crops a tensor.

    Pads or crops a tensor of shape (B, C, [Z], Y, X) to new shape.

    Parameters
    ----------
        x (torch.Tensor): Input image of shape (B, C, [Z], Y, X)
        size (list or tuple): Desired size ([Z*], Y*, X*)
        mode (str): Mode, either 'pad' or 'crop'

    Returns
    -------
        The padded or cropped tensor
    """
    # TODO: Support cropping/padding on selected dimensions
    assert (x.dim() == 4 and len(size) == 2) or (x.dim() == 5 and len(size) == 3)
    # assert x.dim() in [4,5] and len(size) == 2

    size = tuple(size)
    x_size = x.size()[2:]

    if mode == "pad":
        cond = any(x_size[i] > size[i] for i in range(len(size)))
    elif mode == "crop":
        cond = any(x_size[i] < size[i] for i in range(len(size)))

    if cond:
        raise ValueError(f"Trying to {mode} from size {x_size} to size {size}")

    diffs = [abs(x - s) for x, s in zip(x_size, size)]
    d1 = [d // 2 for d in diffs]
    d2 = [d - (d // 2) for d in diffs]

    if mode == "pad":
        if x.dim() == 4:
            padding = [d1[1], d2[1], d1[0], d2[0], 0, 0, 0, 0]
        elif x.dim() == 5:
            padding = [d1[2], d2[2], d1[1], d2[1], d1[0], d2[0], 0, 0, 0, 0]
        return nn.functional.pad(x, padding)
    elif mode == "crop":
        if x.dim() == 4:
            return x[:, :, d1[0] : (x_size[0] - d2[0]), d1[1] : (x_size[1] - d2[1])]
        elif x.dim() == 5:
            return x[
                :,
                :,
                d1[0] : (x_size[0] - d2[0]),
                d1[1] : (x_size[1] - d2[1]),
                d1[2] : (x_size[2] - d2[2]),
            ]


def pad_img_tensor(x: torch.Tensor, size: Iterable[int]) -> torch.Tensor:
    """Pads a tensor

    Pads a tensor of shape (B, C, [Z], Y, X) to desired spatial dimensions.

    Parameters
    ----------
        x (torch.Tensor): Input image of shape (B, C, [Z], Y, X)
        size (list or tuple): Desired size  ([Z*], Y*, X*)

    Returns
    -------
        The padded tensor
    """
    return _pad_crop_img(x, size, "pad")


def crop_img_tensor(x, size) -> torch.Tensor:
    """Crops a tensor.
    Crops a tensor of shape (batch, channels, h, w) to a desired height and width
    given by a tuple.
    Args:
        x (torch.Tensor): Input image
        size (list or tuple): Desired size (height, width)

    Returns
    -------
        The cropped tensor
    """
    return _pad_crop_img(x, size, "crop")


class StableExponential:
    """
    Here, the idea is that everything is done on the tensor which you've given in the constructor.
    when exp() is called, what that means is that we want to compute self._tensor.exp()
    when log() is called, we want to compute torch.log(self._tensor.exp())

    What is done here is that definition of exp() has been changed. This, naturally, has changed the result of log.
    but the log is still the mathematical log, that is, it takes the math.log() on whatever comes out of exp().
    """

    def __init__(self, tensor):
        self._raw_tensor = tensor
        posneg_dic = self.posneg_separation(self._raw_tensor)
        self.pos_f, self.neg_f = posneg_dic["filter"]
        self.pos_data, self.neg_data = posneg_dic["value"]

    def posneg_separation(self, tensor):
        pos = tensor > 0
        pos_tensor = torch.clip(tensor, min=0)

        neg = tensor <= 0
        neg_tensor = torch.clip(tensor, max=0)

        return {"filter": [pos, neg], "value": [pos_tensor, neg_tensor]}

    def exp(self):
        return torch.exp(self.neg_data) * self.neg_f + (1 + self.pos_data) * self.pos_f

    def log(self):
        """
        Note that if you have the output from exp(). You could simply apply torch.log() on it and that should give
        identical numbers.
        """
        return self.neg_data * self.neg_f + torch.log(1 + self.pos_data) * self.pos_f


class StableLogVar:

    def __init__(self, logvar, enable_stable=True, var_eps=1e-6):
        """
        Args:
            var_eps: var() has this minimum value.
        """
        self._lv = logvar
        self._enable_stable = enable_stable
        self._eps = var_eps

    def get(self):
        if self._enable_stable is False:
            return self._lv

        return torch.log(self.get_var())

    def get_var(self):
        if self._enable_stable is False:
            return torch.exp(self._lv)
        return StableExponential(self._lv).exp() + self._eps

    def get_std(self):
        return torch.sqrt(self.get_var())

    def centercrop_to_size(self, size):
        if self._lv.shape[-1] == size:
            return

        diff = self._lv.shape[-1] - size
        assert diff > 0 and diff % 2 == 0
        self._lv = F.center_crop(self._lv, (size, size))


class StableMean:

    def __init__(self, mean):
        self._mean = mean

    def get(self):
        return self._mean

    def centercrop_to_size(self, size):
        if self._mean.shape[-1] == size:
            return

        diff = self._mean.shape[-1] - size
        assert diff > 0 and diff % 2 == 0
        self._mean = F.center_crop(self._mean, (size, size))


def allow_numpy(func):
    """
    All optional arguments are passed as is. positional arguments are checked. if they are numpy array,
    they are converted to torch Tensor.
    """

    def numpy_wrapper(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.Tensor(arg)
            new_args.append(arg)
        new_args = tuple(new_args)

        output = func(*new_args, **kwargs)
        return output

    return numpy_wrapper


class Interpolate(nn.Module):
    """Wrapper for torch.nn.functional.interpolate."""

    def __init__(self, size=None, scale=None, mode="bilinear", align_corners=False):
        super().__init__()
        assert (size is None) == (scale is not None)
        self.size = size
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        out = F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return out


def kl_normal_mc(z, p_mulv, q_mulv):
    """
    One-sample estimation of element-wise KL between two diagonal
    multivariate normal distributions. Any number of dimensions,
    broadcasting supported (be careful).
    :param z:
    :param p_mulv:
    :param q_mulv:
    :return:
    """
    assert isinstance(p_mulv, tuple)
    assert isinstance(q_mulv, tuple)
    p_mu, p_lv = p_mulv
    q_mu, q_lv = q_mulv

    p_std = p_lv.get_std()
    q_std = q_lv.get_std()

    p_distrib = Normal(p_mu.get(), p_std)
    q_distrib = Normal(q_mu.get(), q_std)
    return q_distrib.log_prob(z) - p_distrib.log_prob(z)


def free_bits_kl(
    kl: torch.Tensor, free_bits: float, batch_average: bool = False, eps: float = 1e-6
) -> torch.Tensor:
    """
    Computes free-bits version of KL divergence.
    Ensures that the KL doesn't go to zero for any latent dimension.
    Hence, it contributes to use latent variables more efficiently,
    leading to better representation learning.

    NOTE:
    Takes in the KL with shape (batch size, layers), returns the KL with
    free bits (for optimization) with shape (layers,), which is the average
    free-bits KL per layer in the current batch.
    If batch_average is False (default), the free bits are per layer and
    per batch element. Otherwise, the free bits are still per layer, but
    are assigned on average to the whole batch. In both cases, the batch
    average is returned, so it's simply a matter of doing mean(clamp(KL))
    or clamp(mean(KL)).

    Args:
        kl (torch.Tensor)
        free_bits (float)
        batch_average (bool, optional))
        eps (float, optional)

    Returns
    -------
        The KL with free bits
    """
    assert kl.dim() == 2
    if free_bits < eps:
        return kl.mean(0)
    if batch_average:
        return kl.mean(0).clamp(min=free_bits)
    return kl.clamp(min=free_bits).mean(0)
