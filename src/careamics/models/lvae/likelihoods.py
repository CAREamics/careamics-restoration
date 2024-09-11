"""
Script containing modules for defining different likelihood functions (as nn.Module).
"""

from __future__ import annotations

import math
from typing import Literal, Union, TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from torch import nn

from careamics.config.likelihood_model import (
    GaussianLikelihoodConfig,
    NMLikelihoodConfig,
)

if TYPE_CHECKING:
    from careamics.models.lvae.noise_models import (
        GaussianMixtureNoiseModel,
        MultiChannelNoiseModel,
    )

    NoiseModel = Union[GaussianMixtureNoiseModel, MultiChannelNoiseModel]


def likelihood_factory(
    config: Union[GaussianLikelihoodConfig, NMLikelihoodConfig, None]
):
    """
    Factory function for creating likelihood modules.

    Parameters
    ----------
    config: Union[GaussianLikelihoodConfig, NMLikelihoodConfig]
        The configuration object for the likelihood module.

    Returns
    -------
    nn.Module
        The likelihood module.
    """
    if config is None:
        return None

    if isinstance(config, GaussianLikelihoodConfig):
        return GaussianLikelihood(
            predict_logvar=config.predict_logvar,
            logvar_lowerbound=config.logvar_lowerbound,
        )
    elif isinstance(config, NMLikelihoodConfig):
        return NoiseModelLikelihood(
            data_mean=config.data_mean,
            data_std=config.data_std,
            noiseModel=config.noise_model,
        )
    else:
        raise ValueError(f"Invalid likelihood model type: {config.model_type}")


# TODO: is it really worth to have this class? Or it just adds complexity? --> REFACTOR
class LikelihoodModule(nn.Module):
    """
    The base class for all likelihood modules.
    It defines the fundamental structure and methods for specialized likelihood models.
    """

    def distr_params(self, x: Any) -> None:
        return None

    def set_params_to_same_device_as(self, correct_device_tensor: Any) -> None:
        pass

    @staticmethod
    def logvar(params: Any) -> None:
        return None

    @staticmethod
    def mean(params: Any) -> None:
        return None

    @staticmethod
    def mode(params: Any) -> None:
        return None

    @staticmethod
    def sample(params: Any) -> None:
        return None

    def log_likelihood(self, x: Any, params: Any) -> None:
        return None

    def get_mean_lv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]: ...

    def forward(
        self, input_: torch.Tensor, x: Union[torch.Tensor, None]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Parameters:
        -----------
        input_: torch.Tensor
            The output of the top-down pass (e.g., reconstructed image in HDN,
            or the unmixed images in 'Split' models).
        x: Union[torch.Tensor, None]
            The target tensor. If None, the log-likelihood is not computed.
        """
        distr_params = self.distr_params(input_)
        mean = self.mean(distr_params)
        mode = self.mode(distr_params)
        sample = self.sample(distr_params)
        logvar = self.logvar(distr_params)

        if x is None:
            ll = None
        else:
            ll = self.log_likelihood(x, distr_params)

        dct = {
            "mean": mean,
            "mode": mode,
            "sample": sample,
            "params": distr_params,
            "logvar": logvar,
        }

        return ll, dct


class GaussianLikelihood(LikelihoodModule):
    r"""A specialized `LikelihoodModule` for Gaussian likelihood.

    Specifically, in the LVAE model, the likelihood is defined as:
        p(x|z_1) = N(x|\mu_{p,1}, \sigma_{p,1}^2)
    """

    def __init__(
        self,
        predict_logvar: Union[Literal["pixelwise"], None] = None,
        logvar_lowerbound: Union[float, None] = None,
    ):
        """Constructor.

        Parameters
        ----------
        predict_logvar: Union[Literal["pixelwise"], None], optional
            If `pixelwise`, log-variance is computed for each pixel, else log-variance
            is not computed. Default is `None`.
        logvar_lowerbound: float, optional
            The lowerbound value for log-variance. Default is `None`.
        """
        super().__init__()

        self.predict_logvar = predict_logvar
        self.logvar_lowerbound = logvar_lowerbound
        assert self.predict_logvar in [None, "pixelwise"]

        print(
            f"[{self.__class__.__name__}] PredLVar:{self.predict_logvar} LowBLVar:{self.logvar_lowerbound}"
        )

    def get_mean_lv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Given the output of the top-down pass, compute the mean and log-variance of the
        Gaussian distribution defining the likelihood.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor to the likelihood module, i.e., the output of the top-down
            pass.

        Returns
        -------
        tuple of (torch.tensor, optional torch.tensor)
            The first element of the tuple is the mean, the second element is the
            log-variance. If the attribute `predict_logvar` is `None` then the second
            element will be `None`.
        """

        # if LadderVAE.predict_logvar is None, dim 1 of `x`` has no. of target channels
        if self.predict_logvar is None:
            return x, None

        # Get pixel-wise mean and logvar
        # if LadderVAE.predict_logvar is not None,
        #   dim 1 has double no. of target channels
        mean, lv = x.chunk(2, dim=1)

        # Optionally, clip log-var to a lower bound
        if self.logvar_lowerbound is not None:
            lv = torch.clip(lv, min=self.logvar_lowerbound)

        return mean, lv

    def distr_params(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Get parameters (mean, log-var) of the Gaussian distribution defined by the likelihood.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor to the likelihood module, i.e., the output
            the LVAE 'output_layer'. Shape is: (B, 2 * C, [Z], Y, X) in case
            `predict_logvar` is not None, or (B, C, [Z], Y, X) otherwise.
        """
        mean, lv = self.get_mean_lv(x)
        params = {
            "mean": mean,
            "logvar": lv,
        }
        return params

    @staticmethod
    def mean(params: dict[str, torch.Tensor]) -> torch.Tensor:
        return params["mean"]

    @staticmethod
    def mode(params: dict[str, torch.Tensor]) -> torch.Tensor:
        return params["mean"]

    @staticmethod
    def sample(params: dict[str, torch.Tensor]) -> torch.Tensor:
        # p = Normal(params['mean'], (params['logvar'] / 2).exp())
        # return p.rsample()
        return params["mean"]

    @staticmethod
    def logvar(params: dict[str, torch.Tensor]) -> torch.Tensor:
        return params["logvar"]

    def log_likelihood(
        self, x: torch.Tensor, params: dict[str, Union[torch.Tensor, None]]
    ):
        """Compute Gaussian log-likelihood

        Parameters
        ----------
        x: torch.Tensor
            The target tensor. Shape is (B, C, [Z], Y, X).
        params: dict[str, Union[torch.Tensor, None]]
            The tensors obtained by chunking the output of the top-down pass,
            here used as parameters of the Gaussian distribution.

        Returns
        -------
        torch.Tensor
            The log-likelihood tensor. Shape is (B, C, [Z], Y, X).
        """
        if self.predict_logvar is not None:
            logprob = log_normal(x, params["mean"], params["logvar"])
        else:
            logprob = -0.5 * (params["mean"] - x) ** 2
        return logprob


def log_normal(
    x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """
    Compute the log-probability at `x` of a Gaussian distribution
    with parameters `(mean, exp(logvar))`.

    NOTE: In the case of LVAE, the log-likeihood formula becomes:
        \\mathbb{E}_{z_1\\sim{q_\\phi}}[\\log{p_\theta(x|z_1)}]=-\frac{1}{2}(\\mathbb{E}_{z_1\\sim{q_\\phi}}[\\log{2\\pi\\sigma_{p,0}^2(z_1)}] +\\mathbb{E}_{z_1\\sim{q_\\phi}}[\frac{(x-\\mu_{p,0}(z_1))^2}{\\sigma_{p,0}^2(z_1)}])

    Parameters
    ----------
    x: torch.Tensor
        The ground-truth tensor. Shape is (batch, channels, dim1, dim2).
    mean: torch.Tensor
        The inferred mean of distribution. Shape is (batch, channels, dim1, dim2).
    logvar: torch.Tensor
        The inferred log-variance of distribution. Shape has to be either scalar or broadcastable.
    """
    var = torch.exp(logvar)
    log_prob = -0.5 * (
        ((x - mean) ** 2) / var + logvar + torch.tensor(2 * math.pi).log()
    )
    return log_prob


class NoiseModelLikelihood(LikelihoodModule):

    def __init__(
        self,
        data_mean: Union[np.ndarray, torch.Tensor],
        data_std: Union[np.ndarray, torch.Tensor],
        noiseModel: NoiseModel,
    ):
        """Constructor.

        Parameters
        ----------
        data_mean: Union[np.ndarray, torch.Tensor]
            The mean of the data, used to unnormalize data for noise model evaluation.
        data_std: Union[np.ndarray, torch.Tensor]
            The standard deviation of the data, used to unnormalize data for noise
            model evaluation.
        noiseModel: NoiseModel
            The noise model instance used to compute the likelihood.
        """
        super().__init__()
        self.data_mean = torch.Tensor(data_mean)
        self.data_std = torch.Tensor(data_std)
        self.noiseModel = noiseModel

    def _set_params_to_same_device_as(
        self, correct_device_tensor: torch.Tensor
    ) -> None:
        """Set the parameters to the same device as the input tensor.

        Parameters
        ----------
        correct_device_tensor: torch.Tensor
            The tensor whose device is used to set the parameters.
        """
        if self.data_mean.device != correct_device_tensor.device:
            self.data_mean = self.data_mean.to(correct_device_tensor.device)
            self.data_std = self.data_std.to(correct_device_tensor.device)

    def get_mean_lv(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return x, None

    def distr_params(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mean, lv = self.get_mean_lv(x)
        params = {
            "mean": mean,
            "logvar": lv,
        }
        return params

    @staticmethod
    def mean(params: dict[str, torch.Tensor]) -> torch.Tensor:
        return params["mean"]

    @staticmethod
    def mode(params: dict[str, torch.Tensor]) -> torch.Tensor:
        return params["mean"]

    @staticmethod
    def sample(params: dict[str, torch.Tensor]) -> torch.Tensor:
        return params["mean"]

    def log_likelihood(self, x: torch.Tensor, params: dict[str, torch.Tensor]):
        """Compute the log-likelihood given the parameters `params` obtained
        from the reconstruction tensor and the target tensor `x`.

        Parameters
        ----------
        x: torch.Tensor
            The target tensor. Shape is (B, C, [Z], Y, X).
        params: dict[str, Union[torch.Tensor, None]]
            The tensors obtained from output of the top-down pass.
            Here, "mean" correspond to the whole output, while logvar is `None`.

        Returns
        -------
        torch.Tensor
            The log-likelihood tensor. Shape is (B, C, [Z], Y, X).
        """
        self._set_params_to_same_device_as(x)
        predicted_s_denormalized = params["mean"] * self.data_std + self.data_mean
        x_denormalized = x * self.data_std + self.data_mean
        likelihoods = self.noiseModel.likelihood(
            x_denormalized, predicted_s_denormalized
        )
        logprob = torch.log(likelihoods)
        return logprob
