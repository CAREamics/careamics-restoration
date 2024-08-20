"""Algorithm configuration."""

from __future__ import annotations

from pprint import pformat
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from careamics.config.support import SupportedAlgorithm, SupportedLoss

from .architectures import CustomModel, LVAEModel
from .likelihood_model import GaussianLikelihoodModel, NMLikelihoodModel
from .nm_model import MultiChannelNmModel
from .optimizer_models import LrSchedulerModel, OptimizerModel


class VAEAlgorithmConfig(BaseModel):
    """Algorithm configuration.

    This Pydantic model validates the parameters governing the components of the
    training algorithm: which algorithm, loss function, model architecture, optimizer,
    and learning rate scheduler to use.

    Currently, we only support N2V, CARE, N2N and custom models. The `n2v` algorithm is
    only compatible with `n2v` loss and `UNet` architecture. The `custom` algorithm
    allows you to register your own architecture and select it using its name as
    `name` in the custom pydantic model.

    Attributes
    ----------
    algorithm : algorithm: Literal["musplit", "denoisplit", "custom"]
        Algorithm to use.
    loss : Literal["musplit", "denoisplit", "denoisplit_musplit"]
        Loss function to use.
    model : Union[LVAEModel, CustomModel]
        Model architecture to use.
    noise_model: Optional[MultiChannelNmModel]
        Noise model to use.
    noise_model_likelihood_model: Optional[NMLikelihoodModel]
        Noise model likelihood model to use.
    gaussian_likelihood_model: Optional[GaussianLikelihoodModel]
        Gaussian likelihood model to use.
    optimizer : OptimizerModel, optional
        Optimizer to use.
    lr_scheduler : LrSchedulerModel, optional
        Learning rate scheduler to use.

    Raises
    ------
    ValueError
        Algorithm parameter type validation errors.
    ValueError
        If the algorithm, loss and model are not compatible.

    Examples
    --------
    # TODO add once finalized
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        protected_namespaces=(),  # allows to use model_* as a field name
        validate_assignment=True,
        extra="allow",
    )

    # Mandatory fields
    # defined in SupportedAlgorithm
    # TODO: Use supported Enum classes for typing?
    #   - values can still be passed as strings and they will be cast to Enum
    algorithm_type: Literal["vae"]
    algorithm: Literal["musplit", "denoisplit", "custom"]
    loss: Literal["musplit", "denoisplit", "denoisplit_musplit"]
    model: Union[LVAEModel, CustomModel] = Field(discriminator="architecture")

    noise_model: Optional[MultiChannelNmModel] = None
    noise_model_likelihood_model: Optional[NMLikelihoodModel] = None
    gaussian_likelihood_model: Optional[GaussianLikelihoodModel] = None

    # Optional fields
    optimizer: OptimizerModel = OptimizerModel()
    """Optimizer to use, defined in SupportedOptimizer."""

    lr_scheduler: LrSchedulerModel = LrSchedulerModel()

    @model_validator(mode="after")
    def algorithm_cross_validation(self: Self) -> Self:
        """Validate the algorithm model based on `algorithm`.

        Returns
        -------
        Self
            The validated model.
        """
        # musplit
        if self.algorithm == SupportedAlgorithm.MUSPLIT:
            if self.loss != SupportedLoss.MUSPLIT:
                raise ValueError(
                    f"Algorithm {self.algorithm} only supports loss `musplit`."
                )
            
        if self.algorithm == SupportedAlgorithm.DENOISPLIT:
            if self.loss in [
                SupportedLoss.DENOISPLIT, 
                SupportedLoss.DENOISPLIT_MUSPLIT
            ]:
                raise ValueError(
                    f"Algorithm {self.algorithm} only supports loss `denoisplit` "
                    "or `denoisplit_musplit."
                )
            if self.loss == SupportedLoss.DENOISPLIT and self.model.predict_logvar is not None:
                raise ValueError(
                    "Algorithm `denoisplit` with loss `denoisplit` only supports "
                    "`predict_logvar` as `None`."
                )

        # TODO: what if algorithm is not musplit or denoisplit (HDN?)
        return self
    
    @model_validator(mode="after")
    def output_channels_validation(self: Self) -> Self:
        """Validate the consistency between the number of output channels and the
        number of noise models.

        Returns
        -------
        Self
            The validated model.
        """
        assert self.model.output_channels == len(self.noise_model.noise_models), (
            f"Number of output channels ({self.model.output_channels}) must match "
            f"the number of noise models ({len(self.noise_model.noise_models)})."
        )
        return self
    
    @model_validator(mode="after")
    def predict_logvar_validation(self: Self) -> Self:
        """Validate the consistency of `predict_logvar` in the different places
        in which it is used.

        Returns
        -------
        Self
            The validated model.
        """
        assert (
            self.model.predict_logvar == self.gaussian_likelihood_model.predict_logvar,
            f"Model `predict_logvar` ({self.model.predict_logvar}) must match "
            "Gaussian likelihood model `predict_logvar` "
            f"({self.gaussian_likelihood_model.predict_logvar}).",
        )
        return self
        

    def __str__(self) -> str:
        """Pretty string representing the configuration.

        Returns
        -------
        str
            Pretty string.
        """
        return pformat(self.model_dump())
