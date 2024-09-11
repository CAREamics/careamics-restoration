"""Module containing `FCNAlgorithmConfig` class."""

from pprint import pformat
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from careamics.config.architectures import CustomModel, UNetModel
from careamics.config.optimizer_models import LrSchedulerModel, OptimizerModel


class FCNAlgorithmConfig(BaseModel):
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
    algorithm : {"n2v", "care", "n2n", "custom"}
        Algorithm to use.
    loss : {"n2v", "mae", "mse"}
        Loss function to use.
    model : UNetModel or CustomModel
        Model architecture to use.
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
    Minimum example:
    >>> from careamics.config import FCNAlgorithmConfig
    >>> config_dict = {
    ...     "algorithm": "n2v",
    ...     "loss": "n2v",
    ...     "model": {
    ...         "architecture": "UNet",
    ...     }
    ... }
    >>> config = FCNAlgorithmConfig(**config_dict)
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        protected_namespaces=(),  # allows to use model_* as a field name
        validate_assignment=True,
        extra="allow",
    )

    # Mandatory fields
    algorithm: Literal["n2v", "care", "n2n", "custom"]
    """Name of the algorithm, as defined in SupportedAlgorithm. Use `custom` for custom
    model architecture."""

    loss: Literal["n2v", "mae", "mse"]
    """Loss function to use, as defined in SupportedLoss."""

    model: Union[UNetModel, CustomModel] = Field(discriminator="architecture")
    """Model architecture to use, along with its parameters. Compatible architectures
    are defined in SupportedArchitecture, and their Pydantic models in
    `careamics.config.architectures`."""
    # TODO supported architectures are now all the architectures but does not warn users
    # of the compatibility with the algorithm

    # Optional fields
    optimizer: OptimizerModel = OptimizerModel()
    """Optimizer to use, defined in SupportedOptimizer."""

    lr_scheduler: LrSchedulerModel = LrSchedulerModel()
    """Learning rate scheduler to use, defined in SupportedLrScheduler."""

    @model_validator(mode="after")
    def algorithm_cross_validation(self: Self) -> Self:
        """Validate the algorithm model based on `algorithm`.

        N2V:
        - loss must be n2v
        - model must be a `UNetModel`

        Returns
        -------
        Self
            The validated model.
        """
        # N2V
        if self.algorithm == "n2v":
            # n2v is only compatible with the n2v loss
            if self.loss != "n2v":
                raise ValueError(
                    f"Algorithm {self.algorithm} only supports loss `n2v`."
                )

            # n2v is only compatible with the UNet model
            if not isinstance(self.model, UNetModel):
                raise ValueError(
                    f"Model for algorithm {self.algorithm} must be a `UNetModel`."
                )

            # n2v requires the number of input and output channels to be the same
            if self.model.in_channels != self.model.num_classes:
                raise ValueError(
                    "N2V requires the same number of input and output channels. Make "
                    "sure that `in_channels` and `num_classes` are the same."
                )

        if self.algorithm == "care" or self.algorithm == "n2n":
            if self.loss == "n2v":
                raise ValueError("Supervised algorithms do not support loss `n2v`.")

        if (self.algorithm == "custom") != (self.model.architecture == "custom"):
            raise ValueError(
                "Algorithm and model architecture must be both `custom` or not."
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

    @classmethod
    def get_compatible_algorithms(cls) -> list[str]:
        """Get the list of compatible algorithms.

        Returns
        -------
        list of str
            List of compatible algorithms.
        """
        return ["n2v", "care", "n2n"]
