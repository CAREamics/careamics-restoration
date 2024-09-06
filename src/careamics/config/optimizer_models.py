"""Optimizers and schedulers Pydantic models."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from torch import optim
from typing_extensions import Self

from careamics.utils.torch_utils import filter_parameters

from .support import SupportedOptimizer


class OptimizerModel(BaseModel):
    """Torch optimizer Pydantic model.

    Only parameters supported by the corresponding torch optimizer will be taken
    into account. For more details, check:
    https://pytorch.org/docs/stable/optim.html#algorithms

    Note that mandatory parameters (see the specific Optimizer signature in the
    link above) must be provided. For example, SGD requires `lr`.

    Attributes
    ----------
    name : {"Adam", "SGD"}
        Name of the optimizer.
    parameters : dict
        Parameters of the optimizer (see torch documentation).
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        validate_assignment=True,
    )

    # Mandatory field
    name: Literal["Adam", "SGD", "Adamax"] = Field(
        default="Adam", validate_default=True
    )
    """Name of the optimizer, supported optimizers are defined in SupportedOptimizer."""

    # Optional parameters, empty dict default value to allow filtering dictionary
    parameters: dict = Field(
        default={
            "lr": 1e-4,
        },
        validate_default=True,
    )
    """Parameters of the optimizer, see PyTorch documentation for more details."""

    @field_validator("parameters")
    @classmethod
    def filter_parameters(cls, user_params: dict, values: ValidationInfo) -> dict:
        """
        Validate optimizer parameters.

        This method filters out unknown parameters, given the optimizer name.

        Parameters
        ----------
        user_params : dict
            Parameters passed on to the torch optimizer.
        values : ValidationInfo
            Pydantic field validation info, used to get the optimizer name.

        Returns
        -------
        dict
            Filtered optimizer parameters.

        Raises
        ------
        ValueError
            If the optimizer name is not specified.
        """
        optimizer_name = values.data["name"]

        # retrieve the corresponding optimizer class
        optimizer_class = getattr(optim, optimizer_name)

        # filter the user parameters according to the optimizer's signature
        parameters = filter_parameters(optimizer_class, user_params)

        return parameters

    @model_validator(mode="after")
    def sgd_lr_parameter(self) -> Self:
        """
        Check that SGD optimizer has the mandatory `lr` parameter specified.

        This is specific for PyTorch < 2.2.

        Returns
        -------
        Self
            Validated optimizer.

        Raises
        ------
        ValueError
            If the optimizer is SGD and the lr parameter is not specified.
        """
        if self.name == SupportedOptimizer.SGD and "lr" not in self.parameters:
            raise ValueError(
                "SGD optimizer requires `lr` parameter, check that it has correctly "
                "been specified in `parameters`."
            )

        return self


class LrSchedulerModel(BaseModel):
    """Torch learning rate scheduler Pydantic model.

    Only parameters supported by the corresponding torch lr scheduler will be taken
    into account. For more details, check:
    https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

    Note that mandatory parameters (see the specific LrScheduler signature in the
    link above) must be provided. For example, StepLR requires `step_size`.

    Attributes
    ----------
    name : {"ReduceLROnPlateau", "StepLR"}
        Name of the learning rate scheduler.
    parameters : dict
        Parameters of the learning rate scheduler (see torch documentation).
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        validate_assignment=True,
    )

    # Mandatory field
    name: Literal["ReduceLROnPlateau", "StepLR"] = Field(default="ReduceLROnPlateau")
    """Name of the learning rate scheduler, supported schedulers are defined in
    SupportedScheduler."""

    # Optional parameters
    parameters: dict = Field(default={}, validate_default=True)
    """Parameters of the learning rate scheduler, see PyTorch documentation for more
    details."""

    @field_validator("parameters")
    @classmethod
    def filter_parameters(cls, user_params: dict, values: ValidationInfo) -> dict:
        """Filter parameters based on the learning rate scheduler's signature.

        Parameters
        ----------
        user_params : dict
            User parameters.
        values : ValidationInfo
            Pydantic field validation info, used to get the scheduler name.

        Returns
        -------
        dict
            Filtered scheduler parameters.

        Raises
        ------
        ValueError
            If the scheduler is StepLR and the step_size parameter is not specified.
        """
        # retrieve the corresponding scheduler class
        scheduler_class = getattr(optim.lr_scheduler, values.data["name"])

        # filter the user parameters according to the scheduler's signature
        parameters = filter_parameters(scheduler_class, user_params)

        if values.data["name"] == "StepLR" and "step_size" not in parameters:
            raise ValueError(
                "StepLR scheduler requires `step_size` parameter, check that it has "
                "correctly been specified in `parameters`."
            )

        return parameters
