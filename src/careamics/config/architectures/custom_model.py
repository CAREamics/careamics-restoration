"""Custom architecture Pydantic model."""

from __future__ import annotations

import inspect
from pprint import pformat
from typing import Any, Literal

from pydantic import ConfigDict, field_validator, model_validator
from torch.nn import Module
from typing_extensions import Self

from .architecture_model import ArchitectureModel
from .register_model import get_custom_model


class CustomModel(ArchitectureModel):
    """Custom model configuration.

    This Pydantic model allows storing parameters for a custom model. In order for the
    model to be valid, the specific model needs to be registered using the
    `register_model` decorator, and its name correctly passed to this model
    configuration (see Examples).

    Attributes
    ----------
    architecture : Literal["custom"]
        Discriminator for the custom model, must be set to "custom".
    name : str
        Name of the custom model.
    parameters : CustomParametersModel
        All parameters, required for the initialization of the torch module have to be
        passed here.

    Raises
    ------
    ValueError
        If the custom model `name` is unknown.
    ValueError
        If the custom model is not a torch Module subclass.
    ValueError
        If the custom model parameters are not valid.

    Examples
    --------
    >>> from torch import nn, ones
    >>> from careamics.config import CustomModel, register_model
    >>> # Register a custom model
    >>> @register_model(name="my_linear")
    ... class LinearModel(nn.Module):
    ...    def __init__(self, in_features, out_features, *args, **kwargs):
    ...        super().__init__()
    ...        self.in_features = in_features
    ...        self.out_features = out_features
    ...        self.weight = nn.Parameter(ones(in_features, out_features))
    ...        self.bias = nn.Parameter(ones(out_features))
    ...    def forward(self, input):
    ...        return (input @ self.weight) + self.bias
    ...
    >>> # Create a configuration
    >>> config_dict = {
    ...     "architecture": "custom",
    ...     "name": "my_linear",
    ...     "in_features": 10,
    ...     "out_features": 5,
    ... }
    >>> config = CustomModel(**config_dict)
    """

    # pydantic model config
    model_config = ConfigDict(
        extra="allow",
    )

    # discriminator used for choosing the pydantic model in Model
    architecture: Literal["custom"]
    """Name of the architecture."""

    name: str
    """Name of the custom model."""

    @field_validator("name")
    @classmethod
    def custom_model_is_known(cls, value: str) -> str:
        """Check whether the custom model is known.

        Parameters
        ----------
        value : str
            Name of the custom model as registered using the `@register_model`
            decorator.

        Returns
        -------
        str
            The custom model name.
        """
        # delegate error to get_custom_model
        model = get_custom_model(value)

        # check if it is a torch Module subclass
        if not issubclass(model, Module):
            raise ValueError(
                f'Retrieved class {model} with name "{value}" is not a '
                f"torch.nn.Module subclass."
            )

        return value

    @model_validator(mode="after")
    def check_parameters(self: Self) -> Self:
        """Validate model by instantiating the model with the parameters.

        Returns
        -------
        Self
            The validated model.
        """
        # instantiate model
        try:
            get_custom_model(self.name)(**self.model_dump())
        except Exception as e:
            raise ValueError(
                f"while passing parameters to the model {e}. Verify that all "
                f"mandatory parameters are provided, and that either the {e} accepts "
                f"*args and **kwargs in its __init__() method, or that no additional"
                f"parameter is provided. Trace: "
                f"filename: {inspect.trace()[-1].filename}, function: "
                f"{inspect.trace()[-1].function}, line: {inspect.trace()[-1].lineno}"
            ) from None

        return self

    def __str__(self) -> str:
        """Pretty string representing the configuration.

        Returns
        -------
        str
            Pretty string.
        """
        return pformat(self.model_dump())

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Dump the model configuration.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments from Pydantic BaseModel model_dump method.

        Returns
        -------
        dict[str, Any]
            Model configuration.
        """
        model_dict = super().model_dump()

        # remove the name key
        model_dict.pop("name")

        return model_dict
