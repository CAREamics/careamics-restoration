"""Pydantic CAREamics configuration."""

from __future__ import annotations

import re
from pprint import pformat
from typing import Any, Literal, Union

from bioimageio.spec.generic.v0_3 import CiteEntry
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from careamics.config.algorithms import UNetBasedAlgorithm, VAEBasedAlgorithm
from careamics.config.data import GeneralDataConfig
from careamics.config.training_model import TrainingConfig


class Configuration(BaseModel):
    """
    CAREamics configuration.

    The configuration defines all parameters used to build and train a CAREamics model.
    These parameters are validated to ensure that they are compatible with each other.

    It contains three sub-configurations:

    - AlgorithmModel: configuration for the algorithm training, which includes the
        architecture, loss function, optimizer, and other hyperparameters.
    - DataModel: configuration for the dataloader, which includes the type of data,
        transformations, mean/std and other parameters.
    - TrainingModel: configuration for the training, which includes the number of
        epochs or the callbacks.

    Attributes
    ----------
    experiment_name : str
        Name of the experiment, used when saving logs and checkpoints.
    algorithm : AlgorithmModel
        Algorithm configuration.
    data : DataModel
        Data configuration.
    training : TrainingModel
        Training configuration.

    Methods
    -------
    set_3D(is_3D: bool, axes: str, patch_size: List[int]) -> None
        Switch configuration between 2D and 3D.
    model_dump(
        exclude_defaults: bool = False, exclude_none: bool = True, **kwargs: Dict
        ) -> Dict
        Export configuration to a dictionary.

    Raises
    ------
    ValueError
        Configuration parameter type validation errors.
    ValueError
        If the experiment name contains invalid characters or is empty.
    ValueError
        If the algorithm is 3D but there is not "Z" in the data axes, or 2D algorithm
        with "Z" in data axes.
    ValueError
        Algorithm, data or training validation errors.

    Notes
    -----
    We provide convenience methods to create standards configurations, for instance:
    >>> from careamics.config import create_n2v_configuration
    >>> config = create_n2v_configuration(
    ...     experiment_name="n2v_experiment",
    ...     data_type="array",
    ...     axes="YX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100
    ... )

    The configuration can be exported to a dictionary using the model_dump method:
    >>> config_dict = config.model_dump()

    Configurations can also be exported or imported from yaml files:
    >>> from careamics.config import save_configuration, load_configuration
    >>> path_to_config = save_configuration(config, my_path / "config.yml")
    >>> other_config = load_configuration(path_to_config)

    Examples
    --------
    Minimum example:
    >>> from careamics import configuration_factory
    >>> config_dict = {
    ...         "experiment_name": "N2V_experiment",
    ...         "algorithm_config": {
    ...             "algorithm": "n2v",
    ...             "loss": "n2v",
    ...             "model": {
    ...                 "architecture": "UNet",
    ...             },
    ...         },
    ...         "training_config": {
    ...             "num_epochs": 200,
    ...         },
    ...         "data_config": {
    ...             "data_type": "tiff",
    ...             "patch_size": [64, 64],
    ...             "axes": "SYX",
    ...         },
    ...     }
    >>> config = configuration_factory(config_dict)
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        # extra="forbid", # TODO
    )

    # version
    version: Literal["0.1.0"] = "0.1.0"
    """CAREamics configuration version."""

    # required parameters
    experiment_name: str
    """Name of the experiment, used to name logs and checkpoints."""

    # Sub-configurations
    algorithm_config: Union[UNetBasedAlgorithm, VAEBasedAlgorithm] = Field(
        discriminator="algorithm"
    )
    """Algorithm configuration, holding all parameters required to configure the
    model."""

    data_config: GeneralDataConfig
    """Data configuration, holding all parameters required to configure the training
    data loader."""

    training_config: TrainingConfig
    """Training configuration, holding all parameters required to configure the
    training process."""

    @field_validator("experiment_name")
    @classmethod
    def no_symbol(cls, name: str) -> str:
        """
        Validate experiment name.

        A valid experiment name is a non-empty string with only contains letters,
        numbers, underscores, dashes and spaces.

        Parameters
        ----------
        name : str
            Name to validate.

        Returns
        -------
        str
            Validated name.

        Raises
        ------
        ValueError
            If the name is empty or contains invalid characters.
        """
        if len(name) == 0 or name.isspace():
            raise ValueError("Experiment name is empty.")

        # Validate using a regex that it contains only letters, numbers, underscores,
        # dashes and spaces
        if not re.match(r"^[a-zA-Z0-9_\- ]*$", name):
            raise ValueError(
                f"Experiment name contains invalid characters (got {name}). "
                f"Only letters, numbers, underscores, dashes and spaces are allowed."
            )

        return name

    @model_validator(mode="after")
    def validate_3D(self: Self) -> Self:
        """
        Change algorithm dimensions to match data.axes.

        Returns
        -------
        Self
            Validated configuration.
        """
        if "Z" in self.data_config.axes and not self.algorithm_config.model.is_3D():
            # change algorithm to 3D
            self.algorithm_config.model.set_3D(True)
        elif "Z" not in self.data_config.axes and self.algorithm_config.model.is_3D():
            # change algorithm to 2D
            self.algorithm_config.model.set_3D(False)

        return self

    def __str__(self) -> str:
        """
        Pretty string reprensenting the configuration.

        Returns
        -------
        str
            Pretty string.
        """
        return pformat(self.model_dump())

    def set_3D(self, is_3D: bool, axes: str, patch_size: list[int]) -> None:
        """
        Set 3D flag and axes.

        Parameters
        ----------
        is_3D : bool
            Whether the algorithm is 3D or not.
        axes : str
            Axes of the data.
        patch_size : list[int]
            Patch size.
        """
        # set the flag and axes (this will not trigger validation at the config level)
        self.algorithm_config.model.set_3D(is_3D)
        self.data_config.set_3D(axes, patch_size)

        # cheap hack: trigger validation
        self.algorithm_config = self.algorithm_config

    def get_algorithm_friendly_name(self) -> str:
        """
        Get the algorithm name.

        Returns
        -------
        str
            Algorithm name.
        """
        raise ValueError("Unknown algorithm.")

    def get_algorithm_description(self) -> str:
        """
        Return a description of the algorithm.

        This method is used to generate the README of the BioImage Model Zoo export.

        Returns
        -------
        str
            Description of the algorithm.
        """
        raise ValueError("No algorithm description available.")

    def get_algorithm_citations(self) -> list[CiteEntry]:
        """
        Return a list of citation entries of the current algorithm.

        This is used to generate the model description for the BioImage Model Zoo.

        Returns
        -------
        List[CiteEntry]
            List of citation entries.
        """
        raise ValueError("No algorithm citations available.")

    def get_algorithm_references(self) -> str:
        """
        Get the algorithm references.

        This is used to generate the README of the BioImage Model Zoo export.

        Returns
        -------
        str
            Algorithm references.
        """
        raise ValueError("No algorithm references available.")

    def get_algorithm_keywords(self) -> list[str]:
        """
        Get algorithm keywords.

        Returns
        -------
        list[str]
            List of keywords.
        """
        return ["CAREamics"]

    def model_dump(
        self,
        *,
        mode: Literal["json", "python"] | str = "python",
        include: Any | None = None,
        exclude: Any | None = None,
        context: Any | None = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = True,
        round_trip: bool = False,
        warnings: bool | Literal["none", "warn", "error"] = True,
        serialize_as_any: bool = False,
    ) -> dict:
        """
        Override model_dump method in order to set default values.

        As opposed to the parent model_dump method, this method sets exclude none by
        default.

        Parameters
        ----------
        mode : Literal['json', 'python'] | str, default='python'
            The serialization format.
        include : Any | None, default=None
            Attributes to include.
        exclude : Any | None, default=None
            Attributes to exclude.
        context : Any | None, default=None
            Additional context to pass to the serialization functions.
        by_alias : bool, default=False
            Whether to use attribute aliases.
        exclude_unset : bool, default=False
            Whether to exclude fields that are not set.
        exclude_defaults : bool, default=False
            Whether to exclude fields that have default values.
        exclude_none : bool, default=true
            Whether to exclude fields that have None values.
        round_trip : bool, default=False
            Whether to dump and load the data to ensure that the output is a valid
            representation.
        warnings : bool | Literal['none', 'warn', 'error'], default=True
            Whether to emit warnings.
        serialize_as_any : bool, default=False
            Whether to serialize all types as Any.

        Returns
        -------
        dict
            Dictionary containing the model parameters.
        """
        dictionary = super().model_dump(
            mode=mode,
            include=include,
            exclude=exclude,
            context=context,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            round_trip=round_trip,
            warnings=warnings,
            serialize_as_any=serialize_as_any,
        )

        return dictionary
