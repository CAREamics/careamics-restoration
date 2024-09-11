"""Pydantic CAREamics configuration."""

from __future__ import annotations

import re
from pathlib import Path
from pprint import pformat
from typing import Literal, Union

import yaml
from bioimageio.spec.generic.v0_3 import CiteEntry
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from .data_model import DataConfig
from .fcn_algorithm_model import FCNAlgorithmConfig
from .references import (
    CARE,
    CUSTOM,
    N2N,
    N2V,
    N2V2,
    STRUCT_N2V,
    STRUCT_N2V2,
    CAREDescription,
    CARERef,
    N2NDescription,
    N2NRef,
    N2V2Description,
    N2V2Ref,
    N2VDescription,
    N2VRef,
    StructN2V2Description,
    StructN2VDescription,
    StructN2VRef,
)
from .support import SupportedAlgorithm, SupportedPixelManipulation, SupportedTransform
from .training_model import TrainingConfig
from .transformations.n2v_manipulate_model import (
    N2VManipulateModel,
)
from .vae_algorithm_model import VAEAlgorithmConfig


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
    set_N2V2(use_n2v2: bool) -> None
        Switch N2V algorithm between N2V and N2V2.
    set_structN2V(
        mask_axis: Literal["horizontal", "vertical", "none"], mask_span: int) -> None
        Set StructN2V parameters.
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
    We provide convenience methods to create standards configurations, for instance
    for N2V, in the `careamics.config.configuration_factory` module.
    >>> from careamics.config.configuration_factory import create_n2v_configuration
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
    >>> from careamics.config import Configuration
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
    >>> config = Configuration(**config_dict)
    """

    model_config = ConfigDict(
        validate_assignment=True,
        set_arbitrary_types_allowed=True,
    )

    # version
    version: Literal["0.1.0"] = "0.1.0"
    """CAREamics configuration version."""

    # required parameters
    experiment_name: str
    """Name of the experiment, used to name logs and checkpoints."""

    # Sub-configurations
    algorithm_config: Union[FCNAlgorithmConfig, VAEAlgorithmConfig] = Field(
        discriminator="algorithm"
    )
    """Algorithm configuration, holding all parameters required to configure the
    model."""

    data_config: DataConfig
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

        Only for non-custom algorithms.

        Returns
        -------
        Self
            Validated configuration.
        """
        if self.algorithm_config.algorithm != SupportedAlgorithm.CUSTOM:
            if "Z" in self.data_config.axes and not self.algorithm_config.model.is_3D():
                # change algorithm to 3D
                self.algorithm_config.model.set_3D(True)
            elif (
                "Z" not in self.data_config.axes and self.algorithm_config.model.is_3D()
            ):
                # change algorithm to 2D
                self.algorithm_config.model.set_3D(False)

        return self

    @model_validator(mode="after")
    def validate_algorithm_and_data(self: Self) -> Self:
        """
        Validate algorithm and data compatibility.

        In particular, the validation does the following:

        - If N2V is used, it enforces the presence of N2V_Maniuplate in the transforms
        - If N2V2 is used, it enforces the correct manipulation strategy

        Returns
        -------
        Self
            Validated configuration.
        """
        if self.algorithm_config.algorithm == SupportedAlgorithm.N2V:
            # missing N2V_MANIPULATE
            if not self.data_config.has_n2v_manipulate():
                self.data_config.transforms.append(
                    N2VManipulateModel(
                        name=SupportedTransform.N2V_MANIPULATE.value,
                    )
                )

            median = SupportedPixelManipulation.MEDIAN.value
            uniform = SupportedPixelManipulation.UNIFORM.value
            strategy = median if self.algorithm_config.model.n2v2 else uniform
            self.data_config.set_N2V2_strategy(strategy)
        else:
            # remove N2V manipulate if present
            if self.data_config.has_n2v_manipulate():
                self.data_config.remove_n2v_manipulate()

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

    def set_N2V2(self, use_n2v2: bool) -> None:
        """
        Switch N2V algorithm between N2V and N2V2.

        Parameters
        ----------
        use_n2v2 : bool
            Whether to use N2V2 or not.

        Raises
        ------
        ValueError
            If the algorithm is not N2V.
        """
        if self.algorithm_config.algorithm == SupportedAlgorithm.N2V:
            self.algorithm_config.model.n2v2 = use_n2v2
            strategy = (
                SupportedPixelManipulation.MEDIAN.value
                if use_n2v2
                else SupportedPixelManipulation.UNIFORM.value
            )
            self.data_config.set_N2V2_strategy(strategy)
        else:
            raise ValueError("N2V2 can only be set for N2V algorithm.")

    def set_structN2V(
        self, mask_axis: Literal["horizontal", "vertical", "none"], mask_span: int
    ) -> None:
        """
        Set StructN2V parameters.

        Parameters
        ----------
        mask_axis : Literal["horizontal", "vertical", "none"]
            Axis of the structural mask.
        mask_span : int
            Span of the structural mask.
        """
        self.data_config.set_structN2V_mask(mask_axis, mask_span)

    def get_algorithm_flavour(self) -> str:
        """
        Get the algorithm name.

        Returns
        -------
        str
            Algorithm name.
        """
        if self.algorithm_config.algorithm == SupportedAlgorithm.N2V:
            use_n2v2 = self.algorithm_config.model.n2v2
            use_structN2V = self.data_config.transforms[-1].struct_mask_axis != "none"

            # return the n2v flavour
            if use_n2v2 and use_structN2V:
                return STRUCT_N2V2
            elif use_n2v2:
                return N2V2
            elif use_structN2V:
                return STRUCT_N2V
            else:
                return N2V
        elif self.algorithm_config.algorithm == SupportedAlgorithm.N2N:
            return N2N
        elif self.algorithm_config.algorithm == SupportedAlgorithm.CARE:
            return CARE
        else:
            return CUSTOM

    def get_algorithm_description(self) -> str:
        """
        Return a description of the algorithm.

        This method is used to generate the README of the BioImage Model Zoo export.

        Returns
        -------
        str
            Description of the algorithm.
        """
        algorithm_flavour = self.get_algorithm_flavour()

        if algorithm_flavour == CUSTOM:
            return f"Custom algorithm, named {self.algorithm_config.model.name}"
        else:  # currently only N2V flavours
            if algorithm_flavour == N2V:
                return N2VDescription().description
            elif algorithm_flavour == N2V2:
                return N2V2Description().description
            elif algorithm_flavour == STRUCT_N2V:
                return StructN2VDescription().description
            elif algorithm_flavour == STRUCT_N2V2:
                return StructN2V2Description().description
            elif algorithm_flavour == N2N:
                return N2NDescription().description
            elif algorithm_flavour == CARE:
                return CAREDescription().description

        return ""

    def get_algorithm_citations(self) -> list[CiteEntry]:
        """
        Return a list of citation entries of the current algorithm.

        This is used to generate the model description for the BioImage Model Zoo.

        Returns
        -------
        List[CiteEntry]
            List of citation entries.
        """
        if self.algorithm_config.algorithm == SupportedAlgorithm.N2V:
            use_n2v2 = self.algorithm_config.model.n2v2
            use_structN2V = self.data_config.transforms[-1].struct_mask_axis != "none"

            # return the (struct)N2V(2) references
            if use_n2v2 and use_structN2V:
                return [N2VRef, N2V2Ref, StructN2VRef]
            elif use_n2v2:
                return [N2VRef, N2V2Ref]
            elif use_structN2V:
                return [N2VRef, StructN2VRef]
            else:
                return [N2VRef]
        elif self.algorithm_config.algorithm == SupportedAlgorithm.N2N:
            return [N2NRef]
        elif self.algorithm_config.algorithm == SupportedAlgorithm.CARE:
            return [CARERef]

        raise ValueError("Citation not available for custom algorithm.")

    def get_algorithm_references(self) -> str:
        """
        Get the algorithm references.

        This is used to generate the README of the BioImage Model Zoo export.

        Returns
        -------
        str
            Algorithm references.
        """
        if self.algorithm_config.algorithm == SupportedAlgorithm.N2V:
            use_n2v2 = self.algorithm_config.model.n2v2
            use_structN2V = self.data_config.transforms[-1].struct_mask_axis != "none"

            references = [
                N2VRef.text + " doi: " + N2VRef.doi,
                N2V2Ref.text + " doi: " + N2V2Ref.doi,
                StructN2VRef.text + " doi: " + StructN2VRef.doi,
            ]

            # return the (struct)N2V(2) references
            if use_n2v2 and use_structN2V:
                return "".join(references)
            elif use_n2v2:
                references.pop(-1)
                return "".join(references)
            elif use_structN2V:
                references.pop(-2)
                return "".join(references)
            else:
                return references[0]

        return ""

    def get_algorithm_keywords(self) -> list[str]:
        """
        Get algorithm keywords.

        Returns
        -------
        list[str]
            List of keywords.
        """
        if self.algorithm_config.algorithm == SupportedAlgorithm.N2V:
            use_n2v2 = self.algorithm_config.model.n2v2
            use_structN2V = self.data_config.transforms[-1].struct_mask_axis != "none"

            keywords = [
                "denoising",
                "restoration",
                "UNet",
                "3D" if "Z" in self.data_config.axes else "2D",
                "CAREamics",
                "pytorch",
                N2V,
            ]

            if use_n2v2:
                keywords.append(N2V2)
            if use_structN2V:
                keywords.append(STRUCT_N2V)
        else:
            keywords = ["CAREamics"]

        return keywords

    def model_dump(
        self,
        exclude_defaults: bool = False,
        exclude_none: bool = True,
        **kwargs: dict,
    ) -> dict:
        """
        Override model_dump method in order to set default values.

        Parameters
        ----------
        exclude_defaults : bool, optional
            Whether to exclude fields with default values or not, by default
            True.
        exclude_none : bool, optional
            Whether to exclude fields with None values or not, by default True.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        dict
            Dictionary containing the model parameters.
        """
        dictionary = super().model_dump(
            exclude_none=exclude_none, exclude_defaults=exclude_defaults, **kwargs
        )

        return dictionary


def load_configuration(path: Union[str, Path]) -> Configuration:
    """
    Load configuration from a yaml file.

    Parameters
    ----------
    path : str or Path
        Path to the configuration.

    Returns
    -------
    Configuration
        Configuration.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    """
    # load dictionary from yaml
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Configuration file {path} does not exist in " f" {Path.cwd()!s}"
        )

    dictionary = yaml.load(Path(path).open("r"), Loader=yaml.SafeLoader)

    return Configuration(**dictionary)


def save_configuration(config: Configuration, path: Union[str, Path]) -> Path:
    """
    Save configuration to path.

    Parameters
    ----------
    config : Configuration
        Configuration to save.
    path : str or Path
        Path to a existing folder in which to save the configuration or to an existing
        configuration file.

    Returns
    -------
    Path
        Path object representing the configuration.

    Raises
    ------
    ValueError
        If the path does not point to an existing directory or .yml file.
    """
    # make sure path is a Path object
    config_path = Path(path)

    # check if path is pointing to an existing directory or .yml file
    if config_path.exists():
        if config_path.is_dir():
            config_path = Path(config_path, "config.yml")
        elif config_path.suffix != ".yml" and config_path.suffix != ".yaml":
            raise ValueError(
                f"Path must be a directory or .yml or .yaml file (got {config_path})."
            )
    else:
        if config_path.suffix != ".yml" and config_path.suffix != ".yaml":
            raise ValueError(
                f"Path must be a directory or .yml or .yaml file (got {config_path})."
            )

    # save configuration as dictionary to yaml
    with open(config_path, "w") as f:
        # dump configuration
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)

    return config_path
