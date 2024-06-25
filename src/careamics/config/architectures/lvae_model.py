"""LVAE Pydantic model."""

from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from .architecture_model import ArchitectureModel


class LVAEModel(ArchitectureModel):
    """LVAE model."""

    model_config = ConfigDict(
        use_enum_values=True, protected_namespaces=(), validate_assignment=True
    )

    architecture: Literal["LVAE"]
    input_shape: int = Field(default=64, ge=8, le=1024)
    # multiscale_count = Field() # TODO clarify
    # 0 - off, 1 - len(z_dims)
    z_dims: tuple = Field(default=(128, 128, 128, 128), validate_default=True)
    num_channels: int = Field(default=1, ge=1, validate_default=True)
    encoder_n_filters: int = Field(default=64, ge=8, le=1024, validate_default=True)
    decoder_n_filters: int = Field(default=64, ge=8, le=1024, validate_default=True)
    encoder_dropout: float = Field(default=0.1, ge=0.0, le=0.9, validate_default=True)
    decoder_dropout: float = Field(default=0.1, ge=0.0, le=0.9, validate_default=True)
    nonlinearity: Literal[
        "None", "Sigmoid", "Softmax", "Tanh", "ReLU", "LeakyReLU", "ELU"
    ] = Field(default="ELU", validate_default=True)

    predict_logvar: bool = Field(default=False, validate_default=True)
    enable_noise_model: bool = Field(default=True, validate_default=True)
    analytical_kl: bool = Field(default=False, validate_default=True)

    @field_validator("encoder_n_filters")
    @classmethod
    def validate_encoder_n_filters(cls, encoder_n_filters: int) -> int:
        """
        Validate that num_channels_init is even.

        Parameters
        ----------
        encoder_n_filters : int
            Number of channels.

        Returns
        -------
        int
            Validated number of channels.

        Raises
        ------
        ValueError
            If the number of channels is odd.
        """
        # if odd
        if encoder_n_filters % 2 != 0:
            raise ValueError(
                f"Number of channels for the bottom layer must be even"
                f" (got {encoder_n_filters})."
            )

        return encoder_n_filters

    @field_validator("decoder_n_filters")
    @classmethod
    def validate_decoder_n_filters(cls, decoder_n_filters: int) -> int:
        """
        Validate that num_channels_init is even.

        Parameters
        ----------
        decoder_n_filters : int
            Number of channels.

        Returns
        -------
        int
            Validated number of channels.

        Raises
        ------
        ValueError
            If the number of channels is odd.
        """
        # if odd
        if decoder_n_filters % 2 != 0:
            raise ValueError(
                f"Number of channels for the bottom layer must be even"
                f" (got {decoder_n_filters})."
            )

        return decoder_n_filters

    @field_validator("z_dims")
    def validate_z_dims(cls, z_dims: tuple) -> tuple:
        """
        Validate the z_dims.

        Parameters
        ----------
        z_dims : tuple
            Tuple of z dimensions.

        Returns
        -------
        tuple
            Validated z dimensions.

        Raises
        ------
        ValueError
            If the number of z dimensions is not 4.
        """
        if len(z_dims) < 2:
            raise ValueError(
                f"Number of z dimensions must be at least 2 (got {len(z_dims)})."
            )

        return z_dims

    def set_3D(self, is_3D: bool) -> None:
        """
        Set 3D model by setting the `conv_dims` parameters.

        Parameters
        ----------
        is_3D : bool
            Whether the algorithm is 3D or not.
        """
        raise NotImplementedError("VAE is not implemented yet.")

    def is_3D(self) -> bool:
        """
        Return whether the model is 3D or not.

        Returns
        -------
        bool
            Whether the model is 3D or not.
        """
        raise NotImplementedError("VAE is not implemented yet.")