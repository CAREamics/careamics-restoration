"""LVAE Pydantic model."""

from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from .architecture_model import ArchitectureModel


# TODO: it is quite confusing to call this LVAEModel, as it is basically a config
class LVAEModel(ArchitectureModel):
    """LVAE model."""

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    architecture: Literal["LVAE"]
    input_shape: int = Field(default=64, ge=8, le=1024)
    multiscale_count: int = Field(default=5)  # TODO clarify
    # 0 - off, len(z_dims) + 1 # TODO can/should be le to z_dims len + 1
    z_dims: list = Field(default=[128, 128, 128, 128])
    output_channels: int = Field(default=1, ge=1)
    encoder_n_filters: int = Field(default=64, ge=8, le=1024)
    decoder_n_filters: int = Field(default=64, ge=8, le=1024)
    encoder_dropout: float = Field(default=0.1, ge=0.0, le=0.9)
    decoder_dropout: float = Field(default=0.1, ge=0.0, le=0.9)
    nonlinearity: Literal[
        "None", "Sigmoid", "Softmax", "Tanh", "ReLU", "LeakyReLU", "ELU"
    ] = Field(
        default="ELU",
    )

    predict_logvar: Literal[None, "pixelwise"] = None

    analytical_kl: bool = Field(
        default=False,
    )

    @field_validator("encoder_n_filters")
    @classmethod
    def validate_encoder_even(cls, encoder_n_filters: int) -> int:
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
    def validate_decoder_even(cls, decoder_n_filters: int) -> int:
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

    @model_validator(mode="after")
    def validate_multiscale_count(cls, self: Self) -> Self:
        """
        Validate the multiscale count.

        Parameters
        ----------
        self : Self
            The model.

        Returns
        -------
        Self
            The validated model.
        """
        # if self.multiscale_count != 0:
        #     if self.multiscale_count != len(self.z_dims) - 1:
        #         raise ValueError(
        #             f"Multiscale count must be 0 or equal to the number of Z "
        #             f"dims - 1 (got {self.multiscale_count} and {len(self.z_dims)})."
        #         )

        return self

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
