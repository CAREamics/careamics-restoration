"""N2V manipulation transform."""

from typing import Any, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from careamics.config.support import SupportedPixelManipulation, SupportedStructAxis
from careamics.transforms.transform import Transform

from .pixel_manipulation import median_manipulate, uniform_manipulate
from .struct_mask_parameters import StructMaskParameters


class N2VManipulate(Transform):
    """
    Default augmentation for the N2V model.

    This transform expects C(Z)YX dimensions.

    Parameters
    ----------
    roi_size : int, optional
        Size of the replacement area, by default 11.
    masked_pixel_percentage : float, optional
        Percentage of pixels to mask, by default 0.2.
    strategy : Literal[ "uniform", "median" ], optional
        Replaccement strategy, uniform or median, by default uniform.
    remove_center : bool, optional
        Whether to remove central pixel from patch, by default True.
    struct_mask_axis : Literal["horizontal", "vertical", "none"], optional
        StructN2V mask axis, by default "none".
    struct_mask_span : int, optional
        StructN2V mask span, by default 5.
    seed : Optional[int], optional
        Random seed, by default None.

    Attributes
    ----------
    masked_pixel_percentage : float
        Percentage of pixels to mask.
    roi_size : int
        Size of the replacement area.
    strategy : Literal[ "uniform", "median" ]
        Replaccement strategy, uniform or median.
    remove_center : bool
        Whether to remove central pixel from patch.
    struct_mask : Optional[StructMaskParameters]
        StructN2V mask parameters.
    rng : Generator
        Random number generator.
    """

    def __init__(
        self,
        roi_size: int = 11,
        masked_pixel_percentage: float = 0.2,
        strategy: Literal[
            "uniform", "median"
        ] = SupportedPixelManipulation.UNIFORM.value,
        remove_center: bool = True,
        struct_mask_axis: Literal["horizontal", "vertical", "none"] = "none",
        struct_mask_span: int = 5,
        seed: Optional[int] = None,
    ):
        """Constructor.

        Parameters
        ----------
        roi_size : int, optional
            Size of the replacement area, by default 11.
        masked_pixel_percentage : float, optional
            Percentage of pixels to mask, by default 0.2.
        strategy : Literal[ "uniform", "median" ], optional
            Replaccement strategy, uniform or median, by default uniform.
        remove_center : bool, optional
            Whether to remove central pixel from patch, by default True.
        struct_mask_axis : Literal["horizontal", "vertical", "none"], optional
            StructN2V mask axis, by default "none".
        struct_mask_span : int, optional
            StructN2V mask span, by default 5.
        seed : Optional[int], optional
            Random seed, by default None.
        """
        self.masked_pixel_percentage = masked_pixel_percentage
        self.roi_size = roi_size
        self.strategy = strategy
        self.remove_center = remove_center  # TODO is this ever used?

        if struct_mask_axis == SupportedStructAxis.NONE:
            self.struct_mask: Optional[StructMaskParameters] = None
        else:
            self.struct_mask = StructMaskParameters(
                axis=0 if struct_mask_axis == SupportedStructAxis.HORIZONTAL else 1,
                span=struct_mask_span,
            )

        # numpy random generator
        self.rng = np.random.default_rng(seed=seed)

    def __call__(
        self, patch: NDArray, *args: Any, **kwargs: Any
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Apply the transform to the image.

        Parameters
        ----------
        patch : np.ndarray
            Image patch, 2D or 3D, shape C(Z)YX.
        *args : Any
            Additional arguments, unused.
        **kwargs : Any
            Additional keyword arguments, unused.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Masked patch, original patch, and mask.
        """
        masked = np.zeros_like(patch)
        mask = np.zeros_like(patch)
        if self.strategy == SupportedPixelManipulation.UNIFORM:
            # Iterate over the channels to apply manipulation separately
            for c in range(patch.shape[0]):
                masked[c, ...], mask[c, ...] = uniform_manipulate(
                    patch=patch[c, ...],
                    mask_pixel_percentage=self.masked_pixel_percentage,
                    subpatch_size=self.roi_size,
                    remove_center=self.remove_center,
                    struct_params=self.struct_mask,
                    rng=self.rng,
                )
        elif self.strategy == SupportedPixelManipulation.MEDIAN:
            # Iterate over the channels to apply manipulation separately
            for c in range(patch.shape[0]):
                masked[c, ...], mask[c, ...] = median_manipulate(
                    patch=patch[c, ...],
                    mask_pixel_percentage=self.masked_pixel_percentage,
                    subpatch_size=self.roi_size,
                    struct_params=self.struct_mask,
                    rng=self.rng,
                )
        else:
            raise ValueError(f"Unknown masking strategy ({self.strategy}).")

        # TODO: Output does not match other transforms, how to resolve?
        #     - Don't include in Compose and apply after if algorithm is N2V?
        #     - or just don't return patch? but then mask is in the target position
        # TODO why return patch?
        return masked, patch, mask
