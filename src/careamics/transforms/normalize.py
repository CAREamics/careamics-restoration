"""Normalization and denormalization transforms for image patches."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from careamics.transforms.transform import Transform


def _reshape_stats(stats: list[float], ndim: int) -> NDArray:
    """Reshape stats to match the number of dimensions of the input image.

    This allows to broadcast the stats (mean or std) to the image dimensions, and
    thus directly perform a vectorial calculation.

    Parameters
    ----------
    stats : list of float
        List of stats, mean or standard deviation.
    ndim : int
        Number of dimensions of the image, including the C channel.

    Returns
    -------
    NDArray
        Reshaped stats.
    """
    return np.array(stats)[(..., *[np.newaxis] * (ndim - 1))]


class Normalize(Transform):
    """
    Normalize an image or image patch.

    Normalization is a zero mean and unit variance. This transform expects C(Z)YX
    dimensions.

    Not that an epsilon value of 1e-6 is added to the standard deviation to avoid
    division by zero and that it returns a float32 image.

    Parameters
    ----------
    image_means : list of float
        Mean value per channel.
    image_stds : list of float
        Standard deviation value per channel.
    target_means : list of float, optional
        Target mean value per channel, by default None.
    target_stds : list of float, optional
        Target standard deviation value per channel, by default None.

    Attributes
    ----------
    image_means : list of float
        Mean value per channel.
    image_stds : list of float
        Standard deviation value per channel.
    target_means :list of float, optional
        Target mean value per channel, by default None.
    target_stds : list of float, optional
        Target standard deviation value per channel, by default None.
    """

    def __init__(
        self,
        image_means: list[float],
        image_stds: list[float],
        target_means: Optional[list[float]] = None,
        target_stds: Optional[list[float]] = None,
    ):
        """Constructor.

        Parameters
        ----------
        image_means : list of float
            Mean value per channel.
        image_stds : list of float
            Standard deviation value per channel.
        target_means : list of float, optional
            Target mean value per channel, by default None.
        target_stds : list of float, optional
            Target standard deviation value per channel, by default None.
        """
        self.image_means = image_means
        self.image_stds = image_stds
        self.target_means = target_means
        self.target_stds = target_stds

        self.eps = 1e-6

    def __call__(
        self,
        patch: np.ndarray,
        target: Optional[NDArray] = None,
        **additional_arrays: NDArray,
    ) -> tuple[NDArray, Optional[NDArray], dict[str, NDArray]]:
        """Apply the transform to the source patch and the target (optional).

        Parameters
        ----------
        patch : NDArray
            Patch, 2D or 3D, shape C(Z)YX.
        target : NDArray, optional
            Target for the patch, by default None.
        **additional_arrays : NDArray
            Additional arrays that will be transformed identically to `patch` and
            `target`.

        Returns
        -------
        tuple of NDArray
            Transformed patch and target, the target can be returned as `None`.
        """
        if len(self.image_means) != patch.shape[0]:
            raise ValueError(
                f"Number of means (got a list of size {len(self.image_means)}) and "
                f"number of channels (got shape {patch.shape} for C(Z)YX) do not match."
            )
        if len(additional_arrays) != 0:
            raise NotImplementedError(
                "Transforming additional arrays is currently not supported for "
                "`Normalize`."
            )

        # reshape mean and std and apply the normalization to the patch
        means = _reshape_stats(self.image_means, patch.ndim)
        stds = _reshape_stats(self.image_stds, patch.ndim)
        norm_patch = self._apply(patch, means, stds)

        # same for the target patch
        if (
            target is not None
            and self.target_means is not None
            and self.target_stds is not None
        ):
            target_means = _reshape_stats(self.target_means, target.ndim)
            target_stds = _reshape_stats(self.target_stds, target.ndim)
            norm_target = self._apply(target, target_means, target_stds)
        else:
            norm_target = None

        return norm_patch, norm_target, additional_arrays

    def _apply(self, patch: NDArray, mean: NDArray, std: NDArray) -> NDArray:
        """
        Apply the transform to the image.

        Parameters
        ----------
        patch : NDArray
            Image patch, 2D or 3D, shape C(Z)YX.
        mean : NDArray
            Mean values.
        std : NDArray
            Standard deviations.

        Returns
        -------
        NDArray
            Normalized image patch.
        """
        return ((patch - mean) / (std + self.eps)).astype(np.float32)


class Denormalize:
    """
    Denormalize an image.

    Denormalization is performed expecting a zero mean and unit variance input. This
    transform expects C(Z)YX dimensions.

    Note that an epsilon value of 1e-6 is added to the standard deviation to avoid
    division by zero during the normalization step, which is taken into account during
    denormalization.

    Parameters
    ----------
    image_means : list or tuple of float
        Mean value per channel.
    image_stds : list or tuple of float
        Standard deviation value per channel.

    """

    def __init__(
        self,
        image_means: list[float],
        image_stds: list[float],
    ):
        """Constructor.

        Parameters
        ----------
        image_means : list of float
            Mean value per channel.
        image_stds : list of float
            Standard deviation value per channel.
        """
        self.image_means = image_means
        self.image_stds = image_stds

        self.eps = 1e-6

    def __call__(self, patch: NDArray) -> NDArray:
        """Reverse the normalization operation for a batch of patches.

        Parameters
        ----------
        patch : NDArray
            Patch, 2D or 3D, shape BC(Z)YX.

        Returns
        -------
        NDArray
            Transformed array.
        """
        if len(self.image_means) != patch.shape[1]:
            raise ValueError(
                f"Number of means (got a list of size {len(self.image_means)}) and "
                f"number of channels (got shape {patch.shape} for BC(Z)YX) do not "
                f"match."
            )

        means = _reshape_stats(self.image_means, patch.ndim)
        stds = _reshape_stats(self.image_stds, patch.ndim)

        denorm_array = self._apply(
            patch,
            np.swapaxes(means, 0, 1),  # swap axes as C channel is axis 1
            np.swapaxes(stds, 0, 1),
        )

        return denorm_array.astype(np.float32)

    def _apply(self, array: NDArray, mean: NDArray, std: NDArray) -> NDArray:
        """
        Apply the transform to the image.

        Parameters
        ----------
        array : NDArray
            Image patch, 2D or 3D, shape C(Z)YX.
        mean : NDArray
            Mean values.
        std : NDArray
            Standard deviations.

        Returns
        -------
        NDArray
            Denormalized image array.
        """
        return array * (std + self.eps) + mean
