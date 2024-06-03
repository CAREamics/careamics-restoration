from typing import List, Optional, Tuple

import numpy as np

from careamics.transforms.transform import Transform


class Normalize(Transform):
    """
    Normalize an image or image patch.

    Normalization is a zero mean and unit variance. This transform expects C(Z)YX
    dimensions.

    Not that an epsilon value of 1e-6 is added to the standard deviation to avoid
    division by zero and that it returns a float32 image.

    Attributes
    ----------
    mean : float
        Mean value.
    std : float
        Standard deviation value.
    """

    def __init__(
        self,
        image_means: List[float],
        image_stds: List[float],
        target_means: Optional[List[float]] = None,
        target_stds: Optional[List[float]] = None,
    ):
        self.image_means = image_means
        self.image_stds = image_stds
        self.target_means = target_means
        self.target_stds = target_stds
        self.eps = 1e-6

    def __call__(
        self, patch: np.ndarray, target: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply the transform to the source patch and the target (optional).

        Parameters
        ----------
        patch : np.ndarray
            Patch, 2D or 3D, shape C(Z)YX.
        target : Optional[np.ndarray], optional
            Target for the patch, by default None

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            Transformed patch and target.
        """
        norm_patch = np.zeros_like(patch, dtype=np.float32)
        norm_target = (
            np.zeros_like(target, dtype=np.float32) if target is not None else None
        )

        for i in range(patch.shape[0]):
            norm_patch[i] = self._apply(
                patch[i], self.image_means[i], self.image_stds[i]
            )
            if target is not None:
                norm_target[i] = self._apply(
                    target[i], self.target_means[i], self.target_stds[i]
                )

        return norm_patch, norm_target

    def _apply(self, patch: np.ndarray, mean: float, std: float) -> np.ndarray:
        return ((patch - mean) / (std + self.eps)).astype(np.float32)


class Denormalize:
    """
    Denormalize an image or image patch.

    Denormalization is performed expecting a zero mean and unit variance input. This
    transform expects C(Z)YX dimensions.

    Not that an epsilon value of 1e-6 is added to the standard deviation to avoid
    division by zero during the normalization step, which is taken into account during
    denormalization.

    Attributes
    ----------
    mean : float
        Mean value.
    std : float
        Standard deviation value.
    """

    def __init__(
        self,
        image_means: List[float],
        image_stds: List[float],
    ):
        self.image_means = image_means
        self.image_stds = image_stds
        self.eps = 1e-6

    def __call__(self, patch: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply the transform to the source patch and the target (optional).

        Parameters
        ----------
        patch : np.ndarray
            Patch, 2D or 3D, shape C(Z)YX.
        target : Optional[np.ndarray], optional
            Target for the patch, by default None

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            Transformed patch and target.
        """
        norm_patch = np.zeros_like(patch, dtype=np.float32)

        for i in range(patch.shape[0]):
            norm_patch[i] = self._apply(
                patch[i], self.image_means[i], self.image_stds[i]
            )

        return norm_patch

    def _apply(self, patch: np.ndarray, mean: float, std: float) -> np.ndarray:
        """
        Apply the transform to the image.

        Parameters
        ----------
        patch : np.ndarray
            Image or image patch, 2D or 3D, shape C(Z)YX.
        """
        return patch * (std + self.eps) + mean
