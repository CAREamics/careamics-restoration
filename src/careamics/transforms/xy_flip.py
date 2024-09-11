"""XY flip transform."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from careamics.transforms.transform import Transform


class XYFlip(Transform):
    """Flip image along X and Y axis, one at a time.

    This transform randomly flips one of the last two axes.

    This transform expects C(Z)YX dimensions.

    Attributes
    ----------
    axis_indices : List[int]
        Indices of the axes that can be flipped.
    rng : np.random.Generator
        Random number generator.
    p : float
        Probability of applying the transform.
    seed : Optional[int]
        Random seed.

    Parameters
    ----------
    flip_x : bool, optional
        Whether to flip along the X axis, by default True.
    flip_y : bool, optional
        Whether to flip along the Y axis, by default True.
    p : float, optional
        Probability of applying the transform, by default 0.5.
    seed : Optional[int], optional
        Random seed, by default None.
    """

    def __init__(
        self,
        flip_x: bool = True,
        flip_y: bool = True,
        p: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        flip_x : bool, optional
            Whether to flip along the X axis, by default True.
        flip_y : bool, optional
            Whether to flip along the Y axis, by default True.
        p : float
            Probability of applying the transform, by default 0.5.
        seed : Optional[int], optional
            Random seed, by default None.
        """
        if p < 0 or p > 1:
            raise ValueError("Probability must be in [0, 1].")

        if not flip_x and not flip_y:
            raise ValueError("At least one axis must be flippable.")

        # probability to apply the transform
        self.p = p

        # "flippable" axes
        self.axis_indices = []

        if flip_y:
            self.axis_indices.append(-2)
        if flip_x:
            self.axis_indices.append(-1)

        # numpy random generator
        self.rng = np.random.default_rng(seed=seed)

    def __call__(
        self,
        patch: NDArray,
        target: Optional[NDArray] = None,
        **additional_arrays: NDArray,
    ) -> tuple[NDArray, Optional[NDArray], dict[str, NDArray]]:
        """Apply the transform to the source patch and the target (optional).

        Parameters
        ----------
        patch : np.ndarray
            Patch, 2D or 3D, shape C(Z)YX.
        target : Optional[np.ndarray], optional
            Target for the patch, by default None.
        **additional_arrays : NDArray
            Additional arrays that will be transformed identically to `patch` and
            `target`.

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            Transformed patch and target.
        """
        if self.rng.random() > self.p:
            return patch, target, additional_arrays

        # choose an axis to flip
        axis = self.rng.choice(self.axis_indices)

        patch_transformed = self._apply(patch, axis)
        target_transformed = self._apply(target, axis) if target is not None else None
        additional_transformed = {
            key: self._apply(array, axis) for key, array in additional_arrays.items()
        }

        return patch_transformed, target_transformed, additional_transformed

    def _apply(self, patch: NDArray, axis: int) -> NDArray:
        """Apply the transform to the image.

        Parameters
        ----------
        patch : np.ndarray
            Image patch, 2D or 3D, shape C(Z)YX.
        axis : int
            Axis to flip.

        Returns
        -------
        np.ndarray
            Flipped image patch.
        """
        return np.ascontiguousarray(np.flip(patch, axis=axis))
