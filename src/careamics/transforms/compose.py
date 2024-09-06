"""A class chaining transforms together."""

from typing import Dict, List, Optional, Tuple, cast

import numpy as np

from careamics.config.data_model import TRANSFORMS_UNION

from .n2v_manipulate import N2VManipulate
from .normalize import Normalize
from .xy_flip import XYFlip
from .xy_random_rotate90 import XYRandomRotate90

ALL_TRANSFORMS = {
    "Normalize": Normalize,
    "N2VManipulate": N2VManipulate,
    "XYFlip": XYFlip,
    "XYRandomRotate90": XYRandomRotate90,
}


def get_all_transforms() -> Dict[str, type]:
    """Return all the transforms accepted by CAREamics.

    Returns
    -------
    dict
        A dictionary with all the transforms accepted by CAREamics, where the keys are
        the transform names and the values are the transform classes.
    """
    return ALL_TRANSFORMS


class Compose:
    """A class chaining transforms together.

    Parameters
    ----------
    transform_list : List[TRANSFORMS_UNION]
        A list of dictionaries where each dictionary contains the name of a
        transform and its parameters.

    Attributes
    ----------
    _callable_transforms : Callable
        A callable that applies the transforms to the input data.
    """

    def __init__(self, transform_list: List[TRANSFORMS_UNION]) -> None:
        """Instantiate a Compose object.

        Parameters
        ----------
        transform_list : List[TRANSFORMS_UNION]
            A list of dictionaries where each dictionary contains the name of a
            transform and its parameters.
        """
        # retrieve all available transforms
        all_transforms = get_all_transforms()

        # instantiate all transforms
        self.transforms = [
            all_transforms[t.name](**t.model_dump()) for t in transform_list
        ]

    def _chain_transforms(
        self, patch: np.ndarray, target: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Chain transforms on the input data.

        Parameters
        ----------
        patch : np.ndarray
            Input data.
        target : Optional[np.ndarray]
            Target data, by default None.

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            The output of the transformations.
        """
        params = (patch, target)

        for t in self.transforms:
            params = t(*params)

        return params

    def __call__(
        self, patch: np.ndarray, target: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, ...]:
        """Apply the transforms to the input data.

        Parameters
        ----------
        patch : np.ndarray
            The input data.
        target : Optional[np.ndarray], optional
            Target data, by default None.

        Returns
        -------
        Tuple[np.ndarray, ...]
            The output of the transformations.
        """
        return cast(Tuple[np.ndarray, ...], self._chain_transforms(patch, target))
