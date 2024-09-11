import numpy as np
import pytest

from careamics.config.transformations import (
    N2VManipulateModel,
    NormalizeModel,
    XYFlipModel,
    XYRandomRotate90Model,
)
from careamics.dataset.dataset_utils.running_stats import compute_normalization_stats
from careamics.transforms import Compose, Normalize, XYFlip, XYRandomRotate90


def test_empty_compose(ordered_array):
    array = ordered_array((4, 5, 5))
    source = array[2:, ...]
    target = array[:2, ...]

    # instantiate Compose
    compose = Compose([])

    # apply the composed transform
    source_transformed, target_transformed = compose(source, target)

    # check the results
    assert (source_transformed == source).all()
    assert (target_transformed == target).all()


def test_compose_with_target(ordered_array):
    seed = 24
    array = ordered_array((4, 5, 5))
    source = array[2:, ...]
    target = array[:2, ...]

    # transform lists
    transform_list = [XYFlip(seed=seed), XYRandomRotate90(seed=seed)]
    transform_list_pydantic = [
        XYFlipModel(name="XYFlip", seed=seed),
        XYRandomRotate90Model(name="XYRandomRotate90", seed=seed),
    ]

    # instantiate Compose
    compose = Compose(transform_list_pydantic)

    # apply the composed transform
    source_transformed, target_transformed = compose(source, target)

    # apply them manually
    t1_source, t1_target, _ = transform_list[0](source, target)
    t2_source, t2_target, _ = transform_list[1](t1_source, t1_target)

    # check the results
    assert (source_transformed == t2_source).all()
    assert (target_transformed == t2_target).all()


def test_compose_n2v(ordered_array):
    seed = 24
    array = ordered_array((2, 2, 5, 5))

    # transform lists
    means, stds = compute_normalization_stats(image=array)

    transform_list_pydantic = [
        NormalizeModel(image_means=means, image_stds=stds),
        XYFlipModel(seed=seed),
        XYRandomRotate90Model(seed=seed),
        N2VManipulateModel(),
    ]

    # apply the transforms
    normalize = Normalize(image_means=means, image_stds=stds)
    xyflip = XYFlip(seed=seed)
    xyrotate = XYRandomRotate90(seed=seed)
    array_aug, *_ = xyrotate(
        *xyflip(*normalize(array[0])[:2])[  # ignore additional_arrays (element no. 3)
            :2
        ]  # ignore additional_arrays (element no. 3)
    )

    # instantiate Compose
    compose = Compose(transform_list_pydantic)

    # apply the composed transform
    results = compose(array[0])
    assert len(results) == 3  # output of n2v_manipulate
    assert (results[1] == array_aug).all()
    assert (
        results[0][np.where(results[2] == 1)] != array_aug[np.where(results[2] == 1)]
    ).all()
    assert (
        results[0][np.where(results[2] != 1)] == array_aug[np.where(results[2] != 1)]
    ).all()


@pytest.mark.parametrize(
    "shape",
    [
        # 2D
        (1, 2, 2),
        (2, 2, 2),
        # 3D
        (1, 2, 2, 2),
        (2, 2, 2, 2),
    ],
)
def test_random_composition(ordered_array, shape):
    """Test that all the transforms can be composed in arbitrary order.

    This exclude the N2VManipulate transform, which needs to be the last one.
    """
    rng = np.random.default_rng(seed=42)

    # create array
    array = ordered_array(shape)
    mask = ordered_array(shape) + 5

    # apply transform in random order
    for _ in range(10):
        flip_x = rng.choice([True, False])

        transforms = [
            NormalizeModel(
                image_means=[0.5 for _ in range(array.shape[0])],
                image_stds=[0.5 for _ in range(array.shape[0])],
                target_means=[0.5 for _ in range(array.shape[0])],
                target_stds=[0.5 for _ in range(array.shape[0])],
            ),
            XYFlipModel(flip_x=flip_x, seed=42),
            XYRandomRotate90Model(seed=42),
        ]

        # randomly sort the transforms
        rng.shuffle(transforms)

        # apply compose
        compose = Compose(transforms)
        result_array, result_mask = compose(array, mask)
        assert not np.array_equal(result_array, array)
        assert not np.array_equal(result_mask, mask)


def test_compose_additional_arrays(ordered_array):
    """Test additional arrays get tranformed in the same way as the patch."""
    seed = 24
    # create inputs
    shape = (2, 2, 5, 5)
    array = ordered_array(shape)
    additional_arrays = {"arr": ordered_array(shape)}

    # create tranforms
    transforms = [
        XYFlipModel(name="XYFlip", seed=seed),
        XYRandomRotate90Model(name="XYRandomRotate90", seed=seed),
    ]

    compose = Compose(transforms)

    augmented, _, additional_augmented = compose.transform_with_additional_arrays(
        array, **additional_arrays
    )
    assert np.array_equal(augmented, additional_augmented["arr"])
