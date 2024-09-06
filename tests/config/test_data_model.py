import numpy as np
import pytest
import yaml

from careamics.config.data_model import DataConfig
from careamics.config.support import (
    SupportedPixelManipulation,
    SupportedStructAxis,
    SupportedTransform,
)
from careamics.config.transformations import (
    N2VManipulateModel,
    NormalizeModel,
    XYFlipModel,
    XYRandomRotate90Model,
)
from careamics.transforms import get_all_transforms


@pytest.mark.parametrize("ext", ["nd2", "jpg", "png ", "zarr", "npy"])
def test_wrong_extensions(minimum_data: dict, ext: str):
    """Test that supported model raises ValueError for unsupported extensions."""
    minimum_data["data_type"] = ext

    # instantiate DataModel model
    with pytest.raises(ValueError):
        DataConfig(**minimum_data)


@pytest.mark.parametrize("mean, std", [(0, 124.5), (12.6, 0.1)])
def test_mean_std_non_negative(minimum_data: dict, mean, std):
    """Test that non negative mean and std are accepted."""
    minimum_data["image_means"] = [mean]
    minimum_data["image_stds"] = [std]
    minimum_data["target_means"] = [mean]
    minimum_data["target_stds"] = [std]

    data_model = DataConfig(**minimum_data)
    assert data_model.image_means == [mean]
    assert data_model.image_stds == [std]
    assert data_model.target_means == [mean]
    assert data_model.target_stds == [std]


def test_mean_std_both_specified_or_none(minimum_data: dict):
    """Test an error is raised if std is specified but mean is None."""
    # No error if both are None
    DataConfig(**minimum_data)

    # Error if only mean is defined
    minimum_data["image_means"] = [10.4]
    with pytest.raises(ValueError):
        DataConfig(**minimum_data)

    # Error if only std is defined
    minimum_data.pop("image_means")
    minimum_data["image_stds"] = [10.4]
    with pytest.raises(ValueError):
        DataConfig(**minimum_data)

    # No error if both are specified
    minimum_data["image_means"] = [10.4]
    minimum_data["image_stds"] = [10.4]
    DataConfig(**minimum_data)

    # Error if target mean is defined but target std is None
    minimum_data["target_stds"] = [10.4, 11]
    with pytest.raises(ValueError):
        DataConfig(**minimum_data)


def test_set_mean_and_std(minimum_data: dict):
    """Test that mean and std can be set after initialization."""
    # they can be set both, when they None
    mean = [4.07]
    std = [14.07]
    data = DataConfig(**minimum_data)
    data.set_means_and_stds(mean, std)
    assert data.image_means == mean
    assert data.image_stds == std

    # Set also target mean and std
    data.set_means_and_stds(mean, std, mean, std)
    assert data.target_means == mean
    assert data.target_stds == std


def test_normalize_not_accepted(minimum_data: dict):
    """Test that normalize is not accepted, because it is mandatory and applied else
    where."""
    minimum_data["image_means"] = [10.4]
    minimum_data["image_stds"] = [3.2]
    minimum_data["transforms"] = [
        NormalizeModel(image_means=[0.485], image_stds=[0.229])
    ]

    with pytest.raises(ValueError):
        DataConfig(**minimum_data)


def test_patch_size(minimum_data: dict):
    """Test that non-zero even patch size are accepted."""
    # 2D
    data_model = DataConfig(**minimum_data)

    # 3D
    minimum_data["patch_size"] = [16, 8, 8]
    minimum_data["axes"] = "ZYX"

    data_model = DataConfig(**minimum_data)
    assert data_model.patch_size == minimum_data["patch_size"]


@pytest.mark.parametrize(
    "patch_size", [[12], [0, 12, 12], [12, 12, 13], [16, 10, 16], [12, 12, 12, 12]]
)
def test_wrong_patch_size(minimum_data: dict, patch_size):
    """Test that wrong patch sizes are not accepted (zero or odd, dims 1 or > 3)."""
    minimum_data["axes"] = "ZYX" if len(patch_size) == 3 else "YX"
    minimum_data["patch_size"] = patch_size

    with pytest.raises(ValueError):
        DataConfig(**minimum_data)


def test_set_3d(minimum_data: dict):
    """Test that 3D can be set."""
    data = DataConfig(**minimum_data)
    assert "Z" not in data.axes
    assert len(data.patch_size) == 2

    # error if changing Z manually
    with pytest.raises(ValueError):
        data.axes = "ZYX"

    # or patch size
    data = DataConfig(**minimum_data)
    with pytest.raises(ValueError):
        data.patch_size = [64, 64, 64]

    # set 3D
    data = DataConfig(**minimum_data)
    data.set_3D("ZYX", [64, 64, 64])
    assert "Z" in data.axes
    assert len(data.patch_size) == 3


@pytest.mark.parametrize(
    "transforms",
    [
        [
            {"name": SupportedTransform.XY_FLIP.value},
            {"name": SupportedTransform.N2V_MANIPULATE.value},
        ],
        [
            {"name": SupportedTransform.XY_FLIP.value},
        ],
        [
            {"name": SupportedTransform.XY_FLIP.value},
            {"name": SupportedTransform.XY_RANDOM_ROTATE90.value},
            {"name": SupportedTransform.N2V_MANIPULATE.value},
        ],
    ],
)
def test_passing_supported_transforms(minimum_data: dict, transforms):
    """Test that list of supported transforms can be passed."""
    minimum_data["transforms"] = transforms
    model = DataConfig(**minimum_data)

    supported = {
        "XYFlip": XYFlipModel,
        "XYRandomRotate90": XYRandomRotate90Model,
        "N2VManipulate": N2VManipulateModel,
    }

    for ind, t in enumerate(transforms):
        assert t["name"] == model.transforms[ind].name
        assert isinstance(model.transforms[ind], supported[t["name"]])


@pytest.mark.parametrize(
    "transforms",
    [
        [
            {"name": SupportedTransform.N2V_MANIPULATE.value},
            {"name": SupportedTransform.XY_FLIP.value},
        ],
        [
            {"name": SupportedTransform.N2V_MANIPULATE.value},
        ],
        [
            {"name": SupportedTransform.XY_FLIP.value},
            {"name": SupportedTransform.N2V_MANIPULATE.value},
            {"name": SupportedTransform.XY_RANDOM_ROTATE90.value},
        ],
    ],
)
def test_n2vmanipulate_last_transform(minimum_data: dict, transforms):
    """Test that N2V Manipulate is moved to the last position if it is not."""
    minimum_data["transforms"] = transforms
    model = DataConfig(**minimum_data)
    assert model.transforms[-1].name == SupportedTransform.N2V_MANIPULATE.value


def test_multiple_n2v_manipulate(minimum_data: dict):
    """Test that passing multiple n2v manipulate raises an error."""
    minimum_data["transforms"] = [
        {"name": SupportedTransform.N2V_MANIPULATE.value},
        {"name": SupportedTransform.N2V_MANIPULATE.value},
    ]
    with pytest.raises(ValueError):
        DataConfig(**minimum_data)


def test_remove_n2v_manipulate(minimum_data: dict):
    """Test that N2V Manipulate can be removed."""
    minimum_data["transforms"] = [
        {"name": SupportedTransform.XY_FLIP.value},
        {"name": SupportedTransform.N2V_MANIPULATE.value},
    ]
    model = DataConfig(**minimum_data)
    model.remove_n2v_manipulate()
    assert len(model.transforms) == 1
    assert model.transforms[-1].name == SupportedTransform.XY_FLIP.value


def test_add_n2v_manipulate(minimum_data: dict):
    """Test that N2V Manipulate can be added."""
    minimum_data["transforms"] = [
        {"name": SupportedTransform.XY_FLIP.value},
    ]
    model = DataConfig(**minimum_data)
    model.add_n2v_manipulate()
    assert len(model.transforms) == 2
    assert model.transforms[-1].name == SupportedTransform.N2V_MANIPULATE.value

    # test that adding twice doesn't change anything
    model.add_n2v_manipulate()
    assert len(model.transforms) == 2
    assert model.transforms[-1].name == SupportedTransform.N2V_MANIPULATE.value


def test_correct_transform_parameters(minimum_data: dict):
    """Test that the transforms have the correct parameters.

    This is important to know that the transforms are not all instantiated as
    a generic transform.
    """
    minimum_data["transforms"] = [
        {"name": SupportedTransform.XY_FLIP.value},
        {"name": SupportedTransform.XY_RANDOM_ROTATE90.value},
        {"name": SupportedTransform.N2V_MANIPULATE.value},
    ]
    model = DataConfig(**minimum_data)

    # N2VManipulate
    params = model.transforms[-1].model_dump()
    assert "roi_size" in params
    assert "masked_pixel_percentage" in params
    assert "strategy" in params
    assert "struct_mask_axis" in params
    assert "struct_mask_span" in params


def test_passing_empty_transforms(minimum_data: dict):
    """Test that empty list of transforms can be passed."""
    minimum_data["transforms"] = []
    DataConfig(**minimum_data)


def test_passing_incorrect_element(minimum_data: dict):
    """Test that incorrect element in the list of transforms raises an error (
    e.g. passing un object rather than a string)."""
    minimum_data["transforms"] = [
        {"name": get_all_transforms()[SupportedTransform.XY_FLIP.value]()},
    ]
    with pytest.raises(ValueError):
        DataConfig(**minimum_data)


def test_set_n2v_strategy(minimum_data: dict):
    """Test that the N2V strategy can be set."""
    uniform = SupportedPixelManipulation.UNIFORM.value
    median = SupportedPixelManipulation.MEDIAN.value

    data = DataConfig(**minimum_data)
    assert data.transforms[-1].name == SupportedTransform.N2V_MANIPULATE.value
    assert data.transforms[-1].strategy == uniform

    data.set_N2V2_strategy(median)
    assert data.transforms[-1].strategy == median

    data.set_N2V2_strategy(uniform)
    assert data.transforms[-1].strategy == uniform


def test_set_n2v_strategy_wrong_value(minimum_data: dict):
    """Test that passing a wrong strategy raises an error."""
    data = DataConfig(**minimum_data)
    with pytest.raises(ValueError):
        data.set_N2V2_strategy("wrong_value")


def test_set_struct_mask(minimum_data: dict):
    """Test that the struct mask can be set."""
    none = SupportedStructAxis.NONE.value
    vertical = SupportedStructAxis.VERTICAL.value
    horizontal = SupportedStructAxis.HORIZONTAL.value

    data = DataConfig(**minimum_data)
    assert data.transforms[-1].name == SupportedTransform.N2V_MANIPULATE.value
    assert data.transforms[-1].struct_mask_axis == none
    assert data.transforms[-1].struct_mask_span == 5

    data.set_structN2V_mask(vertical, 3)
    assert data.transforms[-1].struct_mask_axis == vertical
    assert data.transforms[-1].struct_mask_span == 3

    data.set_structN2V_mask(horizontal, 7)
    assert data.transforms[-1].struct_mask_axis == horizontal
    assert data.transforms[-1].struct_mask_span == 7

    data.set_structN2V_mask(none, 11)
    assert data.transforms[-1].struct_mask_axis == none
    assert data.transforms[-1].struct_mask_span == 11


def test_set_struct_mask_wrong_value(minimum_data: dict):
    """Test that passing a wrong struct mask axis raises an error."""
    data = DataConfig(**minimum_data)
    with pytest.raises(ValueError):
        data.set_structN2V_mask("wrong_value", 3)

    with pytest.raises(ValueError):
        data.set_structN2V_mask(SupportedStructAxis.VERTICAL.value, 1)


def test_export_to_yaml_float32_stats(tmp_path, minimum_data: dict):
    """Test exporting and loading the pydantic model when the statistics are
    np.float32."""
    data = DataConfig(**minimum_data)

    # set np.float32 stats values
    data.set_means_and_stds([np.float32(1234.5678)], [np.float32(21.73)])

    # export to yaml
    config_path = tmp_path / "data_config.yaml"
    with open(config_path, "w") as f:
        # dump configuration
        yaml.dump(data.model_dump(), f, default_flow_style=False, sort_keys=False)

    # load model
    dictionary = yaml.load(config_path.open("r"), Loader=yaml.SafeLoader)
    read_data = DataConfig(**dictionary)
    assert read_data.model_dump() == data.model_dump()
