from pathlib import Path

import pytest

from careamics.config import (
    Configuration,
    load_configuration,
    save_configuration,
)
from careamics.config.support import (
    SupportedAlgorithm,
    SupportedPixelManipulation,
    SupportedTransform,
)


@pytest.mark.parametrize("name", ["Sn4K3", "C4_M e-L"])
def test_valid_names(minimum_configuration: dict, name: str):
    """Test valid names (letters, numbers, spaces, dashes and underscores)."""
    minimum_configuration["experiment_name"] = name
    myconf = Configuration(**minimum_configuration)
    assert myconf.experiment_name == name


@pytest.mark.parametrize("name", ["", "   ", "#", "/", "^", "%", ",", ".", "a=b"])
def test_invalid_names(minimum_configuration: dict, name: str):
    """Test that invalid names raise an error."""
    minimum_configuration["experiment_name"] = name
    with pytest.raises(ValueError):
        Configuration(**minimum_configuration)


def test_3D_algorithm_and_data_compatibility(minimum_configuration: dict):
    """Test that errors are raised if algorithm `is_3D` and data axes are
    incompatible.
    """
    # 3D but no Z in axes
    minimum_configuration["algorithm_config"]["model"]["conv_dims"] = 3
    config = Configuration(**minimum_configuration)
    assert config.algorithm_config.model.conv_dims == 2

    # 2D but Z in axes
    minimum_configuration["algorithm_config"]["model"]["conv_dims"] = 2
    minimum_configuration["data_config"]["axes"] = "ZYX"
    minimum_configuration["data_config"]["patch_size"] = [64, 64, 64]
    config = Configuration(**minimum_configuration)
    assert config.algorithm_config.model.conv_dims == 3


def test_set_3D(minimum_configuration: dict):
    """Test the set 3D method."""
    conf = Configuration(**minimum_configuration)

    # set to 3D
    conf.set_3D(True, "ZYX", [64, 64, 64])
    assert conf.data_config.axes == "ZYX"
    assert conf.data_config.patch_size == [64, 64, 64]
    assert conf.algorithm_config.model.conv_dims == 3

    # set to 2D
    conf.set_3D(False, "SYX", [64, 64])
    assert conf.data_config.axes == "SYX"
    assert conf.data_config.patch_size == [64, 64]
    assert conf.algorithm_config.model.conv_dims == 2


def test_algorithm_and_data_default_transforms(minimum_configuration: dict):
    """Test that the default data transforms are compatible with n2v."""
    minimum_configuration["algorithm_config"] = {
        "algorithm": "n2v",
        "loss": "n2v",
        "model": {
            "architecture": "UNet",
        },
    }
    Configuration(**minimum_configuration)


@pytest.mark.parametrize(
    "algorithm, strategy",
    [
        ("n2v", SupportedPixelManipulation.UNIFORM.value),
        ("n2v", SupportedPixelManipulation.MEDIAN.value),
        ("n2v2", SupportedPixelManipulation.UNIFORM.value),
        ("n2v2", SupportedPixelManipulation.MEDIAN.value),
    ],
)
def test_n2v2_and_transforms(minimum_configuration: dict, algorithm, strategy):
    """Test that the manipulation strategy is corrected if the data transforms are
    incompatible with n2v2."""
    use_n2v2 = algorithm == "n2v2"
    minimum_configuration["algorithm_config"] = {
        "algorithm": "n2v",
        "loss": "n2v",
        "model": {
            "architecture": "UNet",
            "n2v2": use_n2v2,
        },
    }

    expected_strategy = (
        SupportedPixelManipulation.MEDIAN.value
        if use_n2v2
        else SupportedPixelManipulation.UNIFORM.value
    )

    # missing ManipulateN2V
    minimum_configuration["data_config"]["transforms"] = [
        {"name": SupportedTransform.XY_FLIP.value}
    ]
    config = Configuration(**minimum_configuration)
    assert len(config.data_config.transforms) == 2
    assert (
        config.data_config.transforms[-1].name
        == SupportedTransform.N2V_MANIPULATE.value
    )
    assert config.data_config.transforms[-1].strategy == expected_strategy

    # passing ManipulateN2V with the wrong strategy
    minimum_configuration["data_config"]["transforms"] = [
        {
            "name": SupportedTransform.N2V_MANIPULATE.value,
            "strategy": strategy,
        }
    ]
    config = Configuration(**minimum_configuration)
    assert config.data_config.transforms[-1].strategy == expected_strategy


def test_setting_n2v2(minimum_configuration: dict):
    # make sure we use n2v
    minimum_configuration["algorithm_config"][
        "algorithm"
    ] = SupportedAlgorithm.N2V.value

    # test config
    config = Configuration(**minimum_configuration)
    assert config.algorithm_config.algorithm == SupportedAlgorithm.N2V.value
    assert not config.algorithm_config.model.n2v2
    assert (
        config.data_config.transforms[-1].strategy
        == SupportedPixelManipulation.UNIFORM.value
    )

    # set N2V2
    config.set_N2V2(True)
    assert config.algorithm_config.model.n2v2
    assert (
        config.data_config.transforms[-1].strategy
        == SupportedPixelManipulation.MEDIAN.value
    )

    # set back to N2V
    config.set_N2V2(False)
    assert not config.algorithm_config.model.n2v2
    assert (
        config.data_config.transforms[-1].strategy
        == SupportedPixelManipulation.UNIFORM.value
    )


def test_config_to_yaml(tmp_path: Path, minimum_configuration: dict):
    """Test that we can export a config to yaml and load it back"""

    # test that we can instantiate a config
    myconf = Configuration(**minimum_configuration)

    # export to yaml
    yaml_path = save_configuration(myconf, tmp_path)
    assert yaml_path.exists()

    # load from yaml
    my_other_conf = load_configuration(yaml_path)
    assert my_other_conf == myconf


def test_config_to_yaml_wrong_path(tmp_path: Path, minimum_configuration: dict):
    """Test that an error is raised when the path is not a directory and not a .yml"""

    # test that we can instantiate a config
    myconf = Configuration(**minimum_configuration)

    # export to yaml
    yaml_path = tmp_path / "tmp.txt"
    with pytest.raises(ValueError):
        save_configuration(myconf, yaml_path)

    # existing file
    yaml_path.touch()
    with pytest.raises(ValueError):
        save_configuration(myconf, yaml_path)
