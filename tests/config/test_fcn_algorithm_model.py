import pytest

from careamics.config.fcn_algorithm_model import FCNAlgorithmConfig
from careamics.config.support import (
    SupportedAlgorithm,
    SupportedArchitecture,
    SupportedLoss,
)


def test_all_algorithms_are_supported():
    """Test that all algorithms defined in the Literal are supported."""
    # list of supported algorithms
    algorithms = list(SupportedAlgorithm)

    # Algorithm json schema to extract the literal value
    schema = FCNAlgorithmConfig.model_json_schema()

    # check that all algorithms are supported
    for algo in schema["properties"]["algorithm"]["enum"]:
        assert algo in algorithms


# TODO: this should not support musplit and denoisplit losses
def test_all_losses_are_supported():
    """Test that all losses defined in the Literal are supported."""
    # list of supported losses
    losses = list(SupportedLoss)

    # Algorithm json schema
    schema = FCNAlgorithmConfig.model_json_schema()

    # check that all losses are supported
    for loss in schema["properties"]["loss"]["enum"]:
        assert loss in losses


def test_model_discriminator(minimum_algorithm_n2v):
    """Test that discriminator permits correct assignment."""
    for model_name in SupportedArchitecture:
        # TODO change once VAE are implemented
        if model_name.value == "UNet":
            minimum_algorithm_n2v["model"]["architecture"] = model_name.value

            algo = FCNAlgorithmConfig(**minimum_algorithm_n2v)
            assert algo.model.architecture == model_name.value


@pytest.mark.parametrize(
    "algorithm, loss, model",
    [
        ("n2v", "n2v", {"architecture": "UNet", "n2v2": False}),
        ("n2n", "mae", {"architecture": "UNet", "n2v2": False}),
        ("care", "mae", {"architecture": "UNet", "n2v2": False}),
    ],
)
def test_algorithm_constraints(algorithm: str, loss: str, model: dict):
    """Test that constraints are passed for each algorithm."""
    FCNAlgorithmConfig(algorithm=algorithm, loss=loss, model=model)


def test_n_channels_n2v():
    """Check that an error is raised if n2v has different number of channels in
    input and output."""
    model = {
        "architecture": "UNet",
        "in_channels": 1,
        "num_classes": 2,
        "n2v2": False,
    }
    loss = "n2v"

    with pytest.raises(ValueError):
        FCNAlgorithmConfig(algorithm="n2v", loss=loss, model=model)


@pytest.mark.parametrize(
    "algorithm, n_in, n_out",
    [
        ("n2v", 2, 2),
        ("n2n", 3, 3),
        ("care", 1, 2),
    ],
)
def test_comaptiblity_of_number_of_channels(algorithm, n_in, n_out):
    """Check that no error is thrown when instantiating the algorithm with a valid
    number of in and out channels."""
    model = {
        "architecture": "UNet",
        "in_channels": n_in,
        "num_classes": n_out,
        "n2v2": False,
    }
    loss = "n2v" if algorithm == "n2v" else "mae"

    FCNAlgorithmConfig(algorithm=algorithm, loss=loss, model=model)


def test_custom_model(custom_model_parameters):
    """Test that a custom model can be instantiated."""
    # create algorithm configuration
    FCNAlgorithmConfig(
        algorithm=SupportedAlgorithm.CUSTOM.value,
        loss="mse",
        model=custom_model_parameters,
    )


def test_custom_model_wrong_algorithm(custom_model_parameters):
    """Test that a custom model fails if the algorithm is not custom."""
    # create algorithm configuration
    with pytest.raises(ValueError):
        FCNAlgorithmConfig(
            algorithm=SupportedAlgorithm.CARE.value,
            loss="mse",
            model=custom_model_parameters,
        )
