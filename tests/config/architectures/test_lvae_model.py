import pytest

from careamics.config.architectures import LVAEModel
from careamics.config.support import SupportedActivation


def test_instantiation():
    """Test that LVAEModel can be instantiated."""
    model_params = {
        "architecture": "LVAE",
    }

    # instantiate model
    LVAEModel(**model_params)


def test_architecture_missing():
    """Test that LVAEModel requires architecture."""
    model_params = {
        "input_shape": 64,
    }

    with pytest.raises(ValueError):
        LVAEModel(**model_params)


@pytest.mark.parametrize("encoder_n_filters", [8, 16, 32, 96, 128])
def test_encoder_n_filters(encoder_n_filters: int):
    """Test that LVAEModel accepts num_channels_init as an even number and
    minimum 8."""
    model_params = {"architecture": "LVAE", "encoder_n_filters": encoder_n_filters}

    # instantiate model
    LVAEModel(**model_params)


@pytest.mark.parametrize("decoder_n_filters", [8, 16, 32, 96, 128])
def test_decoder_n_filters(decoder_n_filters: int):
    """Test that LVAEModel accepts num_channels_init as an even number and
    minimum 8."""
    model_params = {"architecture": "LVAE", "decoder_n_filters": decoder_n_filters}

    # instantiate model
    LVAEModel(**model_params)


@pytest.mark.parametrize("n_filters", [2, 17, 127])
def test_wrong_num_filters(n_filters: int):
    """Test that wrong num_channels_init causes an error."""
    model_params = {"architecture": "LVAE", "encoder_n_filters": n_filters}
    with pytest.raises(ValueError):
        LVAEModel(**model_params)

    model_params = {"architecture": "LVAE", "decoder_n_filters": n_filters}
    with pytest.raises(ValueError):
        LVAEModel(**model_params)


def test_activations():
    """Test that LVAEModel accepts all activations."""
    for act in SupportedActivation:
        model_params = {
            "architecture": "LVAE",
            "nonlinearity": act.value,
        }

        # instantiate model
        LVAEModel(**model_params)


def test_all_activations_are_supported():
    """Test that all activations defined in the Literal are supported."""
    # list of supported activations
    activations = list(SupportedActivation)

    # Algorithm json schema
    schema = LVAEModel.model_json_schema()

    # check that all activations are supported
    for act in schema["properties"]["nonlinearity"]["enum"]:
        assert act in activations


def test_activation_wrong_values():
    """Test that wrong values are not accepted."""
    model_params = {
        "architecture": "LVAE",
        "nonlinearity": "wrong",
    }

    with pytest.raises(ValueError):
        LVAEModel(**model_params)


def test_parameters_wrong_values_by_assigment():
    """Test that wrong values are not accepted through assignment."""
    model_params = {
        "architecture": "LVAE",
        "z_dims": (128, 128, 128),
        "encoder_n_filters": 32,
    }
    model = LVAEModel(**model_params)

    # z_dims
    model.z_dims = model_params["z_dims"]
    with pytest.raises(ValueError):
        model.depth = -1

    # number of channels in the encoder
    model.encoder_n_filters = model_params["encoder_n_filters"]
    with pytest.raises(ValueError):
        model.encoder_n_filters = 2


def test_model_dump():
    """Test that default values are excluded from model dump."""
    model_params = {
        "architecture": "LVAE",  # default value
        "z_dims": (128, 128, 128, 128),  # default value
        "nonlinearity": "ReLU",  # non-default value
        "decoder_n_filters": 32,  # non-default value
    }
    model = LVAEModel(**model_params)

    # dump model
    model_dict = model.model_dump(exclude_defaults=True)

    # check that default values are excluded except the architecture
    assert "architecture" not in model_dict
    assert len(model_dict) == 2

    # check that we get all the optional values with the exclude_defaults flag
    model_dict = model.model_dump(exclude_defaults=False)
    assert len(model_dict) == len(dict(model)) - 1
