import pytest

from careamics.config.architectures import (
    clear_custom_models,
    get_custom_model,
    register_model,
)


# register a model
@register_model(name="mymodel")
class MyModel:
    model_name: str
    model_id: int


def test_register_model():
    """Test the register_model decorator."""

    # get custom model
    model = get_custom_model("mymodel")

    # check if it is a subclass of MyModel
    assert issubclass(model, MyModel)


def test_wrong_model():
    """Test that an error is raised if an unknown model is requested."""
    get_custom_model("mymodel")

    with pytest.raises(ValueError):
        get_custom_model("unknown_model")


@pytest.mark.skip("This tests prevents other tests with custom models to pass.")
def test_clear_custom_models():
    """Test that the custom models are cleared."""
    # retrieve model
    get_custom_model("mymodel")

    # clear custom models
    clear_custom_models()

    # request the model again
    with pytest.raises(ValueError):
        get_custom_model("mymodel")
