from torch import nn, ones

from careamics.config.architectures import (
    CustomModel,
    UNetModel,
    register_model,
)
from careamics.config.support import SupportedArchitecture
from careamics.models import model_factory
from careamics.models.unet import UNet


def test_model_registry_unet():
    """Test that"""
    model_config = {
        "architecture": "UNet",
    }

    # instantiate model
    model = model_factory(UNetModel(**model_config))
    assert isinstance(model, UNet)


def test_model_registry_custom():
    """Test that a custom model can be retrieved and instantiated."""

    # create and register a custom model
    @register_model(name="linear_model")
    class LinearModel(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(ones(in_features, out_features))
            self.bias = nn.Parameter(ones(out_features))

        def forward(self, input):
            return (input @ self.weight) + self.bias

    model_config = {
        "architecture": SupportedArchitecture.CUSTOM.value,
        "name": "linear_model",
        "in_features": 10,
        "out_features": 5,
    }

    # instantiate model
    model = model_factory(CustomModel(**model_config))
    assert isinstance(model, LinearModel)
    assert model.in_features == 10
    assert model.out_features == 5
