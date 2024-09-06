import pytest
import torch

from careamics.config import FCNAlgorithmConfig
from careamics.lightning.lightning_module import (
    FCNModule,
    create_careamics_module,
)

# TODO: rename to test_FCN_lightining_module.py


def test_careamics_module(minimum_algorithm_n2v):
    """Test that the minimum algorithm allows instantiating a the Lightning API
    intermediate layer."""
    algo_config = FCNAlgorithmConfig(**minimum_algorithm_n2v)

    # extract model parameters
    model_parameters = algo_config.model.model_dump(exclude_none=True)

    # instantiate FCNModule
    create_careamics_module(
        algorithm_type=algo_config.algorithm_type,
        algorithm=algo_config.algorithm,
        loss=algo_config.loss,
        architecture=algo_config.model.architecture,
        model_parameters=model_parameters,
        optimizer=algo_config.optimizer.name,
        optimizer_parameters=algo_config.optimizer.parameters,
        lr_scheduler=algo_config.lr_scheduler.name,
        lr_scheduler_parameters=algo_config.lr_scheduler.parameters,
    )


def test_careamics_fcn(minimum_algorithm_n2v):
    """Test that the minimum algorithm allows instantiating a CAREamicsKiln."""
    algo_config = FCNAlgorithmConfig(**minimum_algorithm_n2v)

    # instantiate CAREamicsKiln
    FCNModule(algo_config)


@pytest.mark.parametrize(
    "shape",
    [
        (8, 8),
        (16, 16),
        (32, 32),
    ],
)
def test_careamics_kiln_unet_2D_depth_2_shape(shape):
    algo_dict = {
        "algorithm_type": "fcn",
        "algorithm": "n2n",
        "model": {
            "architecture": "UNet",
            "conv_dims": 2,
            "in_channels": 1,
            "num_classes": 1,
            "depth": 2,
        },
        "loss": "mae",
    }
    algo_config = FCNAlgorithmConfig(**algo_dict)

    # instantiate CAREamicsKiln
    model = FCNModule(algo_config)
    # set model to evaluation mode to avoid batch dimension error
    model.model.eval()
    # test forward pass
    x = torch.rand((1, 1, *shape))
    y: torch.Tensor = model.forward(x)
    assert y.shape == x.shape


@pytest.mark.parametrize(
    "shape",
    [
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
    ],
)
def test_careamics_kiln_unet_2D_depth_3_shape(shape):
    algo_dict = {
        "algorithm_type": "fcn",
        "algorithm": "n2n",
        "model": {
            "architecture": "UNet",
            "conv_dims": 2,
            "in_channels": 1,
            "num_classes": 1,
            "depth": 3,
        },
        "loss": "mae",
    }
    algo_config = FCNAlgorithmConfig(**algo_dict)

    # instantiate CAREamicsKiln
    model = FCNModule(algo_config)
    # set model to evaluation mode to avoid batch dimension error
    model.model.eval()
    # test forward pass
    x = torch.rand((1, 1, *shape))
    y: torch.Tensor = model.forward(x)
    assert y.shape == x.shape


@pytest.mark.parametrize(
    "shape",
    [
        (8, 32, 16),
        (16, 32, 16),
        (8, 32, 32),
        (32, 64, 64),
    ],
)
def test_careamics_kiln_unet_depth_2_3D(shape):
    algo_dict = {
        "algorithm_type": "fcn",
        "algorithm": "n2n",
        "model": {
            "architecture": "UNet",
            "conv_dims": 3,
            "in_channels": 1,
            "num_classes": 1,
            "depth": 2,
        },
        "loss": "mae",
    }
    algo_config = FCNAlgorithmConfig(**algo_dict)

    # instantiate CAREamicsKiln
    model = FCNModule(algo_config)
    # set model to evaluation mode to avoid batch dimension error
    model.model.eval()
    # test forward pass
    x = torch.rand((1, 1, *shape))
    y: torch.Tensor = model.forward(x)
    assert y.shape == x.shape


@pytest.mark.parametrize(
    "shape",
    [
        (8, 64, 64),
        (16, 64, 64),
        (16, 128, 128),
        (32, 128, 128),
    ],
)
def test_careamics_kiln_unet_depth_3_3D(shape):
    algo_dict = {
        "algorithm_type": "fcn",
        "algorithm": "n2n",
        "model": {
            "architecture": "UNet",
            "conv_dims": 3,
            "in_channels": 1,
            "num_classes": 1,
            "depth": 3,
        },
        "loss": "mae",
    }
    algo_config = FCNAlgorithmConfig(**algo_dict)

    # instantiate CAREamicsKiln
    model = FCNModule(algo_config)
    # set model to evaluation mode to avoid batch dimension error
    model.model.eval()
    # test forward pass
    x = torch.rand((1, 1, *shape))
    y: torch.Tensor = model.forward(x)
    assert y.shape == x.shape


@pytest.mark.parametrize(
    "shape",
    [
        (8, 64, 64),
        (16, 64, 64),
        (16, 128, 128),
        (32, 128, 128),
    ],
)
def test_careamics_kiln_unet_depth_3_3D_n2v2(shape):
    algo_dict = {
        "algorithm_type": "fcn",
        "algorithm": "n2v",
        "model": {
            "architecture": "UNet",
            "conv_dims": 3,
            "in_channels": 1,
            "num_classes": 1,
            "depth": 3,
            "n2v2": True,
        },
        "loss": "n2v",
    }
    algo_config = FCNAlgorithmConfig(**algo_dict)

    # instantiate CAREamicsKiln
    model = FCNModule(algo_config)
    # set model to evaluation mode to avoid batch dimension error
    model.model.eval()
    # test forward pass
    x = torch.rand((1, 1, *shape))
    y: torch.Tensor = model.forward(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("n_channels", [1, 3, 4])
def test_careamics_kiln_unet_depth_2_channels_2D(n_channels):
    algo_dict = {
        "algorithm_type": "fcn",
        "algorithm": "n2n",
        "model": {
            "architecture": "UNet",
            "conv_dims": 2,
            "in_channels": n_channels,
            "num_classes": n_channels,
            "depth": 2,
        },
        "loss": "mae",
    }
    algo_config = FCNAlgorithmConfig(**algo_dict)

    # instantiate CAREamicsKiln
    model = FCNModule(algo_config)
    # set model to evaluation mode to avoid batch dimension error
    model.model.eval()
    # test forward pass
    x = torch.rand((1, n_channels, 32, 32))
    y: torch.Tensor = model.forward(x)
    assert y.shape == x.shape


@pytest.mark.parametrize(
    "n_channels,independent_channels",
    [(1, False), (1, True), (3, False), (3, True), (4, False), (4, True)],
)
def test_careamics_kiln_unet_depth_3_channels_2D(n_channels, independent_channels):
    algo_dict = {
        "algorithm_type": "fcn",
        "algorithm": "n2n",
        "model": {
            "architecture": "UNet",
            "conv_dims": 2,
            "in_channels": n_channels,
            "num_classes": n_channels,
            "depth": 3,
            "independent_channels": independent_channels,
        },
        "loss": "mae",
    }
    algo_config = FCNAlgorithmConfig(**algo_dict)

    # instantiate CAREamicsKiln
    model = FCNModule(algo_config)
    # set model to evaluation mode to avoid batch dimension error
    model.model.eval()
    # test forward pass
    x = torch.rand((1, n_channels, 64, 64))
    y: torch.Tensor = model.forward(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("n_channels", [1, 3, 4])
def test_careamics_kiln_unet_depth_2_channels_3D(n_channels):
    algo_dict = {
        "algorithm_type": "fcn",
        "algorithm": "n2n",
        "model": {
            "architecture": "UNet",
            "conv_dims": 3,
            "in_channels": n_channels,
            "num_classes": n_channels,
            "depth": 2,
        },
        "loss": "mae",
    }
    algo_config = FCNAlgorithmConfig(**algo_dict)

    # instantiate CAREamicsKiln
    model = FCNModule(algo_config)
    # set model to evaluation mode to avoid batch dimension error
    model.model.eval()
    # test forward pass
    x = torch.rand((2, n_channels, 16, 32, 32))
    y: torch.Tensor = model.forward(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("n_channels", [1, 3, 4])
def test_careamics_kiln_unet_depth_3_channels_3D(n_channels):
    algo_dict = {
        "algorithm_type": "fcn",
        "algorithm": "n2n",
        "model": {
            "architecture": "UNet",
            "conv_dims": 3,
            "in_channels": n_channels,
            "num_classes": n_channels,
            "depth": 3,
        },
        "loss": "mae",
    }
    algo_config = FCNAlgorithmConfig(**algo_dict)

    # instantiate CAREamicsKiln
    model = FCNModule(algo_config)
    # set model to evaluation mode to avoid batch dimension error
    model.model.eval()
    # test forward pass
    x = torch.rand((1, n_channels, 16, 64, 64))
    y: torch.Tensor = model.forward(x)
    assert y.shape == x.shape
