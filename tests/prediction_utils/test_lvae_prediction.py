import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from careamics.config.inference_model import InferenceConfig
from careamics.config.tile_information import TileInformation
from careamics.dataset import InMemoryTiledPredDataset
from careamics.dataset.tiling.collate_tiles import collate_tiles
from careamics.models.lvae.likelihoods import GaussianLikelihood
from careamics.models.lvae.lvae import LadderVAE
from careamics.prediction_utils import convert_outputs
from careamics.prediction_utils.lvae_prediction import (
    lvae_predict_mmse_tiled_batch,
    lvae_predict_tiled_batch,
)


@pytest.fixture
def minimum_lvae_params():
    return {
        "input_shape": 64,
        "output_channels": 2,
        "multiscale_count": None,
        "z_dims": [128, 128, 128, 128],
        "encoder_n_filters": 64,
        "decoder_n_filters": 64,
        "encoder_dropout": 0.1,
        "decoder_dropout": 0.1,
        "nonlinearity": "ELU",
        "predict_logvar": "pixelwise",
        "analytical_kl": False,
    }


@pytest.fixture
def gaussian_likelihood_params():
    return {"predict_logvar": "pixelwise", "logvar_lowerbound": -5}


# TODO: Test with mock LCMultiChDloader


@pytest.mark.skip(
    reason=(
        "Doesn't make sense to use `InMemoryTiledPredDataset` dataset because it does "
        "not handle lateral context inputs."
    )
)
@pytest.mark.parametrize("predict_logvar", ["pixelwise", None])
@pytest.mark.parametrize("output_channels", [2, 3])
def test_smoke_careamics_dset_lvae_prediction(
    minimum_lvae_params, gaussian_likelihood_params, predict_logvar, output_channels
):
    minimum_lvae_params["predict_logvar"] = predict_logvar
    minimum_lvae_params["output_channels"] = output_channels
    gaussian_likelihood_params["predict_logvar"] = predict_logvar
    input_shape = minimum_lvae_params["input_shape"]

    # initialize model
    model = LadderVAE(**minimum_lvae_params)
    # initialize likelihood
    likelihood_obj = GaussianLikelihood(**gaussian_likelihood_params)

    # create predict dataset
    inference_dict = {
        "data_type": "array",
        "axes": "SYX",
        "tile_size": [input_shape, input_shape],
        "tile_overlap": [(input_shape // 4) * 2, (input_shape // 4) * 2],  # ensure even
        "image_means": [2.0],
        "image_stds": [1.0],
        "tta_transforms": False,
    }
    inference_config = InferenceConfig(**inference_dict)
    N_samples = 3
    data_shape = (N_samples, input_shape * 4 + 23, input_shape * 4 + 23)
    data = np.random.random_sample(size=data_shape)
    dataset = InMemoryTiledPredDataset(inference_config, data)
    dataloader = DataLoader(dataset, collate_fn=collate_tiles)

    tiled_predictions = []
    log_vars = []
    for batch in dataloader:
        y, log_var = lvae_predict_tiled_batch(model, likelihood_obj, batch)
        tiled_predictions.append(y)
        log_vars.append(log_var)

        # y is a 2-tuple, second element is tile info, similar for logvar
        assert y[0].shape == (1, output_channels, input_shape, input_shape)
        if predict_logvar == "pixelwise":
            assert log_var[0].shape == (1, output_channels, input_shape, input_shape)
        elif predict_logvar is None:
            assert log_var is None

    prediction_shape = (1, output_channels, *data_shape[-2:])
    predictions = convert_outputs(tiled_predictions, tiled=True)
    for prediction in predictions:
        assert prediction.shape == prediction_shape


@pytest.mark.parametrize("predict_logvar", ["pixelwise", None])
@pytest.mark.parametrize("output_channels", [2, 3])
def test_lvae_predict_single_sample(
    minimum_lvae_params, gaussian_likelihood_params, predict_logvar, output_channels
):
    """Test predictions of a single sample."""
    minimum_lvae_params["predict_logvar"] = predict_logvar
    minimum_lvae_params["output_channels"] = output_channels
    gaussian_likelihood_params["predict_logvar"] = predict_logvar

    input_shape = minimum_lvae_params["input_shape"]

    # initialize model
    model = LadderVAE(**minimum_lvae_params)
    # initialize likelihood
    likelihood_obj = GaussianLikelihood(**gaussian_likelihood_params)

    # dummy input
    x = torch.rand(size=(1, 1, input_shape, input_shape))
    tile_info = TileInformation(
        array_shape=(1, input_shape * 4, input_shape * 4),
        last_tile=False,
        overlap_crop_coords=((8, 8 + input_shape), (8, 8 + input_shape)),
        stitch_coords=((0, input_shape), (0, input_shape)),
        sample_id=0,
    )
    input_ = (x, [tile_info])  # simulate output of datasets
    # prediction
    y_tiled, log_var_tiled = lvae_predict_tiled_batch(model, likelihood_obj, input_)
    y = y_tiled[0]

    assert y.shape == (1, output_channels, input_shape, input_shape)
    if predict_logvar == "pixelwise":
        log_var = log_var_tiled[0]
        assert log_var.shape == (1, output_channels, input_shape, input_shape)
    elif predict_logvar is None:
        assert log_var_tiled is None


@pytest.mark.parametrize("predict_logvar", ["pixelwise", None])
@pytest.mark.parametrize("output_channels", [2, 3])
def test_lvae_predict_mmse_tiled_batch(
    minimum_lvae_params, gaussian_likelihood_params, predict_logvar, output_channels
):
    """Test MMSE prediction."""
    minimum_lvae_params["predict_logvar"] = predict_logvar
    minimum_lvae_params["output_channels"] = output_channels
    gaussian_likelihood_params["predict_logvar"] = predict_logvar

    input_shape = minimum_lvae_params["input_shape"]

    # initialize model
    model = LadderVAE(**minimum_lvae_params)
    # initialize likelihood
    likelihood_obj = GaussianLikelihood(**gaussian_likelihood_params)

    # dummy input
    x = torch.rand(size=(1, 1, input_shape, input_shape))
    tile_info = TileInformation(
        array_shape=(1, input_shape * 4, input_shape * 4),
        last_tile=False,
        overlap_crop_coords=((8, 8 + input_shape), (8, 8 + input_shape)),
        stitch_coords=((0, input_shape), (0, input_shape)),
        sample_id=0,
    )
    input_ = (x, [tile_info])  # simulate output of datasets
    # prediction
    y_tiled, y_std_tiled, log_var_tiled = lvae_predict_mmse_tiled_batch(
        model, likelihood_obj, input_, mmse_count=5
    )
    y = y_tiled[0]
    y_std = y_std_tiled[0]

    assert y.shape == (1, output_channels, input_shape, input_shape)
    assert y_std.shape == (1, output_channels, input_shape, input_shape)
    if predict_logvar == "pixelwise":
        log_var = log_var_tiled[0]
        assert log_var.shape == (1, output_channels, input_shape, input_shape)
    elif predict_logvar is None:
        assert log_var_tiled is None
