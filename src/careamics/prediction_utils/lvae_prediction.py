"""Module containing pytorch implementations for obtaining predictions from an LVAE."""

from typing import Any, Optional

import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from careamics.dataset.tiling.lvae_tiled_patching import compute_tile_info_legacy
from careamics.lvae_training.dataset.vae_dataset import MultiChDloader
from careamics.models.lvae import LadderVAE as LVAE
from careamics.models.lvae.likelihoods import LikelihoodModule
from careamics.prediction_utils import convert_outputs

# TODO: convert these functions to lightning module `predict_step`
#   -> mmse_count will have to be an instance attribute?


# This function is needed because the output of the datasets (input here) can include
#   auxillary items, such as the TileInformation. This function allows for easier reuse
#   between lvae_predict_single_sample and lvae_predict_mmse.
def lvae_predict_single_sample(
    model: LVAE,
    likelihood_obj: LikelihoodModule,
    input: torch.Tensor,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Generate a single sample prediction from an LVAE model, for a given input.

    Parameters
    ----------
    model : LVAE
        Trained LVAE model.
    likelihood_obj : LikelihoodModule
        Instance of a likelihood class.
    input : torch.tensor
        Input to generate prediction for. Expected shape is (S, C, Y, X).

    Returns
    -------
    tuple of (torch.tensor, optional torch.tensor)
        The first element is the sample prediction, and the second element is the
        log-variance. The log-variance will be None if `model.predict_logvar is None`.
    """
    model.eval()  # Not in original predict code: effects batch_norm and dropout layers
    with torch.no_grad():
        output: torch.Tensor
        output, _ = model(input)  # 2nd item is top-down data dict

    # presently, get_mean_lv just splits the output in 2 if predict_logvar=True,
    #   optionally clips the logvavr if logvar_lowerbound is not None
    # TODO: consider refactoring to remove use of the likelihood object
    sample_prediction, log_var = likelihood_obj.get_mean_lv(output)

    # TODO: output denormalization using target stats that will be saved in data config
    # -> Don't think we need this, saw it in a random bit of code somewhere.

    return sample_prediction, log_var


def lvae_predict_tiled_batch(
    model: LVAE,
    likelihood_obj: LikelihoodModule,
    input: tuple[Any, ...],
) -> tuple[tuple[Any, ...], Optional[tuple[Any, ...]]]:
    # TODO: fix docstring return types, ... too many output options
    """
    Generate a single sample prediction from an LVAE model, for a given input.

    Parameters
    ----------
    model : LVAE
        Trained LVAE model.
    likelihood_obj : LikelihoodModule
        Instance of a likelihood class.
    input : torch.tensor | tuple of (torch.tensor, Any, ...)
        Input to generate prediction for. This can include auxilary inputs such as
        `TileInformation`, but the model input is always the first item of the tuple.
        Expected shape of the model input is (S, C, Y, X).

    Returns
    -------
    tuple of ((torch.tensor, Any, ...), optional tuple of (torch.tensor, Any, ...))
        The first element is the sample prediction, and the second element is the
        log-variance. The log-variance will be None if `model.predict_logvar is None`.
        Any auxillary data included in the input will also be include with both the
        sample prediction and the log-variance.
    """
    x: torch.Tensor
    aux: list[Any]
    x, *aux = input

    sample_prediction, log_var = lvae_predict_single_sample(
        model=model, likelihood_obj=likelihood_obj, input=x
    )

    log_var_output = (log_var, *aux) if log_var is not None else None
    return (sample_prediction, *aux), log_var_output


def lvae_predict_mmse_tiled_batch(
    model: LVAE,
    likelihood_obj: LikelihoodModule,
    input: tuple[Any, ...],
    mmse_count: int,
) -> tuple[tuple[Any, ...], tuple[Any], Optional[tuple[Any, ...]]]:
    # TODO: fix docstring return types, ... hard to make readable
    """
    Generate the MMSE (minimum mean squared error) prediction, for a given input.

    This is calculated from the mean of multiple single sample predictions.

    Parameters
    ----------
    model : LVAE
        Trained LVAE model.
    likelihood_obj : LikelihoodModule
        Instance of a likelihood class.
    input : torch.tensor | tuple of (torch.tensor, Any, ...)
        Input to generate prediction for. This can include auxilary inputs such as
        `TileInformation`, but the model input is always the first item of the tuple.
        Expected shape of the model input is (S, C, Y, X).
    mmse_count : int
        Number of samples to generate to calculate MMSE (minimum mean squared error).

    Returns
    -------
    tuple of (tuple of (torch.Tensor[Any], Any, ...))
        A tuple of 3 elements. The first element contains the MMSE prediction, the
        second contains the standard deviation of the samples used to create the MMSE
        prediction. Finally the last element contains the log-variance of the
        likelihood, this will be `None` if `likelihood.predict_logvar` is `None`.
        Any auxillary data included in the input will also be include with all of the
        MMSE prediction, the standard deviation, and the log-variance.
    """
    if mmse_count <= 0:
        raise ValueError("MMSE count must be greater than zero.")

    x: torch.Tensor
    aux: list[Any]
    x, *aux = input

    input_shape = x.shape
    output_shape = (input_shape[0], model.target_ch, *input_shape[2:])
    log_var: Optional[torch.Tensor] = None
    # pre-declare empty array to fill with individual sample predictions
    sample_predictions = torch.zeros(size=(mmse_count, *output_shape))
    for mmse_idx in range(mmse_count):
        sample_prediction, lv = lvae_predict_single_sample(
            model=model, likelihood_obj=likelihood_obj, input=x
        )
        # only keep the log variance of the first sample prediction
        if mmse_idx == 0:
            log_var = lv

        # store sample predictions
        sample_predictions[mmse_idx, ...] = sample_prediction

    mmse_prediction = torch.mean(sample_predictions, dim=0)
    mmse_prediction_std = torch.std(sample_predictions, dim=0)

    log_var_output = (log_var, *aux) if log_var is not None else None
    return (mmse_prediction, *aux), (mmse_prediction_std, *aux), log_var_output


def lvae_predict_mmse_dataset(
    model: LVAE,
    likelihood_obj: LikelihoodModule,
    dataset: MultiChDloader,
    mmse_count: int,
    batch_size: int,
    num_workers: int,
) -> tuple[list[NDArray], list[NDArray], list[NDArray]]:
    """
    Predict the MMSE result for a whole dataset.

    Parameters
    ----------
    model : LVAE
        Trained LVAE model.
    likelihood_obj : LikelihoodModule
        Instance of a likelihood class.
    dataset : MultiChDloader
        The dataset to predict on.
    mmse_count : int
        Number of samples to generate to calculate MMSE (minimum mean squared error).
    batch_size : int
        The batch size to predict with, passed to the PyTorch dataloader.
    num_workers : int
        The number of workers to predict with, passed to the PyTorch dataloader.

    Returns
    -------
    tuple[list[NDArray], list[NDArray], list[NDArray]]
        The stiched results. First list is the MMSE images, second list is the standard
        deviations and third list is the log variance.
    """
    dataloader = DataLoader(
        dataset,
        pin_memory=False,
        num_workers=num_workers,
        shuffle=False,
        batch_size=batch_size,
    )

    mmse_batches = []
    mmse_std_batches = []
    log_var_batches = []
    for batch_idx, batch in enumerate(dataloader):
        # TODO: move input to correct device, i.e. cuda if available
        input_: torch.Tensor
        input_, _ = batch[:2]

        # create tile information (simulating CAREamics dataset output)
        dataset_idx_start = (batch_idx // batch_size) + (batch_idx % batch_size)
        dataset_idx_iter = range(dataset_idx_start, dataset_idx_start + input_.shape[0])
        tile_infos = tuple(
            compute_tile_info_legacy(dataset.idx_manager, idx)
            for idx in dataset_idx_iter
        )

        # predict mmse
        mmse_batch, mmse_std_batch, log_var_batch = lvae_predict_mmse_tiled_batch(
            model, likelihood_obj, (input_, tile_infos), mmse_count
        )
        # keep results
        mmse_batches.append(mmse_batch)
        mmse_std_batches.append(mmse_std_batch)
        log_var_batches.append(log_var_batch)

    # stitch tiles
    mmse_stitched_images = convert_outputs(mmse_batches, tiled=True)
    mmse_std_stitched_images = convert_outputs(mmse_std_batches, tiled=True)
    log_var_stitched_images = convert_outputs(log_var_batches, tiled=True)

    return mmse_stitched_images, mmse_std_stitched_images, log_var_stitched_images
