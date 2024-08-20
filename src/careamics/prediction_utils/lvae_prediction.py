"""Module containing pytorch implementations for obtaining predictions from an LVAE."""

from typing import Any, Optional, Union, overload

import torch

from careamics.models import LVAE
from careamics.models.lvae.likelihoods import LikelihoodModule

# TODO: convert these functions to lightning module `predict_step`
#   -> mmse_count will have to be an instance attribute?


# This function is needed because the output of the datasets (input here) can include
#   auxillary items, such as the TileInformation. This function allows for easier reuse
#   between lvae_predict_single_sample and lvae_predict_mmse.
def _pure_lvae_predict_single_sample(
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


@overload
def lvae_predict_single_sample(  # numpydoc ignore=GL08
    model: LVAE, likelihood_obj: LikelihoodModule, input: tuple[Any]
) -> tuple[tuple[Any], Optional[tuple[Any]]]: ...


@overload
def lvae_predict_single_sample(  # numpydoc ignore=GL08
    model: LVAE, likelihood_obj: LikelihoodModule, input: torch.Tensor
) -> tuple[torch.Tensor, Optional[torch.Tensor]]: ...


def lvae_predict_single_sample(
    model: LVAE,
    likelihood_obj: LikelihoodModule,
    input: Union[torch.Tensor, tuple[Any]],
) -> Union[
    tuple[torch.Tensor, Optional[torch.Tensor]], tuple[tuple[Any], Optional[tuple[Any]]]
]:
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
    if isinstance(input, tuple):
        x, *aux = input
    elif isinstance(input, torch.Tensor):
        x = input
        aux = []
    else:
        raise ValueError(f"Unknown input type '{type(input)}'")

    sample_prediction, log_var = _pure_lvae_predict_single_sample(
        model=model, likelihood_obj=likelihood_obj, input=x
    )

    # TODO: don't like how this is handled
    #   this is how it is done in FCNModule.predict_step
    if len(aux) == 0:
        return sample_prediction, log_var
    else:
        log_var_output = (log_var, *aux) if log_var is not None else None
        return (sample_prediction, *aux), log_var_output


@overload
def lvae_predict_mmse(  # numpydoc ignore=GL08
    model: LVAE, likelihood_obj: LikelihoodModule, input: tuple[Any], mmse_count: int
) -> tuple[tuple[Any], Optional[tuple[Any]]]: ...


@overload
def lvae_predict_mmse(  # numpydoc ignore=GL08
    model: LVAE, likelihood_obj: LikelihoodModule, input: torch.Tensor, mmse_count: int
) -> tuple[torch.Tensor, Optional[torch.Tensor]]: ...


def lvae_predict_mmse(
    model: LVAE,
    likelihood_obj: LikelihoodModule,
    input: Union[torch.Tensor, tuple[Any]],
    mmse_count: int,
) -> Union[
    tuple[torch.Tensor, Optional[torch.Tensor]], tuple[tuple[Any], Optional[tuple[Any]]]
]:
    # TODO: fix docstring return types, ... too many output options
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
    tuple of ((torch.tensor, Any, ...), optional tuple of (torch.tensor, Any, ...))
        The first element is the MMSE prediction, and the second element is the
        log-variance. The log-variance will be None if `model.predict_logvar is None`.
        Any auxillary data included in the input will also be include with both the
        MMSE prediction and the log-variance.
    """
    if isinstance(input, tuple):
        x, *aux = input
    elif isinstance(input, torch.Tensor):
        x = input
        aux = []
    else:
        raise ValueError(f"Unknown input type '{type(input)}'")

    if mmse_count <= 0:
        raise ValueError("MMSE count must be greater than zero.")

    input_shape = x.shape
    output_shape = (input_shape[0], model.target_ch, *input_shape[2:])
    log_var: Optional[torch.Tensor] = None
    # pre-declare empty array to fill with individual sample predictions
    sample_predictions = torch.zeros(size=(mmse_count, *output_shape))
    for mmse_idx in range(mmse_count):
        sample_prediction, lv = _pure_lvae_predict_single_sample(
            model=model, likelihood_obj=likelihood_obj, input=x
        )
        # only keep the log variance of the first sample prediction
        # TODO: confirm
        if mmse_idx == 0:
            log_var = lv

        # store sample predictions
        sample_predictions[mmse_idx, ...] = sample_prediction

    # TODO: don't like how this is handled
    #   this is how it is done in FCNModule.predict_step
    # take the mean of the sample predictions
    mmse_prediction = torch.mean(sample_predictions, dim=0)
    if len(aux) == 0:
        return mmse_prediction, log_var
    else:
        log_var_output = (log_var, *aux) if log_var is not None else None
        return (mmse_prediction, *aux), log_var_output
