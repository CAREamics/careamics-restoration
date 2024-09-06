"""Activations for CAREamics models."""

from typing import Callable, Union

import torch.nn as nn

from ..config.support import SupportedActivation


def get_activation(activation: Union[SupportedActivation, str]) -> Callable:
    """
    Get activation function.

    Parameters
    ----------
    activation : str
        Activation function name.

    Returns
    -------
    Callable
        Activation function.
    """
    if activation == SupportedActivation.RELU:
        return nn.ReLU()
    elif activation == SupportedActivation.ELU:
        return nn.ELU()
    elif activation == SupportedActivation.LEAKYRELU:
        return nn.LeakyReLU()
    elif activation == SupportedActivation.TANH:
        return nn.Tanh()
    elif activation == SupportedActivation.SIGMOID:
        return nn.Sigmoid()
    elif activation == SupportedActivation.SOFTMAX:
        return nn.Softmax(dim=1)
    elif activation == SupportedActivation.NONE:
        return nn.Identity()
    else:
        raise ValueError(f"Activation {activation} not supported.")
