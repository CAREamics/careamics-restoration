"""Losses supported by CAREamics."""

from careamics.utils import BaseEnum


# TODO register loss with custom_loss decorator?
class SupportedLoss(str, BaseEnum):
    """Supported losses.

    Attributes
    ----------
    MSE : str
        Mean Squared Error loss.
    MAE : str
        Mean Absolute Error loss.
    N2V : str
        Noise2Void loss.
    """

    MSE = "mse"
    MAE = "mae"
    N2V = "n2v"
    # PN2V = "pn2v"
    # HDN = "hdn"
    MUSPLIT = "musplit"
    DENOISPLIT = "denoisplit"
    DENOISPLIT_MUSPLIT = "denoisplit_musplit"
    # CE = "ce"
    # DICE = "dice"
