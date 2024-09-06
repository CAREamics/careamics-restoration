"""Callback saving CAREamics configuration as hyperparameters in the model."""

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from careamics.config import Configuration


class HyperParametersCallback(Callback):
    """
    Callback allowing saving CAREamics configuration as hyperparameters in the model.

    This allows saving the configuration as dictionary in the checkpoints, and
    loading it subsequently in a CAREamist instance.

    Parameters
    ----------
    config : Configuration
        CAREamics configuration to be saved as hyperparameter in the model.

    Attributes
    ----------
    config : Configuration
        CAREamics configuration to be saved as hyperparameter in the model.
    """

    def __init__(self, config: Configuration) -> None:
        """
        Constructor.

        Parameters
        ----------
        config : Configuration
            CAREamics configuration to be saved as hyperparameter in the model.
        """
        self.config = config

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Update the hyperparameters of the model with the configuration on train start.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning trainer, unused.
        pl_module : LightningModule
            PyTorch Lightning module.
        """
        pl_module.hparams.update(self.config.model_dump())
