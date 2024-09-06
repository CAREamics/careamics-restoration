"""Optimizers and schedulers supported by CAREamics."""

from careamics.utils import BaseEnum


class SupportedOptimizer(str, BaseEnum):
    """Supported optimizers.

    Attributes
    ----------
    Adam : str
        Adam optimizer.
    SGD : str
        Stochastic Gradient Descent optimizer.
    """

    # ASGD = "ASGD"
    # Adadelta = "Adadelta"
    # Adagrad = "Adagrad"
    ADAM = "Adam"
    # AdamW = "AdamW"
    ADAMAX = "Adamax"
    # LBFGS = "LBFGS"
    # NAdam = "NAdam"
    # RAdam = "RAdam"
    # RMSprop = "RMSprop"
    # Rprop = "Rprop"
    SGD = "SGD"
    # SparseAdam = "SparseAdam"


class SupportedScheduler(str, BaseEnum):
    """Supported schedulers.

    Attributes
    ----------
    ReduceLROnPlateau : str
        Reduce learning rate on plateau.
    StepLR : str
        Step learning rate.
    """

    # ChainedScheduler = "ChainedScheduler"
    # ConstantLR = "ConstantLR"
    # CosineAnnealingLR = "CosineAnnealingLR"
    # CosineAnnealingWarmRestarts = "CosineAnnealingWarmRestarts"
    # CyclicLR = "CyclicLR"
    # ExponentialLR = "ExponentialLR"
    # LambdaLR = "LambdaLR"
    # LinearLR = "LinearLR"
    # MultiStepLR = "MultiStepLR"
    # MultiplicativeLR = "MultiplicativeLR"
    # OneCycleLR = "OneCycleLR"
    # PolynomialLR = "PolynomialLR"
    REDUCE_LR_ON_PLATEAU = "ReduceLROnPlateau"
    # SequentialLR = "SequentialLR"
    STEP_LR = "StepLR"
