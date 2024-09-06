"""
Here there are functions to define a config file.
"""

import os

import ml_collections

from careamics.lvae_training.dataset.data_utils import DataType
from careamics.models.lvae.utils import LossType


def _init_config():
    """
    Create a default config object with all the required fields.
    """
    config = ml_collections.ConfigDict()

    config.data = ml_collections.ConfigDict()

    config.model = ml_collections.ConfigDict()

    config.loss = ml_collections.ConfigDict()

    config.training = ml_collections.ConfigDict()

    config.workdir = os.getcwd()
    config.datadir = ""
    return config


def get_config():
    config = _init_config()

    data = config.data
    data.image_size = 128  # the patch size
    # data.grid_size = 32 # the retained sub-patch when doing inner tiling
    data.multiscale_lowres_count = (
        None  # todo: this one will be an issue in current careamics
    )
    data.num_channels = 2  # in careamics probably in lvae pydantic model

    model = config.model  # all in lvae pydantic model
    model.z_dims = [128, 128, 128, 128]
    model.n_filters = 64
    model.dropout = 0.1
    model.nonlin = "elu"
    model.enable_noise_model = True
    model.analytical_kl = False
    model.predict_logvar = None

    loss = config.loss  # in algorithm config
    loss.loss_type = LossType.Elbo  # LossType.Elbo or LossType.DenoiSplitMuSplit
    loss.kl_loss_formulation = ""  # '', 'usplit', 'denoisplit'

    training = config.training
    training.lr = 0.001  # in algorithm config
    training.lr_scheduler_patience = 30
    training.batch_size = 32  # in data config
    training.earlystop_patience = (
        200  # in training config in the callbacks (early stopping)
    )
    training.max_epochs = 400  # training config
    training.pre_trained_ckpt_fpath = ""  # this is through the careamics API

    # Set of attributes not to include in the PyDantic data model
    training.num_workers = (
        4  # this is in the data config, passed in the dataloader parameters
    )
    training.grad_clip_norm_value = 0.5  # Taken from https://github.com/openai/vdvae/blob/main/hps.py#L38 # this maybe should be in a new trainer_parameters dict in the training config pydantic model
    training.gradient_clip_algorithm = "value"
    training.precision = 32
    data.data_type = DataType.BioSR_MRC
    data.ch1_fname = "ER/GT_all.mrc"
    data.ch2_fname = "Microtubules/GT_all.mrc"
    model.noise_model_ch1_fpath = "/group/jug/ashesh/training_pre_eccv/noise_model/2402/429/GMMNoiseModel_ER-GT_all__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz"
    model.noise_model_ch2_fpath = "/group/jug/ashesh/training_pre_eccv/noise_model/2402/434/GMMNoiseModel_Microtubules-GT_all__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz"
    # Parameters to apply synthetic noise to data (e.g., used with BioSR data for denoiSplit)
    data.poisson_noise_factor = 1000
    data.enable_gaussian_noise = True
    data.synthetic_gaussian_scale = 4450
    data.input_has_dependant_noise = True

    return config
