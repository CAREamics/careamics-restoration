"""
This script is meant to load data, initialize the model, and provide the logic for training it.
"""

import glob
import os
import socket
import sys
from typing import Dict

import pytorch_lightning as pl
import torch
from absl import app, flags
from ml_collections.config_flags import config_flags
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
print(sys.path)

from careamics.lvae_training.dataset.data_modules import (
    LCMultiChDloader,
    MultiChDloader,
)
from careamics.lvae_training.dataset.data_utils import DataSplitType
from careamics.lvae_training.lightning_module import LadderVAELight
from careamics.lvae_training.train_utils import *

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False
)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_string(
    "logdir", "/group/jug/federico/wandb_backup/", "The folder name for storing logging"
)
flags.DEFINE_string(
    "datadir", "/group/jug/federico/careamics_training/data/BioSR", "Data directory."
)
flags.DEFINE_boolean("use_max_version", False, "Overwrite the max version of the model")
flags.DEFINE_string(
    "load_ckptfpath",
    "",
    "The path to a previous ckpt from which the weights should be loaded",
)
flags.mark_flags_as_required(["workdir", "config", "mode"])


def create_dataset(
    config,
    datadir,
    eval_datasplit_type=DataSplitType.Val,
    raw_data_dict=None,
    skip_train_dataset=False,
    kwargs_dict=None,
):

    if kwargs_dict is None:
        kwargs_dict = {}

    datapath = datadir

    # Hard-coded parameters (used to be in the config file)
    normalized_input = True
    use_one_mu_std = True
    train_aug_rotate = False
    enable_random_cropping = True
    lowres_supervision = False

    # 1) Data loader for Lateral Contextualization
    if (
        "multiscale_lowres_count" in config.data
        and config.data.multiscale_lowres_count is not None
    ):
        # Get padding attributes
        if "padding_kwargs" not in kwargs_dict:
            padding_kwargs = {}
            if "padding_mode" in config.data and config.data.padding_mode is not None:
                padding_kwargs["mode"] = config.data.padding_mode
            else:
                padding_kwargs["mode"] = "reflect"
            if "padding_value" in config.data and config.data.padding_value is not None:
                padding_kwargs["constant_values"] = config.data.padding_value
            else:
                padding_kwargs["constant_values"] = None
        else:
            padding_kwargs = kwargs_dict.pop("padding_kwargs")

        train_data = (
            None
            if skip_train_dataset
            else LCMultiChDloader(
                config.data,
                datapath,
                datasplit_type=DataSplitType.Train,
                val_fraction=0.1,
                test_fraction=0.1,
                normalized_input=normalized_input,
                use_one_mu_std=use_one_mu_std,
                enable_rotation_aug=train_aug_rotate,
                enable_random_cropping=enable_random_cropping,
                num_scales=config.data.multiscale_lowres_count,
                lowres_supervision=lowres_supervision,
                padding_kwargs=padding_kwargs,
                **kwargs_dict,
                allow_generation=True,
            )
        )
        max_val = train_data.get_max_val()

        val_data = LCMultiChDloader(
            config.data,
            datapath,
            datasplit_type=eval_datasplit_type,
            val_fraction=0.1,
            test_fraction=0.1,
            normalized_input=normalized_input,
            use_one_mu_std=use_one_mu_std,
            enable_rotation_aug=False,  # No rotation aug on validation
            enable_random_cropping=False,
            # No random cropping on validation. Validation is evaluated on determistic grids
            num_scales=config.data.multiscale_lowres_count,
            lowres_supervision=lowres_supervision,
            padding_kwargs=padding_kwargs,
            allow_generation=False,
            **kwargs_dict,
            max_val=max_val,
        )
    # 2) Vanilla data loader
    else:
        train_data_kwargs = {"allow_generation": True, **kwargs_dict}
        val_data_kwargs = {"allow_generation": False, **kwargs_dict}

        train_data_kwargs["enable_random_cropping"] = enable_random_cropping
        val_data_kwargs["enable_random_cropping"] = False

        train_data = (
            None
            if skip_train_dataset
            else MultiChDloader(
                data_config=config.data,
                fpath=datapath,
                datasplit_type=DataSplitType.Train,
                val_fraction=0.1,
                test_fraction=0.1,
                normalized_input=normalized_input,
                use_one_mu_std=use_one_mu_std,
                enable_rotation_aug=train_aug_rotate,
                **train_data_kwargs,
            )
        )

        max_val = train_data.get_max_val()
        val_data = MultiChDloader(
            data_config=config.data,
            fpath=datapath,
            datasplit_type=eval_datasplit_type,
            val_fraction=0.1,
            test_fraction=0.1,
            normalized_input=normalized_input,
            use_one_mu_std=use_one_mu_std,
            enable_rotation_aug=False,  # No rotation aug on validation
            max_val=max_val,
            **val_data_kwargs,
        )

    # For normalizing, we should be using the training data's mean and std.
    mean_val, std_val = train_data.compute_mean_std()
    train_data.set_mean_std(mean_val, std_val)
    val_data.set_mean_std(mean_val, std_val)

    return train_data, val_data


def create_model_and_train(
    config: ml_collections.ConfigDict,
    data_mean: Dict[str, torch.Tensor],
    data_std: Dict[str, torch.Tensor],
    logger: WandbLogger,
    checkpoint_callback: ModelCheckpoint,
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    # tensorboard previous files.
    for filename in glob.glob(config.workdir + "/events*"):
        os.remove(filename)

    # checkpoints
    for filename in glob.glob(config.workdir + "/*.ckpt"):
        os.remove(filename)

    if "num_targets" in config.model:
        target_ch = config.model.num_targets
    else:
        target_ch = config.data.get("num_channels", 2)

    # Instantiate the model (lightning wrapper)
    model = LadderVAELight(
        data_mean=data_mean, data_std=data_std, config=config, target_ch=target_ch
    )

    # Load pre-trained weights if any
    if config.training.pre_trained_ckpt_fpath:
        print("Starting with pre-trained model", config.training.pre_trained_ckpt_fpath)
        checkpoint = torch.load(config.training.pre_trained_ckpt_fpath)
        _ = model.load_state_dict(checkpoint["state_dict"], strict=False)

    estop_monitor = config.model.get("monitor", "val_loss")
    estop_mode = MetricMonitor(estop_monitor).mode()

    callbacks = [
        EarlyStopping(
            monitor=estop_monitor,
            min_delta=1e-6,
            patience=config.training.earlystop_patience,
            verbose=True,
            mode=estop_mode,
        ),
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger.experiment.config.update(config.to_dict())
    # wandb.init(config=config)
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=config.training.max_epochs,
        gradient_clip_val=config.training.grad_clip_norm_value,
        gradient_clip_algorithm=config.training.gradient_clip_algorithm,
        logger=logger,
        callbacks=callbacks,
        # limit_train_batches = config.training.limit_train_batches,
        precision=config.training.precision,
    )
    trainer.fit(model, train_loader, val_loader)


def train_network(
    train_loader: DataLoader,
    val_loader: DataLoader,
    data_mean: Dict[str, torch.Tensor],
    data_std: Dict[str, torch.Tensor],
    config: ml_collections.ConfigDict,
    model_name: str,
    logdir: str,
):
    ckpt_monitor = config.model.get("monitor", "val_loss")
    ckpt_mode = MetricMonitor(ckpt_monitor).mode()
    checkpoint_callback = ModelCheckpoint(
        monitor=ckpt_monitor,
        dirpath=config.workdir,
        filename=model_name + "_best",
        save_last=True,
        save_top_k=1,
        mode=ckpt_mode,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = model_name + "_last"
    logger = WandbLogger(
        name=os.path.join(config.hostname, config.exptname),
        save_dir=logdir,
        project="Disentanglement",
    )

    create_model_and_train(
        config=config,
        data_mean=data_mean,
        data_std=data_std,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        train_loader=train_loader,
        val_loader=val_loader,
    )


def main(argv):
    config = FLAGS.config

    assert os.path.exists(FLAGS.workdir)
    cur_workdir, relative_path = get_workdir(
        config, FLAGS.workdir, FLAGS.use_max_version
    )
    print(f"Saving training to {cur_workdir}")

    config.workdir = cur_workdir
    config.exptname = relative_path
    config.hostname = socket.gethostname()
    config.datadir = FLAGS.datadir
    config.training.pre_trained_ckpt_fpath = FLAGS.load_ckptfpath

    if FLAGS.mode == "train":
        set_logger(workdir=cur_workdir)
        raw_data_dict = None

        # From now on, config cannot be changed.
        config = ml_collections.FrozenConfigDict(config)
        log_config(config, cur_workdir)

        train_data, val_data = create_dataset(
            config, FLAGS.datadir, raw_data_dict=raw_data_dict
        )

        mean_dict, std_dict = get_mean_std_dict_for_model(config, train_data)

        batch_size = config.training.batch_size
        shuffle = True
        train_dloader = DataLoader(
            train_data,
            pin_memory=False,
            num_workers=config.training.num_workers,
            shuffle=shuffle,
            batch_size=batch_size,
        )
        val_dloader = DataLoader(
            val_data,
            pin_memory=False,
            num_workers=config.training.num_workers,
            shuffle=False,
            batch_size=batch_size,
        )

        train_network(
            train_loader=train_dloader,
            val_loader=val_dloader,
            data_mean=mean_dict,
            data_std=std_dict,
            config=config,
            model_name="BaselineVAECL",
            logdir=FLAGS.logdir,
        )

    elif FLAGS.mode == "eval":
        pass
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
    app.run(main)
