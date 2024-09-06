"""Test PredictionWriterCallback class."""

import os
from pathlib import Path
from typing import Union
from unittest.mock import Mock, patch

import numpy as np
import pytest
import tifffile
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from careamics.config import Configuration
from careamics.config.support import SupportedData
from careamics.dataset import IterablePredDataset
from careamics.lightning import (
    create_careamics_module,
    create_predict_datamodule,
    create_train_datamodule,
)
from careamics.lightning.callbacks import PredictionWriterCallback
from careamics.lightning.callbacks.prediction_writer_callback import (
    WriteStrategy,
    create_write_strategy,
)
from careamics.prediction_utils import convert_outputs


@pytest.fixture
def write_strategy():
    """Mock `WriteFunc`."""
    return Mock(spec=WriteStrategy)


@pytest.fixture
def dirpath(tmp_path: Path):
    """Directory path."""
    return tmp_path / "predictions"


@pytest.fixture
def prediction_writer_callback(
    write_strategy: WriteStrategy, dirpath: Union[Path, str]
):
    """Initialized `PredictionWriterCallback`."""
    return PredictionWriterCallback(write_strategy=write_strategy, dirpath=dirpath)


# TODO: smoke test with tiff (& example custom save func?)


def test_smoke_n2v_tiled_tiff(tmp_path, minimum_configuration):
    rng = np.random.default_rng(42)

    # training data
    train_array = rng.integers(0, 255, (32, 32)).astype(np.float32)
    val_array = rng.integers(0, 255, (32, 32)).astype(np.float32)

    # write train array to tiff
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    file_name = "image.tiff"
    train_file = train_dir / file_name
    tifffile.imwrite(train_file, train_array)

    cfg = Configuration(**minimum_configuration)

    # create lightning module
    model = create_careamics_module(
        algorithm_type=cfg.algorithm_config.algorithm_type,
        algorithm=cfg.algorithm_config.algorithm,
        loss=cfg.algorithm_config.loss,
        architecture=cfg.algorithm_config.model.architecture,
    )

    # create data module
    data = create_train_datamodule(
        train_data=train_array,
        val_data=val_array,
        data_type=cfg.data_config.data_type,
        patch_size=cfg.data_config.patch_size,
        axes=cfg.data_config.axes,
        batch_size=cfg.data_config.batch_size,
    )

    # create prediction writer callback params
    write_strategy = create_write_strategy(write_type="tiff", tiled=True)
    dirpath = tmp_path / "predictions"

    # create trainer
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmp_path,
        callbacks=[
            ModelCheckpoint(
                dirpath=tmp_path / "checkpoints",
                filename="test_lightning_api",
            ),
            PredictionWriterCallback(write_strategy=write_strategy, dirpath=dirpath),
        ],
    )

    # train
    trainer.fit(model, datamodule=data)

    # predict
    means, stds = data.get_data_statistics()
    predict_data = create_predict_datamodule(
        data_type="tiff",
        pred_data=train_dir,
        axes=cfg.data_config.axes,
        image_means=means,
        image_stds=stds,
        tile_size=(8, 8),
        tile_overlap=(2, 2),
    )

    # predict
    predicted = trainer.predict(model, datamodule=predict_data)
    predicted_images = convert_outputs(predicted, tiled=True)

    # assert predicted file exists
    assert (dirpath / file_name).is_file()

    # open file
    save_data = tifffile.imread(dirpath / file_name)
    # save data has singleton channel axis
    np.testing.assert_array_equal(save_data, predicted_images[0][0], verbose=True)


def test_smoke_n2v_untiled_tiff(tmp_path, minimum_configuration):
    rng = np.random.default_rng(42)

    # training data
    train_array = rng.integers(0, 255, (32, 32)).astype(np.float32)
    val_array = rng.integers(0, 255, (32, 32)).astype(np.float32)

    # write train array to tiff
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    file_name = "image.tiff"
    train_file = train_dir / file_name
    tifffile.imwrite(train_file, train_array)

    cfg = Configuration(**minimum_configuration)

    # create lightning module
    model = create_careamics_module(
        algorithm_type=cfg.algorithm_config.algorithm_type,
        algorithm=cfg.algorithm_config.algorithm,
        loss=cfg.algorithm_config.loss,
        architecture=cfg.algorithm_config.model.architecture,
    )

    # create data module
    data = create_train_datamodule(
        train_data=train_array,
        val_data=val_array,
        data_type=cfg.data_config.data_type,
        patch_size=cfg.data_config.patch_size,
        axes=cfg.data_config.axes,
        batch_size=cfg.data_config.batch_size,
    )

    # create prediction writer callback params
    write_strategy = create_write_strategy(write_type="tiff", tiled=False)
    dirpath = tmp_path / "predictions"

    # create trainer
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmp_path,
        callbacks=[
            ModelCheckpoint(
                dirpath=tmp_path / "checkpoints",
                filename="test_lightning_api",
            ),
            PredictionWriterCallback(write_strategy=write_strategy, dirpath=dirpath),
        ],
    )

    # train
    trainer.fit(model, datamodule=data)

    # predict
    means, stds = data.get_data_statistics()
    predict_data = create_predict_datamodule(
        data_type="tiff",
        pred_data=train_dir,
        axes=cfg.data_config.axes,
        image_means=means,
        image_stds=stds,
    )

    # predict
    predicted = trainer.predict(model, datamodule=predict_data)
    predicted_images = convert_outputs(predicted, tiled=False)

    # assert predicted file exists
    assert (dirpath / file_name).is_file()

    # open file
    save_data = tifffile.imread(dirpath / file_name)
    # save data has singleton channel axis
    np.testing.assert_array_equal(save_data, predicted_images[0][0], verbose=True)


def test_initialization(prediction_writer_callback, write_strategy, dirpath):
    """Test `PredictionWriterCallback` initializes as expected."""
    assert prediction_writer_callback._writing_predictions is True
    assert prediction_writer_callback.write_strategy is write_strategy
    assert prediction_writer_callback.dirpath == Path(dirpath).resolve()


def test_set_dirpath_absolute_path(prediction_writer_callback):
    """Test initialization of dirpath with absolute path."""
    absolute_path = Path("/absolute/path").absolute()
    prediction_writer_callback.dirpath = absolute_path
    assert prediction_writer_callback._dirpath == absolute_path


def test_set_dirpath_relative_path(prediction_writer_callback):
    """Test initialization of dirpath with relatice path."""
    relative_path = "relative/path"
    # patch pathlib.Path.cwd to return
    mock_cwd = Path("/current/working/dir")
    with patch("pathlib.Path.cwd", return_value=mock_cwd):
        prediction_writer_callback.dirpath = relative_path
        assert prediction_writer_callback._dirpath == mock_cwd / relative_path


@pytest.mark.parametrize("initial_value", [True, False])
def test_writing_predictions_context(prediction_writer_callback, initial_value):
    """
    Test that the context manager for writing predictions works as expected.
    """
    # initialize value
    prediction_writer_callback._writing_predictions = initial_value

    temp_value = True
    with prediction_writer_callback.writing_predictions(temp_value):
        assert prediction_writer_callback._writing_predictions == temp_value
    # make sure it is restored to it's initial value
    assert prediction_writer_callback._writing_predictions == initial_value

    # for value is false
    temp_value = False
    with prediction_writer_callback.writing_predictions(temp_value):
        assert prediction_writer_callback._writing_predictions == temp_value
    assert prediction_writer_callback._writing_predictions == initial_value


@pytest.mark.parametrize("initial_value", [True, False])
def test_set_writing_predictions(prediction_writer_callback, initial_value):
    """
    Test that `set_writing_predictions` changes the value of `_writing_predictions.
    """
    # initialize value
    prediction_writer_callback._writing_predictions = initial_value

    new_value = True
    prediction_writer_callback.set_writing_predictions(new_value)
    assert prediction_writer_callback._writing_predictions == new_value

    new_value = False
    prediction_writer_callback.set_writing_predictions(new_value)
    assert prediction_writer_callback._writing_predictions == new_value


@pytest.mark.parametrize("writing_predictions", [True, False])
def test_setup_prediction_directory_creation(
    prediction_writer_callback, dirpath, writing_predictions
):
    """
    Test prediction directory is created when `setup` is called at `stage="predict"`.
    """
    trainer = Mock(spec=Trainer)
    pl_module = Mock(spec=LightningModule)
    stage = "predict"

    prediction_writer_callback._writing_predictions = writing_predictions
    prediction_writer_callback.setup(trainer=trainer, pl_module=pl_module, stage=stage)

    # makes directory only if writing predictions is turned on
    assert os.path.isdir(dirpath) == writing_predictions


def test_write_on_batch_end(prediction_writer_callback):
    """
    Test `PredictionWriterCallback.write_on_batch_end`.

    Check `prediction_writer_callback.write_strategy.write_batch` is called as
    expected.
    """
    # mock write_on_batch_end inputs.
    trainer = Mock(spec=Trainer)
    pl_module = Mock(spec=LightningModule)
    prediction = Mock()
    batch_indices = [0, 1, 2]
    batch = Mock()
    batch_idx = 0
    dataloader_idx = 0

    # Mocking the dataset to be of type IterablePredDataset
    mock_dataset = Mock(spec=IterablePredDataset)
    trainer.predict_dataloaders = [Mock(spec=DataLoader)]
    trainer.predict_dataloaders[dataloader_idx].dataset = mock_dataset

    prediction_writer_callback.write_on_batch_end(
        trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    )
    # The different strategy functions are tested individually
    #   so they do not need to be tested here
    prediction_writer_callback.write_strategy.write_batch.assert_called_once_with(
        trainer=trainer,
        pl_module=pl_module,
        prediction=prediction,
        batch_indices=batch_indices,
        batch=batch,
        batch_idx=batch_idx,
        dataloader_idx=dataloader_idx,
        dirpath=prediction_writer_callback.dirpath,
    )


def test_write_on_batch_end_writing_predictions_off(prediction_writer_callback):
    """
    Test that `write_batch` acts as expected when writing predictions is set to off.

    Ensure `PredictionWriterCallback.write_strategy.write_batch` is not called when
    `PredictionWriterCallback.writing_predictions=False`.
    """
    prediction_writer_callback._writing_predictions = False

    trainer = Mock(spec=Trainer)
    pl_module = Mock(spec=LightningModule)
    prediction = Mock()
    batch_indices = [0, 1, 2]
    batch = Mock()
    batch_idx = 0
    dataloader_idx = 0

    # Mocking the dataset to be of type IterablePredDataset or IterableTiledPredDataset
    mock_dataset = Mock(spec=IterablePredDataset)
    trainer.predict_dataloaders = [Mock(spec=DataLoader)]
    trainer.predict_dataloaders[dataloader_idx].dataset = mock_dataset

    prediction_writer_callback.write_on_batch_end(
        trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    )
    prediction_writer_callback.write_strategy.write_batch.assert_not_called()


def test_from_write_func_params(write_strategy, dirpath):
    """
    Test `from_write_func_param` classmethod initializes `PredictWriterCallback`.
    """

    write_type = SupportedData.TIFF
    tiled = True
    write_func = None
    write_extension = None
    write_func_kwargs = None

    # mock create_write_strategy as it is already tested individually.
    with patch(
        "careamics.lightning.callbacks"
        ".prediction_writer_callback.prediction_writer_callback.create_write_strategy"
    ) as mock_create_write_strategy:
        write_strategy = Mock(spec=WriteStrategy)
        mock_create_write_strategy.return_value = write_strategy
        callback = PredictionWriterCallback.from_write_func_params(
            write_type=write_type,
            tiled=tiled,
            write_func=write_func,
            write_extension=write_extension,
            write_func_kwargs=write_func_kwargs,
            dirpath=dirpath,
        )
        mock_create_write_strategy.assert_called_once_with(
            write_type=write_type,
            tiled=tiled,
            write_func=write_func,
            write_extension=write_extension,
            write_func_kwargs=write_func_kwargs,
        )
        assert callback.write_strategy == write_strategy
        assert callback.dirpath == Path(dirpath).resolve()
