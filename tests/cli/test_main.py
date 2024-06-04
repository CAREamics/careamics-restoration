from pathlib import Path

import numpy as np
import tifffile
from typer.testing import CliRunner
import yaml

from careamics.config import Configuration
from careamics.config.support import SupportedData
from careamics.cli.main import app

runner = CliRunner()


def test_train(tmp_path: Path, minimum_configuration: dict):

    # create & save config
    config_path = tmp_path / "config.yaml"
    config = Configuration(**minimum_configuration)
    config.data_config.data_type = SupportedData.TIFF.value
    with open(config_path, "w") as file:
        yaml.dump(config.model_dump(), file, indent=2)

    # training data
    train_array = np.random.rand(32, 32)
    # save files
    train_file = tmp_path / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    # invoke command 
    result = runner.invoke(
        app,
        [
            "train",
            str(config_path),
            "-ts",
            str(train_file),
            "-wd",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0


def test_predict():
    result = runner.invoke(app, ["predict"])
    result.exit_code == 2 # assert exits with error (NotImplementedError)