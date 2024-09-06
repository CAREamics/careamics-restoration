from typing import Optional

from pydantic import ConfigDict, computed_field

from careamics.lvae_training.dataset.vae_data_config import VaeDatasetConfig


class LCVaeDatasetConfig(VaeDatasetConfig):
    model_config = ConfigDict(validate_assignment=True)

    num_scales: int = 1
    """The number of resolutions at which we want the input. The target is formed at the
    highest resolution."""
