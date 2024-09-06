"""Bioimage.io utils."""

from pathlib import Path
from typing import Union


def get_unzip_path(zip_path: Union[Path, str]) -> Path:
    """Generate unzipped folder path from the bioimage.io model path.

    Parameters
    ----------
    zip_path : Path
        Path to the bioimage.io model.

    Returns
    -------
    Path
        Path to the unzipped folder.
    """
    zip_path = Path(zip_path)

    return zip_path.parent / (str(zip_path.name) + ".unzip")


def create_env_text(pytorch_version: str) -> str:
    """Create environment yaml content for the bioimage model.

    This installs an environment with the specified pytorch version and the latest
    changes to careamics.

    Parameters
    ----------
    pytorch_version : str
        Pytorch version.

    Returns
    -------
    str
        Environment text.
    """
    env = (
        f"name: careamics\n"
        f"dependencies:\n"
        f"  - python=3.10\n"
        f"  - pytorch={pytorch_version}\n"
        f"  - torchvision={pytorch_version}\n"
        f"  - pip\n"
        f"  - pip:\n"
        f"    - git+https://github.com/CAREamics/careamics.git\n"
    )

    return env
