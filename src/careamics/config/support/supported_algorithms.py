"""Algorithms supported by CAREamics."""

from __future__ import annotations

from careamics.utils import BaseEnum


class SupportedAlgorithm(str, BaseEnum):
    """Algorithms available in CAREamics."""

    N2V = "n2v"
    """Noise2Void algorithm, a self-supervised approach based on blind denoising."""

    CARE = "care"
    """Content-aware image restoration, a supervised algorithm used for a variety
    of tasks."""

    N2N = "n2n"
    """Noise2Noise algorithm, a self-supervised denoising scheme based on comparing
    noisy images of the same sample."""

    MUSPLIT = "musplit"
    """An image splitting approach based on ladder VAE architectures."""

    DENOISPLIT = "denoisplit"
    """An image splitting and denoising approach based on ladder VAE architectures."""

    CUSTOM = "custom"
    """Custom algorithm, used for cases where a custom architecture is provided."""

    # PN2V = "pn2v"
    # HDN = "hdn"
    # SEG = "segmentation"
