"""Module containing different strategies for writing predictions."""

from pathlib import Path
from typing import Any, Optional, Protocol, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from careamics.config.tile_information import TileInformation
from careamics.dataset import IterablePredDataset, IterableTiledPredDataset
from careamics.file_io import WriteFunc
from careamics.prediction_utils import stitch_prediction_single

from .file_path_utils import create_write_file_path, get_sample_file_path


class WriteStrategy(Protocol):
    """Protocol for write strategy classes."""

    _write_filenames: Optional[list[Path]]
    _n_samples_per_file: Optional[list[int]]

    def write_batch(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,  # TODO: change to expected type
        batch_indices: Optional[Sequence[int]],
        batch: Any,  # TODO: change to expected type
        batch_idx: int,
        dataloader_idx: int,
        dirpath: Path,
    ) -> None:
        """
        WriteStrategy subclasses must contain this function to write a batch.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning Trainer.
        pl_module : LightningModule
            PyTorch Lightning LightningModule.
        prediction : Any
            Predictions on `batch`.
        batch_indices : sequence of int
            Indices identifying the samples in the batch.
        batch : Any
            Input batch.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index.
        dirpath : Path
            Path to directory to save predictions to.
        """


class CacheTiles:
    """
    A write strategy that will cache tiles.

    Tiles are cached until a whole image is predicted on. Then the stitched
    prediction is saved.

    Parameters
    ----------
    write_func : WriteFunc
        Function used to save predictions.
    write_extension : str
        Extension added to prediction file paths.
    write_func_kwargs : dict of {str: Any}
        Extra kwargs to pass to `write_func`.

    Attributes
    ----------
    write_func : WriteFunc
        Function used to save predictions.
    write_extension : str
        Extension added to prediction file paths.
    write_func_kwargs : dict of {str: Any}
        Extra kwargs to pass to `write_func`.
    tile_cache : list of numpy.ndarray
        Tiles cached for stitching prediction.
    tile_info_cache : list of TileInformation
        Cached tile information for stitching prediction.
    """

    def __init__(
        self,
        write_func: WriteFunc,
        write_extension: str,
        write_func_kwargs: dict[str, Any],
    ) -> None:
        """
        A write strategy that will cache tiles.

        Tiles are cached until a whole image is predicted on. Then the stitched
        prediction is saved.

        Parameters
        ----------
        write_func : WriteFunc
            Function used to save predictions.
        write_extension : str
            Extension added to prediction file paths.
        write_func_kwargs : dict of {str: Any}
            Extra kwargs to pass to `write_func`.
        """
        super().__init__()

        self.write_func: WriteFunc = write_func
        self.write_extension: str = write_extension
        self.write_func_kwargs: dict[str, Any] = write_func_kwargs

        self._write_filenames: Optional[list[Path]] = None
        self._n_samples_per_file: Optional[list[int]] = None

        # where tiles will be cached until a whole image has been predicted
        self.tile_cache: list[NDArray] = []
        self.tile_info_cache: list[TileInformation] = []

    @property
    def last_tiles(self) -> list[bool]:
        """
        List of bool to determine whether each tile in the cache is the last tile.

        Returns
        -------
        list of bool
            Whether each tile in the tile cache is the last tile.
        """
        return [tile_info.last_tile for tile_info in self.tile_info_cache]

    def write_batch(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: tuple[NDArray, list[TileInformation]],
        batch_indices: Optional[Sequence[int]],
        batch: tuple[NDArray, list[TileInformation]],
        batch_idx: int,
        dataloader_idx: int,
        dirpath: Path,
    ) -> None:
        """
        Cache tiles until the last tile is predicted; save the stitched prediction.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning Trainer.
        pl_module : LightningModule
            PyTorch Lightning LightningModule.
        prediction : Any
            Predictions on `batch`.
        batch_indices : sequence of int
            Indices identifying the samples in the batch.
        batch : Any
            Input batch.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index.
        dirpath : Path
            Path to directory to save predictions to.
        """
        dataloaders: Union[DataLoader, list[DataLoader]] = trainer.predict_dataloaders
        dataloader: DataLoader = (
            dataloaders[dataloader_idx]
            if isinstance(dataloaders, list)
            else dataloaders
        )
        dataset: IterableTiledPredDataset = dataloader.dataset
        if not isinstance(dataset, IterableTiledPredDataset):
            raise TypeError("Prediction dataset is not `IterableTiledPredDataset`.")

        # cache tiles (batches are split into single samples)
        self.tile_cache.extend(np.split(prediction[0], prediction[0].shape[0]))
        self.tile_info_cache.extend(prediction[1])

        # save stitched prediction
        if self._has_last_tile():

            # get image tiles and remove them from the cache
            tiles, tile_infos = self._get_image_tiles()
            self._clear_cache()

            # stitch prediction
            prediction_image = stitch_prediction_single(
                tiles=tiles, tile_infos=tile_infos
            )

            # write prediction
            sample_id = tile_infos[0].sample_id  # need this to select correct file name
            input_file_path = get_sample_file_path(dataset=dataset, sample_id=sample_id)
            file_path = create_write_file_path(
                dirpath=dirpath,
                file_path=input_file_path,
                write_extension=self.write_extension,
            )
            self.write_func(
                file_path=file_path, img=prediction_image[0], **self.write_func_kwargs
            )

    def _has_last_tile(self) -> bool:
        """
        Whether a last tile is contained in the cached tiles.

        Returns
        -------
        bool
            Whether a last tile is contained in the cached tiles.
        """
        return any(self.last_tiles)

    def _clear_cache(self) -> None:
        """Remove the tiles in the cache up to the first last tile."""
        index = self._last_tile_index()
        self.tile_cache = self.tile_cache[index + 1 :]
        self.tile_info_cache = self.tile_info_cache[index + 1 :]

    def _last_tile_index(self) -> int:
        """
        Find the index of the last tile in the tile cache.

        Returns
        -------
        int
            Index of last tile.

        Raises
        ------
        ValueError
            If there is no last tile in the tile cache.
        """
        last_tiles = self.last_tiles
        if not any(last_tiles):
            raise ValueError("No last tile in the tile cache.")
        index = np.where(last_tiles)[0][0]
        return index

    def _get_image_tiles(self) -> tuple[list[NDArray], list[TileInformation]]:
        """
        Get the tiles corresponding to a single image.

        Returns
        -------
        tuple of (list of numpy.ndarray, list of TileInformation)
            Tiles and tile information to stitch together a full image.
        """
        index = self._last_tile_index()
        tiles = self.tile_cache[: index + 1]
        tile_infos = self.tile_info_cache[: index + 1]
        return tiles, tile_infos


class WriteTilesZarr:
    """Strategy to write tiles to Zarr file."""

    def write_batch(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
        dirpath: Path,
    ) -> None:
        """
        Write tiles to zarr file.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning Trainer.
        pl_module : LightningModule
            PyTorch Lightning LightningModule.
        prediction : Any
            Predictions on `batch`.
        batch_indices : sequence of int
            Indices identifying the samples in the batch.
        batch : Any
            Input batch.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index.
        dirpath : Path
            Path to directory to save predictions to.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError


class WriteImage:
    """
    A strategy for writing image predictions (i.e. un-tiled predictions).

    Parameters
    ----------
    write_func : WriteFunc
        Function used to save predictions.
    write_extension : str
        Extension added to prediction file paths.
    write_func_kwargs : dict of {str: Any}
        Extra kwargs to pass to `write_func`.

    Attributes
    ----------
    write_func : WriteFunc
        Function used to save predictions.
    write_extension : str
        Extension added to prediction file paths.
    write_func_kwargs : dict of {str: Any}
        Extra kwargs to pass to `write_func`.
    """

    def __init__(
        self,
        write_func: WriteFunc,
        write_extension: str,
        write_func_kwargs: dict[str, Any],
    ) -> None:
        """
        A strategy for writing image predictions (i.e. un-tiled predictions).

        Parameters
        ----------
        write_func : WriteFunc
            Function used to save predictions.
        write_extension : str
            Extension added to prediction file paths.
        write_func_kwargs : dict of {str: Any}
            Extra kwargs to pass to `write_func`.
        """
        super().__init__()

        self.write_func: WriteFunc = write_func
        self.write_extension: str = write_extension
        self.write_func_kwargs: dict[str, Any] = write_func_kwargs

        self._write_filenames: Optional[list[Path]] = None
        self._n_samples_per_file: Optional[list[int]] = None

    def write_batch(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: NDArray,
        batch_indices: Optional[Sequence[int]],
        batch: NDArray,
        batch_idx: int,
        dataloader_idx: int,
        dirpath: Path,
    ) -> None:
        """
        Save full images.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning Trainer.
        pl_module : LightningModule
            PyTorch Lightning LightningModule.
        prediction : Any
            Predictions on `batch`.
        batch_indices : sequence of int
            Indices identifying the samples in the batch.
        batch : Any
            Input batch.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index.
        dirpath : Path
            Path to directory to save predictions to.

        Raises
        ------
        TypeError
            If trainer prediction dataset is not `IterablePredDataset`.
        """
        dls: Union[DataLoader, list[DataLoader]] = trainer.predict_dataloaders
        dl: DataLoader = dls[dataloader_idx] if isinstance(dls, list) else dls
        ds: IterablePredDataset = dl.dataset
        if not isinstance(ds, IterablePredDataset):
            raise TypeError("Prediction dataset is not `IterablePredDataset`.")

        for i in range(prediction.shape[0]):
            prediction_image = prediction[0]
            sample_id = batch_idx * dl.batch_size + i
            input_file_path = get_sample_file_path(dataset=ds, sample_id=sample_id)
            file_path = create_write_file_path(
                dirpath=dirpath,
                file_path=input_file_path,
                write_extension=self.write_extension,
            )
            self.write_func(
                file_path=file_path, img=prediction_image, **self.write_func_kwargs
            )
