"""
Utility functions needed by dataloader & co.
"""

import os
from dataclasses import dataclass
from typing import List

import numpy as np
from skimage.io import imread, imsave

from careamics.lvae_training.dataset.vae_data_config import DataSplitType, DataType


def load_tiff(path):
    """
    Returns a 4d numpy array: num_imgs*h*w*num_channels
    """
    data = imread(path, plugin="tifffile")
    return data


def save_tiff(path, data):
    imsave(path, data, plugin="tifffile")


def load_tiffs(paths):
    data = [load_tiff(path) for path in paths]
    return np.concatenate(data, axis=0)


def split_in_half(s, e):
    n = e - s
    s1 = list(np.arange(n // 2))
    s2 = list(np.arange(n // 2, n))
    return [x + s for x in s1], [x + s for x in s2]


def adjust_for_imbalance_in_fraction_value(
    val: List[int],
    test: List[int],
    val_fraction: float,
    test_fraction: float,
    total_size: int,
):
    """
    here, val and test are divided almost equally. Here, we need to take into account their respective fractions
    and pick elements rendomly from one array and put in the other array.
    """
    if val_fraction == 0:
        test += val
        val = []
    elif test_fraction == 0:
        val += test
        test = []
    else:
        diff_fraction = test_fraction - val_fraction
        if diff_fraction > 0:
            imb_count = int(diff_fraction * total_size / 2)
            val = list(np.random.RandomState(seed=955).permutation(val))
            test += val[:imb_count]
            val = val[imb_count:]
        elif diff_fraction < 0:
            imb_count = int(-1 * diff_fraction * total_size / 2)
            test = list(np.random.RandomState(seed=955).permutation(test))
            val += test[:imb_count]
            test = test[imb_count:]
    return val, test


def get_train_val_data(
    data_config,
    fpath,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
    allow_generation=False,  # TODO: what is this
):
    """
    Load the data from the given path and split them in training, validation and test sets.

    Ensure that the shape of data should be N*H*W*C: N is number of data points. H,W are the image dimensions.
    C is the number of channels.
    """
    if data_config.data_type == DataType.SeparateTiffData:
        fpath1 = os.path.join(fpath, data_config.ch1_fname)
        fpath2 = os.path.join(fpath, data_config.ch2_fname)
        fpaths = [fpath1, fpath2]
        fpath0 = ""
        if "ch_input_fname" in data_config:
            fpath0 = os.path.join(fpath, data_config.ch_input_fname)
            fpaths = [fpath0] + fpaths

        print(
            f"Loading from {fpath} Channels: "
            f"{fpath1},{fpath2}, inp:{fpath0} Mode:{DataSplitType.name(datasplit_type)}"
        )

        data = np.concatenate([load_tiff(fpath)[..., None] for fpath in fpaths], axis=3)
        if data_config.data_type == DataType.PredictedTiffData:
            assert len(data.shape) == 5 and data.shape[-1] == 1
            data = data[..., 0].copy()
        # data = data[::3].copy()
        # NOTE: This was not the correct way to do it. It is so because the noise present in the input was directly related
        # to the noise present in the channels and so this is not the way we would get the data.
        # We need to add the noise independently to the input and the target.

        # if data_config.get('poisson_noise_factor', False):
        #     data = np.random.poisson(data)
        # if data_config.get('enable_gaussian_noise', False):
        #     synthetic_scale = data_config.get('synthetic_gaussian_scale', 0.1)
        #     print('Adding Gaussian noise with scale', synthetic_scale)
        #     noise = np.random.normal(0, synthetic_scale, data.shape)
        #     data = data + noise

        if datasplit_type == DataSplitType.All:
            return data.astype(np.float32)

        train_idx, val_idx, test_idx = get_datasplit_tuples(
            val_fraction, test_fraction, len(data), starting_test=True
        )
        if datasplit_type == DataSplitType.Train:
            return data[train_idx].astype(np.float32)
        elif datasplit_type == DataSplitType.Val:
            return data[val_idx].astype(np.float32)
        elif datasplit_type == DataSplitType.Test:
            return data[test_idx].astype(np.float32)

    elif data_config.data_type == DataType.BioSR_MRC:
        num_channels = data_config.num_channels
        fpaths = []
        data_list = []
        for i in range(num_channels):
            fpath1 = os.path.join(fpath, getattr(data_config, f"ch{i + 1}_fname"))
            fpaths.append(fpath1)
            data = get_mrc_data(fpath1)[..., None]
            data_list.append(data)

        dirname = os.path.dirname(os.path.dirname(fpaths[0])) + "/"

        msg = ",".join([x[len(dirname) :] for x in fpaths])
        print(
            f"Loaded from {dirname} Channels:{len(fpaths)} {msg} Mode:{datasplit_type}"
        )
        N = data_list[0].shape[0]
        for data in data_list:
            N = min(N, data.shape[0])

        cropped_data = []
        for data in data_list:
            cropped_data.append(data[:N])

        data = np.concatenate(cropped_data, axis=3)

        if datasplit_type == DataSplitType.All:
            return data.astype(np.float32)

        train_idx, val_idx, test_idx = get_datasplit_tuples(
            val_fraction, test_fraction, len(data), starting_test=True
        )
        if datasplit_type == DataSplitType.Train:
            return data[train_idx].astype(np.float32)
        elif datasplit_type == DataSplitType.Val:
            return data[val_idx].astype(np.float32)
        elif datasplit_type == DataSplitType.Test:
            return data[test_idx].astype(np.float32)


def get_datasplit_tuples(
    val_fraction: float,
    test_fraction: float,
    total_size: int,
    starting_test: bool = False,
):
    if starting_test:
        # test => val => train
        test = list(range(0, int(total_size * test_fraction)))
        val = list(range(test[-1] + 1, test[-1] + 1 + int(total_size * val_fraction)))
        train = list(range(val[-1] + 1, total_size))
    else:
        # {test,val}=> train
        test_val_size = int((val_fraction + test_fraction) * total_size)
        train = list(range(test_val_size, total_size))

        if test_val_size == 0:
            test = []
            val = []
            return train, val, test

        # Split the test and validation in chunks.
        chunksize = max(1, min(3, test_val_size // 2))

        nchunks = test_val_size // chunksize

        test = []
        val = []
        s = 0
        for i in range(nchunks):
            if i % 2 == 0:
                val += list(np.arange(s, s + chunksize))
            else:
                test += list(np.arange(s, s + chunksize))
            s += chunksize

        if i % 2 == 0:
            test += list(np.arange(s, test_val_size))
        else:
            p1, p2 = split_in_half(s, test_val_size)
            test += p1
            val += p2

    val, test = adjust_for_imbalance_in_fraction_value(
        val, test, val_fraction, test_fraction, total_size
    )

    return train, val, test


def get_mrc_data(fpath):
    # HXWXN
    _, data = read_mrc(fpath)
    data = data[None]
    data = np.swapaxes(data, 0, 3)
    return data[..., 0]


@dataclass
class GridIndexManager:
    data_shape: tuple
    grid_shape: tuple
    patch_shape: tuple
    trim_boundary: bool

    # Vera: patch is centered on index in the grid, grid size not used in training,
    # used only during val / test, grid size controls the overlap of the patches
    # in training you only get random patches every time
    # For borders - just cropped the data, so it perfectly divisible

    def __post_init__(self):
        assert len(self.data_shape) == len(
            self.grid_shape
        ), f"Data shape:{self.data_shape} and grid size:{self.grid_shape} must have the same dimension"
        assert len(self.data_shape) == len(
            self.patch_shape
        ), f"Data shape:{self.data_shape} and patch shape:{self.patch_shape} must have the same dimension"
        innerpad = np.array(self.patch_shape) - np.array(self.grid_shape)
        for dim, pad in enumerate(innerpad):
            if pad < 0:
                raise ValueError(
                    f"Patch shape:{self.patch_shape} must be greater than or equal to grid shape:{self.grid_shape} in dimension {dim}"
                )
            if pad % 2 != 0:
                raise ValueError(
                    f"Patch shape:{self.patch_shape} must have even padding in dimension {dim}"
                )

    def patch_offset(self):
        return (np.array(self.patch_shape) - np.array(self.grid_shape)) // 2

    def get_individual_dim_grid_count(self, dim: int):
        """
        Returns the number of the grid in the specified dimension, ignoring all other dimensions.
        """
        assert dim < len(
            self.data_shape
        ), f"Dimension {dim} is out of bounds for data shape {self.data_shape}"
        assert dim >= 0, "Dimension must be greater than or equal to 0"

        if self.grid_shape[dim] == 1 and self.patch_shape[dim] == 1:
            return self.data_shape[dim]
        elif self.trim_boundary is False:
            return int(np.ceil(self.data_shape[dim] / self.grid_shape[dim]))
        else:
            excess_size = self.patch_shape[dim] - self.grid_shape[dim]
            return int(
                np.floor((self.data_shape[dim] - excess_size) / self.grid_shape[dim])
            )

    def total_grid_count(self):
        """
        Returns the total number of grids in the dataset.
        """
        return self.grid_count(0) * self.get_individual_dim_grid_count(0)

    def grid_count(self, dim: int):
        """
        Returns the total number of grids for one value in the specified dimension.
        """
        assert dim < len(
            self.data_shape
        ), f"Dimension {dim} is out of bounds for data shape {self.data_shape}"
        assert dim >= 0, "Dimension must be greater than or equal to 0"
        if dim == len(self.data_shape) - 1:
            return 1

        return self.get_individual_dim_grid_count(dim + 1) * self.grid_count(dim + 1)

    def get_grid_index(self, dim: int, coordinate: int):
        """
        Returns the index of the grid in the specified dimension.
        """
        assert dim < len(
            self.data_shape
        ), f"Dimension {dim} is out of bounds for data shape {self.data_shape}"
        assert dim >= 0, "Dimension must be greater than or equal to 0"
        assert (
            coordinate < self.data_shape[dim]
        ), f"Coordinate {coordinate} is out of bounds for data shape {self.data_shape}"

        if self.grid_shape[dim] == 1 and self.patch_shape[dim] == 1:
            return coordinate
        elif self.trim_boundary is False:
            return np.floor(coordinate / self.grid_shape[dim])
        else:
            excess_size = (self.patch_shape[dim] - self.grid_shape[dim]) // 2
            # can be <0 if coordinate is in [0,grid_shape[dim]]
            return max(0, np.floor((coordinate - excess_size) / self.grid_shape[dim]))

    def dataset_idx_from_grid_idx(self, grid_idx: tuple):
        """
        Returns the index of the grid in the dataset.
        """
        assert len(grid_idx) == len(
            self.data_shape
        ), f"Dimension indices {grid_idx} must have the same dimension as data shape {self.data_shape}"
        index = 0
        for dim in range(len(grid_idx)):
            index += grid_idx[dim] * self.grid_count(dim)
        return index

    def get_patch_location_from_dataset_idx(self, dataset_idx: int):
        """
        Returns the patch location of the grid in the dataset.
        """
        location = self.get_location_from_dataset_idx(dataset_idx)
        offset = self.patch_offset()
        return tuple(np.array(location) - np.array(offset))

    def get_dataset_idx_from_grid_location(self, location: tuple):
        assert len(location) == len(
            self.data_shape
        ), f"Location {location} must have the same dimension as data shape {self.data_shape}"
        grid_idx = [
            self.get_grid_index(dim, location[dim]) for dim in range(len(location))
        ]
        return self.dataset_idx_from_grid_idx(tuple(grid_idx))

    def get_gridstart_location_from_dim_index(self, dim: int, dim_index: int):
        """
        Returns the grid-start coordinate of the grid in the specified dimension.
        """
        assert dim < len(
            self.data_shape
        ), f"Dimension {dim} is out of bounds for data shape {self.data_shape}"
        assert dim >= 0, "Dimension must be greater than or equal to 0"
        assert dim_index < self.get_individual_dim_grid_count(
            dim
        ), f"Dimension index {dim_index} is out of bounds for data shape {self.data_shape}"

        if self.grid_shape[dim] == 1 and self.patch_shape[dim] == 1:
            return dim_index
        elif self.trim_boundary is False:
            return dim_index * self.grid_shape[dim]
        else:
            excess_size = (self.patch_shape[dim] - self.grid_shape[dim]) // 2
            return dim_index * self.grid_shape[dim] + excess_size

    def get_location_from_dataset_idx(self, dataset_idx: int):
        grid_idx = []
        for dim in range(len(self.data_shape)):
            grid_idx.append(dataset_idx // self.grid_count(dim))
            dataset_idx = dataset_idx % self.grid_count(dim)
        location = [
            self.get_gridstart_location_from_dim_index(dim, grid_idx[dim])
            for dim in range(len(self.data_shape))
        ]
        return tuple(location)

    def on_boundary(self, dataset_idx: int, dim: int):
        """
        Returns True if the grid is on the boundary in the specified dimension.
        """
        assert dim < len(
            self.data_shape
        ), f"Dimension {dim} is out of bounds for data shape {self.data_shape}"
        assert dim >= 0, "Dimension must be greater than or equal to 0"

        if dim > 0:
            dataset_idx = dataset_idx % self.grid_count(dim - 1)

        dim_index = dataset_idx // self.grid_count(dim)
        return (
            dim_index == 0 or dim_index == self.get_individual_dim_grid_count(dim) - 1
        )

    def next_grid_along_dim(self, dataset_idx: int, dim: int):
        """
        Returns the index of the grid in the specified dimension in the specified direction.
        """
        assert dim < len(
            self.data_shape
        ), f"Dimension {dim} is out of bounds for data shape {self.data_shape}"
        assert dim >= 0, "Dimension must be greater than or equal to 0"
        new_idx = dataset_idx + self.grid_count(dim)
        if new_idx >= self.total_grid_count():
            return None
        return new_idx

    def prev_grid_along_dim(self, dataset_idx: int, dim: int):
        """
        Returns the index of the grid in the specified dimension in the specified direction.
        """
        assert dim < len(
            self.data_shape
        ), f"Dimension {dim} is out of bounds for data shape {self.data_shape}"
        assert dim >= 0, "Dimension must be greater than or equal to 0"
        new_idx = dataset_idx - self.grid_count(dim)
        if new_idx < 0:
            return None


class IndexSwitcher:
    """
    The idea is to switch from valid indices for target to invalid indices for target.
    If index in invalid for the target, then we return all zero vector as target.
    This combines both logic:
    1. Using less amount of total data.
    2. Using less amount of target data but using full data.
    """

    def __init__(self, idx_manager, data_config, patch_size) -> None:
        self.idx_manager = idx_manager
        self._data_shape = self.idx_manager.get_data_shape()
        self._training_validtarget_fraction = data_config.get(
            "training_validtarget_fraction", 1.0
        )
        self._validtarget_ceilT = int(
            np.ceil(self._data_shape[0] * self._training_validtarget_fraction)
        )
        self._patch_size = patch_size
        assert (
            data_config.deterministic_grid is True
        ), "This only works when the dataset has deterministic grid. Needed randomness comes from this class."
        assert (
            "grid_size" in data_config and data_config.grid_size == 1
        ), "We need a one to one mapping between index and h, w, t"

        self._h_validmax, self._w_validmax = self.get_reduced_frame_size(
            self._data_shape[:3], self._training_validtarget_fraction
        )
        if self._h_validmax < self._patch_size or self._w_validmax < self._patch_size:
            print(
                "WARNING: The valid target size is smaller than the patch size. This will result in all zero target. so, we are ignoring this frame for target."
            )
            self._h_validmax = 0
            self._w_validmax = 0

        print(
            f"[{self.__class__.__name__}] Target Indices: [0,{self._validtarget_ceilT - 1}]. Index={self._validtarget_ceilT - 1} has shape [:{self._h_validmax},:{self._w_validmax}].  Available data: {self._data_shape[0]}"
        )

    def get_valid_target_index(self):
        """
        Returns an index which corresponds to a frame which is expected to have a target.
        """
        _, h, w, _ = self._data_shape
        framepixelcount = h * w
        targetpixels = np.array(
            [framepixelcount] * (self._validtarget_ceilT - 1)
            + [self._h_validmax * self._w_validmax]
        )
        targetpixels = targetpixels / np.sum(targetpixels)
        t = np.random.choice(self._validtarget_ceilT, p=targetpixels)
        # t = np.random.randint(0, self._validtarget_ceilT) if self._validtarget_ceilT >= 1 else 0
        h, w = self.get_valid_target_hw(t)
        index = self.idx_manager.idx_from_hwt(h, w, t)
        # print('Valid', index, h,w,t)
        return index

    def get_invalid_target_index(self):
        # if self._validtarget_ceilT == 0:
        # TODO: There may not be enough data for this to work. The better way is to skip using 0 for invalid target.
        # t = np.random.randint(1, self._data_shape[0])
        # elif self._validtarget_ceilT < self._data_shape[0]:
        #     t = np.random.randint(self._validtarget_ceilT, self._data_shape[0])
        # else:
        #     t = self._validtarget_ceilT - 1
        # 5
        # 1.2 => 2
        total_t, h, w, _ = self._data_shape
        framepixelcount = h * w
        available_h = h - self._h_validmax
        if available_h < self._patch_size:
            available_h = 0
        available_w = w - self._w_validmax
        if available_w < self._patch_size:
            available_w = 0

        targetpixels = np.array(
            [available_h * available_w]
            + [framepixelcount] * (total_t - self._validtarget_ceilT)
        )
        t_probab = targetpixels / np.sum(targetpixels)
        t = np.random.choice(
            np.arange(self._validtarget_ceilT - 1, total_t), p=t_probab
        )

        h, w = self.get_invalid_target_hw(t)
        index = self.idx_manager.idx_from_hwt(h, w, t)
        # print('Invalid', index, h,w,t)
        return index

    def get_valid_target_hw(self, t):
        """
        This is the opposite of get_invalid_target_hw. It returns a h,w which is valid for target.
        This is only valid for single frame setup.
        """
        if t == self._validtarget_ceilT - 1:
            h = np.random.randint(0, self._h_validmax - self._patch_size)
            w = np.random.randint(0, self._w_validmax - self._patch_size)
        else:
            h = np.random.randint(0, self._data_shape[1] - self._patch_size)
            w = np.random.randint(0, self._data_shape[2] - self._patch_size)
        return h, w

    def get_invalid_target_hw(self, t):
        """
        This is the opposite of get_valid_target_hw. It returns a h,w which is not valid for target.
        This is only valid for single frame setup.
        """
        if t == self._validtarget_ceilT - 1:
            h = np.random.randint(
                self._h_validmax, self._data_shape[1] - self._patch_size
            )
            w = np.random.randint(
                self._w_validmax, self._data_shape[2] - self._patch_size
            )
        else:
            h = np.random.randint(0, self._data_shape[1] - self._patch_size)
            w = np.random.randint(0, self._data_shape[2] - self._patch_size)
        return h, w

    def _get_tidx(self, index):
        if isinstance(index, int) or isinstance(index, np.int64):
            idx = index
        else:
            idx = index[0]
        return self.idx_manager.get_t(idx)

    def index_should_have_target(self, index):
        tidx = self._get_tidx(index)
        if tidx < self._validtarget_ceilT - 1:
            return True
        elif tidx > self._validtarget_ceilT - 1:
            return False
        else:
            h, w, _ = self.idx_manager.hwt_from_idx(index)
            return (
                h + self._patch_size < self._h_validmax
                and w + self._patch_size < self._w_validmax
            )

    @staticmethod
    def get_reduced_frame_size(data_shape_nhw, fraction):
        n, h, w = data_shape_nhw

        framepixelcount = h * w
        targetpixelcount = int(n * framepixelcount * fraction)

        # We are currently supporting this only when there is just one frame.
        # if np.ceil(pixelcount / framepixelcount) > 1:
        #     return None, None

        lastframepixelcount = targetpixelcount % framepixelcount
        assert data_shape_nhw[1] == data_shape_nhw[2]
        if lastframepixelcount > 0:
            new_size = int(np.sqrt(lastframepixelcount))
            return new_size, new_size
        else:
            assert (
                targetpixelcount / framepixelcount >= 1
            ), "This is not possible in euclidean space :D (so this is a bug)"
            return h, w


rec_header_dtd = [
    ("nx", "i4"),  # Number of columns
    ("ny", "i4"),  # Number of rows
    ("nz", "i4"),  # Number of sections
    ("mode", "i4"),  # Types of pixels in the image. Values used by IMOD:
    #  0 = unsigned or signed bytes depending on flag in imodFlags
    #  1 = signed short integers (16 bits)
    #  2 = float (32 bits)
    #  3 = short * 2, (used for complex data)
    #  4 = float * 2, (used for complex data)
    #  6 = unsigned 16-bit integers (non-standard)
    # 16 = unsigned char * 3 (for rgb data, non-standard)
    ("nxstart", "i4"),  # Starting point of sub-image (not used in IMOD)
    ("nystart", "i4"),
    ("nzstart", "i4"),
    ("mx", "i4"),  # Grid size in X, Y and Z
    ("my", "i4"),
    ("mz", "i4"),
    ("xlen", "f4"),  # Cell size; pixel spacing = xlen/mx, ylen/my, zlen/mz
    ("ylen", "f4"),
    ("zlen", "f4"),
    ("alpha", "f4"),  # Cell angles - ignored by IMOD
    ("beta", "f4"),
    ("gamma", "f4"),
    # These need to be set to 1, 2, and 3 for pixel spacing to be interpreted correctly
    ("mapc", "i4"),  # map column  1=x,2=y,3=z.
    ("mapr", "i4"),  # map row     1=x,2=y,3=z.
    ("maps", "i4"),  # map section 1=x,2=y,3=z.
    # These need to be set for proper scaling of data
    ("amin", "f4"),  # Minimum pixel value
    ("amax", "f4"),  # Maximum pixel value
    ("amean", "f4"),  # Mean pixel value
    ("ispg", "i4"),  # space group number (ignored by IMOD)
    (
        "next",
        "i4",
    ),  # number of bytes in extended header (called nsymbt in MRC standard)
    ("creatid", "i2"),  # used to be an ID number, is 0 as of IMOD 4.2.23
    ("extra_data", "V30"),  # (not used, first two bytes should be 0)
    # These two values specify the structure of data in the extended header; their meaning depend on whether the
    # extended header has the Agard format, a series of 4-byte integers then real numbers, or has data
    # produced by SerialEM, a series of short integers. SerialEM stores a float as two shorts, s1 and s2, by:
    # value = (sign of s1)*(|s1|*256 + (|s2| modulo 256)) * 2**((sign of s2) * (|s2|/256))
    ("nint", "i2"),
    # Number of integers per section (Agard format) or number of bytes per section (SerialEM format)
    ("nreal", "i2"),  # Number of reals per section (Agard format) or bit
    # Number of reals per section (Agard format) or bit
    # flags for which types of short data (SerialEM format):
    # 1 = tilt angle * 100  (2 bytes)
    # 2 = piece coordinates for montage  (6 bytes)
    # 4 = Stage position * 25    (4 bytes)
    # 8 = Magnification / 100 (2 bytes)
    # 16 = Intensity * 25000  (2 bytes)
    # 32 = Exposure dose in e-/A2, a float in 4 bytes
    # 128, 512: Reserved for 4-byte items
    # 64, 256, 1024: Reserved for 2-byte items
    # If the number of bytes implied by these flags does
    # not add up to the value in nint, then nint and nreal
    # are interpreted as ints and reals per section
    ("extra_data2", "V20"),  # extra data (not used)
    ("imodStamp", "i4"),  # 1146047817 indicates that file was created by IMOD
    ("imodFlags", "i4"),  # Bit flags: 1 = bytes are stored as signed
    # Explanation of type of data
    ("idtype", "i2"),  # ( 0 = mono, 1 = tilt, 2 = tilts, 3 = lina, 4 = lins)
    ("lens", "i2"),
    # ("nd1", "i2"),  # for idtype = 1, nd1 = axis (1, 2, or 3)
    # ("nd2", "i2"),
    ("nphase", "i4"),
    ("vd1", "i2"),  # vd1 = 100. * tilt increment
    ("vd2", "i2"),  # vd2 = 100. * starting angle
    # Current angles are used to rotate a model to match a new rotated image.  The three values in each set are
    # rotations about X, Y, and Z axes, applied in the order Z, Y, X.
    ("triangles", "f4", 6),  # 0,1,2 = original:  3,4,5 = current
    ("xorg", "f4"),  # Origin of image
    ("yorg", "f4"),
    ("zorg", "f4"),
    ("cmap", "S4"),  # Contains "MAP "
    (
        "stamp",
        "u1",
        4,
    ),  # First two bytes have 17 and 17 for big-endian or 68 and 65 for little-endian
    ("rms", "f4"),  # RMS deviation of densities from mean density
    ("nlabl", "i4"),  # Number of labels with useful data
    ("labels", "S80", 10),  # 10 labels of 80 charactors
]


def read_mrc(filename, filetype="image"):
    fd = open(filename, "rb")
    header = np.fromfile(fd, dtype=rec_header_dtd, count=1)

    nx, ny, nz = header["nx"][0], header["ny"][0], header["nz"][0]

    if header[0][3] == 1:
        data_type = "int16"
    elif header[0][3] == 2:
        data_type = "float32"
    elif header[0][3] == 4:
        data_type = "single"
        nx = nx * 2
    elif header[0][3] == 6:
        data_type = "uint16"

    data = np.ndarray(shape=(nx, ny, nz))
    imgrawdata = np.fromfile(fd, data_type)
    fd.close()

    if filetype == "image":
        for iz in range(nz):
            data_2d = imgrawdata[nx * ny * iz : nx * ny * (iz + 1)]
            data[:, :, iz] = data_2d.reshape(nx, ny, order="F")
    else:
        data = imgrawdata

    return header, data
