from typing import Any, Optional
from enum import Enum

from pydantic import BaseModel, ConfigDict, computed_field


# TODO: get rid of unnecessary enums
class DataType(Enum):
    MNIST = 0
    Places365 = 1
    NotMNIST = 2
    OptiMEM100_014 = 3
    CustomSinosoid = 4
    Prevedel_EMBL = 5
    AllenCellMito = 6
    SeparateTiffData = 7
    CustomSinosoidThreeCurve = 8
    SemiSupBloodVesselsEMBL = 9
    Pavia2 = 10
    Pavia2VanillaSplitting = 11
    ExpansionMicroscopyMitoTub = 12
    ShroffMitoEr = 13
    HTIba1Ki67 = 14
    BSD68 = 15
    BioSR_MRC = 16
    TavernaSox2Golgi = 17
    Dao3Channel = 18
    ExpMicroscopyV2 = 19
    Dao3ChannelWithInput = 20
    TavernaSox2GolgiV2 = 21
    TwoDset = 22
    PredictedTiffData = 23
    Pavia3SeqData = 24
    # Here, we have 16 splitting tasks.
    NicolaData = 25


class DataSplitType(Enum):
    All = 0
    Train = 1
    Val = 2
    Test = 3


class GridAlignement(Enum):
    """
    A patch is formed by padding the grid with content. If the grids are 'Center' aligned, then padding is to done equally on all 4 sides.
    On the other hand, if grids are 'LeftTop' aligned, padding is to be done on the right and bottom end of the grid.
    In the former case, one needs (patch_size - grid_size)//2 amount of content on the right end of the frame.
    In the latter case, one needs patch_size - grid_size amount of content on the right end of the frame.
    """

    LeftTop = 0
    Center = 1


# TODO: for all bool params check if they are taking different values in Disentangle repo
# TODO: check if any bool logic can be removed
class VaeDatasetConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    data_type: Optional[DataType]
    """Type of the dataset, should be one of DataType"""

    depth3D: Optional[int] = 1
    """Number of slices in 3D. If data is 2D depth3D is equal to 1"""

    datasplit_type: Optional[DataSplitType] = None
    """Whether to return training, validation or test split, should be one of 
    DataSplitType"""

    num_channels: Optional[int] = 2
    """Number of channels in the input"""

    # TODO: remove ch*_fname parameters, should be parsed automatically from a name list
    ch1_fname: Optional[str] = None
    ch2_fname: Optional[str] = None
    ch_input_fname: Optional[str] = None

    input_is_sum: Optional[bool] = False
    """Whether the input is the sum or average of channels"""

    input_idx: Optional[int] = None
    """Index of the channel where the input is stored in the data"""

    target_idx_list: Optional[list[int]] = None
    """Indices of the channels where the targets are stored in the data"""

    # TODO: where are there used?
    start_alpha: Optional[Any] = None
    end_alpha: Optional[Any] = None

    image_size: int
    """Size of one patch of data"""

    grid_size: Optional[int] = None
    """Frame is divided into square grids of this size. A patch centered on a grid 
    having size `image_size` is returned. Grid size not used in training,
    used only during val / test, grid size controls the overlap of the patches"""

    empty_patch_replacement_enabled: Optional[bool] = False
    """Whether to replace the content of one of the channels
    with background with given probability"""
    empty_patch_replacement_channel_idx: Optional[Any] = None
    empty_patch_replacement_probab: Optional[Any] = None
    empty_patch_max_val_threshold: Optional[Any] = None

    uncorrelated_channels: Optional[bool] = False
    """Replace the content in one of the channels with given probability to make 
    channel content 'uncorrelated'"""
    uncorrelated_channel_probab: Optional[float] = 0.5

    poisson_noise_factor: Optional[float] = -1
    """The added poisson noise factor"""

    synthetic_gaussian_scale: Optional[float] = 0.1

    # TODO: set to True in training code, recheck
    input_has_dependant_noise: Optional[bool] = False

    # TODO: sometimes max_val differs between runs with fixed seeds with noise enabled
    enable_gaussian_noise: Optional[bool] = False
    """Whether to enable gaussian noise"""

    # TODO: is this parameter used?
    allow_generation: bool = False

    # TODO: both used in IndexSwitcher, insure correct passing
    training_validtarget_fraction: Any = None
    deterministic_grid: Any = None

    # TODO: why is this not used?
    enable_rotation_aug: Optional[bool] = False

    grid_alignment: GridAlignement = GridAlignement.LeftTop

    max_val: Optional[float] = None
    """Maximum data in the dataset. Is calculated for train split, and should be 
    externally set for val and test splits."""

    trim_boundary: Optional[bool] = True
    """Whether to trim boundary of the image"""

    overlapping_padding_kwargs: Any = None
    """Parameters for np.pad method"""

    # TODO: remove this parameter, controls debug print
    print_vars: Optional[bool] = False

    # Hard-coded parameters (used to be in the config file)
    normalized_input: bool = True
    """If this is set to true, then one mean and stdev is used
                for both channels. Otherwise, two different mean and stdev are used."""
    use_one_mu_std: Optional[bool] = True

    # TODO: is this parameter used?
    train_aug_rotate: Optional[bool] = False
    enable_random_cropping: Optional[bool] = True

    # TODO: not used?
    multiscale_lowres_count: Optional[int] = None

    @computed_field
    @property
    def padding_kwargs(self) -> dict:
        kwargs_dict = {}
        padding_kwargs = {}
        if (
            self.multiscale_lowres_count is not None
            and self.multiscale_lowres_count is not None
        ):
            # Get padding attributes
            if "padding_kwargs" not in kwargs_dict:
                padding_kwargs = {}
                padding_kwargs["mode"] = "constant"
                padding_kwargs["constant_values"] = 0
            else:
                padding_kwargs = kwargs_dict.pop("padding_kwargs")
        return padding_kwargs
