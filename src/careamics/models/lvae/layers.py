"""
Script containing the common basic blocks (nn.Module) reused by the LadderVAE architecture.

Hierarchy in the model blocks:

"""

from copy import deepcopy
from typing import Callable, Dict, Iterable, Literal, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal

from .utils import (
    StableLogVar,
    StableMean,
    crop_img_tensor,
    kl_normal_mc,
    pad_img_tensor,
)


class ResidualBlock(nn.Module):
    """
    Residual block with 2 convolutional layers.

    Some architectural notes:
        - The number of input, intermediate, and output channels is the same,
        - Padding is always 'same',
        - The 2 convolutional layers have the same groups,
        - No stride allowed,
        - Kernel sizes must be odd.

    The output isgiven by: `out = gate(f(x)) + x`.
    The presence of the gating mechanism is optional, and f(x) has different
    structures depending on the `block_type` argument.
    Specifically, `block_type` is a string specifying the block's structure, with:
        a = activation
        b = batch norm
        c = conv layer
        d = dropout.
    For example, "bacdbacd" defines a block with 2x[batchnorm, activation, conv, dropout].
    """

    default_kernel_size = (3, 3)

    def __init__(
        self,
        channels: int,
        nonlin: Callable,
        kernel: Union[int, Iterable[int]] = None,
        groups: int = 1,
        batchnorm: bool = True,
        block_type: str = None,
        dropout: float = None,
        gated: bool = None,
        skip_padding: bool = False,
        conv2d_bias: bool = True,
    ):
        """
        Constructor.

        Parameters
        ----------
        channels: int
            The number of input and output channels (they are the same).
        nonlin: Callable
            The non-linearity function used in the block (e.g., `nn.ReLU`).
        kernel: Union[int, Iterable[int]], optional
            The kernel size used in the convolutions of the block.
            It can be either a single integer or a pair of integers defining the squared kernel.
            Default is `None`.
        groups: int, optional
            The number of groups to consider in the convolutions. Default is 1.
        batchnorm: bool, optional
            Whether to use batchnorm layers. Default is `True`.
        block_type: str, optional
            A string specifying the block structure, check class docstring for more info.
            Default is `None`.
        dropout: float, optional
            The dropout probability in dropout layers. If `None` dropout is not used.
            Default is `None`.
        gated: bool, optional
            Whether to use gated layer. Default is `None`.
        skip_padding: bool, optional
            Whether to skip padding in convolutions. Default is `False`.
        conv2d_bias: bool, optional
            Whether to use bias term in convolutions. Default is `True`.
        """
        super().__init__()

        # Set kernel size & padding
        if kernel is None:
            kernel = self.default_kernel_size
        elif isinstance(kernel, int):
            kernel = (kernel, kernel)
        elif len(kernel) != 2:
            raise ValueError("kernel has to be None, int, or an iterable of length 2")
        assert all([k % 2 == 1 for k in kernel]), "kernel sizes have to be odd"
        kernel = list(kernel)
        self.skip_padding = skip_padding
        pad = [0] * len(kernel) if self.skip_padding else [k // 2 for k in kernel]
        # print(kernel, pad)

        modules = []
        if block_type == "cabdcabd":
            for i in range(2):
                conv = nn.Conv2d(
                    channels,
                    channels,
                    kernel[i],
                    padding=pad[i],
                    groups=groups,
                    bias=conv2d_bias,
                )
                modules.append(conv)
                modules.append(nonlin)
                if batchnorm:
                    modules.append(nn.BatchNorm2d(channels))
                if dropout is not None:
                    modules.append(nn.Dropout2d(dropout))
        elif block_type == "bacdbac":
            for i in range(2):
                if batchnorm:
                    modules.append(nn.BatchNorm2d(channels))
                modules.append(nonlin)
                conv = nn.Conv2d(
                    channels,
                    channels,
                    kernel[i],
                    padding=pad[i],
                    groups=groups,
                    bias=conv2d_bias,
                )
                modules.append(conv)
                if dropout is not None and i == 0:
                    modules.append(nn.Dropout2d(dropout))
        elif block_type == "bacdbacd":
            for i in range(2):
                if batchnorm:
                    modules.append(nn.BatchNorm2d(channels))
                modules.append(nonlin)
                conv = nn.Conv2d(
                    channels,
                    channels,
                    kernel[i],
                    padding=pad[i],
                    groups=groups,
                    bias=conv2d_bias,
                )
                modules.append(conv)
                modules.append(nn.Dropout2d(dropout))

        else:
            raise ValueError(f"unrecognized block type '{block_type}'")

        self.gated = gated
        if gated:
            modules.append(GateLayer2d(channels, 1, nonlin))

        self.block = nn.Sequential(*modules)

    def forward(self, x):

        out = self.block(x)
        if out.shape != x.shape:
            return out + F.center_crop(x, out.shape[-2:])
        else:
            return out + x


class ResidualGatedBlock(ResidualBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, gated=True)


class GateLayer2d(nn.Module):
    """
    Double the number of channels through a convolutional layer, then use
    half the channels as gate for the other half.
    """

    def __init__(self, channels, kernel_size, nonlin=nn.LeakyReLU):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = kernel_size // 2
        self.conv = nn.Conv2d(channels, 2 * channels, kernel_size, padding=pad)
        self.nonlin = nonlin

    def forward(self, x):
        x = self.conv(x)
        x, gate = torch.chunk(x, 2, dim=1)
        x = self.nonlin(x)  # TODO remove this?
        gate = torch.sigmoid(gate)
        return x * gate


class ResBlockWithResampling(nn.Module):
    """
    Residual block that takes care of resampling (i.e. downsampling or upsampling) steps (by a factor 2).
    It is structured as follows:
        1. `pre_conv`: a downsampling or upsampling strided convolutional layer in case of resampling, or
            a 1x1 convolutional layer that maps the number of channels of the input to `inner_channels`.
        2. `ResidualBlock`
        3. `post_conv`: a 1x1 convolutional layer that maps the number of channels to `c_out`.

    Some implementation notes:
    - Resampling is performed through a strided convolution layer at the beginning of the block.
    - The strided convolution block has fixed kernel size of 3x3 and 1 layer of zero-padding.
    - The number of channels is adjusted at the beginning and end of the block through 1x1 convolutional layers.
    - The number of internal channels is by default the same as the number of output channels, but
      min_inner_channels can override the behaviour.
    """

    def __init__(
        self,
        mode: Literal["top-down", "bottom-up"],
        c_in: int,
        c_out: int,
        min_inner_channels: int = None,
        nonlin: Callable = nn.LeakyReLU,
        resample: bool = False,
        res_block_kernel: Union[int, Iterable[int]] = None,
        groups: int = 1,
        batchnorm: bool = True,
        res_block_type: str = None,
        dropout: float = None,
        gated: bool = None,
        skip_padding: bool = False,
        conv2d_bias: bool = True,
        # lowres_input: bool = False,
    ):
        """
        Constructor.

        Parameters
        ----------
        mode: Literal["top-down", "bottom-up"]
            The type of resampling performed in the initial strided convolution of the block.
            If "bottom-up" downsampling of a factor 2 is done.
            If "top-down" upsampling of a factor 2 is done.
        c_in: int
            The number of input channels.
        c_out: int
            The number of output channels.
        min_inner_channels: int, optional
            The number of channels used in the inner layer of this module.
            Default is `None`, meaning that the number of inner channels is set to `c_out`.
        nonlin: Callable, optional
            The non-linearity function used in the block. Default is `nn.LeakyReLU`.
        resample: bool, optional
            Whether to perform resampling in the first convolutional layer.
            If `False`, the first convolutional layer just maps the input to a tensor with
            `inner_channels` channels through 1x1 convolution. Default is `False`.
        res_block_kernel: Union[int, Iterable[int]], optional
            The kernel size used in the convolutions of the residual block.
            It can be either a single integer or a pair of integers defining the squared kernel.
            Default is `None`.
        groups: int, optional
            The number of groups to consider in the convolutions. Default is 1.
        batchnorm: bool, optional
            Whether to use batchnorm layers. Default is `True`.
        res_block_type: str, optional
            A string specifying the structure of residual block.
            Check `ResidualBlock` doscstring for more information.
            Default is `None`.
        dropout: float, optional
            The dropout probability in dropout layers. If `None` dropout is not used.
            Default is `None`.
        gated: bool, optional
            Whether to use gated layer. Default is `None`.
        skip_padding: bool, optional
            Whether to skip padding in convolutions. Default is `False`.
        conv2d_bias: bool, optional
            Whether to use bias term in convolutions. Default is `True`.
        """
        super().__init__()
        assert mode in ["top-down", "bottom-up"]

        if min_inner_channels is None:
            min_inner_channels = 0
        # inner_channels is the number of channels used in the inner layers
        # of ResBlockWithResampling
        inner_channels = max(c_out, min_inner_channels)

        # Define first conv layer to change num channels and/or up/downsample
        if resample:
            if mode == "bottom-up":  # downsample
                self.pre_conv = nn.Conv2d(
                    in_channels=c_in,
                    out_channels=inner_channels,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    groups=groups,
                    bias=conv2d_bias,
                )
            elif mode == "top-down":  # upsample
                self.pre_conv = nn.ConvTranspose2d(
                    in_channels=c_in,
                    kernel_size=3,
                    out_channels=inner_channels,
                    padding=1,
                    stride=2,
                    groups=groups,
                    output_padding=1,
                    bias=conv2d_bias,
                )
        elif c_in != inner_channels:
            self.pre_conv = nn.Conv2d(
                c_in, inner_channels, 1, groups=groups, bias=conv2d_bias
            )
        else:
            self.pre_conv = None

        # Residual block
        self.res = ResidualBlock(
            channels=inner_channels,
            nonlin=nonlin,
            kernel=res_block_kernel,
            groups=groups,
            batchnorm=batchnorm,
            dropout=dropout,
            gated=gated,
            block_type=res_block_type,
            skip_padding=skip_padding,
            conv2d_bias=conv2d_bias,
        )

        # Define last conv layer to get correct num output channels
        if inner_channels != c_out:
            self.post_conv = nn.Conv2d(
                inner_channels, c_out, 1, groups=groups, bias=conv2d_bias
            )
        else:
            self.post_conv = None

    def forward(self, x):
        if self.pre_conv is not None:
            x = self.pre_conv(x)

        x = self.res(x)

        if self.post_conv is not None:
            x = self.post_conv(x)
        return x


class TopDownDeterministicResBlock(ResBlockWithResampling):

    def __init__(self, *args, upsample: bool = False, **kwargs):
        kwargs["resample"] = upsample
        super().__init__("top-down", *args, **kwargs)


class BottomUpDeterministicResBlock(ResBlockWithResampling):

    def __init__(self, *args, downsample: bool = False, **kwargs):
        kwargs["resample"] = downsample
        super().__init__("bottom-up", *args, **kwargs)


class BottomUpLayer(nn.Module):
    """
    Bottom-up deterministic layer.
    It consists of one or a stack of `BottomUpDeterministicResBlock`'s.
    The outputs are the so-called `bu_values` that are later used in the Decoder to update the
    generative distributions.

    NOTE: When Lateral Contextualization is Enabled (i.e., `enable_multiscale=True`),
    the low-res lateral input is first fed through a BottomUpDeterministicBlock (BUDB)
    (without downsampling), and then merged to the latent tensor produced by the primary flow
    of the `BottomUpLayer` through the `MergeLowRes` layer. It is meaningful to remark that
    the BUDB that takes care of encoding the low-res input can be either shared with the
    primary flow (and in that case it is the "same_size" BUDB (or stack of BUDBs) -> see `self.net`),
    or can be a deep-copy of the primary flow's BUDB.
    This behaviour is controlled by `lowres_separate_branch` parameter.
    """

    def __init__(
        self,
        n_res_blocks: int,
        n_filters: int,
        downsampling_steps: int = 0,
        nonlin: Callable = None,
        batchnorm: bool = True,
        dropout: float = None,
        res_block_type: str = None,
        res_block_kernel: int = None,
        res_block_skip_padding: bool = False,
        gated: bool = None,
        enable_multiscale: bool = False,
        multiscale_lowres_size_factor: int = None,
        lowres_separate_branch: bool = False,
        multiscale_retain_spatial_dims: bool = False,
        decoder_retain_spatial_dims: bool = False,
        output_expected_shape: Iterable[int] = None,
    ):
        """
        Constructor.

        Parameters
        ----------
        n_res_blocks: int
            Number of `BottomUpDeterministicResBlock` modules stacked in this layer.
        n_filters: int
            Number of channels present through out the layers of this block.
        downsampling_steps: int, optional
            Number of downsampling steps that has to be done in this layer (typically 1).
            Default is 0.
        nonlin: Callable, optional
            The non-linearity function used in the block. Default is `None`.
        batchnorm: bool, optional
            Whether to use batchnorm layers. Default is `True`.
        dropout: float, optional
            The dropout probability in dropout layers. If `None` dropout is not used.
            Default is `None`.
        res_block_type: str, optional
            A string specifying the structure of residual block.
            Check `ResidualBlock` doscstring for more information.
            Default is `None`.
        res_block_kernel: Union[int, Iterable[int]], optional
            The kernel size used in the convolutions of the residual block.
            It can be either a single integer or a pair of integers defining the squared kernel.
            Default is `None`.
        res_block_skip_padding: bool, optional
            Whether to skip padding in convolutions in the Residual block. Default is `False`.
        gated: bool, optional
            Whether to use gated layer. Default is `None`.
        enable_multiscale: bool, optional
            Whether to enable multiscale (Lateral Contextualization) or not. Default is `False`.
        multiscale_lowres_size_factor: int, optional
            A factor the expresses the relative size of the primary flow tensor with respect to the
            lower-resolution lateral input tensor. Default in `None`.
        lowres_separate_branch: bool, optional
            Whether the residual block(s) encoding the low-res input should be shared (`False`) or
            not (`True`) with the primary flow "same-size" residual block(s). Default is `False`.
        multiscale_retain_spatial_dims: bool, optional
            Whether to pad the latent tensor resulting from the bottom-up layer's primary flow
            to match the size of the low-res input. Default is `False`.
        decoder_retain_spatial_dims: bool, optional
            Default is `False`.
        output_expected_shape: Iterable[int], optional
            The expected shape of the layer output (only used if `enable_multiscale == True`).
            Default is `None`.
        """
        super().__init__()

        # Define attributes for Lateral Contextualization
        self.enable_multiscale = enable_multiscale
        self.lowres_separate_branch = lowres_separate_branch
        self.multiscale_retain_spatial_dims = multiscale_retain_spatial_dims
        self.multiscale_lowres_size_factor = multiscale_lowres_size_factor
        self.decoder_retain_spatial_dims = decoder_retain_spatial_dims
        self.output_expected_shape = output_expected_shape
        assert self.output_expected_shape is None or self.enable_multiscale is True

        bu_blocks_downsized = []
        bu_blocks_samesize = []
        for _ in range(n_res_blocks):
            do_resample = False
            if downsampling_steps > 0:
                do_resample = True
                downsampling_steps -= 1
            block = BottomUpDeterministicResBlock(
                c_in=n_filters,
                c_out=n_filters,
                nonlin=nonlin,
                downsample=do_resample,
                batchnorm=batchnorm,
                dropout=dropout,
                res_block_type=res_block_type,
                res_block_kernel=res_block_kernel,
                skip_padding=res_block_skip_padding,
                gated=gated,
            )
            if do_resample:
                bu_blocks_downsized.append(block)
            else:
                bu_blocks_samesize.append(block)

        self.net_downsized = nn.Sequential(*bu_blocks_downsized)
        self.net = nn.Sequential(*bu_blocks_samesize)

        # Using the same net for the low resolution (and larger sized image)
        self.lowres_net = self.lowres_merge = None
        if self.enable_multiscale:
            self._init_multiscale(
                n_filters=n_filters,
                nonlin=nonlin,
                batchnorm=batchnorm,
                dropout=dropout,
                res_block_type=res_block_type,
            )

        # msg = f'[{self.__class__.__name__}] McEnabled:{int(enable_multiscale)} '
        # if enable_multiscale:
        #     msg += f'McParallelBeam:{int(multiscale_retain_spatial_dims)} McFactor{multiscale_lowres_size_factor}'
        # print(msg)

    def _init_multiscale(
        self,
        nonlin: Callable = None,
        n_filters: int = None,
        batchnorm: bool = None,
        dropout: float = None,
        res_block_type: str = None,
    ) -> None:
        """
        This method defines the modules responsible of merging compressed lateral inputs to the outputs
        of the primary flow at different hierarchical levels in the multiresolution approach (LC).

        Specifically, the method initializes `lowres_net`, which is a stack of `BottomUpDeterministicBlock`'s
        (w/out downsampling) that takes care of additionally processing the low-res input, and `lowres_merge`,
        which is the module responsible of merging the compressed lateral input to the main flow.

        NOTE: The merge modality is set by default to "residual", meaning that the merge layer
        performs concatenation on dim=1, followed by 1x1 convolution and a Residual Gated block.

        Parameters
        ----------
        nonlin: Callable, optional
            The non-linearity function used in the block. Default is `None`.
        n_filters: int
            Number of channels present through out the layers of this block.
        batchnorm: bool, optional
            Whether to use batchnorm layers. Default is `True`.
        dropout: float, optional
            The dropout probability in dropout layers. If `None` dropout is not used.
            Default is `None`.
        res_block_type: str, optional
            A string specifying the structure of residual block.
            Check `ResidualBlock` doscstring for more information.
            Default is `None`.
        """
        self.lowres_net = self.net
        if self.lowres_separate_branch:
            self.lowres_net = deepcopy(self.net)

        self.lowres_merge = MergeLowRes(
            channels=n_filters,
            merge_type="residual",
            nonlin=nonlin,
            batchnorm=batchnorm,
            dropout=dropout,
            res_block_type=res_block_type,
            multiscale_retain_spatial_dims=self.multiscale_retain_spatial_dims,
            multiscale_lowres_size_factor=self.multiscale_lowres_size_factor,
        )

    def forward(
        self, x: torch.Tensor, lowres_x: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x: torch.Tensor
            The input of the `BottomUpLayer`, i.e., the input image or the output of the
            previous layer.
        lowres_x: torch.Tensor, optional
            The low-res input used for Lateral Contextualization (LC). Default is `None`.
        """
        # The input is fed through the residual downsampling block(s)
        primary_flow = self.net_downsized(x)
        # The downsampling output is fed through additional residual block(s)
        primary_flow = self.net(primary_flow)

        # If LC is not used, simply return output of primary-flow
        if self.enable_multiscale is False:
            assert lowres_x is None
            return primary_flow, primary_flow

        if lowres_x is not None:
            # First encode the low-res lateral input
            lowres_flow = self.lowres_net(lowres_x)
            # Then pass the result through the MergeLowRes layer
            merged = self.lowres_merge(primary_flow, lowres_flow)
        else:
            merged = primary_flow

        if (
            self.multiscale_retain_spatial_dims is False
            or self.decoder_retain_spatial_dims is True
        ):
            return merged, merged

        if self.output_expected_shape is not None:
            expected_shape = self.output_expected_shape
        else:
            fac = self.multiscale_lowres_size_factor
            expected_shape = (merged.shape[-2] // fac, merged.shape[-1] // fac)
            assert merged.shape[-2:] != expected_shape

        # Crop the resulting tensor so that it matches with the Decoder
        value_to_use_in_topdown = crop_img_tensor(merged, expected_shape)
        return merged, value_to_use_in_topdown


class MergeLayer(nn.Module):
    """
    This layer merges two or more 4D input tensors by concatenating along dim=1 and passes the result through:
    a) a convolutional 1x1 layer (`merge_type == "linear"`), or
    b) a convolutional 1x1 layer and then a gated residual block (`merge_type == "residual"`), or
    c) a convolutional 1x1 layer and then an ungated residual block (`merge_type == "residual_ungated"`).
    """

    def __init__(
        self,
        merge_type: Literal["linear", "residual", "residual_ungated"],
        channels: Union[int, Iterable[int]],
        nonlin: Callable = nn.LeakyReLU,
        batchnorm: bool = True,
        dropout: float = None,
        res_block_type: str = None,
        res_block_kernel: int = None,
        res_block_skip_padding: bool = False,
        conv2d_bias: bool = True,
    ):
        """
        Constructor.

        Parameters
        ----------
        merge_type: Literal["linear", "residual", "residual_ungated"]
            The type of merge done in the layer. It can be chosen between "linear", "residual", and "residual_ungated".
            Check the class docstring for more information about the behaviour of different merge modalities.
        channels: Union[int, Iterable[int]]
            The number of channels used in the convolutional blocks of this layer.
            If it is an `int`:
                - 1st 1x1 Conv2d: in_channels=2*channels, out_channels=channels
                - (Optional) ResBlock: in_channels=channels, out_channels=channels
            If it is an Iterable (must have `len(channels)==3`):
                - 1st 1x1 Conv2d: in_channels=sum(channels[:-1]), out_channels=channels[-1]
                - (Optional) ResBlock: in_channels=channels[-1], out_channels=channels[-1]
        nonlin: Callable, optional
            The non-linearity function used in the block. Default is `nn.LeakyReLU`.
        batchnorm: bool, optional
            Whether to use batchnorm layers. Default is `True`.
        dropout: float, optional
            The dropout probability in dropout layers. If `None` dropout is not used.
            Default is `None`.
        res_block_type: str, optional
            A string specifying the structure of residual block.
            Check `ResidualBlock` doscstring for more information.
            Default is `None`.
        res_block_kernel: Union[int, Iterable[int]], optional
            The kernel size used in the convolutions of the residual block.
            It can be either a single integer or a pair of integers defining the squared kernel.
            Default is `None`.
        res_block_skip_padding: bool, optional
            Whether to skip padding in convolutions in the Residual block. Default is `False`.
        conv2d_bias: bool, optional
            Whether to use bias term in convolutions. Default is `True`.
        """
        super().__init__()
        try:
            iter(channels)
        except TypeError:  # it is not iterable
            channels = [channels] * 3
        else:  # it is iterable
            if len(channels) == 1:
                channels = [channels[0]] * 3

        # assert len(channels) == 3

        if merge_type == "linear":
            self.layer = nn.Conv2d(
                sum(channels[:-1]), channels[-1], 1, bias=conv2d_bias
            )
        elif merge_type == "residual":
            self.layer = nn.Sequential(
                nn.Conv2d(
                    sum(channels[:-1]), channels[-1], 1, padding=0, bias=conv2d_bias
                ),
                ResidualGatedBlock(
                    channels[-1],
                    nonlin,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    block_type=res_block_type,
                    kernel=res_block_kernel,
                    conv2d_bias=conv2d_bias,
                    skip_padding=res_block_skip_padding,
                ),
            )
        elif merge_type == "residual_ungated":
            self.layer = nn.Sequential(
                nn.Conv2d(
                    sum(channels[:-1]), channels[-1], 1, padding=0, bias=conv2d_bias
                ),
                ResidualBlock(
                    channels[-1],
                    nonlin,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    block_type=res_block_type,
                    kernel=res_block_kernel,
                    conv2d_bias=conv2d_bias,
                    skip_padding=res_block_skip_padding,
                ),
            )

    def forward(self, *args) -> torch.Tensor:

        # Concatenate the input tensors along dim=1
        x = torch.cat(args, dim=1)

        # Pass the concatenated tensor through the conv layer
        x = self.layer(x)

        return x


class MergeLowRes(MergeLayer):
    """
    Child class of `MergeLayer`, specifically designed to merge the low-resolution patches
    that are used in Lateral Contextualization approach.
    """

    def __init__(self, *args, **kwargs):
        self.retain_spatial_dims = kwargs.pop("multiscale_retain_spatial_dims")
        self.multiscale_lowres_size_factor = kwargs.pop("multiscale_lowres_size_factor")
        super().__init__(*args, **kwargs)

    def forward(self, latent: torch.Tensor, lowres: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        latent: torch.Tensor
            The output latent tensor from previous layer in the LVAE hierarchy.
        lowres: torch.Tensor
            The low-res patch image to be merged to increase the context.
        """
        if self.retain_spatial_dims:
            # Pad latent tensor to match lowres tensor's shape
            latent = pad_img_tensor(latent, lowres.shape[2:])
        else:
            # Crop lowres tensor to match latent tensor's shape
            lh, lw = lowres.shape[-2:]
            h = lh // self.multiscale_lowres_size_factor
            w = lw // self.multiscale_lowres_size_factor
            h_pad = (lh - h) // 2
            w_pad = (lw - w) // 2
            lowres = lowres[:, :, h_pad:-h_pad, w_pad:-w_pad]

        return super().forward(latent, lowres)


class SkipConnectionMerger(MergeLayer):
    """
    A specialized `MergeLayer` module, designed to handle skip connections in the model.
    """

    def __init__(
        self,
        nonlin: Callable,
        channels: Union[int, Iterable[int]],
        batchnorm: bool,
        dropout: float,
        res_block_type: str,
        merge_type: Literal["linear", "residual", "residual_ungated"] = "residual",
        conv2d_bias: bool = True,
        res_block_kernel: int = None,
        res_block_skip_padding: bool = False,
    ):
        """
        Constructor.

        nonlin: Callable, optional
            The non-linearity function used in the block. Default is `nn.LeakyReLU`.
        channels: Union[int, Iterable[int]]
            The number of channels used in the convolutional blocks of this layer.
            If it is an `int`:
                - 1st 1x1 Conv2d: in_channels=2*channels, out_channels=channels
                - (Optional) ResBlock: in_channels=channels, out_channels=channels
            If it is an Iterable (must have `len(channels)==3`):
                - 1st 1x1 Conv2d: in_channels=sum(channels[:-1]), out_channels=channels[-1]
                - (Optional) ResBlock: in_channels=channels[-1], out_channels=channels[-1]
        batchnorm: bool, optional
            Whether to use batchnorm layers. Default is `True`.
        dropout: float, optional
            The dropout probability in dropout layers. If `None` dropout is not used.
            Default is `None`.
        res_block_type: str, optional
            A string specifying the structure of residual block.
            Check `ResidualBlock` doscstring for more information.
            Default is `None`.
        merge_type: Literal["linear", "residual", "residual_ungated"]
            The type of merge done in the layer. It can be chosen between "linear", "residual", and "residual_ungated".
            Check the class docstring for more information about the behaviour of different merge modalities.
        conv2d_bias: bool, optional
            Whether to use bias term in convolutions. Default is `True`.
        res_block_kernel: Union[int, Iterable[int]], optional
            The kernel size used in the convolutions of the residual block.
            It can be either a single integer or a pair of integers defining the squared kernel.
            Default is `None`.
        res_block_skip_padding: bool, optional
            Whether to skip padding in convolutions in the Residual block. Default is `False`.
        """
        super().__init__(
            channels=channels,
            nonlin=nonlin,
            merge_type=merge_type,
            batchnorm=batchnorm,
            dropout=dropout,
            res_block_type=res_block_type,
            res_block_kernel=res_block_kernel,
            conv2d_bias=conv2d_bias,
            res_block_skip_padding=res_block_skip_padding,
        )


class TopDownLayer(nn.Module):
    """
    Top-down inference layer.
    It includes:
        - Stochastic sampling,
        - Computation of KL divergence,
        - A small deterministic ResNet that performs upsampling.

    NOTE 1:
            The algorithm for generative inference approximately works as follows:
                - p_params = output of top-down layer above
                - bu = inferred bottom-up value at this layer
                - q_params = merge(bu, p_params)
                - z = stochastic_layer(q_params)
                - (optional) get and merge skip connection from prev top-down layer
                - top-down deterministic ResNet

    NOTE 2:
        The Top-Down layer can work in two modes: inference and prediction/generative.
        Depending on the particular mode, it follows distinct behaviours:
        - In inference mode, parameters of q(z_i|z_i+1) are obtained from the inference path,
        by merging outcomes of bottom-up and top-down passes. The exception is the top layer,
        in which the parameters of q(z_L|x) are set as the output of the topmost bottom-up layer.
        - On the contrary in predicition/generative mode, parameters of q(z_i|z_i+1) can be obtained
        once again by merging bottom-up and top-down outputs (CONDITIONAL GENERATION), or it is
        possible to directly sample from the prior p(z_i|z_i+1) (UNCONDITIONAL GENERATION).

    NOTE 3:
        When doing unconditional generation, bu_value is not available. Hence the
        merge layer is not used, and z is sampled directly from p_params.

    NOTE 4:
        If this is the top layer, at inference time, the uppermost bottom-up value
        is used directly as q_params, and p_params are defined in this layer
        (while they are usually taken from the previous layer), and can be learned.
    """

    def __init__(
        self,
        z_dim: int,
        n_res_blocks: int,
        n_filters: int,
        is_top_layer: bool = False,
        downsampling_steps: int = None,
        nonlin: Callable = None,
        merge_type: Literal["linear", "residual", "residual_ungated"] = None,
        batchnorm: bool = True,
        dropout: float = None,
        stochastic_skip: bool = False,
        res_block_type: str = None,
        res_block_kernel: int = None,
        res_block_skip_padding: bool = None,
        groups: int = 1,
        gated: bool = None,
        learn_top_prior: bool = False,
        top_prior_param_shape: Iterable[int] = None,
        analytical_kl: bool = False,
        bottomup_no_padding_mode: bool = False,
        topdown_no_padding_mode: bool = False,
        retain_spatial_dims: bool = False,
        restricted_kl: bool = False,
        vanilla_latent_hw: Iterable[int] = None,
        non_stochastic_version: bool = False,
        input_image_shape: Union[None, Tuple[int, int]] = None,
        normalize_latent_factor: float = 1.0,
        conv2d_bias: bool = True,
        stochastic_use_naive_exponential: bool = False,
    ):
        """
        Constructor.

        Parameters
        ----------
        z_dim: int
            The size of the latent space.
        n_res_blocks: int
            The number of TopDownDeterministicResBlock blocks
        n_filters: int
            The number of channels present through out the layers of this block.
        is_top_layer: bool, optional
            Whether the current layer is at the top of the Decoder hierarchy. Default is `False`.
        downsampling_steps: int, optional
            The number of downsampling steps that has to be done in this layer (typically 1).
            Default is `False`.
        nonlin: Callable, optional
            The non-linearity function used in the block (e.g., `nn.ReLU`). Default is `None`.
        merge_type: Literal["linear", "residual", "residual_ungated"], optional
            The type of merge done in the layer. It can be chosen between "linear", "residual",
            and "residual_ungated". Check the `MergeLayer` class docstring for more information
            about the behaviour of different merging modalities. Default is `None`.
        batchnorm: bool, optional
            Whether to use batchnorm layers. Default is `True`.
        dropout: float, optional
            The dropout probability in dropout layers. If `None` dropout is not used.
            Default is `None`.
        stochastic_skip: bool, optional
            Whether to use skip connections between previous top-down layer's output and this layer's stochastic output.
            Stochastic skip connection allows the previous layer's output has a way to directly reach this hierarchical
            level, hence facilitating the gradient flow during backpropagation. Default is `False`.
        res_block_type: str, optional
            A string specifying the structure of residual block.
            Check `ResidualBlock` documentation for more information.
            Default is `None`.
        res_block_kernel: Union[int, Iterable[int]], optional
            The kernel size used in the convolutions of the residual block.
            It can be either a single integer or a pair of integers defining the squared kernel.
            Default is `None`.
        res_block_skip_padding: bool, optional
            Whether to skip padding in convolutions in the Residual block. Default is `None`.
        groups: int, optional
            The number of groups to consider in the convolutions. Default is 1.
        gated: bool, optional
            Whether to use gated layer in `ResidualBlock`. Default is `None`.
        learn_top_prior:
            Whether to set the top prior as learnable.
            If this is set to `False`, in the top-most layer the prior will be N(0,1).
            Otherwise, we will still have a normal distribution whose parameters will be learnt.
            Default is `False`.
        top_prior_param_shape: Iterable[int], optional
            The size of the tensor which expresses the mean and the variance
            of the prior for the top most layer. Default is `None`.
        analytical_kl: bool, optional
            If True, KL divergence is calculated according to the analytical formula.
            Otherwise, an MC approximation using sampled latents is calculated.
            Default is `False`.
        bottomup_no_padding_mode: bool, optional
            Whether padding is used in the different layers of the bottom-up pass.
            It is meaningful to know this in advance in order to assess whether before
            merging `bu_values` and `p_params` tensors any alignment is needed.
            Default is `False`.
        topdown_no_padding_mode: bool, optional
            Whether padding is used in the different layers of the top-down pass.
            It is meaningful to know this in advance in order to assess whether before
            merging `bu_values` and `p_params` tensors any alignment is needed.
            The same information is also needed in handling the skip connections between
            top-down layers. Default is `False`.
        retain_spatial_dims: bool, optional
            If `True`, the size of Encoder's latent space is kept to `input_image_shape` within the topdown layer.
            This implies that the oput spatial size equals the input spatial size.
            To achieve this, we centercrop the intermediate representation.
            Default is `False`.
        restricted_kl: bool, optional
            Whether to compute the restricted version of KL Divergence.
            See `NormalStochasticBlock2d` module for more information about its computation.
            Default is `False`.
        vanilla_latent_hw: Iterable[int], optional
            The shape of the latent tensor used for prediction (i.e., it influences the computation of restricted KL).
            Default is `None`.
        non_stochastic_version: bool, optional
            Whether to replace the stochastic layer that samples a latent variable from the latent distribiution with
            a non-stochastic layer that simply drwas a sample as the mode of the latent distribution.
            Default is `False`.
        input_image_shape: Tuple[int, int], optionalut
            The shape of the input image tensor.
            When `retain_spatial_dims` is set to `True`, this is used to ensure that the shape of this layer
            output has the same shape as the input. Default is `None`.
        normalize_latent_factor: float, optional
            A factor used to normalize the latent tensors `q_params`.
            Specifically, normalization is done by dividing the latent tensor by this factor.
            Default is 1.0.
        conv2d_bias: bool, optional
            Whether to use bias term is the convolutional blocks of this layer.
            Default is `True`.
        stochastic_use_naive_exponential: bool, optional
            If `False`, in the NormalStochasticBlock2d exponentials are computed according
            to the alternative definition provided by `StableExponential` class.
            This should improve numerical stability in the training process.
            Default is `False`.
        """
        super().__init__()

        self.is_top_layer = is_top_layer
        self.z_dim = z_dim
        self.stochastic_skip = stochastic_skip
        self.learn_top_prior = learn_top_prior
        self.analytical_kl = analytical_kl
        self.bottomup_no_padding_mode = bottomup_no_padding_mode
        self.topdown_no_padding_mode = topdown_no_padding_mode
        self.retain_spatial_dims = retain_spatial_dims
        self.latent_shape = input_image_shape if self.retain_spatial_dims else None
        self.non_stochastic_version = non_stochastic_version
        self.normalize_latent_factor = normalize_latent_factor
        self._vanilla_latent_hw = vanilla_latent_hw

        # Define top layer prior parameters, possibly learnable
        if is_top_layer:
            self.top_prior_params = nn.Parameter(
                torch.zeros(top_prior_param_shape), requires_grad=learn_top_prior
            )

        # Downsampling steps left to do in this layer
        dws_left = downsampling_steps

        # Define deterministic top-down block, which is a sequence of deterministic
        # residual blocks with (optional) downsampling.
        block_list = []
        for _ in range(n_res_blocks):
            do_resample = False
            if dws_left > 0:
                do_resample = True
                dws_left -= 1
            block_list.append(
                TopDownDeterministicResBlock(
                    c_in=n_filters,
                    c_out=n_filters,
                    nonlin=nonlin,
                    upsample=do_resample,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    res_block_type=res_block_type,
                    res_block_kernel=res_block_kernel,
                    skip_padding=res_block_skip_padding,
                    gated=gated,
                    conv2d_bias=conv2d_bias,
                    groups=groups,
                )
            )
        self.deterministic_block = nn.Sequential(*block_list)

        # Define stochastic block with 2D convolutions
        if self.non_stochastic_version:
            self.stochastic = NonStochasticBlock2d(
                c_in=n_filters,
                c_vars=z_dim,
                c_out=n_filters,
                transform_p_params=(not is_top_layer),
                groups=groups,
                conv2d_bias=conv2d_bias,
            )
        else:
            self.stochastic = NormalStochasticBlock2d(
                c_in=n_filters,
                c_vars=z_dim,
                c_out=n_filters,
                transform_p_params=(not is_top_layer),
                vanilla_latent_hw=vanilla_latent_hw,
                restricted_kl=restricted_kl,
                use_naive_exponential=stochastic_use_naive_exponential,
            )

        if not is_top_layer:
            # Merge layer: it combines bottom-up inference and top-down
            # generative outcomes to give posterior parameters
            self.merge = MergeLayer(
                channels=n_filters,
                merge_type=merge_type,
                nonlin=nonlin,
                batchnorm=batchnorm,
                dropout=dropout,
                res_block_type=res_block_type,
                res_block_kernel=res_block_kernel,
                conv2d_bias=conv2d_bias,
            )

            # Skip connection that goes around the stochastic top-down layer
            if stochastic_skip:
                self.skip_connection_merger = SkipConnectionMerger(
                    channels=n_filters,
                    nonlin=nonlin,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    res_block_type=res_block_type,
                    merge_type=merge_type,
                    conv2d_bias=conv2d_bias,
                    res_block_kernel=res_block_kernel,
                    res_block_skip_padding=res_block_skip_padding,
                )

        # print(f'[{self.__class__.__name__}] normalize_latent_factor:{self.normalize_latent_factor}')

    def sample_from_q(
        self,
        input_: torch.Tensor,
        bu_value: torch.Tensor,
        var_clip_max: float = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        This method computes the latent inference distribution q(z_i|z_{i+1}) amd samples a latent tensor from it.

        Parameters
        ----------
        input_: torch.Tensor
            The input tensor to the layer, which is the output of the top-down layer above.
        bu_value: torch.Tensor
            The tensor defining the parameters /mu_q and /sigma_q computed during the bottom-up deterministic pass
            at the correspondent hierarchical layer.
        var_clip_max: float, optional
            The maximum value reachable by the log-variance of the latent distribution.
            Values exceeding this threshold are clipped. Default is `None`.
        mask: Union[None, torch.Tensor], optional
            A tensor that is used to mask the sampled latent tensor. Default is `None`.
        """
        if self.is_top_layer:  # In top layer, we don't merge bu_value with p_params
            q_params = bu_value
        else:
            # NOTE: Here the assumption is that the vampprior is only applied on the top layer.
            n_img_prior = None
            p_params = self.get_p_params(input_, n_img_prior)
            q_params = self.merge(bu_value, p_params)

        sample = self.stochastic.sample_from_q(q_params, var_clip_max)

        if mask:
            return sample[mask]

        return sample

    def get_p_params(
        self,
        input_: torch.Tensor,
        n_img_prior: int,
    ) -> torch.Tensor:
        """
        This method returns the parameters of the prior distribution p(z_i|z_{i+1}) for the latent tensor
        depending on the hierarchical level of the layer and other specific conditions.

        Parameters
        ----------
        input_: torch.Tensor
            The input tensor to the layer, which is the output of the top-down layer above.
        n_img_prior: int
            The number of images to be generated from the unconditional prior distribution p(z_L).
        """
        p_params = None

        # If top layer, define p_params as the ones of the prior p(z_L)
        if self.is_top_layer:
            p_params = self.top_prior_params

            # Sample specific number of images by expanding the prior
            if n_img_prior is not None:
                p_params = p_params.expand(n_img_prior, -1, -1, -1)

        # Else the input from the layer above is p_params itself
        else:
            p_params = input_

        return p_params

    def align_pparams_buvalue(
        self, p_params: torch.Tensor, bu_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        In case the padding is not used either (or both) in encoder and decoder, we could have a shape mismatch
        in the spatial dimensions (usually, dim=2 & dim=3).
        This method performs a centercrop to ensure that both remain aligned.

        Parameters
        ----------
        p_params: torch.Tensor
            The tensor defining the parameters /mu_p and /sigma_p for the latent distribution p(z_i|z_{i+1}).
        bu_value: torch.Tensor
            The tensor defining the parameters /mu_q and /sigma_q computed during the bottom-up deterministic pass
            at the correspondent hierarchical layer.
        """
        if bu_value.shape[-2:] != p_params.shape[-2:]:
            assert self.bottomup_no_padding_mode is True  # TODO WTF ?
            if self.topdown_no_padding_mode is False:
                assert bu_value.shape[-1] > p_params.shape[-1]
                bu_value = F.center_crop(bu_value, p_params.shape[-2:])
            else:
                if bu_value.shape[-1] > p_params.shape[-1]:
                    bu_value = F.center_crop(bu_value, p_params.shape[-2:])
                else:
                    p_params = F.center_crop(p_params, bu_value.shape[-2:])
        return p_params, bu_value

    def forward(
        self,
        input_: torch.Tensor = None,
        skip_connection_input: torch.Tensor = None,
        inference_mode: bool = False,
        bu_value: torch.Tensor = None,
        n_img_prior: int = None,
        forced_latent: torch.Tensor = None,
        use_mode: bool = False,
        force_constant_output: bool = False,
        mode_pred: bool = False,
        use_uncond_mode: bool = False,
        var_clip_max: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        input_: torch.Tensor, optional
            The input tensor to the layer, which is the output of the top-down layer above.
            Default is `None`.
        skip_connection_input: torch.Tensor, optional
            The tensor brought by the skip connection between the current and the previous top-down layer.
            Default is `None`.
        inference_mode: bool, optional
            Whether the layer is in inference mode. See NOTE 2 in class description for more info.
            Default is `False`.
        bu_value: torch.Tensor, optional
            The tensor defining the parameters /mu_q and /sigma_q computed during the bottom-up deterministic pass
            at the correspondent hierarchical layer. Default is `None`.
        n_img_prior: int, optional
            The number of images to be generated from the unconditional prior distribution p(z_L).
            Default is `None`.
        forced_latent: torch.Tensor, optional
            A pre-defined latent tensor. If it is not `None`, than it is used as the actual latent tensor and,
            hence, sampling does not happen. Default is `None`.
        use_mode: bool, optional
            Whether the latent tensor should be set as the latent distribution mode.
            In the case of Gaussian, the mode coincides with the mean of the distribution.
            Default is `False`.
        force_constant_output: bool, optional
            Whether to copy the first sample (and rel. distrib parameters) over the whole batch.
            This is used when doing experiment from the prior - q is not used.
            Default is `False`.
        mode_pred: bool, optional
            Whether the model is in prediction mode. Default is `False`.
        use_uncond_mode: bool, optional
            Whether to use the uncoditional distribution p(z) to sample latents in prediction mode.
        var_clip_max: float
            The maximum value reachable by the log-variance of the latent distribution.
            Values exceeding this threshold are clipped.
        """
        # Check consistency of arguments
        inputs_none = input_ is None and skip_connection_input is None
        if self.is_top_layer and not inputs_none:
            raise ValueError("In top layer, inputs should be None")

        p_params = self.get_p_params(input_, n_img_prior)

        # Get the parameters for the latent distribution to sample from
        if inference_mode:  # TODO What's this ?
            if self.is_top_layer:
                q_params = bu_value
                if mode_pred is False:
                    p_params, bu_value = self.align_pparams_buvalue(p_params, bu_value)
            else:
                if use_uncond_mode:
                    q_params = p_params
                else:
                    p_params, bu_value = self.align_pparams_buvalue(p_params, bu_value)
                    q_params = self.merge(bu_value, p_params)
        # In generative mode, q is not used
        else:
            q_params = None

        # NOTE: Sampling is done either from q(z_i | z_{i+1}, x) or p(z_i | z_{i+1})
        # depending on the mode (hence, in practice, by checking whether q_params is None).

        # Normalization of latent space parameters:
        # it is done, purely for stablity. See Very deep VAEs generalize autoregressive models.
        if self.normalize_latent_factor:
            q_params = q_params / self.normalize_latent_factor

        # Sample (and process) a latent tensor in the stochastic layer
        x, data_stoch = self.stochastic(
            p_params=p_params,
            q_params=q_params,
            forced_latent=forced_latent,
            use_mode=use_mode,
            force_constant_output=force_constant_output,
            analytical_kl=self.analytical_kl,
            mode_pred=mode_pred,
            use_uncond_mode=use_uncond_mode,
            var_clip_max=var_clip_max,
        )

        # Merge skip connection from previous layer
        if self.stochastic_skip and not self.is_top_layer:
            if self.topdown_no_padding_mode is True:
                # If no padding is done in the current top-down pass, there may be a shape mismatch between current tensor and skip connection input.
                # As an example, if the output of last TopDownLayer was of size 64*64, due to lack of padding in the current layer, the current tensor
                # might become different in shape, say 60*60.
                # In order to avoid shape mismatch, we do central crop of the skip connection input.
                skip_connection_input = F.center_crop(
                    skip_connection_input, x.shape[-2:]
                )

            x = self.skip_connection_merger(x, skip_connection_input)

        # Save activation before residual block as it can be the skip connection input in the next layer
        x_pre_residual = x

        if self.retain_spatial_dims:
            # when we don't want to do padding in topdown as well, we need to spare some boundary pixels which would be used up.
            extra_len = (self.topdown_no_padding_mode is True) * 3

            # this means that x should be of the same size as config.data.image_size. So, we have to centercrop by a factor of 2 at this point.
            # assert x.shape[-1] >= self.latent_shape[-1] // 2 + extra_len
            # we assume that one topdown layer will have exactly one upscaling layer.
            new_latent_shape = (
                self.latent_shape[0] // 2 + extra_len,
                self.latent_shape[1] // 2 + extra_len,
            )

            # If the LC is not applied on all layers, then this can happen.
            if x.shape[-1] > new_latent_shape[-1]:
                x = F.center_crop(x, new_latent_shape)

        # Last top-down block (sequence of residual blocks)
        x = self.deterministic_block(x)

        if self.topdown_no_padding_mode:
            x = F.center_crop(x, self.latent_shape)

        # Save some metrics that will be used in the loss computation
        keys = [
            "z",
            "kl_samplewise",
            "kl_samplewise_restricted",
            "kl_spatial",
            "kl_channelwise",
            # 'logprob_p',
            "logprob_q",
            "qvar_max",
        ]
        data = {k: data_stoch.get(k, None) for k in keys}
        data["q_mu"] = None
        data["q_lv"] = None
        if data_stoch["q_params"] is not None:
            q_mu, q_lv = data_stoch["q_params"]
            data["q_mu"] = q_mu
            data["q_lv"] = q_lv

        return x, x_pre_residual, data


class NormalStochasticBlock2d(nn.Module):
    """
    Stochastic block used in the Top-Down inference pass.

    Algorithm:
        - map input parameters to q(z) and (optionally) p(z) via convolution
        - sample a latent tensor z ~ q(z)
        - feed z to convolution and return.

    NOTE 1:
        If parameters for q are not given, sampling is done from p(z).

    NOTE 2:
        The restricted KL divergence is obtained by first computing the element-wise KL divergence
        (i.e., the KL computed for each element of the latent tensors). Then, the restricted version
        is computed by summing over the channels and the spatial dimensions associated only to the
        portion of the latent tensor that is used for prediction.
    """

    def __init__(
        self,
        c_in: int,
        c_vars: int,
        c_out: int,
        kernel: int = 3,
        transform_p_params: bool = True,
        vanilla_latent_hw: int = None,
        restricted_kl: bool = False,
        use_naive_exponential: bool = False,
    ):
        """
        Parameters
        ----------
        c_in: int
            The number of channels of the input tensor.
        c_vars: int
            The number of channels of the latent space tensor.
        c_out:  int
            The output of the stochastic layer.
            Note that this is different from the sampled latent z.
        kernel: int, optional
            The size of the kernel used in convolutional layers.
            Default is 3.
        transform_p_params: bool, optional
            Whether a transformation should be applied to the `p_params` tensor.
            The transformation consists in a 2D convolution ()`conv_in_p()`) that
            maps the input to a larger number of channels.
            Default is `True`.
        vanilla_latent_hw: int, optional
            The shape of the latent tensor used for prediction (i.e., it influences the computation of restricted KL).
            Default is `None`.
        restricted_kl: bool, optional
            Whether to compute the restricted version of KL Divergence.
            See NOTE 2 for more information about its computation.
            Default is `False`.
        use_naive_exponential: bool, optional
            If `False`, exponentials are computed according to the alternative definition
            provided by `StableExponential` class. This should improve numerical stability
            in the training process. Default is `False`.
        """
        super().__init__()
        assert kernel % 2 == 1
        pad = kernel // 2
        self.transform_p_params = transform_p_params
        self.c_in = c_in
        self.c_out = c_out
        self.c_vars = c_vars
        self._use_naive_exponential = use_naive_exponential
        self._vanilla_latent_hw = vanilla_latent_hw
        self._restricted_kl = restricted_kl

        if transform_p_params:
            self.conv_in_p = nn.Conv2d(c_in, 2 * c_vars, kernel, padding=pad)
        self.conv_in_q = nn.Conv2d(c_in, 2 * c_vars, kernel, padding=pad)
        self.conv_out = nn.Conv2d(c_vars, c_out, kernel, padding=pad)

    # def forward_swapped(self, p_params, q_mu, q_lv):
    #
    #     if self.transform_p_params:
    #         p_params = self.conv_in_p(p_params)
    #     else:
    #         assert p_params.size(1) == 2 * self.c_vars
    #
    #     # Define p(z)
    #     p_mu, p_lv = p_params.chunk(2, dim=1)
    #     p = Normal(p_mu, (p_lv / 2).exp())
    #
    #     # Define q(z)
    #     q = Normal(q_mu, (q_lv / 2).exp())
    #     # Sample from q(z)
    #     sampling_distrib = q
    #
    #     # Generate latent variable (typically by sampling)
    #     z = sampling_distrib.rsample()
    #
    #     # Output of stochastic layer
    #     out = self.conv_out(z)
    #
    #     data = {
    #         'z': z,  # sampled variable at this layer (batch, ch, h, w)
    #         'p_params': p_params,  # (b, ch, h, w) where b is 1 or batch size
    #     }
    #     return out, data

    def get_z(
        self,
        sampling_distrib: torch.distributions.normal.Normal,
        forced_latent: torch.Tensor,
        use_mode: bool,
        mode_pred: bool,
        use_uncond_mode: bool,
    ) -> torch.Tensor:
        """
        This method enables to sample a latent tensor given the distribution to sample from.

        Latent variable can be obtained is several ways:
            - Sampled from the (Gaussian) latent distribution.
            - Taken as a pre-defined forced latent.
            - Taken as the mode (mean) of the latent distribution.
            - In prediction mode (`mode_pred==True`), can be either sample or taken as the distribution mode.

        Parameters
        ----------
        sampling_distrib: torch.distributions.normal.Normal
            The Gaussian distribution from which latent tensor is sampled.
        forced_latent: torch.Tensor
            A pre-defined latent tensor. If it is not `None`, than it is used as the actual latent tensor and,
            hence, sampling does not happen.
        use_mode: bool
            Whether the latent tensor should be set as the latent distribution mode.
            In the case of Gaussian, the mode coincides with the mean of the distribution.
        mode_pred: bool
            Whether the model is prediction mode.
        use_uncond_mode: bool
            Whether to use the uncoditional distribution p(z) to sample latents in prediction mode.
        """
        if forced_latent is None:
            if use_mode:
                z = sampling_distrib.mean
            else:
                if mode_pred:
                    if use_uncond_mode:
                        z = sampling_distrib.mean
                    else:
                        z = sampling_distrib.rsample()
                else:
                    z = sampling_distrib.rsample()
        else:
            z = forced_latent
        return z

    def sample_from_q(
        self, q_params: torch.Tensor, var_clip_max: float
    ) -> torch.Tensor:
        """
        Given an input parameter tensor defining q(z),
        it processes it by calling `process_q_params()` method and
        sample a latent tensor from the resulting distribution.

        Parameters
        ----------
        q_params: torch.Tensor
            The input tensor to be processed.
        var_clip_max: float
            The maximum value reachable by the log-variance of the latent distribution.
            Values exceeding this threshold are clipped.
        """
        _, _, q = self.process_q_params(q_params, var_clip_max)
        return q.rsample()

    def compute_kl_metrics(
        self,
        p: torch.distributions.normal.Normal,
        p_params: torch.Tensor,
        q: torch.distributions.normal.Normal,
        q_params: torch.Tensor,
        mode_pred: bool,
        analytical_kl: bool,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute KL (analytical or MC estimate) and then process it, extracting composed versions of the metric.
        Specifically, the different versions of the KL loss terms are:
            - `kl_elementwise`: KL term for each single element of the latent tensor [Shape: (batch, ch, h, w)].
            - `kl_samplewise`: KL term associated to each sample in the batch [Shape: (batch, )].
            - `kl_samplewise_restricted`: KL term only associated to the portion of the latent tensor that is
            used for prediction and summed over channel and spatial dimensions [Shape: (batch, )].
            - `kl_channelwise`: KL term associated to each sample and each channel [Shape: (batch, ch, )].
            - `kl_spatial`: KL term summed over the channels, i.e., retaining the spatial dimensions [Shape: (batch, h, w)]

        Parameters
        ----------
        p: torch.distributions.normal.Normal
            The prior generative distribution p(z_i|z_{i+1}) (or p(z_L)).
        p_params: torch.Tensor
            The parameters of the prior generative distribution.
        q: torch.distributions.normal.Normal
            The inference distribution q(z_i|z_{i+1}) (or q(z_L|x)).
        q_params: torch.Tensor
            The parameters of the inference distribution.
        mode_pred: bool
            Whether the model is in prediction mode.
        analytical_kl: bool
            Whether to compute the KL divergence analytically or using Monte Carlo estimation.
        z: torch.Tensor
            The sampled latent tensor.
        """
        kl_samplewise_restricted = None

        if mode_pred is False:  # if not in prediction mode
            # KL term for each single element of the latent tensor [Shape: (batch, ch, h, w)]
            if analytical_kl:
                kl_elementwise = kl_divergence(q, p)
            else:
                kl_elementwise = kl_normal_mc(z, p_params, q_params)

            # KL term only associated to the portion of the latent tensor that is used for prediction and
            # summed over channel and spatial dimensions. [Shape: (batch, )]
            # NOTE: vanilla_latent_hw is the shape of the latent tensor used for prediction, hence
            # the restriction has shape [Shape: (batch, ch, vanilla_latent_hw[0], vanilla_latent_hw[1])]
            if self._restricted_kl:
                pad = (kl_elementwise.shape[-1] - self._vanilla_latent_hw) // 2
                assert pad > 0, "Disable restricted kl since there is no restriction."
                tmp = kl_elementwise[..., pad:-pad, pad:-pad]
                kl_samplewise_restricted = tmp.sum((1, 2, 3))

            # KL term associated to each sample in the batch [Shape: (batch, )]
            kl_samplewise = kl_elementwise.sum((1, 2, 3))

            # KL term associated to each sample and each channel [Shape: (batch, ch, )]
            kl_channelwise = kl_elementwise.sum((2, 3))

            # KL term summed over the channels, i.e., retaining the spatial dimensions [Shape: (batch, h, w)]
            kl_spatial = kl_elementwise.sum(1)
        else:  # if predicting, no need to compute KL
            kl_elementwise = kl_samplewise = kl_spatial = kl_channelwise = None

        kl_dict = {
            "kl_elementwise": kl_elementwise,  # (batch, ch, h, w)
            "kl_samplewise": kl_samplewise,  # (batch, )
            "kl_samplewise_restricted": kl_samplewise_restricted,  # (batch, )
            "kl_channelwise": kl_channelwise,  # (batch, ch)
            "kl_spatial": kl_spatial,  # (batch, h, w)
        }
        return kl_dict

    def process_p_params(
        self, p_params: torch.Tensor, var_clip_max: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.distributions.normal.Normal]:
        """
        Process the input parameters to get the prior distribution p(z_i|z_{i+1}) (or p(z_L)).

        Processing consists in:
            - (optionally) 2D convolution on the input tensor to increase number of channels.
            - split the resulting tensor into two chunks, the mean and the log-variance.
            - (optionally) clip the log-variance to an upper threshold.
            - define the normal distribution p(z) given the parameter tensors above.

        Parameters
        ----------
        p_params: torch.Tensor
            The input tensor to be processed.
        var_clip_max: float
            The maximum value reachable by the log-variance of the latent distribution.
            Values exceeding this threshold are clipped.
        """
        if self.transform_p_params:
            p_params = self.conv_in_p(p_params)
        else:
            assert p_params.size(1) == 2 * self.c_vars

        # Define p(z)
        p_mu, p_lv = p_params.chunk(2, dim=1)
        if var_clip_max is not None:
            p_lv = torch.clip(p_lv, max=var_clip_max)

        p_mu = StableMean(p_mu)
        p_lv = StableLogVar(p_lv, enable_stable=not self._use_naive_exponential)
        p = Normal(p_mu.get(), p_lv.get_std())
        return p_mu, p_lv, p

    def process_q_params(
        self, q_params: torch.Tensor, var_clip_max: float, allow_oddsizes: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.distributions.normal.Normal]:
        """
        Process the input parameters to get the inference distribution q(z_i|z_{i+1}) (or q(z|x)).

        Processing consists in:
            - 2D convolution on the input tensor to increase number of channels.
            - split the resulting tensor into two chunks, the mean and the log-variance.
            - (optionally) clip the log-variance to an upper threshold.
            - (optionally) crop the resulting tensors to ensure that the last spatial dimension is even.
            - define the normal distribution q(z) given the parameter tensors above.

        Parameters
        ----------
        p_params: torch.Tensor
            The input tensor to be processed.
        var_clip_max: float
            The maximum value reachable by the log-variance of the latent distribution.
            Values exceeding this threshold are clipped.
        """
        q_params = self.conv_in_q(q_params)

        q_mu, q_lv = q_params.chunk(2, dim=1)
        if var_clip_max is not None:
            q_lv = torch.clip(q_lv, max=var_clip_max)

        if q_mu.shape[-1] % 2 == 1 and allow_oddsizes is False:
            q_mu = F.center_crop(q_mu, q_mu.shape[-1] - 1)
            q_lv = F.center_crop(q_lv, q_lv.shape[-1] - 1)
            # clip_start = np.random.rand() > 0.5
            # q_mu = q_mu[:, :, 1:, 1:] if clip_start else q_mu[:, :, :-1, :-1]
            # q_lv = q_lv[:, :, 1:, 1:] if clip_start else q_lv[:, :, :-1, :-1]

        q_mu = StableMean(q_mu)
        q_lv = StableLogVar(q_lv, enable_stable=not self._use_naive_exponential)
        q = Normal(q_mu.get(), q_lv.get_std())
        return q_mu, q_lv, q

    def forward(
        self,
        p_params: torch.Tensor,
        q_params: torch.Tensor = None,
        forced_latent: torch.Tensor = None,
        use_mode: bool = False,
        force_constant_output: bool = False,
        analytical_kl: bool = False,
        mode_pred: bool = False,
        use_uncond_mode: bool = False,
        var_clip_max: float = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        p_params: torch.Tensor
            The output tensor of the top-down layer above (i.e., mu_{p,i+1}, sigma_{p,i+1}).
        q_params: torch.Tensor, optional
            The tensor resulting from merging the bu_value tensor at the same hierarchical level
            from the bottom-up pass and the `p_params` tensor. Default is `None`.
        forced_latent: torch.Tensor, optional
            A pre-defined latent tensor. If it is not `None`, than it is used as the actual latent
            tensor and, hence, sampling does not happen. Default is `None`.
        use_mode: bool, optional
            Whether the latent tensor should be set as the latent distribution mode.
            In the case of Gaussian, the mode coincides with the mean of the distribution.
            Default is `False`.
        force_constant_output: bool, optional
            Whether to copy the first sample (and rel. distrib parameters) over the whole batch.
            This is used when doing experiment from the prior - q is not used.
            Default is `False`.
        analytical_kl: bool, optional
            Whether to compute the KL divergence analytically or using Monte Carlo estimation.
            Default is `False`.
        mode_pred: bool, optional
            Whether the model is in prediction mode. Default is `False`.
        use_uncond_mode: bool, optional
            Whether to use the uncoditional distribution p(z) to sample latents in prediction mode.
            Default is `False`.
        var_clip_max: float, optional
            The maximum value reachable by the log-variance of the latent distribution.
            Values exceeding this threshold are clipped. Default is `None`.
        """
        debug_qvar_max = 0

        # Check sampling options consistency
        assert (forced_latent is None) or (not use_mode)

        # Get generative distribution p(z_i|z_{i+1})
        p_mu, p_lv, p = self.process_p_params(p_params, var_clip_max)
        p_params = (p_mu, p_lv)

        if q_params is not None:
            # Get inference distribution q(z_i|z_{i+1})
            # NOTE: At inference time, don't centercrop the q_params even if they are odd in size.
            q_mu, q_lv, q = self.process_q_params(
                q_params, var_clip_max, allow_oddsizes=mode_pred is True
            )
            q_params = (q_mu, q_lv)
            sampling_distrib = q
            debug_qvar_max = torch.max(q_lv.get())

            # Centercrop p_params so that their size matches the one of q_params
            q_size = q_mu.get().shape[-1]
            if p_mu.get().shape[-1] != q_size and mode_pred is False:
                p_mu.centercrop_to_size(q_size)
                p_lv.centercrop_to_size(q_size)
        else:
            sampling_distrib = p

        # Sample latent variable
        z = self.get_z(
            sampling_distrib, forced_latent, use_mode, mode_pred, use_uncond_mode
        )

        # Copy one sample (and distrib parameters) over the whole batch.
        # This is used when doing experiment from the prior - q is not used.
        if force_constant_output:
            z = z[0:1].expand_as(z).clone()
            p_params = (
                p_params[0][0:1].expand_as(p_params[0]).clone(),
                p_params[1][0:1].expand_as(p_params[1]).clone(),
            )

        # Pass the sampled latent througn the output convolutional layer of stochastic block
        out = self.conv_out(z)

        # Compute log p(z)# NOTE: disabling its computation.
        # if mode_pred is False:
        #     logprob_p =  p.log_prob(z).sum((1, 2, 3))
        # else:
        #     logprob_p = None

        if q_params is not None:
            # Compute log q(z)
            logprob_q = q.log_prob(z).sum((1, 2, 3))
            # Compute KL divergence metrics
            kl_dict = self.compute_kl_metrics(
                p, p_params, q, q_params, mode_pred, analytical_kl, z
            )
        else:
            kl_dict = {}
            logprob_q = None

        # Store meaningful quantities to use them in following layers
        data = kl_dict
        data["z"] = z  # sampled variable at this layer (batch, ch, h, w)
        data["p_params"] = p_params  # (b, ch, h, w) where b is 1 or batch size
        data["q_params"] = q_params  # (batch, ch, h, w)
        # data['logprob_p'] = logprob_p  # (batch, )
        data["logprob_q"] = logprob_q  # (batch, )
        data["qvar_max"] = debug_qvar_max

        return out, data


class NonStochasticBlock2d(nn.Module):
    """
    Non-stochastic version of the NormalStochasticBlock2d.
    """

    def __init__(
        self,
        c_vars: int,
        c_in: int,
        c_out: int,
        kernel: int = 3,
        groups: int = 1,
        conv2d_bias: bool = True,
        transform_p_params: bool = True,
    ):
        """
        Constructor.

        Parameters
        ----------
        c_vars: int
            The number of channels of the latent space tensor.
        c_in: int
            The number of channels of the input tensor.
        c_out:  int
            The output of the stochastic layer.
            Note that this is different from the sampled latent z.
        kernel: int, optional
            The size of the kernel used in convolutional layers.
            Default is 3.
        groups: int, optional
            The number of groups to consider in the convolutions of this layer.
            Default is 1.
        conv2d_bias: bool, optional
            Whether to use bias term is the convolutional blocks of this layer.
            Default is `True`.
        transform_p_params: bool, optional
            Whether a transformation should be applied to the `p_params` tensor.
            The transformation consists in a 2D convolution ()`conv_in_p()`) that
            maps the input to a larger number of channels.
            Default is `True`.
        """
        super().__init__()
        assert kernel % 2 == 1
        pad = kernel // 2
        self.transform_p_params = transform_p_params
        self.c_in = c_in
        self.c_out = c_out
        self.c_vars = c_vars

        if transform_p_params:
            self.conv_in_p = nn.Conv2d(
                c_in, 2 * c_vars, kernel, padding=pad, bias=conv2d_bias, groups=groups
            )
        self.conv_in_q = nn.Conv2d(
            c_in, 2 * c_vars, kernel, padding=pad, bias=conv2d_bias, groups=groups
        )
        self.conv_out = nn.Conv2d(
            c_vars, c_out, kernel, padding=pad, bias=conv2d_bias, groups=groups
        )

    def compute_kl_metrics(
        self,
        p: torch.distributions.normal.Normal,
        p_params: torch.Tensor,
        q: torch.distributions.normal.Normal,
        q_params: torch.Tensor,
        mode_pred: bool,
        analytical_kl: bool,
        z: torch.Tensor,
    ) -> Dict[str, None]:
        """
        Compute KL (analytical or MC estimate) and then process it, extracting composed versions of the metric.
        Specifically, the different versions of the KL loss terms are:
            - `kl_elementwise`: KL term for each single element of the latent tensor [Shape: (batch, ch, h, w)].
            - `kl_samplewise`: KL term associated to each sample in the batch [Shape: (batch, )].
            - `kl_samplewise_restricted`: KL term only associated to the portion of the latent tensor that is
            used for prediction and summed over channel and spatial dimensions [Shape: (batch, )].
            - `kl_channelwise`: KL term associated to each sample and each channel [Shape: (batch, ch, )].
            - `kl_spatial`: # KL term summed over the channels, i.e., retaining the spatial dimensions [Shape: (batch, h, w)]

        NOTE: in this class all the KL metrics are set to `None`.

        Parameters
        ----------
        p: torch.distributions.normal.Normal
            The prior generative distribution p(z_i|z_{i+1}) (or p(z_L)).
        p_params: torch.Tensor
            The parameters of the prior generative distribution.
        q: torch.distributions.normal.Normal
            The inference distribution q(z_i|z_{i+1}) (or q(z_L|x)).
        q_params: torch.Tensor
            The parameters of the inference distribution.
        mode_pred: bool
            Whether the model is in prediction mode.
        analytical_kl: bool
            Whether to compute the KL divergence analytically or using Monte Carlo estimation.
        z: torch.Tensor
            The sampled latent tensor.
        """
        kl_dict = {
            "kl_elementwise": None,  # (batch, ch, h, w)
            "kl_samplewise": None,  # (batch, )
            "kl_spatial": None,  # (batch, h, w)
            "kl_channelwise": None,  # (batch, ch)
        }
        return kl_dict

    def process_p_params(self, p_params, var_clip_max):
        if self.transform_p_params:
            p_params = self.conv_in_p(p_params)
        else:

            assert (
                p_params.size(1) == 2 * self.c_vars
            ), f"{p_params.shape} {self.c_vars}"

        # Define p(z)
        p_mu, p_lv = p_params.chunk(2, dim=1)
        return p_mu, None

    def process_q_params(self, q_params, var_clip_max, allow_oddsizes=False):
        # Define q(z)
        q_params = self.conv_in_q(q_params)
        q_mu, q_lv = q_params.chunk(2, dim=1)

        if q_mu.shape[-1] % 2 == 1 and allow_oddsizes is False:
            q_mu = F.center_crop(q_mu, q_mu.shape[-1] - 1)

        return q_mu, None

    def forward(
        self,
        p_params: torch.Tensor,
        q_params: torch.Tensor = None,
        forced_latent: Union[None, torch.Tensor] = None,
        use_mode: bool = False,
        force_constant_output: bool = False,
        analytical_kl: bool = False,
        mode_pred: bool = False,
        use_uncond_mode: bool = False,
        var_clip_max: float = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        p_params: torch.Tensor
            The output tensor of the top-down layer above (i.e., mu_{p,i+1}, sigma_{p,i+1}).
        q_params: torch.Tensor, optional
            The tensor resulting from merging the bu_value tensor at the same hierarchical level
            from the bottom-up pass and the `p_params` tensor. Default is `None`.
        forced_latent: torch.Tensor, optional
            A pre-defined latent tensor. If it is not `None`, than it is used as the actual latent
            tensor and, hence, sampling does not happen. Default is `None`.
        use_mode: bool, optional
            Whether the latent tensor should be set as the latent distribution mode.
            In the case of Gaussian, the mode coincides with the mean of the distribution.
            Default is `False`.
        force_constant_output: bool, optional
            Whether to copy the first sample (and rel. distrib parameters) over the whole batch.
            This is used when doing experiment from the prior - q is not used.
            Default is `False`.
        analytical_kl: bool, optional
            Whether to compute the KL divergence analytically or using Monte Carlo estimation.
            Default is `False`.
        mode_pred: bool, optional
            Whether the model is in prediction mode. Default is `False`.
        use_uncond_mode: bool, optional
            Whether to use the uncoditional distribution p(z) to sample latents in prediction mode.
            Default is `False`.
        var_clip_max: float, optional
            The maximum value reachable by the log-variance of the latent distribution.
            Values exceeding this threshold are clipped. Default is `None`.
        """
        debug_qvar_max = 0
        assert (forced_latent is None) or (not use_mode)

        p_mu, _ = self.process_p_params(p_params, var_clip_max)

        p_params = (p_mu, None)

        if q_params is not None:
            # At inference time, just don't centercrop the q_params even if they are odd in size.
            q_mu, _ = self.process_q_params(
                q_params, var_clip_max, allow_oddsizes=mode_pred is True
            )
            q_params = (q_mu, None)
            debug_qvar_max = torch.Tensor([1]).to(q_mu.device)
            # Sample from q(z)
            sampling_distrib = q_mu
            q_size = q_mu.shape[-1]
            if p_mu.shape[-1] != q_size and mode_pred is False:
                p_mu.centercrop_to_size(q_size)
        else:
            # Sample from p(z)
            sampling_distrib = p_mu

        # Generate latent variable (typically by sampling)
        z = sampling_distrib

        # Copy one sample (and distrib parameters) over the whole batch.
        # This is used when doing experiment from the prior - q is not used.
        if force_constant_output:
            z = z[0:1].expand_as(z).clone()
            p_params = (
                p_params[0][0:1].expand_as(p_params[0]).clone(),
                p_params[1][0:1].expand_as(p_params[1]).clone(),
            )

        # Output of stochastic layer
        out = self.conv_out(z)

        kl_dict = {}
        logprob_q = None

        data = kl_dict
        data["z"] = z  # sampled variable at this layer (batch, ch, h, w)
        data["p_params"] = p_params  # (b, ch, h, w) where b is 1 or batch size
        data["q_params"] = q_params  # (batch, ch, h, w)
        data["logprob_q"] = logprob_q  # (batch, )
        data["qvar_max"] = debug_qvar_max

        return out, data
