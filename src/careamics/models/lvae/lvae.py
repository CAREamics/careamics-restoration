"""
Ladder VAE (LVAE) Model

The current implementation is based on "Interpretable Unsupervised Diversity Denoising and Artefact Removal, Prakash et al."
"""

from collections.abc import Iterable
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from careamics.config.architectures import register_model

from ..activation import get_activation
from .layers import (
    BottomUpDeterministicResBlock,
    BottomUpLayer,
    TopDownDeterministicResBlock,
    TopDownLayer,
)
from .utils import Interpolate, ModelType, crop_img_tensor, pad_img_tensor


@register_model("LVAE")
class LadderVAE(nn.Module):

    def __init__(
        self,
        input_shape: int,
        output_channels: int,
        multiscale_count: int,
        z_dims: List[int],
        encoder_n_filters: int,
        decoder_n_filters: int,
        encoder_dropout: float,
        decoder_dropout: float,
        nonlinearity: str,
        predict_logvar: bool,
        analytical_kl: bool,
    ):
        """
        Constructor.

        Parameters
        ----------

        """
        super().__init__()

        # -------------------------------------------------------
        # Customizable attributes
        self.image_size = input_shape
        self.target_ch = output_channels
        self._multiscale_count = multiscale_count
        self.z_dims = z_dims
        self.encoder_n_filters = encoder_n_filters
        self.decoder_n_filters = decoder_n_filters
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.nonlin = nonlinearity
        self.predict_logvar = predict_logvar
        self.analytical_kl = analytical_kl
        # -------------------------------------------------------

        # -------------------------------------------------------
        # Model attributes -> Hardcoded
        self.model_type = ModelType.LadderVae  # TODO remove !
        self.encoder_blocks_per_layer = 1
        self.decoder_blocks_per_layer = 1
        self.bottomup_batchnorm = True
        self.topdown_batchnorm = True
        self.topdown_conv2d_bias = True
        self.gated = True
        self.encoder_res_block_kernel = 3
        self.decoder_res_block_kernel = 3
        self.encoder_res_block_skip_padding = False
        self.decoder_res_block_skip_padding = False
        self.merge_type = "residual"
        self.no_initial_downscaling = True
        self.skip_bottomk_buvalues = 0
        self.non_stochastic_version = False
        self.stochastic_skip = True
        self.learn_top_prior = True
        self.res_block_type = "bacdbacd"  # TODO remove !
        self.mode_pred = False
        self.logvar_lowerbound = -5
        self._var_clip_max = 20
        self._stochastic_use_naive_exponential = False
        self._enable_topdown_normalize_factor = True

        # Attributes that handle LC -> Hardcoded
        self.enable_multiscale = (
            self._multiscale_count is not None and self._multiscale_count > 1
        )
        self.multiscale_retain_spatial_dims = True
        self.multiscale_lowres_separate_branch = False
        self.multiscale_decoder_retain_spatial_dims = (
            self.multiscale_retain_spatial_dims and self.enable_multiscale
        )

        # Derived attributes
        self.n_layers = len(self.z_dims)
        self.encoder_no_padding_mode = (
            self.encoder_res_block_skip_padding is True
            and self.encoder_res_block_kernel > 1
        )
        self.decoder_no_padding_mode = (
            self.decoder_res_block_skip_padding is True
            and self.decoder_res_block_kernel > 1
        )

        # Others...
        self._tethered_to_input = False
        self._tethered_ch1_scalar = self._tethered_ch2_scalar = None
        if self._tethered_to_input:
            target_ch = 1
            requires_grad = False
            self._tethered_ch1_scalar = nn.Parameter(
                torch.ones(1) * 0.5, requires_grad=requires_grad
            )
            self._tethered_ch2_scalar = nn.Parameter(
                torch.ones(1) * 2.0, requires_grad=requires_grad
            )
        # -------------------------------------------------------

        # -------------------------------------------------------
        # Data attributes
        self.color_ch = 1
        self.img_shape = (self.image_size, self.image_size)
        self.normalized_input = True
        # -------------------------------------------------------

        # -------------------------------------------------------
        # Loss attributes
        self._restricted_kl = False  # HC
        # enabling reconstruction loss on mixed input
        self.mixed_rec_w = 0
        self.nbr_consistency_w = 0

        # -------------------------------------------------------

        # -------------------------------------------------------
        # # Training attributes
        # # can be used to tile the validation predictions
        # self._val_idx_manager = val_idx_manager
        # self._val_frame_creator = None
        # # initialize the learning rate scheduler params.
        # self.lr_scheduler_monitor = self.lr_scheduler_mode = None
        # self._init_lr_scheduler_params(config)
        # self._global_step = 0
        # -------------------------------------------------------

        # -------------------------------------------------------

        # Calculate the downsampling happening in the network
        self.downsample = [1] * self.n_layers
        self.overall_downscale_factor = np.power(2, sum(self.downsample))
        if not self.no_initial_downscaling:  # by default do another downscaling
            self.overall_downscale_factor *= 2

        assert max(self.downsample) <= self.encoder_blocks_per_layer
        assert len(self.downsample) == self.n_layers
        # -------------------------------------------------------

        # -------------------------------------------------------
        ### CREATE MODEL BLOCKS
        # First bottom-up layer: change num channels + downsample by factor 2
        # unless we want to prevent this
        stride = 1 if self.no_initial_downscaling else 2
        self.first_bottom_up = self.create_first_bottom_up(stride)

        # Input Branches for Lateral Contextualization
        self.lowres_first_bottom_ups = None
        self._init_multires()

        # Other bottom-up layers
        self.bottom_up_layers = self.create_bottom_up_layers(
            self.multiscale_lowres_separate_branch
        )

        # Top-down layers
        self.top_down_layers = self.create_top_down_layers()
        self.final_top_down = self.create_final_topdown_layer(
            not self.no_initial_downscaling
        )

        # Likelihood module
        # self.likelihood = self.create_likelihood_module()

        # Output layer --> Project to target_ch many channels
        logvar_ch_needed = self.predict_logvar is not None
        self.output_layer = self.parameter_net = nn.Conv2d(
            self.decoder_n_filters,
            self.target_ch * (1 + logvar_ch_needed),
            kernel_size=3,
            padding=1,
            bias=self.topdown_conv2d_bias,
        )

        # # gradient norms. updated while training. this is also logged.
        # self.grad_norm_bottom_up = 0.0
        # self.grad_norm_top_down = 0.0
        # PSNR computation on validation.
        # self.label1_psnr = RunningPSNR()
        # self.label2_psnr = RunningPSNR()

        # msg =f'[{self.__class__.__name__}] Stoc:{not self.non_stochastic_version} RecMode:{self.reconstruction_mode} TethInput:{self._tethered_to_input}'
        # msg += f' TargetCh: {self.target_ch}'
        # print(msg)

    ### SET OF METHODS TO CREATE MODEL BLOCKS
    def create_first_bottom_up(
        self,
        init_stride: int,
        num_res_blocks: int = 1,
    ) -> nn.Sequential:
        """
        This method creates the first bottom-up block of the Encoder.
        Its role is to perform a first image compression step.
        It is composed by a sequence of nn.Conv2d + non-linearity +
        BottomUpDeterministicResBlock (1 or more, default is 1).

        Parameters
        ----------
        init_stride: int
            The stride used by the initial Conv2d block.
        num_res_blocks: int, optional
            The number of BottomUpDeterministicResBlocks to include in the layer, default is 1.
        """
        nonlin = get_activation(self.nonlin)
        modules = [
            nn.Conv2d(
                in_channels=self.color_ch,
                out_channels=self.encoder_n_filters,
                kernel_size=self.encoder_res_block_kernel,
                padding=(
                    0
                    if self.encoder_res_block_skip_padding
                    else self.encoder_res_block_kernel // 2
                ),
                stride=init_stride,
            ),
            nonlin,
        ]

        for _ in range(num_res_blocks):
            modules.append(
                BottomUpDeterministicResBlock(
                    c_in=self.encoder_n_filters,
                    c_out=self.encoder_n_filters,
                    nonlin=nonlin,
                    downsample=False,
                    batchnorm=self.bottomup_batchnorm,
                    dropout=self.encoder_dropout,
                    res_block_type=self.res_block_type,
                    skip_padding=self.encoder_res_block_skip_padding,
                    res_block_kernel=self.encoder_res_block_kernel,
                )
            )

        return nn.Sequential(*modules)

    def create_bottom_up_layers(self, lowres_separate_branch: bool) -> nn.ModuleList:
        """
        This method creates the stack of bottom-up layers of the Encoder
        that are used to generate the so-called `bu_values`.

        NOTE:
            If `self._multiscale_count < self.n_layers`, then LC is done only in the first
            `self._multiscale_count` bottom-up layers (starting from the bottom).

        Parameters
        ----------
        lowres_separate_branch: bool
            Whether the residual block(s) used for encoding the low-res input are shared (`False`) or
            not (`True`) with the "same-size" residual block(s) in the `BottomUpLayer`'s primary flow.
        """
        multiscale_lowres_size_factor = 1
        nonlin = get_activation(self.nonlin)

        bottom_up_layers = nn.ModuleList([])
        for i in range(self.n_layers):
            # Whether this is the top layer
            is_top = i == self.n_layers - 1

            # LC is applied only to the first (_multiscale_count - 1) bottom-up layers
            layer_enable_multiscale = (
                self.enable_multiscale and self._multiscale_count > i + 1
            )

            # This factor determines the factor by which the low-resolution tensor is larger
            # N.B. Only used if layer_enable_multiscale == True, so we updated it only in that case
            multiscale_lowres_size_factor *= 1 + int(layer_enable_multiscale)

            output_expected_shape = (
                (self.img_shape[0] // 2 ** (i + 1), self.img_shape[1] // 2 ** (i + 1))
                if self._multiscale_count > 1
                else None
            )

            # Add bottom-up deterministic layer at level i.
            # It's a sequence of residual blocks (BottomUpDeterministicResBlock), possibly with downsampling between them.
            bottom_up_layers.append(
                BottomUpLayer(
                    n_res_blocks=self.encoder_blocks_per_layer,
                    n_filters=self.encoder_n_filters,
                    downsampling_steps=self.downsample[i],
                    nonlin=nonlin,
                    batchnorm=self.bottomup_batchnorm,
                    dropout=self.encoder_dropout,
                    res_block_type=self.res_block_type,
                    res_block_kernel=self.encoder_res_block_kernel,
                    res_block_skip_padding=self.encoder_res_block_skip_padding,
                    gated=self.gated,
                    lowres_separate_branch=lowres_separate_branch,
                    enable_multiscale=self.enable_multiscale,  # shouldn't the arg be `layer_enable_multiscale` here?
                    multiscale_retain_spatial_dims=self.multiscale_retain_spatial_dims,
                    multiscale_lowres_size_factor=multiscale_lowres_size_factor,
                    decoder_retain_spatial_dims=self.multiscale_decoder_retain_spatial_dims,
                    output_expected_shape=output_expected_shape,
                )
            )

        return bottom_up_layers

    def create_top_down_layers(self) -> nn.ModuleList:
        """
        This method creates the stack of top-down layers of the Decoder.
        In these layer the `bu`_values` from the Encoder are merged with the `p_params` from the previous layer
        of the Decoder to get `q_params`. Then, a stochastic layer generates a sample from the latent distribution
        with parameters `q_params`. Finally, this sample is fed through a TopDownDeterministicResBlock to
        compute the `p_params` for the layer below.

        NOTE 1:
            The algorithm for generative inference approximately works as follows:
                - p_params = output of top-down layer above
                - bu = inferred bottom-up value at this layer
                - q_params = merge(bu, p_params)
                - z = stochastic_layer(q_params)
                - (optional) get and merge skip connection from prev top-down layer
                - top-down deterministic ResNet

        NOTE 2:
            When doing unconditional generation, bu_value is not available. Hence the
            merge layer is not used, and z is sampled directly from p_params.

        Parameters
        ----------
        """
        top_down_layers = nn.ModuleList([])
        nonlin = get_activation(self.nonlin)
        # NOTE: top-down layers are created starting from the bottom-most
        for i in range(self.n_layers):
            # Check if this is the top layer
            is_top = i == self.n_layers - 1

            if self._enable_topdown_normalize_factor:
                normalize_latent_factor = (
                    1 / np.sqrt(2 * (1 + i)) if len(self.z_dims) > 4 else 1.0
                )
            else:
                normalize_latent_factor = 1.0

            top_down_layers.append(
                TopDownLayer(
                    z_dim=self.z_dims[i],
                    n_res_blocks=self.decoder_blocks_per_layer,
                    n_filters=self.decoder_n_filters,
                    is_top_layer=is_top,
                    downsampling_steps=self.downsample[i],
                    nonlin=nonlin,
                    merge_type=self.merge_type,
                    batchnorm=self.topdown_batchnorm,
                    dropout=self.decoder_dropout,
                    stochastic_skip=self.stochastic_skip,
                    learn_top_prior=self.learn_top_prior,
                    top_prior_param_shape=self.get_top_prior_param_shape(),
                    res_block_type=self.res_block_type,
                    res_block_kernel=self.decoder_res_block_kernel,
                    res_block_skip_padding=self.decoder_res_block_skip_padding,
                    gated=self.gated,
                    analytical_kl=self.analytical_kl,
                    restricted_kl=self._restricted_kl,
                    vanilla_latent_hw=self.get_latent_spatial_size(i),
                    # in no_padding_mode, what gets passed from the encoder are not multiples of 2 and so merging operation does not work natively.
                    bottomup_no_padding_mode=self.encoder_no_padding_mode,
                    topdown_no_padding_mode=self.decoder_no_padding_mode,
                    retain_spatial_dims=self.multiscale_decoder_retain_spatial_dims,
                    non_stochastic_version=self.non_stochastic_version,
                    input_image_shape=self.img_shape,
                    normalize_latent_factor=normalize_latent_factor,
                    conv2d_bias=self.topdown_conv2d_bias,
                    stochastic_use_naive_exponential=self._stochastic_use_naive_exponential,
                )
            )
        return top_down_layers

    def create_final_topdown_layer(self, upsample: bool) -> nn.Sequential:
        """
        This method creates the final top-down layer of the Decoder.

        Parameters
        ----------
        upsample: bool
            Whether to upsample the input of the final top-down layer
            by bilinear interpolation with `scale_factor=2`.
        """
        # Final top-down layer
        modules = list()

        if upsample:
            modules.append(Interpolate(scale=2))

        for i in range(self.decoder_blocks_per_layer):
            modules.append(
                TopDownDeterministicResBlock(
                    c_in=self.decoder_n_filters,
                    c_out=self.decoder_n_filters,
                    nonlin=get_activation(self.nonlin),
                    batchnorm=self.topdown_batchnorm,
                    dropout=self.decoder_dropout,
                    res_block_type=self.res_block_type,
                    res_block_kernel=self.decoder_res_block_kernel,
                    skip_padding=self.decoder_res_block_skip_padding,
                    gated=self.gated,
                    conv2d_bias=self.topdown_conv2d_bias,
                )
            )
        return nn.Sequential(*modules)

    def _init_multires(
        self, config=None
    ) -> nn.ModuleList:  # TODO config: ml_collections.ConfigDict refactor
        """
        This method defines the input block/branch to encode/compress low-res lateral inputs at different hierarchical levels
        in the multiresolution approach (LC). The role of the input branches is similar to the one of the first bottom-up layer
        in the primary flow of the Encoder, namely to compress the lateral input image to a degree that is compatible with the
        one of the primary flow.

        NOTE 1: Each input branch consists of a sequence of Conv2d + non-linearity + BottomUpDeterministicResBlock.
        It is meaningful to observe that the `BottomUpDeterministicResBlock` shares the same model attributes with the blocks
        in the primary flow of the Encoder (e.g., c_in, c_out, dropout, etc. etc.). Moreover, it does not perform downsampling.

        NOTE 2: `_multiscale_count` attribute defines the total number of inputs to the bottom-up pass.
        In other terms if we have the input patch and n_LC additional lateral inputs, we will have a total of (n_LC + 1) inputs.
        """
        stride = 1 if self.no_initial_downscaling else 2
        nonlin = get_activation(self.nonlin)
        if self._multiscale_count is None:
            self._multiscale_count = 1

        msg = "Multiscale count({}) should not exceed the number of bottom up layers ({}) by more than 1"
        msg = msg.format(self._multiscale_count, self.n_layers)
        assert (
            self._multiscale_count <= 1 or self._multiscale_count <= 1 + self.n_layers
        ), msg

        msg = (
            "if multiscale is enabled, then we are just working with monocrome images."
        )
        assert self._multiscale_count == 1 or self.color_ch == 1, msg

        lowres_first_bottom_ups = []
        for _ in range(1, self._multiscale_count):
            first_bottom_up = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.color_ch,
                    out_channels=self.encoder_n_filters,
                    kernel_size=5,
                    padding=2,
                    stride=stride,
                ),
                nonlin,
                BottomUpDeterministicResBlock(
                    c_in=self.encoder_n_filters,
                    c_out=self.encoder_n_filters,
                    nonlin=nonlin,
                    downsample=False,
                    batchnorm=self.bottomup_batchnorm,
                    dropout=self.encoder_dropout,
                    res_block_type=self.res_block_type,
                    skip_padding=self.encoder_res_block_skip_padding,
                ),
            )
            lowres_first_bottom_ups.append(first_bottom_up)

        self.lowres_first_bottom_ups = (
            nn.ModuleList(lowres_first_bottom_ups)
            if len(lowres_first_bottom_ups)
            else None
        )

    ### SET OF FORWARD-LIKE METHODS
    def bottomup_pass(self, inp: torch.Tensor) -> List[torch.Tensor]:
        """
        Wrapper of _bottomup_pass().
        """
        return self._bottomup_pass(
            inp,
            self.first_bottom_up,
            self.lowres_first_bottom_ups,
            self.bottom_up_layers,
        )

    def _bottomup_pass(
        self,
        inp: torch.Tensor,
        first_bottom_up: nn.Sequential,
        lowres_first_bottom_ups: nn.ModuleList,
        bottom_up_layers: nn.ModuleList,
    ) -> List[torch.Tensor]:
        """
        This method defines the forward pass through the LVAE Encoder, the so-called
        Bottom-Up pass.

        Parameters
        ----------
        inp: torch.Tensor
            The input tensor to the bottom-up pass of shape (B, 1+n_LC, H, W), where n_LC
            is the number of lateral low-res inputs used in the LC approach.
            In particular, the first channel corresponds to the input patch, while the
            remaining ones are associated to the lateral low-res inputs.
        first_bottom_up: nn.Sequential
            The module defining the first bottom-up layer of the Encoder.
        lowres_first_bottom_ups: nn.ModuleList
            The list of modules defining Lateral Contextualization.
        bottom_up_layers: nn.ModuleList
            The list of modules defining the stack of bottom-up layers of the Encoder.
        """
        if self._multiscale_count > 1:
            x = first_bottom_up(inp[:, :1])
        else:
            x = first_bottom_up(inp)

        # Loop from bottom to top layer, store all deterministic nodes we
        # need for the top-down pass in bu_values list
        bu_values = []
        for i in range(self.n_layers):
            lowres_x = None
            if self._multiscale_count > 1 and i + 1 < inp.shape[1]:
                lowres_x = lowres_first_bottom_ups[i](inp[:, i + 1 : i + 2])

            x, bu_value = bottom_up_layers[i](x, lowres_x=lowres_x)
            bu_values.append(bu_value)

        return bu_values

    def topdown_pass(
        self,
        bu_values: torch.Tensor = None,
        n_img_prior: torch.Tensor = None,
        mode_layers: Iterable[int] = None,
        constant_layers: Iterable[int] = None,
        forced_latent: List[torch.Tensor] = None,
        top_down_layers: nn.ModuleList = None,
        final_top_down_layer: nn.Sequential = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        This method defines the forward pass through the LVAE Decoder, the so-called
        Top-Down pass.

        Parameters
        ----------
        bu_values: torch.Tensor, optional
            Output of the bottom-up pass. It will have values from multiple layers of the ladder.
        n_img_prior: optional
            When `bu_values` is `None`, `n_img_prior` indicates the number of images to generate
            from the prior (so bottom-up pass is not used at all here).
        mode_layers: Iterable[int], optional
            A sequence of indexes associated to the layers in which sampling is disabled and
            the mode (mean value) is used instead. Set to `None` to avoid this behaviour.
        constant_layers: Iterable[int], optional
            A sequence of indexes associated to the layers in which a single instance's z is
            copied over the entire batch (bottom-up path is not used, so only prior is used here).
            Set to `None` to avoid this behaviour.
        forced_latent: List[torch.Tensor], optional
            A list of tensors that are used as fixed latent variables (hence, sampling doesn't take
            place in this case).
        top_down_layers: nn.ModuleList, optional
            A list of top-down layers to use in the top-down pass. If `None`, the method uses the
            default layers defined in the constructor.
        final_top_down_layer: nn.Sequential, optional
            The last top-down layer of the top-down pass. If `None`, the method uses the default
            layers defined in the constructor.
        """
        if top_down_layers is None:
            top_down_layers = self.top_down_layers
        if final_top_down_layer is None:
            final_top_down_layer = self.final_top_down

        # Default: no layer is sampled from the distribution's mode
        if mode_layers is None:
            mode_layers = []
        if constant_layers is None:
            constant_layers = []
        prior_experiment = len(mode_layers) > 0 or len(constant_layers) > 0

        # If the bottom-up inference values are not given, don't do
        # inference, sample from prior instead
        inference_mode = bu_values is not None

        # Check consistency of arguments
        if inference_mode != (n_img_prior is None):
            msg = (
                "Number of images for top-down generation has to be given "
                "if and only if we're not doing inference"
            )
            raise RuntimeError(msg)
        if (
            inference_mode
            and prior_experiment
            and (self.non_stochastic_version is False)
        ):
            msg = (
                "Prior experiments (e.g. sampling from mode) are not"
                " compatible with inference mode"
            )
            raise RuntimeError(msg)

        # Sampled latent variables at each layer
        z = [None] * self.n_layers

        # KL divergence of each layer
        kl = [None] * self.n_layers
        # Kl divergence restricted, only for the LC enabled setup denoiSplit.
        kl_restricted = [None] * self.n_layers

        # mean from which z is sampled.
        q_mu = [None] * self.n_layers
        # log(var) from which z is sampled.
        q_lv = [None] * self.n_layers

        # Spatial map of KL divergence for each layer
        kl_spatial = [None] * self.n_layers

        debug_qvar_max = [None] * self.n_layers

        kl_channelwise = [None] * self.n_layers

        if forced_latent is None:
            forced_latent = [None] * self.n_layers

        # log p(z) where z is the sample in the topdown pass
        # logprob_p = 0.

        # Top-down inference/generation loop
        out = out_pre_residual = None
        for i in reversed(range(self.n_layers)):

            # If available, get deterministic node from bottom-up inference
            try:
                bu_value = bu_values[i]
            except TypeError:
                bu_value = None

            # Whether the current layer should be sampled from the mode
            use_mode = i in mode_layers
            constant_out = i in constant_layers

            # Input for skip connection
            skip_input = out  # TODO or n? or both?

            # Full top-down layer, including sampling and deterministic part
            out, out_pre_residual, aux = top_down_layers[i](
                input_=out,
                skip_connection_input=skip_input,
                inference_mode=inference_mode,
                bu_value=bu_value,
                n_img_prior=n_img_prior,
                use_mode=use_mode,
                force_constant_output=constant_out,
                forced_latent=forced_latent[i],
                mode_pred=self.mode_pred,
                var_clip_max=self._var_clip_max,
            )

            # Save useful variables
            z[i] = aux["z"]  # sampled variable at this layer (batch, ch, h, w)
            kl[i] = aux["kl_samplewise"]  # (batch, )
            kl_restricted[i] = aux["kl_samplewise_restricted"]
            kl_spatial[i] = aux["kl_spatial"]  # (batch, h, w)
            q_mu[i] = aux["q_mu"]
            q_lv[i] = aux["q_lv"]

            kl_channelwise[i] = aux["kl_channelwise"]
            debug_qvar_max[i] = aux["qvar_max"]
            # if self.mode_pred is False:
            #     logprob_p += aux['logprob_p'].mean()  # mean over batch
            # else:
            #     logprob_p = None

        # Final top-down layer
        out = final_top_down_layer(out)

        # Store useful variables in a dict to return them
        data = {
            "z": z,  # list of tensors with shape (batch, ch[i], h[i], w[i])
            "kl": kl,  # list of tensors with shape (batch, )
            "kl_restricted": kl_restricted,  # list of tensors with shape (batch, )
            "kl_spatial": kl_spatial,  # list of tensors w shape (batch, h[i], w[i])
            "kl_channelwise": kl_channelwise,  # list of tensors with shape (batch, ch[i])
            # 'logprob_p': logprob_p,  # scalar, mean over batch
            "q_mu": q_mu,
            "q_lv": q_lv,
            "debug_qvar_max": debug_qvar_max,
        }
        return out, data

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        x: torch.Tensor
            The input tensor of shape (B, C, H, W).
        """
        img_size = x.size()[2:]

        # Pad input to size equal to the closest power of 2
        x_pad = self.pad_input(x)

        # Bottom-up inference: return list of length n_layers (bottom to top)
        bu_values = self.bottomup_pass(x_pad)
        for i in range(0, self.skip_bottomk_buvalues):
            bu_values[i] = None

        mode_layers = range(self.n_layers) if self.non_stochastic_version else None

        # Top-down inference/generation
        out, td_data = self.topdown_pass(bu_values, mode_layers=mode_layers)

        if out.shape[-1] > img_size[-1]:
            # Restore original image size
            out = crop_img_tensor(out, img_size)

        out = self.output_layer(out)
        if self._tethered_to_input:
            assert out.shape[1] == 1
            ch2 = self.get_other_channel(out, x_pad)
            out = torch.cat([out, ch2], dim=1)

        return out, td_data

    ### SET OF UTILS METHODS
    # def sample_prior(
    #         self,
    #         n_imgs,
    #         mode_layers=None,
    #         constant_layers=None
    #     ):

    #     # Generate from prior
    #     out, _ = self.topdown_pass(n_img_prior=n_imgs, mode_layers=mode_layers, constant_layers=constant_layers)
    #     out = crop_img_tensor(out, self.img_shape)

    #     # Log likelihood and other info (per data point)
    #     _, likelihood_data = self.likelihood(out, None)

    #     return likelihood_data['sample']

    # ### ???
    # def sample_from_q(self, x, masks=None):
    #     """
    #     This method performs the bottomup_pass() and samples from the
    #     obtained distribution.
    #     """
    #     img_size = x.size()[2:]

    #     # Pad input to make everything easier with conv strides
    #     x_pad = self.pad_input(x)

    #     # Bottom-up inference: return list of length n_layers (bottom to top)
    #     bu_values = self.bottomup_pass(x_pad)
    #     return self._sample_from_q(bu_values, masks=masks)
    # ### ???

    # def _sample_from_q(self, bu_values, top_down_layers=None, final_top_down_layer=None, masks=None):
    #     if top_down_layers is None:
    #         top_down_layers = self.top_down_layers
    #     if final_top_down_layer is None:
    #         final_top_down_layer = self.final_top_down
    #     if masks is None:
    #         masks = [None] * len(bu_values)

    #     msg = "Multiscale is not supported as of now. You need the output from the previous layers to do this."
    #     assert self.n_layers == 1, msg
    #     samples = []
    #     for i in reversed(range(self.n_layers)):
    #         bu_value = bu_values[i]

    #         # Note that the first argument can be set to None since we are just dealing with one level
    #         sample = top_down_layers[i].sample_from_q(None, bu_value, var_clip_max=self._var_clip_max, mask=masks[i])
    #         samples.append(sample)

    #     return samples

    def reset_for_different_output_size(self, output_size: int) -> None:
        """Reset shape of output and latent tensors for different output size.

        Used during evaluation to reset expected shapes of tensors when
        input/output shape changes.
        For instance, it is needed when the model was trained on, say, 64x64 sized
        patches, but prediction is done on 128x128 patches.
        """
        for i in range(self.n_layers):
            sz = output_size // 2 ** (1 + i)
            self.bottom_up_layers[i].output_expected_shape = (sz, sz)
            self.top_down_layers[i].latent_shape = (output_size, output_size)

    def pad_input(self, x):
        """
        Pads input x so that its sizes are powers of 2
        :param x:
        :return: Padded tensor
        """
        size = self.get_padded_size(x.size())
        x = pad_img_tensor(x, size)
        return x

    ### SET OF GETTERS
    def get_padded_size(self, size):
        """
        Returns the smallest size (H, W) of the image with actual size given
        as input, such that H and W are powers of 2.
        :param size: input size, tuple either (N, C, H, w) or (H, W)
        :return: 2-tuple (H, W)
        """
        # Make size argument into (heigth, width)
        if len(size) == 4:
            size = size[2:]
        if len(size) != 2:
            msg = (
                "input size must be either (N, C, H, W) or (H, W), but it "
                f"has length {len(size)} (size={size})"
            )
            raise RuntimeError(msg)

        if self.multiscale_decoder_retain_spatial_dims is True:
            # In this case, we can go much more deeper and so this is not required
            # (in the way it is. ;). More work would be needed if this was to be correctly implemented )
            return list(size)

        # Overall downscale factor from input to top layer (power of 2)
        dwnsc = self.overall_downscale_factor

        # Output smallest powers of 2 that are larger than current sizes
        padded_size = list(((s - 1) // dwnsc + 1) * dwnsc for s in size)

        return padded_size

    def get_latent_spatial_size(self, level_idx: int):
        """
        level_idx: 0 is the bottommost layer, the highest resolution one.
        """
        actual_downsampling = level_idx + 1
        dwnsc = 2**actual_downsampling
        sz = self.get_padded_size(self.img_shape)
        h = sz[0] // dwnsc
        w = sz[1] // dwnsc
        assert h == w
        return h

    def get_top_prior_param_shape(self, n_imgs: int = 1):
        # TODO num channels depends on random variable we're using

        # Compute the total downscaling performed in the Encoder
        if self.multiscale_decoder_retain_spatial_dims is False:
            dwnsc = self.overall_downscale_factor
        else:
            # LC allow the encoder latents to keep the same (H, W) size at different levels
            actual_downsampling = self.n_layers + 1 - self._multiscale_count
            dwnsc = 2**actual_downsampling

        sz = self.get_padded_size(self.img_shape)
        h = sz[0] // dwnsc
        w = sz[1] // dwnsc
        c = self.z_dims[-1] * 2  # mu and logvar
        top_layer_shape = (n_imgs, c, h, w)
        return top_layer_shape

    def get_other_channel(self, ch1, input):
        assert self.data_std["target"].squeeze().shape == (2,)
        assert self.data_mean["target"].squeeze().shape == (2,)
        assert self.target_ch == 2
        ch1_un = (
            ch1[:, :1] * self.data_std["target"][:, :1]
            + self.data_mean["target"][:, :1]
        )
        input_un = input * self.data_std["input"] + self.data_mean["input"]
        ch2_un = self._tethered_ch2_scalar * (
            input_un - ch1_un * self._tethered_ch1_scalar
        )
        ch2 = (ch2_un - self.data_mean["target"][:, -1:]) / self.data_std["target"][
            :, -1:
        ]
        return ch2
