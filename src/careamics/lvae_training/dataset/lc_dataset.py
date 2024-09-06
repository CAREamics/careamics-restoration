"""
A place for Datasets and Dataloaders.
"""

from typing import Tuple, Union

import numpy as np
from skimage.transform import resize

from .lc_dataset_config import LCVaeDatasetConfig
from .vae_dataset import MultiChDloader


class LCMultiChDloader(MultiChDloader):

    def __init__(
        self,
        data_config: LCVaeDatasetConfig,
        fpath: str,
        val_fraction=None,
        test_fraction=None,
    ):
        """
        Args:
            num_scales: The number of resolutions at which we want the input. Note that the target is formed at the
                        highest resolution.
        """
        self._padding_kwargs = (
            data_config.padding_kwargs  # mode=padding_mode, constant_values=constant_value
        )
        self._uncorrelated_channel_probab = data_config.uncorrelated_channel_probab

        if data_config.overlapping_padding_kwargs is not None:
            assert (
                self._padding_kwargs == data_config.overlapping_padding_kwargs
            ), "During evaluation, overlapping_padding_kwargs should be same as padding_args. \
                It should be so since we just use overlapping_padding_kwargs when it is not None"

        else:
            overlapping_padding_kwargs = data_config.padding_kwargs

        super().__init__(
            data_config, fpath, val_fraction=val_fraction, test_fraction=test_fraction
        )
        self.num_scales = data_config.num_scales
        assert self.num_scales is not None
        self._scaled_data = [self._data]
        self._scaled_noise_data = [self._noise_data]

        assert isinstance(self.num_scales, int) and self.num_scales >= 1
        assert isinstance(self._padding_kwargs, dict)
        assert "mode" in self._padding_kwargs

        for _ in range(1, self.num_scales):
            shape = self._scaled_data[-1].shape
            assert len(shape) == 4
            new_shape = (shape[0], shape[1] // 2, shape[2] // 2, shape[3])
            ds_data = resize(
                self._scaled_data[-1].astype(np.float32), new_shape
            ).astype(self._scaled_data[-1].dtype)
            # NOTE: These asserts are important. the resize method expects np.float32. otherwise, one gets weird results.
            assert (
                ds_data.max() / self._scaled_data[-1].max() < 5
            ), "Downsampled image should not have very different values"
            assert (
                ds_data.max() / self._scaled_data[-1].max() > 0.2
            ), "Downsampled image should not have very different values"

            self._scaled_data.append(ds_data)
            # do the same for noise
            if self._noise_data is not None:
                noise_data = resize(self._scaled_noise_data[-1], new_shape)
                self._scaled_noise_data.append(noise_data)

    def reduce_data(
        self, t_list=None, h_start=None, h_end=None, w_start=None, w_end=None
    ):
        assert t_list is not None
        assert h_start is None
        assert h_end is None
        assert w_start is None
        assert w_end is None

        self._data = self._data[t_list].copy()
        self._scaled_data = [
            self._scaled_data[i][t_list].copy() for i in range(len(self._scaled_data))
        ]

        if self._noise_data is not None:
            self._noise_data = self._noise_data[t_list].copy()
            self._scaled_noise_data = [
                self._scaled_noise_data[i][t_list].copy()
                for i in range(len(self._scaled_noise_data))
            ]

        self.N = len(t_list)
        self.set_img_sz(self._img_sz, self._grid_sz)
        print(
            f"[{self.__class__.__name__}] Data reduced. New data shape: {self._data.shape}"
        )

    def _init_msg(self):
        msg = super()._init_msg()
        msg += f" Pad:{self._padding_kwargs}"
        if self._uncorrelated_channels:
            msg += f" UncorrChProbab:{self._uncorrelated_channel_probab}"
        return msg

    def _load_scaled_img(
        self, scaled_index, index: Union[int, Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(index, int):
            idx = index
        else:
            idx, _ = index

        # tidx = self.idx_manager.get_t(idx)
        patch_loc_list = self.idx_manager.get_patch_location_from_dataset_idx(idx)
        nidx = patch_loc_list[0]

        imgs = self._scaled_data[scaled_index][nidx]
        imgs = tuple([imgs[None, ..., i] for i in range(imgs.shape[-1])])
        if self._noise_data is not None:
            noisedata = self._scaled_noise_data[scaled_index][nidx]
            noise = tuple([noisedata[None, ..., i] for i in range(noisedata.shape[-1])])
            factor = np.sqrt(2) if self._input_is_sum else 1.0
            imgs = tuple([img + noise[0] * factor for img in imgs])
        return imgs

    def _crop_img(self, img: np.ndarray, patch_start_loc: Tuple):
        """
        Here, h_start, w_start could be negative. That simply means we need to pick the content from 0. So,
        the cropped image will be smaller than self._img_sz * self._img_sz
        """
        max_len_vals = list(self.idx_manager.data_shape[1:-1])
        max_len_vals[-2:] = img.shape[-2:]
        return self._crop_img_with_padding(
            img, patch_start_loc, max_len_vals=max_len_vals
        )

    def _get_img(self, index: int):
        """
        Returns the primary patch along with low resolution patches centered on the primary patch.
        """
        # Noise_tuples is populated when there is synthetic noise in training
        # Should have similar type of noise with the noise model
        # Starting with microsplit, dump the noise, use it instead as an augmentation if nessesary
        img_tuples, noise_tuples = self._load_img(index)
        assert self._img_sz is not None
        h, w = img_tuples[0].shape[-2:]
        if self._enable_random_cropping:
            patch_start_loc = self._get_random_hw(h, w)
            if self._5Ddata:
                patch_start_loc = (
                    np.random.choice(img_tuples[0].shape[-3] - self._depth3D),
                ) + patch_start_loc
        else:
            patch_start_loc = self._get_deterministic_loc(index)

        # LC logic is located here, the function crops the image of the highest resolution
        cropped_img_tuples = [
            self._crop_flip_img(img, patch_start_loc, False, False)
            for img in img_tuples
        ]
        cropped_noise_tuples = [
            self._crop_flip_img(noise, patch_start_loc, False, False)
            for noise in noise_tuples
        ]
        patch_start_loc = list(patch_start_loc)
        h_start, w_start = patch_start_loc[-2], patch_start_loc[-1]
        h_center = h_start + self._img_sz // 2
        w_center = w_start + self._img_sz // 2
        allres_versions = {
            i: [cropped_img_tuples[i]] for i in range(len(cropped_img_tuples))
        }
        for scale_idx in range(1, self.num_scales):
            # Returning the image of the lower resolution
            scaled_img_tuples = self._load_scaled_img(scale_idx, index)

            h_center = h_center // 2
            w_center = w_center // 2

            h_start = h_center - self._img_sz // 2
            w_start = w_center - self._img_sz // 2
            patch_start_loc[-2:] = [h_start, w_start]
            scaled_cropped_img_tuples = [
                self._crop_flip_img(img, patch_start_loc, False, False)
                for img in scaled_img_tuples
            ]
            for ch_idx in range(len(img_tuples)):
                allres_versions[ch_idx].append(scaled_cropped_img_tuples[ch_idx])

        output_img_tuples = tuple(
            [
                np.concatenate(allres_versions[ch_idx])
                for ch_idx in range(len(img_tuples))
            ]
        )
        return output_img_tuples, cropped_noise_tuples

    def __getitem__(self, index: Union[int, Tuple[int, int]]):
        img_tuples, noise_tuples = self._get_img(index)
        if self._uncorrelated_channels:
            assert (
                self._input_idx is None
            ), "Uncorrelated channels is not implemented when there is a separate input channel."
            if np.random.rand() < self._uncorrelated_channel_probab:
                img_tuples_new = [None] * len(img_tuples)
                img_tuples_new[0] = img_tuples[0]
                for i in range(1, len(img_tuples)):
                    new_index = np.random.randint(len(self))
                    img_tuples_tmp, _ = self._get_img(new_index)
                    img_tuples_new[i] = img_tuples_tmp[i]
                img_tuples = img_tuples_new

        if self._is_train:
            if self._empty_patch_replacement_enabled:
                if np.random.rand() < self._empty_patch_replacement_probab:
                    img_tuples = self.replace_with_empty_patch(img_tuples)

        if self._enable_rotation:
            img_tuples, noise_tuples = self._rotate(img_tuples, noise_tuples)

        # add noise to input, if noise is present combine it with the image
        # factor is for the compute input not to have too much noise because the average of two gaussians
        if len(noise_tuples) > 0:
            factor = np.sqrt(2) if self._input_is_sum else 1.0
            input_tuples = []
            for x in img_tuples:
                # NOTE: other LC levels already have noise added. So, we just need to add noise to the highest resolution.
                x[0] = x[0] + noise_tuples[0] * factor
                input_tuples.append(x)
        else:
            input_tuples = img_tuples

        # Compute the input by sum / average the channels
        # Alpha is an amount of weight which is applied to the channels when combining them
        # How to sample alpha is still under research
        inp, alpha = self._compute_input(input_tuples)
        target_tuples = [img[:1] for img in img_tuples]
        # add noise to target.
        if len(noise_tuples) >= 1:
            target_tuples = [
                x + noise for x, noise in zip(target_tuples, noise_tuples[1:])
            ]

        target = self._compute_target(target_tuples, alpha)

        output = [inp, target]

        if self._return_alpha:
            output.append(alpha)

        if isinstance(index, int):
            return tuple(output)

        _, grid_size = index
        output.append(grid_size)
        return tuple(output)
