"""
This script provides methods to evaluate the performance of the LVAE model.
It includes functions to:
    - make predictions,
    - quantify the performance of the model
    - create plots to visualize the results.
"""

import math
import os
from typing import Dict, List, Literal, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader
from tqdm import tqdm

from careamics.lightning import VAEModule
from careamics.losses.lvae.losses import (
    get_reconstruction_loss,
    reconstruction_loss_musplit_denoisplit,
)
from careamics.models.lvae.utils import ModelType
from careamics.utils.metrics import scale_invariant_psnr, RunningPSNR


# ------------------------------------------------------------------------------------------------
# Function of plotting: TODO -> moved them to another file, plot_utils.py
def clean_ax(ax):
    """
    Helper function to remove ticks from axes in plots.
    """
    # 2D or 1D axes are of type np.ndarray
    if isinstance(ax, np.ndarray):
        for one_ax in ax:
            clean_ax(one_ax)
        return

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.tick_params(left=False, right=False, top=False, bottom=False)


def get_eval_output_dir(
    saveplotsdir: str, patch_size: int, mmse_count: int = 50
) -> str:
    """
    Given the path to a root directory to save plots, patch size, and mmse count,
    it returns the specific directory to save the plots.
    """
    eval_out_dir = os.path.join(
        saveplotsdir, f"eval_outputs/patch_{patch_size}_mmse_{mmse_count}"
    )
    os.makedirs(eval_out_dir, exist_ok=True)
    print(eval_out_dir)
    return eval_out_dir


def get_psnr_str(tar_hsnr, pred, col_idx):
    """
    Compute PSNR between the ground truth (`tar_hsnr`) and the predicted image (`pred`).
    """
    return f"{scale_invariant_psnr(tar_hsnr[col_idx][None], pred[col_idx][None]).item():.1f}"


def add_psnr_str(ax_, psnr):
    """
    Add psnr string to the axes
    """
    textstr = f"PSNR\n{psnr}"
    props = dict(boxstyle="round", facecolor="gray", alpha=0.5)
    # place a text box in upper left in axes coords
    ax_.text(
        0.05,
        0.95,
        textstr,
        transform=ax_.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
        color="white",
    )


def get_last_index(bin_count, quantile):
    cumsum = np.cumsum(bin_count)
    normalized_cumsum = cumsum / cumsum[-1]
    for i in range(1, len(normalized_cumsum)):
        if normalized_cumsum[-i] < quantile:
            return i - 1
    return None


def get_first_index(bin_count, quantile):
    cumsum = np.cumsum(bin_count)
    normalized_cumsum = cumsum / cumsum[-1]
    for i in range(len(normalized_cumsum)):
        if normalized_cumsum[i] > quantile:
            return i
    return None


def show_for_one(
    idx,
    val_dset,
    highsnr_val_dset,
    model,
    calibration_stats,
    mmse_count=5,
    patch_size=256,
    num_samples=2,
    baseline_preds=None,
):
    """
    Given an index, it plots the input, target, reconstructed images and the difference image.
    Note the the difference image is computed with respect to a ground truth image, obtained from the high SNR dataset.
    """
    highsnr_val_dset.set_img_sz(patch_size, 64)
    highsnr_val_dset.disable_noise()
    _, tar_hsnr = highsnr_val_dset[idx]
    inp, tar, recon_img_list = get_predictions(
        idx, val_dset, model, mmse_count=mmse_count, patch_size=patch_size
    )
    plot_crops(
        inp,
        tar,
        tar_hsnr,
        recon_img_list,
        calibration_stats,
        num_samples=num_samples,
        baseline_preds=baseline_preds,
    )


def plot_crops(
    inp,
    tar,
    tar_hsnr,
    recon_img_list,
    calibration_stats,
    num_samples=2,
    baseline_preds=None,
):
    """ """
    if baseline_preds is None:
        baseline_preds = []
    if len(baseline_preds) > 0:
        for i in range(len(baseline_preds)):
            if baseline_preds[i].shape != tar_hsnr.shape:
                print(
                    f"Baseline prediction {i} shape {baseline_preds[i].shape} does not match target shape {tar_hsnr.shape}"
                )
                print("This happens when we want to predict the edges of the image.")
                return

    # color_ch_list = ['goldenrod', 'cyan']
    # color_pred = 'red'
    # insetplot_xmax_value = 10000
    # insetplot_xmin_value = -1000
    # inset_min_labelsize = 10
    # inset_rect = [0.05, 0.05, 0.4, 0.2]

    # Set plot attributes
    img_sz = 3
    ncols = num_samples + len(baseline_preds) + 1 + 1 + 1 + 1 + 1 * (num_samples > 1)
    grid_factor = 5
    grid_img_sz = img_sz * grid_factor
    example_spacing = 1
    c0_extra = 1
    nimgs = 1
    fig_w = ncols * img_sz + 2 * c0_extra / grid_factor
    fig_h = int(img_sz * ncols + (example_spacing * (nimgs - 1)) / grid_factor)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        nrows=int(grid_factor * fig_h),
        ncols=int(grid_factor * fig_w),
        hspace=0.2,
        wspace=0.2,
    )
    params = {"mathtext.default": "regular"}
    plt.rcParams.update(params)

    # plot baselines
    for i in range(2, 2 + len(baseline_preds)):
        for col_idx in range(baseline_preds[0].shape[0]):
            ax_temp = fig.add_subplot(
                gs[
                    col_idx * grid_img_sz : grid_img_sz * (col_idx + 1),
                    i * grid_img_sz + c0_extra : (i + 1) * grid_img_sz + c0_extra,
                ]
            )
            print(tar_hsnr.shape, baseline_preds[i - 2].shape)
            psnr = get_psnr_str(tar_hsnr, baseline_preds[i - 2], col_idx)
            ax_temp.imshow(baseline_preds[i - 2][col_idx], cmap="magma")
            add_psnr_str(ax_temp, psnr)
            clean_ax(ax_temp)

    # plot samples
    sample_start_idx = 2 + len(baseline_preds)
    for i in range(sample_start_idx, ncols - 3):
        for col_idx in range(recon_img_list.shape[1]):
            ax_temp = fig.add_subplot(
                gs[
                    col_idx * grid_img_sz : grid_img_sz * (col_idx + 1),
                    i * grid_img_sz + c0_extra : (i + 1) * grid_img_sz + c0_extra,
                ]
            )
            psnr = get_psnr_str(tar_hsnr, recon_img_list[i - sample_start_idx], col_idx)
            ax_temp.imshow(recon_img_list[i - sample_start_idx][col_idx], cmap="magma")
            add_psnr_str(ax_temp, psnr)
            clean_ax(ax_temp)
            # inset_ax = add_pixel_kde(ax_temp,
            #                       inset_rect,
            #                       [tar_hsnr[col_idx],
            #                        recon_img_list[i - sample_start_idx][col_idx]],
            #                       inset_min_labelsize,
            #                       label_list=['', ''],
            #                       color_list=[color_ch_list[col_idx], color_pred],
            #                       plot_xmax_value=insetplot_xmax_value,
            #                       plot_xmin_value=insetplot_xmin_value)

            # inset_ax.set_xticks([])
            # inset_ax.set_yticks([])

    # difference image
    if num_samples > 1:
        for col_idx in range(recon_img_list.shape[1]):
            ax_temp = fig.add_subplot(
                gs[
                    col_idx * grid_img_sz : grid_img_sz * (col_idx + 1),
                    (ncols - 3) * grid_img_sz
                    + c0_extra : (ncols - 2) * grid_img_sz
                    + c0_extra,
                ]
            )
            ax_temp.imshow(
                recon_img_list[1][col_idx] - recon_img_list[0][col_idx], cmap="coolwarm"
            )
            clean_ax(ax_temp)

    for col_idx in range(recon_img_list.shape[1]):
        # print(recon_img_list.shape)
        ax_temp = fig.add_subplot(
            gs[
                col_idx * grid_img_sz : grid_img_sz * (col_idx + 1),
                c0_extra
                + (ncols - 2) * grid_img_sz : (ncols - 1) * grid_img_sz
                + c0_extra,
            ]
        )
        psnr = get_psnr_str(tar_hsnr, recon_img_list.mean(axis=0), col_idx)
        ax_temp.imshow(recon_img_list.mean(axis=0)[col_idx], cmap="magma")
        add_psnr_str(ax_temp, psnr)
        # inset_ax = add_pixel_kde(ax_temp,
        #                           inset_rect,
        #                           [tar_hsnr[col_idx],
        #                            recon_img_list.mean(axis=0)[col_idx]],
        #                           inset_min_labelsize,
        #                           label_list=['', ''],
        #                           color_list=[color_ch_list[col_idx], color_pred],
        #                           plot_xmax_value=insetplot_xmax_value,
        #                           plot_xmin_value=insetplot_xmin_value)
        # inset_ax.set_xticks([])
        # inset_ax.set_yticks([])

        clean_ax(ax_temp)

        ax_temp = fig.add_subplot(
            gs[
                col_idx * grid_img_sz : grid_img_sz * (col_idx + 1),
                (ncols - 1) * grid_img_sz
                + 2 * c0_extra : (ncols) * grid_img_sz
                + 2 * c0_extra,
            ]
        )
        ax_temp.imshow(tar_hsnr[col_idx], cmap="magma")
        if col_idx == 0:
            legend_ch1_ax = ax_temp
        if col_idx == 1:
            legend_ch2_ax = ax_temp

        # inset_ax = add_pixel_kde(ax_temp,
        #                           inset_rect,
        #                           [tar_hsnr[col_idx],
        #                            ],
        #                           inset_min_labelsize,
        #                           label_list=[''],
        #                           color_list=[color_ch_list[col_idx]],
        #                           plot_xmax_value=insetplot_xmax_value,
        #                           plot_xmin_value=insetplot_xmin_value)
        # inset_ax.set_xticks([])
        # inset_ax.set_yticks([])

        clean_ax(ax_temp)

        ax_temp = fig.add_subplot(
            gs[
                col_idx * grid_img_sz : grid_img_sz * (col_idx + 1),
                grid_img_sz : 2 * grid_img_sz,
            ]
        )
        ax_temp.imshow(tar[0, col_idx].cpu().numpy(), cmap="magma")
        # inset_ax = add_pixel_kde(ax_temp,
        #                           inset_rect,
        #                           [tar[0,col_idx].cpu().numpy(),
        #                            ],
        #                           inset_min_labelsize,
        #                           label_list=[''],
        #                           color_list=[color_ch_list[col_idx]],
        #                           plot_kwargs_list=[{'linestyle':'--'}],
        #                           plot_xmax_value=insetplot_xmax_value,
        #                           plot_xmin_value=insetplot_xmin_value)

        # inset_ax.set_xticks([])
        # inset_ax.set_yticks([])

        clean_ax(ax_temp)

    ax_temp = fig.add_subplot(gs[0:grid_img_sz, 0:grid_img_sz])
    ax_temp.imshow(inp[0, 0].cpu().numpy(), cmap="magma")
    clean_ax(ax_temp)

    # line_ch1 = mlines.Line2D([0, 1], [0, 1], color=color_ch_list[0], linestyle='-', label='$C_1$')
    # line_ch2 = mlines.Line2D([0, 1], [0, 1], color=color_ch_list[1], linestyle='-', label='$C_2$')
    # line_pred = mlines.Line2D([0, 1], [0, 1], color=color_pred, linestyle='-', label='Pred')
    # line_noisych1 = mlines.Line2D([0, 1], [0, 1], color=color_ch_list[0], linestyle='--', label='$C^N_1$')
    # line_noisych2 = mlines.Line2D([0, 1], [0, 1], color=color_ch_list[1], linestyle='--', label='$C^N_2$')
    # legend_ch1 = legend_ch1_ax.legend(handles=[line_ch1, line_noisych1, line_pred], loc='upper right', frameon=False, labelcolor='white',
    #                         prop={'size': 11})
    # legend_ch2 = legend_ch2_ax.legend(handles=[line_ch2, line_noisych2, line_pred], loc='upper right', frameon=False, labelcolor='white',
    #                             prop={'size': 11})

    if calibration_stats is not None:
        smaller_offset = 4
        ax_temp = fig.add_subplot(
            gs[
                grid_img_sz + 1 : 2 * grid_img_sz - smaller_offset + 1,
                smaller_offset - 1 : grid_img_sz - 1,
            ]
        )
        plot_calibration(ax_temp, calibration_stats)


def plot_calibration(ax, calibration_stats):
    """
    To plot calibration statistics (RMV vs RMSE).
    """
    first_idx = get_first_index(calibration_stats[0]["bin_count"], 0.001)
    last_idx = get_last_index(calibration_stats[0]["bin_count"], 0.999)
    ax.plot(
        calibration_stats[0]["rmv"][first_idx:-last_idx],
        calibration_stats[0]["rmse"][first_idx:-last_idx],
        "o",
        label=r"$\hat{C}_0$",
    )

    first_idx = get_first_index(calibration_stats[1]["bin_count"], 0.001)
    last_idx = get_last_index(calibration_stats[1]["bin_count"], 0.999)
    ax.plot(
        calibration_stats[1]["rmv"][first_idx:-last_idx],
        calibration_stats[1]["rmse"][first_idx:-last_idx],
        "o",
        label=r"$\hat{C}_1$",
    )

    ax.set_xlabel("RMV")
    ax.set_ylabel("RMSE")
    ax.legend()


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """
    Adapted from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)
    mid_idx = len(reg_index) // 2
    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)
        a = np.abs(ri - reg_index[mid_idx]) / reg_index[mid_idx]
        # print(a)
        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    matplotlib.colormaps.register(cmap=newcmap, force=True)

    return newcmap


def get_fractional_change(target, prediction, max_val=None):
    """
    Get relative difference between target and prediction.
    """
    if max_val is None:
        max_val = target.max()
    return (target - prediction) / max_val


def get_zero_centered_midval(error):
    """
    When done this way, the midval ensures that the colorbar is centered at 0. (Don't know how, but it works ;))
    """
    vmax = error.max()
    vmin = error.min()
    midval = 1 - vmax / (vmax + abs(vmin))
    return midval


def plot_error(target, prediction, cmap=matplotlib.cm.coolwarm, ax=None, max_val=None):
    """
    Plot the relative difference between target and prediction.
    NOTE: The plot is overlapped to the prediction image (in gray scale).
    NOTE: The colorbar is centered at 0.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Relative difference between target and prediction
    rel_diff = get_fractional_change(target, prediction, max_val=max_val)
    midval = get_zero_centered_midval(rel_diff)
    shifted_cmap = shiftedColorMap(
        cmap, start=0, midpoint=midval, stop=1.0, name="shiftedcmap"
    )
    ax.imshow(prediction, cmap="gray")
    img_err = ax.imshow(rel_diff, cmap=shifted_cmap, alpha=1)
    plt.colorbar(img_err, ax=ax)


# ------------------------------------------------------------------------------------------------


def get_predictions(idx, val_dset, model, mmse_count=50, patch_size=256):
    """
    Given an index and a validation/test set, it returns the input, target and the reconstructed images for that index.
    """
    print(f"Predicting for {idx}")
    val_dset.set_img_sz(patch_size, 64)

    with torch.no_grad():
        # val_dset.enable_noise()
        inp, tar = val_dset[idx]
        # val_dset.disable_noise()

        inp = torch.Tensor(inp[None])
        tar = torch.Tensor(tar[None])
        inp = inp.cuda()
        x_normalized = model.normalize_input(inp)
        tar = tar.cuda()
        tar_normalized = model.normalize_target(tar)

        recon_img_list = []
        for _ in range(mmse_count):
            recon_normalized, td_data = model(x_normalized)
            rec_loss, imgs = model.get_reconstruction_loss(
                recon_normalized,
                x_normalized,
                tar_normalized,
                return_predicted_img=True,
            )
            imgs = model.unnormalize_target(imgs)
            recon_img_list.append(imgs.cpu().numpy()[0])

    recon_img_list = np.array(recon_img_list)
    return inp, tar, recon_img_list


def get_dset_predictions(
    model: VAEModule,
    dset: Dataset,
    batch_size: int,
    loss_type: Literal["musplit", "denoisplit", "denoisplit_musplit"],
    mmse_count: int = 1,
    num_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """Get patch-wise predictions from a model for the entire dataset.

    Parameters
    ----------
    model : VAEModule
        Lightning model used for prediction.
    dset : Dataset
        Dataset to predict on.
    batch_size : int
        Batch size to use for prediction.
    loss_type :
        Type of reconstruction loss used by the model, by default `None`.
    mmse_count : int, optional
        Number of samples to generate for each input and then to average over for
        MMSE estimation, by default 1.
    num_workers : int, optional
        Number of workers to use for DataLoader, by default 4.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float]]
        Tuple containing:
            - predictions: Predicted images for the dataset.
            - predictions_std: Standard deviation of the predicted images.
            - logvar_arr: Log variance of the predicted images.
            - losses: Reconstruction losses for the predictions.
            - psnr: PSNR values for the predictions.
    """
    dloader = DataLoader(
        dset,
        pin_memory=False,
        num_workers=num_workers,
        shuffle=False,
        batch_size=batch_size,
    )

    gauss_likelihood = model.gaussian_likelihood
    nm_likelihood = model.noise_model_likelihood

    predictions = []
    predictions_std = []
    losses = []
    logvar_arr = []
    num_channels = dset[0][1].shape[0]
    patch_psnr_channels = [RunningPSNR() for _ in range(num_channels)]
    with torch.no_grad():
        for batch in tqdm(dloader, desc="Predicting patches"):
            inp, tar = batch
            inp = inp.cuda()
            tar = tar.cuda()

            rec_img_list = []
            for mmse_idx in range(mmse_count):

                # TODO: case of HDN left for future refactoring
                # if model_type == ModelType.Denoiser:
                #     assert model.denoise_channel in [
                #         "Ch1",
                #         "Ch2",
                #         "input",
                #     ], '"all" denoise channel not supported for evaluation. Pick one of "Ch1", "Ch2", "input"'

                #     x_normalized_new, tar_new = model.get_new_input_target(
                #         (inp, tar, *batch[2:])
                #     )
                #     rec, _ = model(x_normalized_new)
                #     rec_loss, imgs = model.get_reconstruction_loss(
                #         rec,
                #         tar,
                #         x_normalized_new,
                #         return_predicted_img=True,
                #     )

                # get model output
                rec, _ = model(inp)

                # get reconstructed img
                if model.model.predict_logvar is None:
                    rec_img = rec
                    logvar = torch.tensor([-1])
                else:
                    rec_img, logvar = torch.chunk(rec, chunks=2, dim=1)
                rec_img_list.append(rec_img.cpu().unsqueeze(0))  # add MMSE dim
                logvar_arr.append(logvar.cpu().numpy())

                # compute reconstruction loss
                if loss_type == "musplit":
                    rec_loss = get_reconstruction_loss(
                        reconstruction=rec, target=tar, likelihood_obj=gauss_likelihood
                    )
                elif loss_type == "denoisplit":
                    rec_loss = get_reconstruction_loss(
                        reconstruction=rec, target=tar, likelihood_obj=nm_likelihood
                    )
                elif loss_type == "denoisplit_musplit":
                    rec_loss = reconstruction_loss_musplit_denoisplit(
                        predictions=rec,
                        targets=tar,
                        gaussian_likelihood=gauss_likelihood,
                        nm_likelihood=nm_likelihood,
                        nm_weight=model.loss_parameters.denoisplit_weight,
                        gaussian_weight=model.loss_parameters.musplit_weight,
                    )
                    rec_loss = {"loss": rec_loss}  # hacky, but ok for now

                # store rec loss values for first pred
                if mmse_idx == 0:
                    try:
                        losses.append(rec_loss["loss"].cpu().numpy())
                    except:
                        losses.append(rec_loss["loss"])

                # update running PSNR
                for i in range(num_channels):
                    patch_psnr_channels[i].update(rec_img[:, i], tar[:, i])

            # aggregate results
            samples = torch.cat(rec_img_list, dim=0)
            mmse_imgs = torch.mean(samples, dim=0)  # avg over MMSE dim
            mmse_std = torch.std(samples, dim=0)
            predictions.append(mmse_imgs.cpu().numpy())
            predictions_std.append(mmse_std.cpu().numpy())

    psnr = [x.get() for x in patch_psnr_channels]
    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(predictions_std, axis=0),
        np.concatenate(logvar_arr),
        np.array(losses),
        psnr,
    )


# ------------------------------------------------------------------------------------------
### Classes and Functions used to stitch predictions
class PatchLocation:
    """
    Encapsulates t_idx and spatial location.
    """

    def __init__(self, h_idx_range, w_idx_range, t_idx):
        self.t = t_idx
        self.h_start, self.h_end = h_idx_range
        self.w_start, self.w_end = w_idx_range

    def __str__(self):
        msg = f"T:{self.t} [{self.h_start}-{self.h_end}) [{self.w_start}-{self.w_end}) "
        return msg


def _get_location(extra_padding, hwt, pred_h, pred_w):
    h_start, w_start, t_idx = hwt
    h_start -= extra_padding
    h_end = h_start + pred_h
    w_start -= extra_padding
    w_end = w_start + pred_w
    return PatchLocation((h_start, h_end), (w_start, w_end), t_idx)


def get_location_from_idx(dset, dset_input_idx, pred_h, pred_w):
    """
    For a given idx of the dataset, it returns where exactly in the dataset, does this prediction lies.
    Note that this prediction also has padded pixels and so a subset of it will be used in the final prediction.
    Which time frame, which spatial location (h_start, h_end, w_start,w_end)
    Args:
        dset:
        dset_input_idx:
        pred_h:
        pred_w:

    Returns
    -------
    """
    extra_padding = dset.per_side_overlap_pixelcount()
    htw = dset.get_idx_manager().hwt_from_idx(
        dset_input_idx, grid_size=dset.get_grid_size()
    )
    return _get_location(extra_padding, htw, pred_h, pred_w)


def remove_pad(pred, loc, extra_padding, smoothening_pixelcount, frame_shape):
    assert smoothening_pixelcount == 0
    if extra_padding - smoothening_pixelcount > 0:
        h_s = extra_padding - smoothening_pixelcount

        # rows
        h_N = frame_shape[0]
        if loc.h_end > h_N:
            assert loc.h_end - extra_padding + smoothening_pixelcount <= h_N
        h_e = extra_padding - smoothening_pixelcount

        w_s = extra_padding - smoothening_pixelcount

        # columns
        w_N = frame_shape[1]
        if loc.w_end > w_N:
            assert loc.w_end - extra_padding + smoothening_pixelcount <= w_N

        w_e = extra_padding - smoothening_pixelcount

        return pred[h_s:-h_e, w_s:-w_e]

    return pred


def update_loc_for_final_insertion(loc, extra_padding, smoothening_pixelcount):
    extra_padding = extra_padding - smoothening_pixelcount
    loc.h_start += extra_padding
    loc.w_start += extra_padding
    loc.h_end -= extra_padding
    loc.w_end -= extra_padding
    return loc


def stitch_predictions(predictions, dset, smoothening_pixelcount=0):
    """
    Args:
        smoothening_pixelcount: number of pixels which can be interpolated
    """
    assert smoothening_pixelcount >= 0 and isinstance(smoothening_pixelcount, int)
    extra_padding = dset.per_side_overlap_pixelcount()
    # if there are more channels, use all of them.
    shape = list(dset.get_data_shape())
    shape[-1] = max(shape[-1], predictions.shape[1])

    output = np.zeros(shape, dtype=predictions.dtype)
    frame_shape = dset.get_data_shape()[1:3]
    for dset_input_idx in range(predictions.shape[0]):
        loc = get_location_from_idx(
            dset, dset_input_idx, predictions.shape[-2], predictions.shape[-1]
        )

        mask = None
        cropped_pred_list = []
        for ch_idx in range(predictions.shape[1]):
            # class i
            cropped_pred_i = remove_pad(
                predictions[dset_input_idx, ch_idx],
                loc,
                extra_padding,
                smoothening_pixelcount,
                frame_shape,
            )

            if mask is None:
                # NOTE: don't need to compute it for every patch.
                assert (
                    smoothening_pixelcount == 0
                ), "For smoothing,enable the get_smoothing_mask. It is disabled since I don't use it and it needs modification to work with non-square images"
                mask = 1
                # mask = _get_smoothing_mask(cropped_pred_i.shape, smoothening_pixelcount, loc, frame_size)

            cropped_pred_list.append(cropped_pred_i)

        loc = update_loc_for_final_insertion(loc, extra_padding, smoothening_pixelcount)
        for ch_idx in range(predictions.shape[1]):
            output[loc.t, loc.h_start : loc.h_end, loc.w_start : loc.w_end, ch_idx] += (
                cropped_pred_list[ch_idx] * mask
            )

    return output


# ------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------
### Classes and Functions used for Calibration
class Calibration:

    def __init__(
        self, num_bins: int = 15, mode: Literal["pixelwise", "patchwise"] = "pixelwise"
    ):
        self._bins = num_bins
        self._bin_boundaries = None
        self._mode = mode
        assert mode in ["pixelwise", "patchwise"]
        self._boundary_mode = "uniform"
        assert self._boundary_mode in ["quantile", "uniform"]
        # self._bin_boundaries = {}

    def logvar_to_std(self, logvar: np.ndarray) -> np.ndarray:
        return np.exp(logvar / 2)

    def compute_bin_boundaries(self, predict_logvar: np.ndarray) -> np.ndarray:
        """
        Compute the bin boundaries for `num_bins` bins and the given logvar values.
        """
        if self._boundary_mode == "quantile":
            boundaries = np.quantile(
                self.logvar_to_std(predict_logvar), np.linspace(0, 1, self._bins + 1)
            )
            return boundaries
        else:
            min_logvar = np.min(predict_logvar)
            max_logvar = np.max(predict_logvar)
            min_std = self.logvar_to_std(min_logvar)
            max_std = self.logvar_to_std(max_logvar)
        return np.linspace(min_std, max_std, self._bins + 1)

    def compute_stats(
        self, pred: np.ndarray, pred_logvar: np.ndarray, target: np.ndarray
    ) -> Dict[int, Dict[str, Union[np.ndarray, List]]]:
        """
        It computes the bin-wise RMSE and RMV for each channel of the predicted image.

        Recall that:
            - RMSE = np.sqrt((pred - target)**2 / num_pixels)
            - RMV = np.sqrt(np.mean(pred_std**2))

        ALGORITHM
        - For each channel:
            - Given the bin boundaries, assign pixels of `std_ch` array to a specific bin index.
            - For each bin index:
                - Compute the RMSE, RMV, and number of pixels for that bin.

        NOTE: each channel of the predicted image/logvar has its own stats.

        Args:
            pred: np.ndarray, shape (n, h, w, c)
            pred_logvar: np.ndarray, shape (n, h, w, c)
            target: np.ndarray, shape (n, h, w, c)
        """
        self._bin_boundaries = {}
        stats = {}
        for ch_idx in range(pred.shape[-1]):
            stats[ch_idx] = {
                "bin_count": [],
                "rmv": [],
                "rmse": [],
                "bin_boundaries": None,
                "bin_matrix": [],
            }
            pred_ch = pred[..., ch_idx]
            logvar_ch = pred_logvar[..., ch_idx]
            std_ch = self.logvar_to_std(logvar_ch)
            target_ch = target[..., ch_idx]
            if self._mode == "pixelwise":
                boundaries = self.compute_bin_boundaries(logvar_ch)
                stats[ch_idx]["bin_boundaries"] = boundaries
                bin_matrix = np.digitize(std_ch.reshape(-1), boundaries)
                bin_matrix = bin_matrix.reshape(std_ch.shape)
                stats[ch_idx]["bin_matrix"] = bin_matrix
                error = (pred_ch - target_ch) ** 2
                for bin_idx in range(self._bins):
                    bin_mask = bin_matrix == bin_idx
                    bin_error = error[bin_mask]
                    bin_size = np.sum(bin_mask)
                    bin_error = (
                        np.sqrt(np.sum(bin_error) / bin_size) if bin_size > 0 else None
                    )  # RMSE
                    bin_var = np.sqrt(np.mean(std_ch[bin_mask] ** 2))  # RMV
                    stats[ch_idx]["rmse"].append(bin_error)
                    stats[ch_idx]["rmv"].append(bin_var)
                    stats[ch_idx]["bin_count"].append(bin_size)
            else:
                raise NotImplementedError("Patchwise mode is not implemented yet.")
        return stats


def nll(x, mean, logvar):
    """
    Log of the probability density of the values x under the Normal
    distribution with parameters mean and logvar.

    :param x: tensor of points, with shape (batch, channels, dim1, dim2)
    :param mean: tensor with mean of distribution, shape
                 (batch, channels, dim1, dim2)
    :param logvar: tensor with log-variance of distribution, shape has to be
                   either scalar or broadcastable
    """
    var = torch.exp(logvar)
    log_prob = -0.5 * (
        ((x - mean) ** 2) / var + logvar + torch.tensor(2 * math.pi).log()
    )
    nll = -log_prob
    return nll


def get_calibrated_factor_for_stdev(
    pred: Union[np.ndarray, torch.Tensor],
    pred_logvar: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    batch_size: int = 32,
    epochs: int = 500,
    lr: float = 0.01,
):
    """
    Here, we calibrate the uncertainty by multiplying the predicted std (mmse estimate or predicted logvar) with a scalar.
    We return the calibrated scalar. This needs to be multiplied with the std.

    NOTE: Why is the input logvar and not std? because the model typically predicts logvar and not std.
    """
    # create a learnable scalar
    scalar = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.Adam([scalar], lr=lr)

    bar = tqdm(range(epochs))
    for _ in bar:
        optimizer.zero_grad()
        # Select a random batch of predictions
        mask = np.random.randint(0, pred.shape[0], batch_size)
        pred_batch = torch.Tensor(pred[mask]).cuda()
        pred_logvar_batch = torch.Tensor(pred_logvar[mask]).cuda()
        target_batch = torch.Tensor(target[mask]).cuda()

        loss = torch.mean(
            nll(target_batch, pred_batch, pred_logvar_batch + torch.log(scalar))
        )
        loss.backward()
        optimizer.step()
        bar.set_description(f"nll: {loss.item()} scalar: {scalar.item()}")

    return np.sqrt(scalar.item())


def plot_calibration(ax, calibration_stats):
    first_idx = get_first_index(calibration_stats[0]["bin_count"], 0.001)
    last_idx = get_last_index(calibration_stats[0]["bin_count"], 0.999)
    ax.plot(
        calibration_stats[0]["rmv"][first_idx:-last_idx],
        calibration_stats[0]["rmse"][first_idx:-last_idx],
        "o",
        label=r"$\hat{C}_0$: Ch1",
    )

    first_idx = get_first_index(calibration_stats[1]["bin_count"], 0.001)
    last_idx = get_last_index(calibration_stats[1]["bin_count"], 0.999)
    ax.plot(
        calibration_stats[1]["rmv"][first_idx:-last_idx],
        calibration_stats[1]["rmse"][first_idx:-last_idx],
        "o",
        label=r"$\hat{C}_1: : Ch2$",
    )

    ax.set_xlabel("RMV")
    ax.set_ylabel("RMSE")
    ax.legend()
