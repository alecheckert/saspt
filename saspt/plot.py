"""
PLOTS TO MAKE:
    - Likelihood-specific functions
        - summarize RBME output on a single set of trajectories
    - Dataset-level functions
        - Show marginal likelihoods by experimental condition
        - Show marginal posterior occupations by experimental condition (heat map)
        - Show marginal posterior occupations by experimental condition (line plot)
            (perhaps enable a "hue" that can overlay sub-conditions, such as 
            molecule concentration)
"""
import os, sys, warnings, numpy as np, pandas as pd, matplotlib, \
    matplotlib.pyplot as plt, matplotlib.gridspec as grd, seaborn as sns
from typing import Tuple, List
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import get_cmap
from scipy.ndimage import gaussian_filter

from .constants import TRACK, FRAME, PY, PX
from .utils import normalize_2d

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'

########################
## PLOTTING UTILITIES ##
########################

def kill_ticks(axes: matplotlib.axes.Axes, spines: bool=False, grid: bool=False):
    """
    Remove the ticks and/or splines and/or grid from a
    matplotlib Axes.
    args
    ----
        axes        :   matplotlib.axes.Axes
        spines      :   bool, also remove spines
        grid        :   boo, also remove the grid
    """
    axes.set_xticks([])
    axes.set_yticks([])
    if spines:
        for s in ['top', 'bottom', 'left', 'right']:
            axes.spines[s].set_visible(False)
    if grid:
        axes.grid(False)

def save_png(out_png: str, dpi: int=600):
    """
    Save a matplotlib.figure.Figure to a PNG.
    args
    ----
        out_png         :   str, out path
        dpi             :   int, resolution
    """
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()

def nanpercentile(arr: np.ndarray, perc: float) -> float:
    """ Thin wrapper on numpy.nanpercentile that returns 1.0 when the
    input is all NaN or empty. """
    if arr.size == 0 or np.isnan(arr).all():
        return 1.0
    else:
        return np.nanpercentile(arr, perc)

def add_log_scale_imshow(axes: matplotlib.axes.Axes, diff_coefs: np.ndarray,
    fontsize: int=None, side: str="x"):
    """
    Add a log axis to a plot produced by matplotlib.axes.Axes.imshow.
    This is specifically used to show the log-values of the diffusion 
    coefficient corresponding to each (linear) x-axis point in some
    of the likelihood plots.
    args
    ----
        axes        :   matplotlib.axes.Axes
        diff_coefs  :   1D ndarray
        fontsize    :   int
        side        :   str, "x" or "y"
    returns
    -------
        None; modifies *axes* directly
    """
    diff_coefs = np.asarray(diff_coefs)
    K = diff_coefs.shape[0]
    d_min = min(diff_coefs)
    d_max = max(diff_coefs)

    # Linear range of the axes
    if side == "x":
        lim = axes.get_xlim()
    elif side == "y":
        lim = axes.get_ylim()

    # xlim = axes.get_xlim()
    lin_span = lim[1] - lim[0]

    # Determine the number of log-10 units (corresponding
    # to major ticks)
    log_diff_coefs = np.log10(diff_coefs)
    first_major_tick = int(log_diff_coefs[0])
    major_tick_values = [first_major_tick]
    c = first_major_tick
    while log_diff_coefs.max() > c:
        c += 1
        major_tick_values.append(c)
    n_major_ticks = len(major_tick_values)

    # Convert between the linear and log scales
    log_span = log_diff_coefs[-1] - log_diff_coefs[0]
    m = lin_span / log_span 
    b = lim[0] - m * log_diff_coefs[0]
    def convert_log_to_lin_coord(log_coord):
        return m * log_coord + b

    # Choose the location of the major ticks
    major_tick_locs = [convert_log_to_lin_coord(coord) \
        for coord in major_tick_values]

    # Major tick labels
    major_tick_labels = ["$10^{%d}$" % int(j) for j in major_tick_values]

    # Minor ticks 
    minor_tick_decile = np.log10(np.arange(1, 11))
    minor_tick_values = []
    for i in range(int(major_tick_values[0])-1, int(major_tick_values[-1])+2):
        minor_tick_values += list(minor_tick_decile + i)
    minor_tick_locs = [convert_log_to_lin_coord(v) for v in minor_tick_values]
    minor_tick_locs = [i for i in minor_tick_locs if ((i >= lim[0]) and (i <= lim[1]))]

    # Set the ticks
    if side == "x":
        axes.set_xticks(major_tick_locs, minor=False)
        axes.set_xticklabels(major_tick_labels, fontsize=fontsize)
        axes.set_xticks(minor_tick_locs, minor=True)
    elif side == "y":
        axes.set_yticks(major_tick_locs, minor=False)
        axes.set_yticklabels(major_tick_labels, fontsize=fontsize)
        axes.set_yticks(minor_tick_locs, minor=True)

###############################
## LIKELIHOOD-SPECIFIC PLOTS ##
###############################

def rbme_posterior_plot(
    out_png: str,
    marginal_likelihood: np.ndarray,
    posterior_occs: np.ndarray,
    diff_coefs: np.ndarray,
    loc_errors: np.ndarray,
    truncate_y_axis: bool=True,
    suptitle: str=None,
):
    """ Visualize the posterior distribution of a state array defined
    with the RBME likelihood function. This likelihood function takes two 
    parameters: diffusion coefficient and localization error. The plot has
    three vertically aligned axes that show the marginal likelihood, mean
    posterior occupations, and marginal posterior occupations respectively.

    args
    ----
        out_png                 :   file to save plot to
        marginal_likelihood     :   2D numpy.ndarray of shape (n_diff_coefs, n_loc_errors)
        posterior_occs          :   2D numpy.ndarray of shape (n_diff_coefs, n_loc_errors)
        diff_coefs              :   1D numpy.ndarray of shape (n_diff_coefs,)
        loc_errors              :   1D numpy.ndarray of shape (n_loc_errors,)
        truncate_y_axis         :   cut off the immobile fraction peak for visibility
        suptitle                :   str, main figure title

    """
    # Catch unsanitary input
    if (not len(diff_coefs.shape) == 1) or (not len(loc_errors.shape) == 1):
        raise ValueError("diff_coefs and loc_errors must be 1-dimensional")
    if not all(map(lambda p: len(p.shape)==2, [marginal_likelihood, posterior_occs])):
        raise ValueError("marginal_likelihood and posterior_occs must be 2-dimensional")
    if not marginal_likelihood.shape == (diff_coefs.shape[0], loc_errors.shape[0]):
        raise ValueError(f"mismatch in shapes of marginal_likelihood " \
            "({marginal_likelihood.shape}), diff_coefs ({diff_coefs.shape}), " \
            "and/or loc_errors ({loc_errors.shape})")
    if not posterior_occs.shape == (diff_coefs.shape[0], loc_errors.shape[0]):
        raise ValueError(f"mismatch in shapes of posterior_occs " \
            "({posterior_occs.shape}), diff_coefs ({diff_coefs.shape}), " \
            "and/or loc_errors ({loc_errors.shape})")

    # Marginalize posterior occupations on localization error
    posterior_occs_marg = posterior_occs.sum(axis=1)
    posterior_occs_marg /= posterior_occs_marg.sum()

    # Plot layout
    fig, ax = plt.subplots(figsize=(6, 5))
    fontsize = 10
    gs = grd.GridSpec(3, 1, height_ratios=(2, 2, 1), width_ratios=None, hspace=0.75)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    ax = [ax0, ax1, ax2]

    # Color map scaling
    vmax0 = np.percentile(marginal_likelihood, 99.5)
    vmax1 = np.percentile(posterior_occs, 99.5)

    p0 = ax[0].imshow(marginal_likelihood.T, cmap="viridis", origin="lower",
        aspect="auto", vmin=0, vmax=vmax0)
    p1 = ax[1].imshow(posterior_occs.T, cmap="viridis", origin="lower",
        aspect="auto", vmin=0, vmax=vmax1)

    # Add log scales for the x-axis
    add_log_scale_imshow(ax[0], diff_coefs, side="x")
    add_log_scale_imshow(ax[1], diff_coefs, side="x")

    # y-ticks for localization error
    n_yticks = 5
    space = loc_errors.shape[0] // n_yticks
    if space > loc_errors.shape[0]:
        space = 1
    try:
        yticks = np.arange(loc_errors.shape[0])[::space]
        yticklabels = ['%.3f' % i for i in loc_errors[::space]]
    except:
        space = 1
        yticks = np.arange(loc_errors.shape[0])[::space]
        yticklabels = ['%.3f' % i for i in loc_errors[::space]]
    for j in range(2):
        ax[j].set_yticks(yticks)
        ax[j].set_yticklabels(yticklabels, fontsize=fontsize)
        ax[j].set_ylabel("Localization error ($\mu$m)", fontsize=fontsize)

    # Show the posterior mean marginalized on localization error
    ax[2].plot(diff_coefs, posterior_occs_marg, color='k')
    ax[2].set_xscale("log")
    ax[2].set_ylabel("Marginal posterior\noccupation", fontsize=fontsize)
    ax[2].set_xlim((0.01, 100.0))
    if truncate_y_axis:
        ax[2].set_ylim((0, posterior_occs_marg[diff_coefs>0.05].max()*2.0))
    else:
        ax[2].set_ylim((0, max(posterior_occs_marg)))
    ax[2].set_xlabel("Diffusion coefficient ($\mu$m$^{2}$ s$^{-1}$)", fontsize=fontsize)

    # Subplot titles
    ax[0].set_title("Naive occupation", fontsize=fontsize)
    ax[1].set_title("Posterior occupation", fontsize=fontsize)

    # Main figure title
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=fontsize)

    # Save, ignoring complaints from a call to matplotlib.pyplot.tight_layout()
    with warnings.catch_warnings(record=False) as w:
        warnings.simplefilter("ignore")
        save_png(out_png, dpi=600)

def rbm_posterior_plot(
    out_png: str,
    marginal_likelihood: np.ndarray,
    posterior_occs: np.ndarray,
    diff_coefs: np.ndarray,
    truncate_y_axis: bool=True,
    fontsize: int=8,
    suptitle: str=None,
):
    """ Plot a 1-dimensional posterior distribution over a grid of 
    logarithmically-spaced diffusion coefficients.

    args
    ----
        out_png                 :   file to save plot to
        marginal_likelihood     :   1D numpy.ndarray of shape
                                    (n_diff_coefs), likelihood function
                                    marginalized over trajectory-state
                                    assignments
        posterior_occs          :   1D numpy.ndarray of shape 
                                    (n_diff_coefs), posterior mean state
                                    occupations
        diff_coefs              :   1D numpy.ndarray of shape 
                                    (n_diff_coefs), diffusion coefficient
                                    for each state in µm2/sec
        truncate_y_axis         :   bool, cut off the immobile fraction
                                    peak if it's too large
        fontsize                :   int
        suptitle                :   str, a title for the plot
    """
    if not all([
        posterior_occs.shape[0] == diff_coefs.shape[0],
        marginal_likelihood.shape[0] == diff_coefs.shape[0]
    ]):
        raise ValueError(f"shape mismatch: {posterior_occs.shape}, " \
            "{marginal_likelihood.shape}, {diff_coefs.shape}")
    posterior_occs = posterior_occs.copy() / np.nansum(posterior_occs)
    fig, ax = plt.subplots(2, 1, figsize=(5, 2.5), sharex=True)
    ax[0].plot(diff_coefs, marginal_likelihood, color='k')
    ax[1].plot(diff_coefs, posterior_occs, color='k')
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Diffusion coefficient ($\mu$m$^{2}$ s$^{-1}$)", fontsize=fontsize)
    ax[0].set_ylabel("Naive\noccupation", fontsize=fontsize)
    ax[1].set_ylabel("Posterior\noccupation", fontsize=fontsize)
    for j in range(2):
        ax[j].set_ylim((0, ax[j].get_ylim()[1]))
        ax[j].yaxis.set_tick_params(labelsize=fontsize)
        ax[j].xaxis.set_tick_params(labelsize=fontsize)
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=fontsize)
    if truncate_y_axis:
        ax[1].set_ylim((0, np.nanmax(posterior_occs[diff_coefs>0.05])*2.0))
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.close()

#######################################
## TRAJECTORY-STATE ASSIGNMENT PLOTS ##
#######################################

def marginal_assignment_probability_plot(
    out_png: str, 
    assignment_probabilities: np.ndarray, 
    assignment_likelihoods: np.ndarray,
    diff_coefs: np.ndarray,
    subplot_extent: Tuple[int]=(0, 6, 0, 1.5),
    fontsize: int=10,
    sort_by_mean_diff_coef: bool=False,
):
    """ Visualize the trajectory-state assignment likelihoods and
    probabilities. The upper subplot shows likelihoods and the lower
    subplot shows probabilities (for instance, the posterior assignment
    probability from a state array run).

    In each subplot, trajectories correspond to columns of a heat map while 
    diffusive states correspond to rows.

    args
    ----
        out_png                 :   output file
        assignment_probabilities:   2D numpy.ndarray of shape (n_diff_coefs,
                                    n_tracks)
        assignment_likelihoods  :   2D numpy.ndarray of shape (n_diff_coefs,
                                    n_tracks)
        diff_coefs              :   1D numpy.ndarray of shape (n_diff_coefs,)
                                    diffusion coefficient in µm2/sec
        subplot_extent          :   4-tuple of int, shape of subplots
        fontsize                :   int
        sort_by_mean_diff_coef  :   bool, plot each trajectory in order of 
                                    increasing mean diffusion coefficient

    """
    figsize = (subplot_extent[1], subplot_extent[3]*2)
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=True)

    # Normalize
    if assignment_probabilities.size > 0:
        assignment_probabilities = assignment_probabilities.copy() / np.nansum(assignment_probabilities, axis=0)

    # Sort values by mean posterior diffusion coefficient
    if sort_by_mean_diff_coef:
        indices = np.argsort((assignment_probabilities.T * diff_coefs).sum(axis=1))
        assignment_probabilities = assignment_probabilities[:,indices]
        assignment_likelihoods = assignment_likelihoods[:,indices]

    # Main plot
    f0 = ax[0].imshow(assignment_likelihoods, origin="lower", cmap='viridis', 
        extent=subplot_extent, vmax=nanpercentile(assignment_likelihoods, 99))
    f1 = ax[1].imshow(assignment_probabilities, origin="lower", cmap='viridis',
        extent=subplot_extent, vmin=0, vmax=nanpercentile(assignment_probabilities, 99))

    # Color scale
    cbar = plt.colorbar(f0, ax=ax[0], shrink=0.75)
    cbar.set_label("Naive probability", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar = plt.colorbar(f1, ax=ax[1], shrink=0.75)
    cbar.set_label("Posterior probability", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    # Ticks and axis labels
    ax[1].set_xticks([])
    ax[1].set_xlabel("Trajectory", fontsize=fontsize)
    add_log_scale_imshow(ax[1], diff_coefs, fontsize=None, side='y')
    for j in range(2):
        ax[j].set_ylabel("Diffusion\ncoefficient\n($\mu$m$^{2}$ s$^{-1}$)",
            fontsize=fontsize)

    # Save
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.close()

def temporal_assignment_probability_plot(
    out_png: str,
    tracks: pd.DataFrame,
    assignment_probabilities: np.ndarray,
    assignment_likelihoods: np.ndarray,
    diff_coefs: np.ndarray,
    subplot_extent: Tuple[int]=(0, 6, 0, 1.5),
    fontsize: int=10,
    tick_fontsize: int=8,
    frame_block_size: int=10,
    normalize: bool=False,
    suptitle: str=None,
):
    """ Plot the trajectory-state assignment probabilities vs. frame index.
    Useful for visualizing density-dependent effects on the assignment 
    probabilities.

    args
    ----
        out_png                 :   output filename
        tracks                  :   pandas.DataFrame, detections
        assignment_probabilities:   2D numpy.ndarray of shape (n_diff_coefs, n_tracks)
        assignment_likelihoods  :   2D numpy.ndarray of shape (n_diff_coefs, n_tracks)
        diff_coefs              :   1D numpy.ndarray of shape (n_diff_coefs,)
        subplot_extent          :   4-tuple of int
        fontsize                :   int
        frame_block_size        :   bin size for aggregating assignment
                                    probabilities/likelihoods
        normalize               :   normalize within each frame block
        suptitle                :   main figure title

    """
    if not all([
        len(assignment_probabilities.shape) == 2,
        len(assignment_likelihoods.shape) == 2,
        len(diff_coefs.shape) == 1,
        assignment_probabilities.shape[0] == diff_coefs.shape[0],
        assignment_likelihoods.shape[0] == diff_coefs.shape[0],
    ]):
        raise ValueError("incompatible shapes for temporal " \
            "assignment probability plot")

    # Number of distinct diffusion coefficients, frames, and frame blocks
    n_diff_coefs = diff_coefs.shape[0]
    n_frames = np.nanmax(tracks[FRAME]) + 1 if len(tracks) > 0 else 1
    n_blocks = (n_frames + frame_block_size - 1) // frame_block_size

    # Index of each trajectory
    track_indices = np.arange(assignment_probabilities.shape[1])

    # Only include track indices represented in this set of trajectories
    if not np.isin(track_indices, tracks[TRACK].unique()).all():
        raise ValueError(f"not all trajectory indices are present in this set of detections")

    # Start frame and stop frame for each trajectory
    start_frames = np.asarray(tracks.groupby(TRACK)[FRAME].first().loc[track_indices])
    stop_frames = np.asarray(tracks.groupby(TRACK)[FRAME].last().loc[track_indices])

    # Posterior probability and likelihood aggregated by frame
    P = np.full((n_diff_coefs, n_blocks), np.nan, dtype=np.float64)
    L = np.full((n_diff_coefs, n_blocks), np.nan, dtype=np.float64)

    for b in range(n_blocks):
        in_block = np.logical_and(
            start_frames<(b+1)*frame_block_size,
            stop_frames>=b*frame_block_size
        )
        if in_block.any():
            P[:,b] = assignment_probabilities[:,in_block].sum(axis=1)
            L[:,b] = assignment_likelihoods[:,in_block].sum(axis=1)

    # Detections per frame
    tracks["frame_block"] = tracks[FRAME] // frame_block_size
    detections_per_frame = np.asarray(
        pd.Series(np.arange(n_blocks)).map(
            tracks.groupby("frame_block").size()
        )
    ) / frame_block_size
    tracks = tracks.drop("frame_block", axis=1)

    # Normalize for each block
    if normalize:
        def norm(x):
            s = x.sum(axis=0)
            nonzero = s > 0
            x[:,nonzero] = x[:,nonzero] / s[nonzero]
            return x 
        P = norm(P)
        L = norm(L)

    # Plot layout
    subplot_extent = (0, 6, 0, 1.5)
    figsize = (subplot_extent[1], subplot_extent[3]*3)
    fig, ax = plt.subplots(3, 1, figsize=figsize)

    # Main plot
    f0 = ax[0].imshow(L, origin="lower", cmap='viridis', extent=subplot_extent,
        vmin=0, vmax=nanpercentile(L, 99))
    f1 = ax[1].imshow(P, origin="lower", cmap='viridis', extent=subplot_extent,
        vmin=0, vmax=nanpercentile(P, 99))

    # Detections per frame
    ax[2].plot(np.arange(n_blocks)*frame_block_size, detections_per_frame,
        color='k')
    ax[2].set_ylabel("Detections\nper frame", fontsize=fontsize)
    ax[2].set_xlabel("Frame", fontsize=fontsize)
    ax[2].set_ylim((0, ax[2].get_ylim()[1]))
    ax[2].yaxis.set_tick_params(labelsize=tick_fontsize)
    ax[2].xaxis.set_tick_params(labelsize=tick_fontsize)

    # Color scale
    cbar = plt.colorbar(f0, ax=ax[0], shrink=0.75)
    if normalize:
        cbar.set_label("Naive\nprobability", fontsize=fontsize)
    else:
        cbar.set_label("Unnormalized\nnaive\nprobability", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    cbar = plt.colorbar(f1, ax=ax[1], shrink=0.75)
    if normalize:
        cbar.set_label("Posterior\nprobability", fontsize=fontsize)
    else:
        cbar.set_label("Unnormalized\nposterior\nprobability", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    # Ticks and axis labels
    xticks = np.arange(n_blocks)
    while xticks.shape[0] > 10:
        xticks = xticks[0::2]
    xticklabels = [f"{x*frame_block_size}" for x in xticks]
    for j in range(2):
        ax[j].set_xticks(xticks * subplot_extent[1] / n_blocks)
        ax[j].set_xticklabels(xticklabels, fontsize=tick_fontsize)
        ax[j].set_xlabel("Frame", fontsize=fontsize)       
        add_log_scale_imshow(ax[j], diff_coefs, fontsize=tick_fontsize, side='y')
        ax[j].set_ylabel("Diffusion\ncoefficient\n($\mu$m$^{2}$ s$^{-1}$)",
            fontsize=fontsize)

    # Main figure title
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=fontsize)

    # Save
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.close()

def spatial_assignment_probability_plot(
    out_png: str,
    tracks: pd.DataFrame,
    assignment_probabilities: np.ndarray,
    assignment_likelihoods: np.ndarray,
    diff_coefs: np.ndarray,
    pixel_size_um: float,
    fontsize: int=10,
    spatial_bin_size: int=2,
    smooth_kernel: float=0.0,
    suptitle: str=None,
):
    """ Experimental plot. Show the prior and posterior mean diffusion
    coefficients as a function of space. 

    Produces a 3-axis plot. The first axis shows detection density, the 
    second shows the mean prior diffusion coefficient, and the third shows
    the mean posterior diffusion coefficient.

    args
    ----
        out_png                 :   output filename
        tracks                  :   pandas.DataFrame, detections
        assignment_probabilities:   2D numpy.ndarray of shape (n_diff_coefs, n_tracks)
        assignment_likelihoods  :   2D numpy.ndarray of shape (n_diff_coefs, n_tracks)
        diff_coefs              :   1D numpy.ndarray of shape (n_diff_coefs,)
        pixel_size_um           :   float, camera pixel size in microns
        fontsize                :   int
        spatial_bin_size        :   int, number of camera pixels per spatial bin
        smooth_kernel           :   float, smoothing kernel in microns
        suptitle                :   str, main figure title if desired
    """
    # Estimate the size of the FOV
    if len(tracks) > 0:
        fov_size = (
            int(np.ceil(tracks[PY].max())) + 1,
            int(np.ceil(tracks[PX].max())) + 1
        )
    else:
        fov_size = (1, 1)

    # Number of distinct diffusion coefficients
    n_dc = diff_coefs.shape[0]

    # Number of trajectories
    n_tracks = assignment_probabilities.shape[1]
    track_indices = np.arange(n_tracks).astype(np.int64)
    tracks = tracks[tracks[TRACK].isin(track_indices)].reset_index(drop=True)

    # Spatial bins
    ybins = np.arange(0, fov_size[0]+spatial_bin_size, spatial_bin_size)
    xbins = np.arange(0, fov_size[1]+spatial_bin_size, spatial_bin_size)

    # Scale smoothing kernel into bin units
    smooth_kernel = smooth_kernel / (pixel_size_um * spatial_bin_size)

    # Bin area in microns
    bin_area = (spatial_bin_size * pixel_size_um)**2

    # Number of detections in each bin
    H = np.histogram2d(tracks[PY], tracks[PX], bins=(xbins, ybins))[0].astype(np.float64)
    H = gaussian_filter(H, smooth_kernel)
    occupied = H > 0

    # Normalize over states for each trajectory
    assignment_probabilities = normalize_2d(assignment_probabilities, 0)
    assignment_likelihoods = normalize_2d(assignment_likelihoods, 0)

    # Mean prior diffusion coefficient in each spatial bin
    mean_diff_coef = pd.Series((assignment_likelihoods.T * diff_coefs).sum(axis=1),
        index=track_indices, name="mean_prior_diff_coef")
    prior_mean = np.histogram2d(tracks[PY], tracks[PX], bins=(xbins, ybins),
        weights=tracks[TRACK].map(mean_diff_coef))[0].astype(np.float64)
    prior_mean = gaussian_filter(prior_mean, smooth_kernel)
    prior_mean[occupied] = prior_mean[occupied] / H[occupied]

    # Mean posterior diffusion coefficient in each spatial bin
    mean_diff_coef = pd.Series((assignment_probabilities.T * diff_coefs).sum(axis=1),
        index=track_indices, name="mean_posterior_diff_coef")
    post_mean = np.histogram2d(tracks[PY], tracks[PX], bins=(xbins, ybins),
        weights=tracks[TRACK].map(mean_diff_coef))[0].astype(np.float64)
    post_mean = gaussian_filter(post_mean, smooth_kernel)
    post_mean[occupied] = post_mean[occupied] / H[occupied]

    # Get localization density
    loc_density = H / bin_area 

    vmin = 0
    vmax = np.mean([np.nanpercentile(post_mean, 99), np.nanpercentile(prior_mean, 99)])

    # Main plot
    fig, ax = plt.subplots(1, 3, figsize=(3*3, 1*3), sharex=True, sharey=True)
    f0 = ax[0].imshow(loc_density, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(f0, ax=ax[0], fraction=0.046, pad=0.04)
    cbar.set_label("Detections / $\mu$m$^{2}$", fontsize=fontsize)

    f1 = ax[1].imshow(prior_mean, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(f1, ax=ax[1], fraction=0.046, pad=0.04)
    cbar.set_label("Naive mean diff. coef. ($\mu$m$^{2}$ s$^{-1}$)", fontsize=fontsize)

    f2 = ax[2].imshow(post_mean, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(f2, ax=ax[2], fraction=0.046, pad=0.04)
    cbar.set_label("Posterior mean diff. coef. ($\mu$m$^{2}$ s$^{-1}$)", fontsize=fontsize)

    # Formatting
    for j in range(3):
        kill_ticks(ax[j], spines=False)

    # Main figure title
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.close()

#########################
## DATASET-LEVEL PLOTS ##
#########################

def heatmap_by_condition(
    out_png: str,
    diff_coefs: np.ndarray,
    dists: dict,
    order: List[str]=[],
    cmap: str="viridis",
    vmin: float=0,
    vmax: float=None,
    subplot_extent: Tuple[int]=(0, 5, 0, 1.5),
    ylabel: str=None,
    cbar_label: str=None,
    suptitle: str=None,
    fontsize: int=8,
    normalize: bool=False,
    order_by_size: bool=False,
):
    """ Plot groups of one-dimensional signals corresponding to different
    experimental conditions as a heat map. The signals are assumed to be
    defined on a logarithmically-spaced support of diffusion coefficients.

    args
    ----
        out_png         :   output file
        diff_coefs      :   1D numpy.ndarray of shape (n_diff_coefs,)
        dists           :   dict keyed by condition name to 2D numpy.ndarray.
                            Each array has shape (n_files, n_diff_coefs),
                            which gives each of the 1D signals corresponding
                            to that condition.
        order           :   list of str, order in which to plot the
                            different experimental conditions
        cmap            :   color map name
        vmin            :   color LUT min
        vmax            :   color LUT max
        subplot_extent  :   dimensions of individual subplots
        ylabel          :   y-axis label   
        cbar_label      :   color bar label
        suptitle        :   main figure title
        normalize       :   normalize each signal for unit intensity
        order_by_size   :   order the signals corresponding to each condition
                            by summed intensity (before normalization)
    """
    # Check compatibility of arguments
    if not all([
        isinstance(dists, dict),
        isinstance(diff_coefs, np.ndarray),
        len(diff_coefs.shape) == 1,
        all(map(lambda a: isinstance(a, np.ndarray), dists.values())),
        all(map(lambda a: a.shape[1] == diff_coefs.shape[0], dists.values()))
    ]):
        raise ValueError("incompatible arguments")

    if len(order) == 0:
        order = list(dists.keys())
        order.sort()
    elif not all(map(lambda c: c in dists.keys(), order)):
        raise ValueError(f"all elements of *order* must be keys in *dists*")

    # Number of conditions
    n_conditions = len(dists)

    # Number of files
    n_files = sum(map(lambda v: v.shape[0], dists.values()))

    # Normalize and order by increasing density, if relevant
    if order_by_size:
        for k in dists.keys():
            i = np.argsort(dists[k].sum(axis=1))
            dists[k] = dists[k][i[::-1],:]
    if normalize:
        for k in dists.keys():
            dists[k] = normalize_2d(dists[k], 1)

    # Color map scaling
    if vmin is None:
        vmin = vmin(map(np.nanmin, dists.values()))
    if vmax is None:
        vmax = nanpercentile(np.concatenate(list(dists.values()), axis=0), 99)

    # Plot layout. We favor stacking conditions vertically rather than
    # horizontally, but otherwise attempt to keep the subplots as square as
    # possible
    ny = int(np.ceil(np.sqrt(n_conditions)))
    nx = int(np.ceil(n_conditions / ny))

    fig, axes = plt.subplots(ny, nx, sharex=True, 
        figsize=(nx*subplot_extent[1], ny*subplot_extent[3]))
    axes = np.asarray(axes).reshape((ny, nx))
    axes = axes.ravel()

    def make_subplot(i: int):
        ax = axes[i]
        name = order[i]
        f = ax.imshow(dists[name], cmap=cmap, vmin=vmin, vmax=vmax, 
            extent=subplot_extent)
        cbar = plt.colorbar(f, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([0])
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_label(cbar_label, fontsize=fontsize)
        add_log_scale_imshow(ax, diff_coefs, fontsize=fontsize, side='x')
        if name != "no_condition":
            ax.set_title(name, fontsize=fontsize)
        ax.set_yticks([])
        if i // nx == ny - 1:
            ax.set_xlabel("Diff. coef. ($\mu$m$^{2}$ s$^{-1}$)", fontsize=fontsize)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=fontsize)

    for i in range(n_conditions):
        make_subplot(i)

    # Empty subplots
    for i in range(n_conditions, ny*nx):
        f = axes[i].imshow(np.full(list(dists.values())[1].shape, np.nan), cmap=cmap,
            vmin=vmin, vmax=vmax, extent=subplot_extent)
        cbar = plt.colorbar(f, ax=axes[i], fraction=0.046, pad=0.04)
        cbar.set_ticks([0])
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_label(cbar_label, fontsize=fontsize)
        add_log_scale_imshow(axes[i], diff_coefs, fontsize=fontsize, side='x')
        axes[i].set_yticks([])
        if i // nx == ny - 1:
            axes[i].set_xlabel("Diff. coef. ($\mu$m$^{2}$ s$^{-1}$)", fontsize=fontsize)
        for s in ['top', 'left', 'right']:
            axes[i].spines[s].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.close()

def lineplot_by_condition(
    out_png: str,
    diff_coefs: np.ndarray,
    dists: dict,
    order: List[str]=[],
    normalize: bool=True,
    fontsize: int=10,
    ylabel: str=None,
    alpha: float=0.5,
):
    # Check compatibility of arguments
    if not all([
        isinstance(dists, dict),
        isinstance(diff_coefs, np.ndarray),
        len(diff_coefs.shape) == 1,
        all(map(lambda a: isinstance(a, np.ndarray), dists.values())),
        all(map(lambda a: a.shape[1] == diff_coefs.shape[0], dists.values()))
    ]):
        raise ValueError("incompatible arguments")

    if len(order) == 0:
        order = list(dists.keys())
        order.sort()
    elif not all(map(lambda c: c in dists.keys(), order)):
        raise ValueError(f"all elements of *order* must be keys in *dists*")

    # Number of conditions
    n_conditions = len(dists)

    # Number of files
    n_files = sum(map(lambda v: v.shape[0], dists.values()))

    # Normalize
    if normalize:
        for k in dists.keys():
            dists[k] = normalize_2d(dists[k], axis=1)

    # Plot layout. We favor stacking conditions vertically rather than
    # horizontally, but otherwise attempt to keep the subplots as square as
    # possible
    ny = int(np.ceil(np.sqrt(n_conditions)))
    nx = int(np.ceil(n_conditions / ny))

    fig, axes = plt.subplots(ny, nx, sharex=True, sharey=True, figsize=(nx*4, ny*1.5))
    colors = sns.color_palette("dark", n_conditions)
    axes = np.asarray(axes).reshape((ny, nx))
    axes = axes.ravel()

    def make_subplot(i: int):
        ax = axes[i]
        name = order[i]
        for j in range(dists[name].shape[0]):
            ax.plot(diff_coefs, dists[name][j,:], color=colors[i], label=None, alpha=alpha)
        ax.plot([], [], color=colors[i], alpha=alpha, label="File")
        ax.plot(diff_coefs, np.nanmean(dists[name], axis=0), color='k', linestyle='--', 
            label="Mean")

        ax.legend(frameon=False, loc='upper right', prop={'size': 6})
        ax.set_title(name, fontsize=fontsize)
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)

        # y-ticks
        ax.set_yticks([])
        if (ylabel is not None) and (i % nx == 0):
            ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_ylim((0, ax.get_ylim()[1]))

        # x-ticks
        ax.set_xscale("log")
        if i // nx == ny - 1:
            ax.set_xlabel("Diff. coef. ($\mu$m$^{2}$ s$^{-1}$)", fontsize=fontsize)

    for i in range(n_conditions):
        make_subplot(i)

    for i in range(n_conditions, ny*nx):
        for s in ['top', 'left', 'right']:
            axes[i].spines[s].set_visible(False)
        if i // nx == ny - 1:
            ax.set_xlabel("Diff. coef. ($\mu$m$^{2}$ s$^{-1}$)", fontsize=fontsize)

    # Save
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.close()



