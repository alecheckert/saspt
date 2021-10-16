#!/usr/bin/env python
"""
defoc.py -- evaluate defocalization probabilities for various diffusion models
"""
import os 
import numpy as np 
import pandas as pd 

# Caching
from functools import lru_cache

# Cubic spline interpolation, for FBM defocalization function
from scipy import interpolate 

# Names of likelihood function
from .constants import RBME, GAMMA, RBME_MARGINAL, FBME

# saspt module directory
PACKAGE_DIR = os.path.split(os.path.abspath(__file__))[0]

# Directory with spline coefficients for FBM defocalization
SPLINE_DIR = os.path.join(PACKAGE_DIR, "splines")


###############
## UTILITIES ##
###############


@lru_cache(maxsize=1)
def load_fbm_defoc_spline(dz=0.7):
    """
    Given a focal depth, get a spline interpolator that enables calculation
    of the fraction of FBMs that defocalize at various frame intervals.
    args
    ----
        dz      :   float, the focal depth in um
    returns
    -------
        5-tuple, the *tck* argument expected by scipy.interpolate's spline
            evaluators -- specifically scipy.interpolate.bisplev
    """
    # Available frame intervals
    avail_dz = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.5, 1.6])

    # Get the closest available focal depth
    m = np.argmin(np.abs(avail_dz - dz))
    sel_dz = avail_dz[m]

    # Path to this file
    path = os.path.join(SPLINE_DIR, "fbm_defoc_splines_dz-%.1f.csv" % sel_dz)

    # Load the spline coefficients
    tcks = load_spline_coefs_multiple_frame_interval(path)
    return tcks


def load_spline_coefs_multiple_frame_interval(path):
    """
    Load multiple sets of bivariate spline coefficients from a file.
    These are in the format required by scipy.interpolate for 
    evaluation of bivariate splines.
    The individual sets of spline coefficients are ;-delimited, while
    the different parts of the coefficient 5-tuple are newline-delimited and
    the individual numbers are ,-delimited.
    args
    ----
        path        :   str, path to a file of the type written by
                        save_spline_coefs_multiple()
    returns
    -------
        list of 5-tuple, the bivariate spline coefficients for each
            frame interval
    """
    with open(path, "r") as f:
        S = f.read().split(";")
    S = [j.split("\n") for j in S]
    result = []
    for lines in S:
        x = np.asarray([float(j) for j in lines[0].split(",")])
        y = np.asarray([float(j) for j in lines[1].split(",")])
        coefs = np.array([float(j) for j in lines[2].split(",")])
        kx = int(lines[3])
        ky = int(lines[4])
        result.append((x, y, coefs, kx, ky))
    return result 


def eval_spline(x, y, tck):
    """
    Evaluate a bivariate spline on the Cartesian product of a set of X points
    and a set of Y points.
    args
    ----
        x       :   1D ndarray, the array of unique x points
        y       :   1D ndarray, the array of unique y points
        tck     :   5-tuple, bivariate spline coefficients of the type
                    read by *load_spline_coefs*
    returns
    -------
        2D ndarray of shape (y.shape[0], x.shape[0]), the evaluated
            bivariate spline at each combination of the input points
    """
    return interpolate.bisplev(x, y, tck).T 


def f_remain_rbm(D, n_frames, frame_interval, dz):
    """
    Calculate the fraction of regular Brownian particles that 
    remain in a microscope's depth of field after some number 
    of frames.
    args
    ----
        D               :   float, diffusion coefficient
                            in um^2 s^-1
        n_frames        :   int, the number of frames
        frame_interval  :   float, seconds
        dz              :   float, depth of field in um
    returns
    -------
        1D ndarray of shape (n_frames,), the probability
            to remain at each frame interval
    """
    if (dz is np.inf) or (dz is None) or (D <= 0.0):
        return np.ones(n_frames, dtype=np.float64)

    # Support for the calculations
    s = (int(dz//2.0)+1) * 2
    support = np.linspace(-s, s, int(((2*s)//0.001)+2))[:-1]
    hz = 0.5 * dz 
    inside = np.abs(support) <= hz 
    outside = ~inside 

    # Define the transfer function for this BM
    g = np.exp(-(support ** 2)/ (4 * D * frame_interval))
    g /= g.sum()
    g_rft = np.fft.rfft(g)   

    # Set up the initial probability density
    pmf = inside.astype(np.float64)
    pmf /= pmf.sum()

    # Propagate over subsequent frame intervals
    result = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        pmf = np.fft.fftshift(np.fft.irfft(
            np.fft.rfft(pmf) * g_rft, n=pmf.shape[0]))
        pmf[outside] = 0.0
        result[t] = pmf.sum()

    return result 


def f_remain_fbm(D, hurst, n_frames, frame_interval, dz, D_type=4):
    """
    Calculate the fraction of fractional Brownian particles that 
    remain in a microscope's depth of field after some number of 
    frames.
    args
    ----
        D               :   float, diffusion coefficient
                            in um^2 s^-1
        hurst           :   float, Hurst parameter
        n_frames        :   int, the number of frames
        frame_interval  :   float, seconds
        dz              :   float, depth of field in um
    returns
    -------
        1D ndarray of shape (n_frames,), the probability
            to remain at each frame interval
    """
    if (dz is np.inf) or (dz is None) or (D <= 0.0):
        return np.ones(n_frames, dtype=np.float64)

    if n_frames > 8:
        raise RuntimeError("no more than 8 frame intervals supported " \
            "for FBM defocalization")

    # Get the dispersion parameter
    if D_type == 1:
        c = np.log(D * np.power(frame_interval, 2*hurst))
    elif D_type == 2:
        c = 2 * hurst * np.log(D * frame_interval)
    elif D_type == 3:
        c = np.log(D * frame_interval / (2 * hurst))
    elif D_type == 4:
        c = np.log(D * frame_interval)

    # Load spline coefficients for this focal depth
    tcks = load_fbm_defoc_spline(dz=dz)

    # Evaluate the probability of defocalization at each frame interval
    return np.asarray([eval_spline(hurst, c, tck) for tck in tcks[:n_frames]])


##############################################
## LIKELIHOOD-SPECIFIC CORRECTION FUNCTIONS ##
##############################################


def defoc_corr_rbm(L, diff_coefs, mode="point", frame_interval=0.00748, dz=0.7,
    normalize=False):
    """
    Apply a defocalization correction to the regular Brownian motion
    likelihood function.
    args
    ----
        L               :   ndarray, the likelihood function as generated
                            by *eval_likelihood*
        diff_coefs      :   1D ndarray, the set of diffusion coefficients
                            corresponding to *L*. The second axis of *L*
                            is assumed to corresponding to the values of 
                            *diff_coefs*.
        mode            :   str, the mode the likelihood was calculated in,
                            either "point" or "binned"
        frame_interval  :   float, frame interval in seconds
        dz              :   float, focal depth in microns
    returns
    -------
        reference to *L* after correction
    """
    diff_coefs = np.asarray(diff_coefs)
    if mode == "point":
        K = diff_coefs.shape[0]

        # For each diffusion coefficient, evaluate the defocalization
        # probability at one frame interval
        frac_remain = np.ones(K, dtype=np.float64)
        if not np.isinf(dz):
            for i, D in enumerate(diff_coefs):
                frac_remain[i] = f_remain_rbm(D, 1, frame_interval, dz)[0]       

    elif mode == "binned":
        K = diff_coefs.shape[0] - 1
        d_mid = np.sqrt(diff_coefs[1:] * diff_coefs[:-1])

        # For each diffusion coefficient, evaluate the defocalization
        # probability at one frame interval
        frac_remain = np.ones(K, dtype=np.float64)
        if not np.isinf(dz):
            for i, D in enumerate(d_mid):
                frac_remain[i] = f_remain_rbm(D, 1, frame_interval, dz)[0]

    # Passed a single aggregate likelihood function
    if len(L.shape) == 1:
        assert L.shape[0] == K, "first axis of aggregate likelihood must " \
            "correspond to the diffusion coefficient"

        # Apply the correction and renormalize
        L /= frac_remain
        if normalize:
            norm = L.sum()
            if norm > 0:
                L /= norm 

    else:
        assert L.shape[0] == K, "first axis of likelihood matrix must " \
            "correspond to the diffusion coefficient"

        # Apply the correction and renormalize
        if len(L.shape) == 2:
            L = (L.T / frac_remain).T
            if normalize: L = L / L.sum(axis=0)

        elif len(L.shape) == 3:
            L = (L.T / frac_remain).T
            if normalize: L = L / L.sum(axis=(0,1))

    return L 


def defoc_corr_rbme(L, diff_coefs, loc_errors, frame_interval=0.00748, dz=0.7,
    normalize=False):
    """
    Apply a defocalization correction to the regular Brownian motion with
    error (RBME) likelihood function. Localization error is not assumed
    to figure into defocalization, so the *loc_errors* argument is ignored.
    args
    ----
        L               :   ndarray, the likelihood function as generated
                            by *eval_likelihood*
        diff_coefs      :   1D ndarray, the set of diffusion coefficients
                            corresponding to *L*. The second axis of *L*
                            is assumed to corresponding to the values of 
                            *diff_coefs*.
        loc_errors      :   1D ndarray, localization error parameters
        frame_interval  :   float, frame interval in seconds
        dz              :   float, focal depth in microns
    returns
    -------
        reference to *L* after correction
    """
    diff_coefs = np.asarray(diff_coefs)
    K = diff_coefs.shape[0]

    # For each diffusion coefficient, evaluate the defocalization
    # probability at one frame interval
    frac_remain = np.ones(K, dtype=np.float64)
    if not np.isinf(dz):
        for i, D in enumerate(diff_coefs):
            frac_remain[i] = f_remain_rbm(D, 1, frame_interval, dz)[0]

    # Passed a single aggregate likelihood function
    if len(L.shape) == 2:
        assert L.shape[0] == K, "first axis of aggregate likelihood must " \
            "correspond to the diffusion coefficient"

        # Apply the correction and renormalize
        L = (L.T / frac_remain).T
        if normalize:
            norm = L.sum()
            if norm > 0:
                L /= norm

    elif len(L.shape) == 3:
        assert L.shape[0] == K, "first axis of likelihood matrix must " \
            "correspond to the diffusion coefficient"
        L = (L.T / frac_remain).T
        if normalize: L = L / L.sum(axis=(0,1))

    return L 


def defoc_corr_fbm(L, diff_coefs, hurst_pars, frame_interval=0.00748, dz=0.7,
    normalize=False):
    """
    Apply a defocalization correction to the fractional Brownian motion 
    likelihood function.
    Both diffusion coefficient and Hurst parameter figure into the
    defocalization function for FBM.
    args
    ----
        L               :   ndarray, the likelihood function as generated
                            by *eval_likelihood*
        diff_coefs      :   1D ndarray, the set of diffusion coefficients
                            corresponding to *L*. The second axis of *L*
                            is assumed to correspond to the values of 
                            *diff_coefs*.
        hurst_pars      :   1D ndarray, the set of Hurst parameters 
                            corresponding to *L*. The third axis of *L*
                            is assumed to correspond to the values of 
                            *hurst_pars*.
        frame_interval  :   float, frame interval in seconds
        dz              :   float, focal depth in microns
    returns
    -------
        reference to *L* after correction
    """
    diff_coefs = np.asarray(diff_coefs)
    hurst_pars = np.asarray(hurst_pars)
    nD = diff_coefs.shape[0]
    nH = hurst_pars.shape[0]

    # Evaluate the defocalization probability at one frame interval
    # for all parameter combinations considered in this likelihood matrix
    frac_remain = np.ones((nD, nH), dtype=np.float64)
    if not np.isinf(dz):
        for i, D in enumerate(diff_coefs):
            for j, H in enumerate(hurst_pars):
                frac_remain[i,j] = f_remain_fbm(D, H, 1, frame_interval, dz)

    # Aggregate likelihood
    if len(L.shape) == 2:

        # Apply the correction
        L /= frac_remain

        # Renormalize
        if normalize:
            norm = L.sum()
            if norm > 0:
                L /= norm

    # Trajectory-wise
    elif len(L.shape) == 3:
        L = (L.T / frac_remain.T).T
        if normalize: L = L / L.sum(axis=(0,1))

    return L 


# Available likelihood corrections
LIKELIHOOD_CORR_FUNCS = {
    GAMMA        : defoc_corr_rbm,
    RBME_MARGINAL: defoc_corr_rbm,
    RBME         : defoc_corr_rbme,
    FBME         : defoc_corr_fbm
}


def defoc_corr(L, support, likelihood, frame_interval, dz, normalize=False, **kwargs):
    """
    Apply a defocalization correction to a likelihood function.
    args
    ----
        L           :   ndarray, the likelihood function as generated
                        by *eval_likelihood*
        support     :   tuple of ndarray, the parameter values for the 
                        support of *L*
        likelihood  :   str, the type of likelihood function
        frame_interval: float, frame interval in seconds
        dz          :   float, focal depth in microns
        kwargs      :   additional kwargs to the defocalization function
                        specific to this likelihood
    returns
    -------
        reference to *L* after correction
    """
    # Check that the likelihood function is supported
    if not likelihood in LIKELIHOOD_CORR_FUNCS.keys():
        raise ValueError("likelihood {} not found in correction functions".format(likelihood))

    # Apply the correction
    return LIKELIHOOD_CORR_FUNCS[likelihood](L, *support, dz=dz,
        frame_interval=frame_interval, normalize=normalize, **kwargs)
