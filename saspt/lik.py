import warnings, numpy as np
from typing import Tuple
from abc import ABC, abstractmethod

from .constants import (
    DEFAULT_DIFF_COEFS, DEFAULT_LOC_ERRORS, DEFAULT_HURST_PARS,
    GAMMA, RBME_MARGINAL, RBME, FBME, TRACK, DR2
)
from .defoc import defoc_corr
from .trajectory_group import TrajectoryGroup
from .parameters import StateArrayParameters

class Likelihood(ABC):
    """ Base class for likelihood functions used in state arrays. Each subclass
    must define attributes that identify the likelihood function and methods 
    required for downstream use in state arrays.

    In more detail, state arrays require likelihood functions that evaluate on a
    discrete grid of parameter values. The Likelihood object passes information
    about its parameter grid to the StateArray object via the *shape*, 
    *parameter_names*, and *parameter_values* attributes. """
    @property
    @abstractmethod
    def name(self) -> str:
        """ A string that identifies this likelihood function """

    @property 
    @abstractmethod 
    def shape(self) -> Tuple[int]:
        """ Shape of the parameter grid on which this likelihood function
        is evaluated. """

    @property 
    @abstractmethod
    def parameter_names(self) -> Tuple[str]:
        """ Names of the parameters defining each axis of the parameter 
        grid """

    @property
    @abstractmethod
    def parameter_values(self) -> Tuple[np.ndarray]:
        """ Values of the parameters on which this likelihood function is
        evaluated. Returns a tuple of 1D np.ndarray defining the values
        along each parameter axis """

    @property 
    @abstractmethod 
    def parameter_units(self) -> Tuple[str]:
        """ Units in which each parameter axis is defined """

    @abstractmethod 
    def __call__(self, trajectories: TrajectoryGroup) -> Tuple[np.ndarray]:
        """ Evaluate the log likelihood function on each trajectory at each 
        point on the parameter grid. Returns two numpy.ndarrays encoding the 
        evaluated log likelihood function as the number of jumps per trajectory.
        The first numpy.ndarray has shape (*self.shape, trajectories.n_tracks) 
        and the second has shape (trajectories.n_tracks). """

    @abstractmethod
    def exp(self, log_L: np.ndarray) -> np.ndarray:
        """ Take the exponent of a log likelihood function in a numerically
        stable way appropriate for this likelihood function """

    @abstractmethod 
    def correct_for_defocalization(self, occs: np.ndarray, normalize: bool) -> np.ndarray:
        """ Correct a set of state occupations on this parameter grid for the 
        effects of defocalization. *occs* and the output of the function are 
        numpy.ndarrays of shape *self.shape*. """

    @abstractmethod
    def marginalize_on_diff_coef(self, occs: np.ndarray) -> np.ndarray:
        """ Marginalize a set of state occupations on the diffusion coefficient.
        May raise NotImplementedError if diffusion coefficient is not a parameter
        for the likelihood function (although this is the case for all 
        likelihoods implemented to date). 

        *occs* is a numpy.ndarray of shape *self.shape* and the return value
        is a 1D numpy.ndarray over the diffusion coefficient axis. """


class RBMELikelihood(Likelihood):
    """ Likelihood function for a regular Brownian motion with localization 
    error (RBME). This likelihood function has two parameters:
        1. the diffusion coefficient (specified in squared microns per sec)
        2. the localization error (specified in microns)

    The vector of all jumps along one axis for an RBME is a multivariate
    normal random vector with mean zero and covariance matrix *C* such that

        C[i,j] = 2 * (D * dt + loc_error^2)   if i == j
        C[i,j] = -loc_error^2                 if abs(i-j) == 1
        C[i,j] = 0                            otherwise

    where *dt* is the frame interval in seconds.

    The parameter grid for the RBMELikelihood object is a set of diffusion
    coefficients and localization errors. Localization error is usually 
    treated as a nuisance parameter (i.e. we marginalize over it after 
    inference).

    init
    ----
        pixel_size_um   :   size of camera pixels in microns
        frame_interval  :   frame interval in seconds
        focal_depth     :   microscope focal depth in microns
        diff_coefs      :   1D numpy.ndarray, the set of diffusion 
                            coefficients to use for the parameter grid
        loc_errors      :   1D numpy.ndarray, the set of localization 
                            errors to use for the parameter grid
    """
    def __init__(self, pixel_size_um: float, frame_interval: float,
        focal_depth: float=np.inf, diff_coefs: np.ndarray=DEFAULT_DIFF_COEFS,
        loc_errors: np.ndarray=DEFAULT_LOC_ERRORS, **kwargs):
        self.pixel_size_um = pixel_size_um 
        self.frame_interval = frame_interval
        self.focal_depth = focal_depth
        self.diff_coefs = np.asarray(diff_coefs)
        self.loc_errors = np.asarray(loc_errors)

    @property
    def name(self) -> str:
        return RBME
    
    @property 
    def shape(self) -> Tuple[int]:
        if not hasattr(self, "_shape"):
            self._shape = (self.diff_coefs.shape[0], self.loc_errors.shape[0])
        return self._shape 

    @property
    def parameter_names(self) -> Tuple[str]:
        if not hasattr(self, "_parameter_names"):
            self._parameter_names = ("diff_coef", "loc_error")
        return self._parameter_names

    @property 
    def parameter_values(self) -> Tuple[np.ndarray]:
        return (self.diff_coefs, self.loc_errors)

    @property 
    def parameter_units(self) -> Tuple[str]:
        if not hasattr(self, "_parameter_units"):
            self._parameter_units = ("µm2/sec", "µm")
        return self._parameter_units

    def __call__(self, trajectories: TrajectoryGroup) -> Tuple[np.ndarray]:
        """ Evaluate the log likelihood function specified by this *Likelihood*
        object on a set of target trajectories. 

        args
        ----
            trajectories    :   saspt.TrajectoryGroup, a set of target 
                                trajectories

        returns
        -------
            (
                3D numpy.ndarray with shape (*self.shape, n_tracks), the log 
                    likelihood function for each point on the parameter grid 
                    evaluated on each of the trajectories;

                1D numpy.ndarray of shape (n_tracks), the number of jumps in
                    each trajectory
            )
        """
        J = trajectories.jumps 
        n_tracks = trajectories.n_tracks

        # Generate the RBME jump covariance matrix
        def make_cov(diff_coef, loc_error, n) -> np.ndarray:
            le2 = loc_error ** 2
            C = 2 * (diff_coef * self.frame_interval + le2) * np.identity(n)
            for i in range(n-1):
                C[i,i+1] = -le2 
                C[i+1,i] = -le2 
            return C

        # Output
        log_L = np.empty((*self.shape, n_tracks), dtype=np.float64)
        jumps_per_track = np.empty(n_tracks, dtype=np.float64)

        # Iterate through the different trajectory lengths
        for n in range(1, trajectories.splitsize+1):

            # Get all track vectors matching this length. *V* is a 3D numpy.ndarray
            # with shape (M, 2, n), where *M* is the number of trajectories with 
            # exactly *n* jumps
            V, track_idx = trajectories.get_track_vectors(n)
            jumps_per_track[track_idx] = n
            y_jumps = V[:,0,:]
            x_jumps = V[:,1,:]

            for i, D in enumerate(self.diff_coefs):
                for j, le in enumerate(self.loc_errors):
                    C = make_cov(D, le, n)
                    C_inv = np.linalg.inv(C)
                    log_norm = n * np.log(2*np.pi) + np.linalg.slogdet(C)[1]
                    y_ll = ((y_jumps @ C_inv) * y_jumps).sum(axis=1)
                    x_ll = ((x_jumps @ C_inv) * x_jumps).sum(axis=1)
                    log_L[i,j,track_idx] = -0.5 * (y_ll + x_ll) - log_norm

        return log_L, jumps_per_track

    def exp(self, log_L: np.ndarray) -> np.ndarray:
        """ Given a log likelihood function, take the exponent to get the
        likelihood function and normalize over all states for each trajectory.

        args
        ----
            log_L   :   3D numpy.ndarray of shape (*self.shape, n_tracks),
                        log likelihoods of each trajectory at each point
                        on the parameter grid

        returns
        -------
            3D numpy.ndarray of shape (*self.shape, n_tracks),
                likelihoods of each track at each point on the
                parameter grid
        """
        if log_L.size > 0:
            log_L[np.isnan(log_L)] = -np.inf
            log_L[log_L>=100.0] = 100.0           
            L = np.exp(log_L - np.nanmax(log_L, axis=(0, 1)))
            return L / L.sum(axis=(0,1))
        else:
            return log_L.copy()

    def correct_for_defocalization(self, occs: np.ndarray, normalize: bool=False) -> np.ndarray:
        """ Apply a defocalization correction suitable for RBMEs to a set of state
        occupations.

        args
        ----
            occs        :   2D numpy.ndarray, state occupations with axes corresponding
                            to self.parameter_values
            normalize   :   bool, normalize over state occupations before returning

        returns
        -------
            2D numpy.ndarray of shape *occs*, the corrected state occupations
        """
        return defoc_corr(occs, self.parameter_values, likelihood=self.name, 
            frame_interval=self.frame_interval, dz=self.focal_depth, normalize=normalize)

    def marginalize_on_diff_coef(self, occs: np.ndarray) -> np.ndarray:
        """ Marginalize a set of state occupations or trajectory-state 
        assignments on the diffusion coefficients.

        args
        ----
            occs    :   numpy.ndarray of shape *self.shape*, state occupations,
                        or numpy.ndarray of shape (*self.shape, n_tracks), 
                        trajectory-state assignments

        returns
        -------
            numpy.ndarray, the state occupations or trajectory-state 
                assignments with parameters other than diffusion coefficient
                marginalized out
        """
        if len(occs.shape) == 2:
            out = occs.sum(axis=1)
            S = out.sum()
            return out / S if S > 0 else out 
        elif len(occs.shape) == 3:
            # Marginalize out localization error
            out = occs.sum(axis=1)
            # Normalization constant for each trajectory
            S = out.sum(axis=0)
            nonzero = S > 0
            out[:,nonzero] = out[:,nonzero] / S[nonzero]
            return out


class GammaLikelihood(Likelihood):
    def __init__(self, pixel_size_um: float, frame_interval: float,
        focal_depth: float, diff_coefs: np.ndarray=DEFAULT_DIFF_COEFS, 
        loc_error: float=0.035, mode: str="point", **kwargs):

        self.pixel_size_um = pixel_size_um
        self.frame_interval = frame_interval
        self.focal_depth = focal_depth  
        self.loc_error = loc_error
        self.diff_coefs = diff_coefs
        self.mode = mode

    @property
    def name(self) -> str:
        return GAMMA

    @property
    def shape(self) -> Tuple[int]:
        if not hasattr(self, "_shape"):
            self._shape = (self.diff_coefs.shape[0],)
        return self._shape

    @property
    def parameter_names(self) -> Tuple[str]:
        if not hasattr(self, "_parameter_names"):
            self._parameter_names = ("diff_coef",)
        return self._parameter_names 

    @property
    def parameter_values(self) -> Tuple[np.ndarray]:
        return (self.diff_coefs,)

    @property
    def parameter_units(self) -> Tuple[str]:
        if not hasattr(self, "_parameter_units"):
            self._parameter_units = ("µm2/sec",)
        return self._parameter_units

    def __call__(self, trajectories: TrajectoryGroup) -> Tuple[np.ndarray]:
        """ Evaluate the log likelihood for each trajectory at each point
        on the parameter grid.

        args
        ----
            trajectories    :   trajectories on which to evaluate the function

        returns
        -------
            (
                2D numpy.ndarray of shape (n_diff_coefs, n_tracks), the log
                    likelihood function;

                1D numpy.ndarray of shape (n_tracks,), the number of jumps
                    in each trajectory
            )
        """
        le2 = self.loc_error ** 2
        K = self.diff_coefs.shape[0]
        n_tracks = trajectories.n_tracks
        alpha = trajectories.jumps_per_track.astype(np.float64)
        sum_r2 = np.array(trajectories.jumps.groupby(TRACK)[DR2].sum())
        log_L = np.zeros((K, n_tracks), dtype=np.float64)
        for j in range(K):
            phi = 4 * (self.diff_coefs[j] * self.frame_interval + le2)
            log_L[j,:] = -(sum_r2 / phi) - alpha * np.log(phi)
        return log_L, alpha

    def exp(self, log_L: np.ndarray) -> np.ndarray:
        """ Given an evaluated log likelihood function, take the exponent
        and normalize to get the likelihood function.

        args
        ----
            log_L       :   2D numpy.ndarray of shape (n_diff_coefs, n_tracks),
                            the result of GammaLikelihood.__call__

        returns
        -------
            2D numpy.ndarray of shape *log_L.shape*, the corresponding
                normalized likelihood function
        """
        if log_L.size > 0:
            L = np.exp(log_L - log_L.max(axis=0))
            L = L / L.sum(axis=0)
            return L
        else:
            return log_L.copy()

    def correct_for_defocalization(self, occs: np.ndarray, normalize: bool=False) -> np.ndarray:
        return defoc_corr(occs, self.parameter_values, likelihood=self.name,
            frame_interval=self.frame_interval, dz=self.focal_depth, normalize=normalize)

    def marginalize_on_diff_coef(self, occs: np.ndarray) -> np.ndarray:
        """ Marginalize a set of state occupations or trajectory-state 
        assignments on the diffusion coefficients.

        args
        ----
            occs    :   numpy.ndarray of shape *self.shape*, state occupations,
                        or numpy.ndarray of shape (*self.shape, n_tracks), 
                        trajectory-state assignments

        returns
        -------
            numpy.ndarray, the state occupations or trajectory-state 
                assignments with parameters other than diffusion coefficient
                marginalized out
        """
        return occs.copy()


class RBMEMarginalLikelihood(Likelihood):
    def __init__(self, pixel_size_um: float, frame_interval: float, focal_depth: float=np.inf,
        diff_coefs: np.ndarray=DEFAULT_DIFF_COEFS, loc_errors: np.ndarray=DEFAULT_LOC_ERRORS,
        **kwargs):

        self.pixel_size_um = pixel_size_um 
        self.frame_interval = frame_interval
        self.focal_depth = focal_depth
        self.diff_coefs = diff_coefs
        self.loc_errors = loc_errors 

    @property
    def name(self) -> str:
        return RBME_MARGINAL

    @property
    def shape(self) -> Tuple[int]:
        if not hasattr(self, "_shape"):
            self._shape = (self.diff_coefs.shape[0],)
        return self._shape

    @property
    def parameter_names(self) -> Tuple[str]:
        if not hasattr(self, "_parameter_names"):
            self._parameter_names = ("diff_coef",)
        return self._parameter_names 

    @property
    def parameter_values(self) -> Tuple[np.ndarray]:
        return (self.diff_coefs,)

    @property
    def parameter_units(self) -> Tuple[str]:
        if not hasattr(self, "_parameter_units"):
            self._parameter_units = ("µm2/sec",)
        return self._parameter_units

    def __call__(self, trajectories: TrajectoryGroup) -> Tuple[np.ndarray]:
        J = trajectories.jumps 
        n_tracks = trajectories.n_tracks

        # Generate the RBME jump covariance matrix
        def make_cov(diff_coef, loc_error, n) -> np.ndarray:
            le2 = loc_error ** 2
            C = 2 * (diff_coef * self.frame_interval + le2) * np.identity(n)
            for i in range(n-1):
                C[i,i+1] = -le2 
                C[i+1,i] = -le2 
            return C

        # Output
        log_L = np.empty((self.diff_coefs.shape[0], self.loc_errors.shape[0], n_tracks), dtype=np.float64)
        jumps_per_track = np.empty(n_tracks, dtype=np.float64)

        # Iterate through the different trajectory lengths
        for n in range(1, trajectories.splitsize+1):

            # Get all track vectors matching this length. *V* is a 3D numpy.ndarray
            # with shape (M, 2, n), where *M* is the number of trajectories with 
            # exactly *n* jumps
            V, track_idx = trajectories.get_track_vectors(n)
            jumps_per_track[track_idx] = n
            y_jumps = V[:,0,:]
            x_jumps = V[:,1,:]

            for i, D in enumerate(self.diff_coefs):
                for j, le in enumerate(self.loc_errors):
                    C = make_cov(D, le, n)
                    C_inv = np.linalg.inv(C)
                    log_norm = n * np.log(2*np.pi) + np.linalg.slogdet(C)[1]
                    y_ll = ((y_jumps @ C_inv) * y_jumps).sum(axis=1)
                    x_ll = ((x_jumps @ C_inv) * x_jumps).sum(axis=1)
                    log_L[i,j,track_idx] = -0.5 * (y_ll + x_ll) - log_norm

        # Marginalize on localization error, letting log(0) = -inf
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            log_L = np.log(self.exp(log_L).sum(axis=1))

        return log_L, jumps_per_track

    def exp(self, log_L: np.ndarray) -> np.ndarray:
        """ Given an evaluated log likelihood function, take the exponent
        and normalize to get the likelihood function.

        args
        ----
            log_L       :   2D numpy.ndarray of shape (n_diff_coefs, n_tracks),
                            the result of RBMEMarginalLikelihood.__call__

        returns
        -------
            2D numpy.ndarray of shape *log_L.shape*, the corresponding
                normalized likelihood function
        """
        if log_L.size > 0:
            L = np.exp(log_L - log_L.max(axis=0))
            L = L / L.sum(axis=0)
            return L
        else:
            return log_L.copy()

    def correct_for_defocalization(self, occs: np.ndarray, normalize: bool=False) -> np.ndarray:
        return defoc_corr(occs, self.parameter_values, likelihood=self.name,
            frame_interval=self.frame_interval, dz=self.focal_depth, normalize=normalize)

    def marginalize_on_diff_coef(self, occs: np.ndarray) -> np.ndarray:
        """ Marginalize a set of state occupations or trajectory-state 
        assignments on the diffusion coefficients.

        args
        ----
            occs    :   numpy.ndarray of shape *self.shape*, state occupations,
                        or numpy.ndarray of shape (*self.shape, n_tracks), 
                        trajectory-state assignments

        returns
        -------
            numpy.ndarray, the state occupations or trajectory-state 
                assignments with parameters other than diffusion coefficient
                marginalized out
        """
        return occs.copy()


class FBMELikelihood(Likelihood):
    def __init__(self, pixel_size_um: float, frame_interval: float, 
        focal_depth: float, diff_coefs: np.ndarray=DEFAULT_DIFF_COEFS,
        hurst_pars: np.ndarray=DEFAULT_HURST_PARS, loc_error: float=0.035, 
        **kwargs):

        self.pixel_size_um = pixel_size_um 
        self.frame_interval = frame_interval
        self.focal_depth = focal_depth 
        self.diff_coefs = np.asarray(diff_coefs)
        self.hurst_pars = np.asarray(hurst_pars)
        self.loc_error = loc_error

    @property
    def name(self) -> str:
        return FBME
    
    @property 
    def shape(self) -> Tuple[int]:
        if not hasattr(self, "_shape"):
            self._shape = (self.diff_coefs.shape[0], self.hurst_pars.shape[0])
        return self._shape 

    @property
    def parameter_names(self) -> Tuple[str]:
        if not hasattr(self, "_parameter_names"):
            self._parameter_names = ("diff_coef", "hurst_parameter")
        return self._parameter_names

    @property 
    def parameter_values(self) -> Tuple[np.ndarray]:
        return (self.diff_coefs, self.hurst_pars)

    @property 
    def parameter_units(self) -> Tuple[str]:
        if not hasattr(self, "_parameter_units"):
            self._parameter_units = ("µm2/sec", "")
        return self._parameter_units

    def __call__(self, trajectories: TrajectoryGroup) -> Tuple[np.ndarray]:
        J = trajectories.jumps 
        n_tracks = trajectories.n_tracks
        le2 = self.loc_error ** 2

        # Generate the FBME jump covariance matrix
        def make_cov(diff_coef, hurst_par, n) -> np.ndarray:
            h2 = hurst_par * 2
            T, S = (np.indices((n, n)) + 1)
            C = diff_coef * self.frame_interval * (
                np.power(np.abs(T - S + 1), h2) + 
                np.power(np.abs(T - S - 1), h2) - 
                2 * np.power(np.abs(T - S), h2)
            )
            C += (2 * le2 * np.identity(n))
            for i in range(n-1):
                C[i,i+1] -= le2 
                C[i+1,i] -= le2 
            return C 

        # Output
        log_L = np.empty((*self.shape, n_tracks), dtype=np.float64)
        jumps_per_track = np.empty(n_tracks, dtype=np.float64)

        # Iterate through the different trajectory lengths
        for n in range(1, trajectories.splitsize+1):

            # Get all track vectors matching this length. *V* is a 3D numpy.ndarray
            # with shape (M, 2, n), where *M* is the number of trajectories with 
            # exactly *n* jumps
            V, track_idx = trajectories.get_track_vectors(n)
            jumps_per_track[track_idx] = n
            y_jumps = V[:,0,:]
            x_jumps = V[:,1,:]

            for i, D in enumerate(self.diff_coefs):
                for j, hp in enumerate(self.hurst_pars):
                    C = make_cov(D, hp, n)
                    C_inv = np.linalg.inv(C)
                    log_norm = n * np.log(2*np.pi) + np.linalg.slogdet(C)[1]
                    y_ll = ((y_jumps @ C_inv) * y_jumps).sum(axis=1)
                    x_ll = ((x_jumps @ C_inv) * x_jumps).sum(axis=1)
                    log_L[i,j,track_idx] = -0.5 * (y_ll + x_ll) - log_norm

        return log_L, jumps_per_track

    def exp(self, log_L: np.ndarray) -> np.ndarray:
        """ Given a log likelihood function, take the exponent to get the
        likelihood function and normalize over all states for each trajectory.

        args
        ----
            log_L   :   3D numpy.ndarray of shape (*self.shape, n_tracks),
                        log likelihoods of each trajectory at each point
                        on the parameter grid

        returns
        -------
            3D numpy.ndarray of shape (*self.shape, n_tracks),
                likelihoods of each track at each point on the
                parameter grid
        """
        if log_L.size > 0:
            log_L[np.isnan(log_L)] = -np.inf
            log_L[log_L>=100.0] = 100.0
            L = np.exp(log_L - np.nanmax(log_L, axis=(0, 1)))
            return L / L.sum(axis=(0,1))
        else:
            return log_L.copy()

    def correct_for_defocalization(self, occs: np.ndarray, normalize: bool=False) -> np.ndarray:
        return defoc_corr(occs, self.parameter_values, likelihood=self.name,
            frame_interval=self.frame_interval, dz=self.focal_depth, normalize=normalize)

    def marginalize_on_diff_coef(self, occs: np.ndarray) -> np.ndarray:
        """ Marginalize a set of state occupations or trajectory-state 
        assignments on the diffusion coefficients.

        args
        ----
            occs    :   numpy.ndarray of shape *self.shape*, state occupations,
                        or numpy.ndarray of shape (*self.shape, n_tracks), 
                        trajectory-state assignments

        returns
        -------
            numpy.ndarray, the state occupations or trajectory-state 
                assignments with parameters other than diffusion coefficient
                marginalized out
        """
        if len(occs.shape) == 2:
            out = occs.sum(axis=1)
            S = out.sum()
            return out / S if S > 0 else out 
        elif len(occs.shape) == 3:
            # Marginalize out Hurst parameter
            out = occs.sum(axis=1)
            # Normalization constant for each trajectory
            S = out.sum(axis=0)
            nonzero = S > 0
            out[:,nonzero] = out[:,nonzero] / S[nonzero]
            return out


####################################
## AVAILABLE LIKELIHOOD FUNCTIONS ##
####################################


LIKELIHOODS = {
    RBME: RBMELikelihood,
    GAMMA: GammaLikelihood,
    RBME_MARGINAL: RBMEMarginalLikelihood,
    FBME: FBMELikelihood,
}


def make_likelihood(likelihood_type: str, **kwargs):
    """ Factory method for Likelihood subclass instances. 

    args
    ----
        likelihood_type     :   a key in *LIKELIHOODS*

    returns
    -------
        new instance of the corresponding Likelihood subclass
    """
    if likelihood_type not in LIKELIHOODS.keys():
        avail = ", ".join(list(LIKELIHOODS.keys()))
        raise KeyError(f"likelihood {likelihood_type} not recognized; options are: {avail}")
    return LIKELIHOODS[likelihood_type](**kwargs)

def make_likelihood_from_params(likelihood_type: str, params: StateArrayParameters,
    **kwargs):
    """ Convenience function; wrapper on *make_likelihood* that extracts
    relevant parameters from a StateArrayParameters instance. 

    args
    ----
        likelihood_type     :   a key in *LIKELIHOODS*
        params              :   parameters for running a state array
        kwargs              :   any additional keyword arguments to the 
                                Likelihood constructor

    returns
    -------
        new instance of the corresponding Likelihood subclass 
    """
    kwargs = {
        'pixel_size_um': params.pixel_size_um,
        'frame_interval': params.frame_interval,
        'focal_depth': params.focal_depth, 
        **kwargs
    }
    return make_likelihood(likelihood_type, **kwargs)
