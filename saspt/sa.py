import os, warnings, numpy as np, pandas as pd 
from tqdm import tqdm
from scipy.special import digamma
from typing import Tuple

from .constants import (
    DEFAULT_MAX_ITER, DEFAULT_CONC_PARAM, DEFAULT_SPLITSIZE,
    DEFAULT_START_FRAME, RBME, RBME_MARGINAL, GAMMA, FBME, FRAME
)
from .trajectory_group import TrajectoryGroup
from .lik import Likelihood, make_likelihood, make_likelihood_from_params
from .parameters import StateArrayParameters
from .plot import (
    rbm_posterior_plot,
    rbme_posterior_plot,
    marginal_assignment_probability_plot,
    temporal_assignment_probability_plot,
    spatial_assignment_probability_plot,
)
from .utils import cartesian_product

class StateArray:
    """ Represents a state array for a particular set of trajectories and a 
    particular likelihood function.

    A state array is a mixture model. Trajectories are assumed to inhabit one
    of a "grid" of states with distinct parameters. The StateArray object
    provides methods to infer the posterior distribution over trajectory-state
    assignments and state occupations, given an observed set of trajectories.

    init
    ----
        trajectories    :   observed set of trajectories in an SPT experiment
        likelihood      :   the likelihood function with respect to which the
                            state array is defined
        params          :   settings for inference

    attributes
    ----------
        n_tracks                :   number of observed trajectories

        n_jumps                 :   number of observed jumps (spot-spot
                                    connections)

        n_detections            :   number of observed spots

        shape                   :   shape of the parameter grid on which the
                                    state occupations are defined

        jumps_per_track         :   number of observed jumps per trajectory

        naive_assignment_probabilities    : naive probabilities for each 
                                    trajectory-state assignment

        posterior_assignment_probabilities: posterior probability for each 
                                    trajectory-state assignment

        prior_occs              :   mean occupation of each state under the
                                    prior distribution over state occupations

        naive_occs              :   mean occupation of each state under the
                                    naive distribution over state occupations

        posterior_occs          :   mean occupation of each state under the 
                                    posterior distribution over state occupations 

        prior_dirichlet_param   :   parameter to the prior Dirichlet distribution
                                    over state occupations

        posterior_dirichlet_param:  parameter to the posterior Dirichlet distribution
                                    over state occupations
    """
    def __init__(self, trajectories: TrajectoryGroup,
        likelihood: Likelihood, params: StateArrayParameters):
        self.trajectories = trajectories
        self.likelihood = likelihood
        self.params = params

    def __repr__(self):
        """ String representation of this StateArray object """    
        return "StateArray:\n  {}".format("\n  ".join([
            f"{a : <18}: {getattr(self, a)}" for a in \
                ["likelihood_type", "n_tracks", "n_jumps", "parameter_names", "shape"]
        ]))

    def __enter__(self):
        return self 

    def __exit__(self, etype, evalue, traceback):
        return etype is None

    @classmethod
    def from_detections(cls, detections: pd.DataFrame, likelihood_type: str, **kwargs):
        """ Convenience method; instantiate a StateArray object by passing 
        a raw set of detections as a pandas.DataFrame.

        args
        ----
            detections          :   pandas.DataFrame indexed by detection with
                                    columns TRACK, FRAME, PY, and PX at minimum
            likelihood_type     :   type of likelihood function to use
            **kwargs            :   any parameters supported by StateArrayParameters

        returns
        -------
            new instance of StateArray
        """
        params = StateArrayParameters(**kwargs)
        trajectories = TrajectoryGroup.from_params(detections, params)
        if params.sample_size < trajectories.n_tracks:
            trajectories = trajectories.subsample(params.sample_size)
        likelihood = make_likelihood_from_params(likelihood_type, params, **kwargs)
        return cls(trajectories, likelihood, params)

    ################
    ## PROPERTIES ##
    ################

    @property 
    def n_tracks(self) -> int:
        return self.trajectories.n_tracks 

    @property 
    def n_jumps(self) -> int:
        return self.trajectories.n_jumps 

    @property 
    def n_detections(self) -> int:
        return self.trajectories.n_detections

    @property 
    def shape(self) -> Tuple[int]:
        return self.likelihood.shape

    @property 
    def likelihood_type(self) -> str:
        return self.likelihood.name

    @property 
    def parameter_names(self) -> str:
        return self.likelihood.parameter_names

    @property 
    def parameter_values(self) -> Tuple[np.ndarray]:
        return self.likelihood.parameter_values

    @property
    def n_states(self) -> int:
        """ Total number of states in the state array """
        if not hasattr(self, "_n_states"):
            self._n_states = np.prod(self.shape)
        return self._n_states

    @property
    def jumps_per_track(self) -> np.ndarray:
        """ Number of jumps per trajectory.

        returns
        -------
            1D numpy.ndarray of shape (n_tracks,)
        """
        if not hasattr(self, "_jumps_per_track"):
            self._calc_assignment_likelihoods()
        return self._jumps_per_track

    @property
    def naive_assignment_probabilities(self):
        """ Naive probability for each trajectory-state assignment.

        This is really just the likelihood function normalized over all
        states for each trajectory. It is equivalent to the posterior
        probability the trajectory-state assignments if

            a. we assume a uniform prior over trajectory-state assignments

            b. we condition on uniform state occupations; that is, the
                occupations of all states in the parameter grid are 
                identical

        returns
        -------
            numpy.ndarray of shape (*self.shape, n_tracks)
        """       
        if not hasattr(self, "_naive_assignment_probabilities"):
            self._calc_assignment_likelihoods()
        return self._naive_assignment_probabilities

    @property 
    def posterior_assignment_probabilities(self) -> np.ndarray:
        """ Posterior probability for each trajectory-state assignment.

        returns
        -------
            numpy.ndarray of shape (*self.shape, n_tracks)
        """
        if not hasattr(self, "_posterior_assignment_probabilities"):
            self._infer_posterior()
        return self._posterior_assignment_probabilities

    @property
    def prior_dirichlet_param(self) -> np.ndarray:
        """ Parameter for the prior Dirichlet distribution over state
        occupations. 

        returns
        -------
            numpy.ndarray of shape *self.shape*
        """
        if not hasattr(self, "_prior_dirichlet_param"):
            self._prior_dirichlet_param = np.full(self.shape,
                self.params.conc_param, dtype=np.float64)
        return self._prior_dirichlet_param

    @property
    def posterior_dirichlet_param(self) -> np.ndarray:
        """ Parameter for the posterior Dirichlet distribution over state
        occupations.

        returns
        -------
            numpy.ndarray of shape *self.shape*.
        """
        if not hasattr(self, "_posterior_dirichlet_param"):
            self._infer_posterior()
        return self._posterior_dirichlet_param

    @property
    def prior_occs(self) -> np.ndarray:
        """ Mean of the prior Dirichlet distribution over state occupations.

        returns
        -------
            numpy.ndarray of shape *self.shape*
        """
        if not hasattr(self, "_prior_occs"):
            if self.params.conc_param <= 0.0:
                self._prior_occs = np.ones(self.shape, dtype=np.float64)
                if self._prior_occs.size > 0:
                    self._prior_occs /= self._prior_occs.sum()
            else:
                S = self.prior_dirichlet_param.sum()
                if S > 0 and self.prior_dirichlet_param.size > 0:
                    self._prior_occs = self.prior_dirichlet_param / S 
                else:
                    self._prior_occs = np.ones(self.shape, dtype=np.float64)
        return self._prior_occs

    @property
    def naive_occs(self) -> np.ndarray:
        """ Naive estimate for the occupations of each state. This is obtained
        by marginalizing the trajectory-state assignment likelihoods over all
        trajectories.

        Normalized, so that naive_occs.sum() == 1.0.

        returns
        -------
            numpy.ndarray of shape *self.shape*
        """
        if not hasattr(self, "_naive_occs"):
            # Scale naive assignment likelihoods by the number of jumps
            # in each trajectory, and marginalize over the assignments
            self._naive_occs = (self.naive_assignment_probabilities * \
                self.jumps_per_track).sum(axis=-1)

            # Apply an appropriate defocalization correction
            self._naive_occs = self.likelihood.correct_for_defocalization(self._naive_occs)

            # Normalize over all states
            if self._naive_occs.sum() > 0:
                self._naive_occs /= self._naive_occs.sum()
        return self._naive_occs

    @property 
    def posterior_occs(self) -> np.ndarray:
        """ Mean of the posterior Dirichlet distribution over state occupations.

        returns
        -------
            numpy.ndarray of shape *self.shape*.
        """
        if not hasattr(self, "_posterior_occs"):
            self._infer_posterior()
        return self._posterior_occs

    @property
    def occupations_dataframe(self) -> pd.DataFrame:
        """ pandas.DataFrame representation of the posterior distribution,
        suitable for saving to a CSV.

        In this representation, each row of the DataFrame corresponds to a 
        distinct state in the state array. We give the marginal likelihood 
        and mean posterior occupation for each state. """
        if not hasattr(self, "_occupations_dataframe"):
            parameters = list(self.likelihood.parameter_names)
            cols = parameters + ["naive_occupation", "mean_posterior_occupation"]
            df = pd.DataFrame(index=np.arange(self.n_states), columns=cols, dtype=np.float64)
            df["naive_occupation"] = self.naive_occs.ravel()
            df["mean_posterior_occupation"] = self.posterior_occs.ravel()
            if self.n_states > 0:
                df[parameters] = cartesian_product(*self.likelihood.parameter_values)
            self._occupations_dataframe = df
        return self._occupations_dataframe

    @property 
    def diff_coefs(self) -> np.ndarray:
        """ Convenience attribute; the set of diffusion coefficients
        supported by the underlying Likelihood function.

        returns
        -------
            1D numpy.ndarray, the diffusion coefficients in µm2/sec
        """
        if not hasattr(self, "_diff_coefs"):
            if "diff_coef" in self.likelihood.parameter_names:
                self._diff_coefs = self.likelihood.diff_coefs
            else:
                self._diff_coefs = np.array([], dtype=np.float64)
        return self._diff_coefs

    #############
    ## METHODS ##
    #############

    def infer_posterior(self) -> Tuple[np.ndarray]:
        """ Attempt to infer the posterior distribution over
        trajectory-state assignments and state occupations for
        this StateArray.

        returns
        -------
            (
                numpy.ndarray of shape (*self.shape, n_tracks), 
                    posterior probabilities for each trajectory-state
                    assignment;

                numpy.ndarray of shape *self.shape*, parameter
                    to the posterior Dirichlet distribution over state
                    occupations;

                numpy.ndarray of shape *self.shape*, mean of the
                    posterior Dirichlet distribution over state occupations
            )
        """
        L = self.naive_assignment_probabilities
        R = L.copy()
        par_indices = tuple(range(len(self.shape)))
        if self.params.progress_bar:
            print("inferring posterior distribution...")
            iterations = tqdm(range(self.params.max_iter))
        else:
            iterations = range(self.params.max_iter)
        for i in iterations:

            # Update posterior state occupations (*n* is the parameter to 
            # a Dirichlet distribution over state occupations)
            n = (R * self.jumps_per_track).sum(axis=-1)
            m = n + self.prior_dirichlet_param

            # Exponent of the expected log occupation under the current
            # posterior estimate
            exp_log_tau = np.exp(digamma(m))

            # Calculate posterior probabilities for each trajectory-state
            # assignment, normalizing over all states for each trajectory
            R = (L.T * exp_log_tau.T).T
            R = R / R.sum(axis=par_indices)

        # Special case: 0 iterations
        if self.params.max_iter == 0:
            n = (R * self.jumps_per_track).sum(axis=-1)

        # Adjust posterior distribution to account for defocalization
        n = self.likelihood.correct_for_defocalization(n, normalize=False)

        # Calculate mean state occupations under the posterior model
        post_mean = n / n.sum() if n.sum() > 0 else self.prior_occs.copy()

        return R, n, post_mean

    def marginalize_on_diff_coef(self, *args, **kwargs):
        """ Inherited from the underlying Likelihood object. """
        return self.likelihood.marginalize_on_diff_coef(*args, **kwargs)

    ##############
    ## PLOTTING ##
    ##############

    def plot_occupations(self, out_png: str, **kwargs):
        """ Make a plot of the posterior distribution suitable for the 
        underlying likelihood function of this StateArray.

        May not be implemented for all likelihood functions.

        args
        ----
            out_png     :   output path for the plot
            kwargs      :   additional keyword arguments to the plotting
                            function
        """
        if self.likelihood.name == RBME:
            rbme_posterior_plot(out_png, self.naive_occs, 
                self.posterior_occs, self.likelihood.diff_coefs,
                self.likelihood.loc_errors, **kwargs)
        elif self.likelihood.name in [RBME_MARGINAL, GAMMA]:
            rbm_posterior_plot(out_png, self.naive_occs, 
                self.posterior_occs, self.likelihood.diff_coefs,
                **kwargs)
        else:
            warnings.warn(f"no posterior plot is implemented for " \
                "likelihood {self.likelihood.name}; not making plot")

    def plot_assignment_probabilities(self, out_png: str, **kwargs):
        """ Make a plot of the posterior probabilities for each
        trajectory-state assignment. 

        The likelihood function must support the diffusion coefficient
        parameter.

        args
        ----
            out_png     :   output path for plot
            kwargs      :   additional keyword arguments to the plotting
                            function
        """
        if "diff_coef" not in self.likelihood.parameter_names:
            warnings.warn(f"no posterior assignment plot is implemented " \
                "for likelihood {self.likelihood.name}; not making plot")
        else:
            scaled_assignment_probabilities = self.naive_assignment_probabilities * \
                self.jumps_per_track
            marginal_assignment_probability_plot(
                out_png,
                self.likelihood.marginalize_on_diff_coef(self.posterior_assignment_probabilities),
                self.likelihood.marginalize_on_diff_coef(scaled_assignment_probabilities),
                self.likelihood.diff_coefs,
                sort_by_mean_diff_coef=True,
                **kwargs
            )

    def plot_temporal_assignment_probabilities(self, out_png: str,
        frame_block_size: int=None, **kwargs):
        """ Make a plot of the posterior probabilities for each 
        trajectory-state assignment as a function of the frame index.

        Useful for understanding whether density (which usually begins
        high and declines throughout an SPT experiment) has a noticeable
        effect on the trajectory-state probabilities.

        args
        ----
            out_png             :   output path for plot
            frame_block_size    :   temporal bin size in frames
            kwargs              :   additional keyword arguments to
                                    plot function
        """
        if "diff_coef" not in self.likelihood.parameter_names:
            warnings.warn(f"no posterior assignment plot is implemented " \
                "for likelihood {self.likelihood.name}; not making plot")
        else:
            P = self.likelihood.marginalize_on_diff_coef(self.posterior_assignment_probabilities)
            L = self.likelihood.marginalize_on_diff_coef(self.naive_assignment_probabilities)
            P = P * self.jumps_per_track
            L = L * self.jumps_per_track

            # Choose a frame block size appropriate for this movie
            if frame_block_size is None:
                n_frames = self.trajectories.detections[FRAME].max() + 1
                base = max(0, np.log10(n_frames) - 2)
                exp_base = 10**int(np.floor(base))
                frame_block_size = exp_base if (base%1.0<0.5) else 3*exp_base

            temporal_assignment_probability_plot(
                out_png,
                self.trajectories.detections,
                P,
                L,
                self.likelihood.diff_coefs,
                frame_block_size=frame_block_size,
                **kwargs
            )

    def plot_spatial_assignment_probabilities(self, out_png: str, **kwargs):
        """ Aggregate the prior and posterior trajectory-state assignment probabilities
        into spatial bins and show alongside the localization density.

        args
        ----
            out_png     :   output path for plot
            kwargs      :   additional keyword arguments to plot function
        """
        if "diff_coef" not in self.likelihood.parameter_names:
            warnings.warn(f"no posterior assignment plot is implemented " \
                "for likelihood {self.likelihood.name}; not making plot")
        else:
            P = self.likelihood.marginalize_on_diff_coef(self.posterior_assignment_probabilities)
            L = self.likelihood.marginalize_on_diff_coef(self.naive_assignment_probabilities)
            spatial_assignment_probability_plot(
                out_png,
                self.trajectories.detections,
                P,
                L,
                self.likelihood.diff_coefs,
                self.params.pixel_size_um,
                **kwargs
            )

    ######################
    ## OBJECT UTILITIES ##
    ######################

    def _infer_posterior(self):
        self._posterior_assignment_probabilities, \
            self._posterior_dirichlet_param, \
            self._posterior_occs = self.infer_posterior()

    def _calc_assignment_likelihoods(self) -> Tuple[np.ndarray]:
        """ Calculate the normalized likelihood and number of jumps for each 
        of the trajectories in this state array. """
        log_L, self._jumps_per_track = self.likelihood(self.trajectories)
        self._naive_assignment_probabilities = self.likelihood.exp(log_L)
