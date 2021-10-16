import os, unittest, numpy as np, pandas as pd
from typing import Tuple

from saspt.constants import TRACK, FRAME, PY, PX, RBME, RBME_MARGINAL, GAMMA, FBME
from saspt.lik import Likelihood, LIKELIHOODS, make_likelihood, RBMELikelihood
from saspt.trajectory_group import TrajectoryGroup

def assert_isinstance_all(type_, objs):
    assert all(map(lambda p: isinstance(p, type_), objs))

class TestLikelihoods(unittest.TestCase):
    def setUp(self):
        # Simple set of trajectories
        self.sample_tracks = pd.DataFrame({
            TRACK:  [     0,    1,    1,   -1,    3,    3,    3,    4,    4],
            FRAME:  [     0,    0,    1,    1,    1,    2,    3,    6,    7],
            PY:     [   0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8],
            PX:     [   0.8,  0.7,  0.6,  0.5,  0.4,  0.3,  0.2,  0.1,  0.0]
        })
        self.kwargs = dict(pixel_size_um=0.16, frame_interval=0.00748, focal_depth=0.7)
        self.trajectory_group = TrajectoryGroup(
            self.sample_tracks,
            pixel_size_um=self.kwargs.get("pixel_size_um"),
            frame_interval=self.kwargs.get("frame_interval"),
            splitsize=4,
            start_frame=0,
        )

    def test_init(self):
        """ Make sure we can initialize each Likelihood subclass without errors """
        for likelihood_type in LIKELIHOODS.keys():
            L = make_likelihood(likelihood_type, **self.kwargs)

            # Is a valid Likelihood subclass
            assert isinstance(L, Likelihood)

            # Enforce attribute typing
            assert isinstance(L.name, str)

            assert isinstance(L.parameter_names, tuple)
            assert_isinstance_all(str, L.parameter_names)

            assert isinstance(L.parameter_units, tuple)
            assert_isinstance_all(str, L.parameter_units)

            assert isinstance(L.parameter_grid, tuple)
            assert_isinstance_all(np.ndarray, L.parameter_grid)

            assert isinstance(L.shape, tuple)
            assert_isinstance_all(int, L.shape)

            # Evaluate on some sample trajectories
            par_axes = tuple(range(len(L.shape)))
            log_L, jumps_per_track = L(self.trajectory_group)
            assert isinstance(log_L, np.ndarray)
            assert log_L.shape == (*L.shape, self.trajectory_group.n_tracks)
            assert jumps_per_track.shape[0] == self.trajectory_group.n_tracks
            assert (~np.isinf(log_L)).any()
            assert (~np.isnan(log_L)).any()
            assert (jumps_per_track >= 1.0).all()

            # Test Likelihood.exp
            exp_log_L = L.exp(log_L)
            assert exp_log_L.shape == (*L.shape, self.trajectory_group.n_tracks)
            assert (np.abs(exp_log_L.sum(axis=par_axes) - 1.0) <= 1.0e-6).all()

            # Test Likelihood.correct_for_defocalization
            np.random.seed(66666)
            occs = np.random.random(size=L.shape)
            occs_corr = L.correct_for_defocalization(occs, normalize=True)
            assert occs_corr.shape == occs.shape
            assert abs(occs_corr.sum() - 1.0) <= 1.0e-6

            # Test Likelihood.marginalize_on_diff_coef
            marg_occs = L.marginalize_on_diff_coef(occs)
            assert len(marg_occs.shape) == 1
            assert marg_occs.shape[0] == len(L.diff_coefs)
            assert abs(marg_occs.sum() - 1.0) < 1.0e-6

    def test_empty_grid(self):
        """ Attempt to initialize each likelihood function with an empty
        parameter grid. """

        kwargs = dict(pixel_size_um=0.16, frame_interval=0.00748, focal_depth=0.7)

        diff_coefs = np.array([], dtype=np.float64)
        loc_errors = np.array([], dtype=np.float64)
        hurst_pars = np.array([], dtype=np.float64)

        supports = {
            RBME: dict(diff_coefs=diff_coefs, loc_errors=loc_errors),
            GAMMA: dict(diff_coefs=diff_coefs),
            RBME_MARGINAL: dict(diff_coefs=diff_coefs, loc_errors=loc_errors),
            FBME: dict(diff_coefs=diff_coefs, hurst_pars=hurst_pars),
        }

        for likelihood_type in LIKELIHOODS.keys():
            support = supports.get(likelihood_type)
            L = make_likelihood(likelihood_type, **self.kwargs, **support)
            assert all(map(lambda p: p.shape[0] == 0, L.parameter_grid))

            # Evaluate on some sample trajectories
            par_axes = tuple(range(len(L.shape)))
            log_L, jumps_per_track = L(self.trajectory_group)
            assert isinstance(log_L, np.ndarray)
            assert log_L.shape == (*L.shape, self.trajectory_group.n_tracks)
            assert jumps_per_track.shape[0] == self.trajectory_group.n_tracks
            assert (jumps_per_track >= 1.0).all()

            # Test Likelihood.exp
            exp_log_L = L.exp(log_L)
            assert exp_log_L.shape == (*L.shape, self.trajectory_group.n_tracks)
