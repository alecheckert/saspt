import os, unittest, numpy as np, pandas as pd

from saspt.sa import StateArray
from saspt.lik import make_likelihood, Likelihood
from saspt.trajectory_group import TrajectoryGroup
from saspt.parameters import StateArrayParameters
from saspt.constants import RBME, TRACK, FRAME, PY, PX

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(TEST_DIR, "fixtures")

class TestStateArray(unittest.TestCase):
    def setUp(self):
        # Master set of parameters
        self.params = StateArrayParameters(
            pixel_size_um=0.16,
            frame_interval=0.01,
            focal_depth=0.7,
            splitsize=10,
            sample_size=100,
            start_frame=0,
            max_iter=10,
            conc_param=1.0,
            progress_bar=False,
        )
        
        # Simple set of trajectories
        self.sample_tracks = pd.DataFrame({
            TRACK:  [     0,    1,    1,   -1,    3,    3,    3,    4,    4],
            FRAME:  [     0,    0,    1,    1,    1,    2,    3,    6,    7],
            PY:     [   0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8],
            PX:     [   0.8,  0.7,  0.6,  0.5,  0.4,  0.3,  0.2,  0.1,  0.0]
        })        

        # Trajectory group for these trajectories
        self.trajectories = TrajectoryGroup.from_params(
            self.sample_tracks, self.params)

        # Test likelihood function
        self.likelihood = make_likelihood(
            likelihood_type=RBME,
            pixel_size_um=self.params.pixel_size_um,
            frame_interval=self.params.frame_interval,
            focal_depth=self.params.focal_depth
        )

    def tearDown(self):
        pass

    def check_shapes(self, SA: StateArray, trajectories: TrajectoryGroup,
        likelihood: Likelihood):
        """ Check that the shapes of the StateArray are consistent with the
        TrajectoryGroup and Likelihood it is derived from """
        assert SA.n_tracks == trajectories.n_tracks
        assert SA.n_jumps == trajectories.n_jumps
        assert SA.n_detections == trajectories.n_detections
        assert SA.shape == likelihood.shape
        assert SA.n_states == np.prod(likelihood.shape)
        assert (SA.jumps_per_track == trajectories.jumps_per_track).all()
        assert SA.naive_assignment_probabilities.shape == (*SA.shape, SA.n_tracks)
        assert SA.prior_dirichlet_param.shape == SA.shape
        assert SA.prior_occs.shape == SA.shape
        assert SA.posterior_assignment_probabilities.shape == (*SA.shape, SA.n_tracks)
        assert SA.posterior_dirichlet_param.shape == SA.shape
        assert SA.posterior_occs.shape == SA.shape
        assert SA.prior_occs.shape == SA.shape

    def check_validity(self, SA: StateArray):
        """ Check the validity of StateArray output, including normalization
        of relevant distributions """
        for attr in [SA.posterior_dirichlet_param, SA.posterior_occs, SA.posterior_assignment_probabilities]:
            assert (~np.isinf(attr)).all()
            assert (~np.isnan(attr)).all()
        assert (np.abs(SA.prior_dirichlet_param - SA.params.conc_param) <= 1.0e-6).all()

        # Correct normalization
        assert (np.abs(SA.naive_assignment_probabilities.sum(axis=(0,1)) - 1.0) <= 1.0e-6).all()
        assert (np.abs(SA.posterior_assignment_probabilities.sum(axis=(0,1)) - 1.0) <= 1.0e-6).all()
        assert abs(SA.posterior_occs.sum() - 1.0) <= 1.0e-6
        assert abs(SA.prior_occs.sum() - 1.0) <= 1.0e-6

    def test_init(self):
        """ Test basic initialization on "normal" input, and check correctness
        of various underlying parameters """
        SA = StateArray(self.trajectories, self.likelihood, self.params)
        
        # Shape checks
        self.check_shapes(SA, self.trajectories, self.likelihood)

        # Validity checks
        self.check_validity(SA)

    def test_no_tracks(self):
        """ Test initialization and inference on input without any trajectories """
        trajectories = self.trajectories.subsample(0)
        assert trajectories.n_tracks == 0

        SA = StateArray(trajectories, self.likelihood, self.params)
        assert SA.n_tracks == 0
        
        # Shape checks
        self.check_shapes(SA, trajectories, self.likelihood)

        # Validity checks
        self.check_validity(SA)

    def test_empty_grid(self):
        """ Test initialization and inference on an empty state grid """
        diff_coefs = np.array([], dtype=np.float64)
        likelihood = make_likelihood(
            likelihood_type=RBME,
            pixel_size_um=self.params.pixel_size_um,
            frame_interval=self.params.frame_interval,
            focal_depth=self.params.focal_depth,
            diff_coefs=diff_coefs,
        )
        SA = StateArray(self.trajectories, likelihood, self.params)
        self.check_shapes(SA, self.trajectories, likelihood)

    def test_posterior_occs_dataframe(self):
        with StateArray(self.trajectories, self.likelihood, self.params) as SA:
            df = SA.posterior_occs_dataframe 
            naive_occs = SA.naive_occs 
            post_occs = SA.posterior_occs

            # Diffusion coefficients are correct
            D = np.array(df["diff_coef"]).reshape(SA.shape)
            for i in range(SA.shape[1]):
                assert (np.abs(D[:,i] - self.likelihood.diff_coefs) < 1.0e-6).all()

            # Localization errors are correct
            LE = np.array(df["loc_error"]).reshape(SA.shape)
            for i in range(SA.shape[0]):
                assert (np.abs(LE[i,:] - self.likelihood.loc_errors) < 1.0e-6).all()

            # Naive state occupations are correct
            NO = np.array(df["naive_occupation"]).reshape(SA.shape)
            assert (np.abs(NO - SA.naive_occs) < 1.0e-6).all()

            # Posterior state occupations are correct
            PO = np.array(df["mean_posterior_occupation"]).reshape(SA.shape)
            assert (np.abs(PO - SA.posterior_occs) < 1.0e-6).all()

            # Naive and posterior state occupations are normalized
            assert abs(NO.sum() - 1.0) < 1.0e-6
            assert abs(PO.sum() - 1.0) < 1.0e-6

        # Works with empty trajectories
        trajectories = self.trajectories.subsample(0)
        with StateArray(trajectories, self.likelihood, self.params) as SA:
            df = SA.posterior_occs_dataframe 

            # Diffusion coefficients are correct
            D = np.array(df["diff_coef"]).reshape(SA.shape)
            for i in range(SA.shape[1]):
                assert (np.abs(D[:,i] - self.likelihood.diff_coefs) < 1.0e-6).all()

            # Localization errors are correct
            LE = np.array(df["loc_error"]).reshape(SA.shape)
            for i in range(SA.shape[0]):
                assert (np.abs(LE[i,:] - self.likelihood.loc_errors) < 1.0e-6).all()           

        # Works with an empty state grid
        likelihood = make_likelihood(
            likelihood_type=RBME,
            pixel_size_um=self.params.pixel_size_um,
            frame_interval=self.params.frame_interval,
            focal_depth=self.params.focal_depth,
            diff_coefs = np.array([]),
        )
        with StateArray(self.trajectories, likelihood, self.params) as SA:
            df = SA.posterior_occs_dataframe 
            assert len(df) == 0
            assert all(map(lambda c: c in df.columns, [
                "diff_coef", "loc_error", "naive_occupation", "mean_posterior_occupation"
            ]))

    def check_plot_func(self, func, out_png, **kwargs):
        if os.path.isfile(out_png):
            os.remove(out_png)
        func(out_png=out_png, **kwargs)
        if os.path.isfile(out_png): 
            os.remove(out_png)

    def test_plot_posterior(self):
        """ Test StateArray.plot_posterior """
        # Normal input 
        SA = StateArray(self.trajectories, self.likelihood, self.params)
        self.check_plot_func(SA.plot_posterior, out_png="_test_out_0.png")

        # Empty input
        trajectories = self.trajectories.subsample(0)
        SA = StateArray(trajectories, self.likelihood, self.params)
        self.check_plot_func(SA.plot_posterior, out_png="_test_out_0.png")

    def test_plot_posterior_assignments(self):
        """ Test StateArray.plot_posterior_assignments """
        # Normal input 
        SA = StateArray(self.trajectories, self.likelihood, self.params)
        self.check_plot_func(SA.plot_posterior_assignments, out_png="_test_out_0.png")

        # Empty input
        trajectories = self.trajectories.subsample(0)
        SA = StateArray(trajectories, self.likelihood, self.params)
        self.check_plot_func(SA.plot_posterior_assignments, out_png="_test_out_0.png")
       
    def test_plot_temporal_posterior_assignments(self):
        """ Test StateArray.plot_temporal_posterior_assignments """
        # Normal input 
        SA = StateArray(self.trajectories, self.likelihood, self.params)
        self.check_plot_func(SA.plot_temporal_posterior_assignments, out_png="_test_out_0.png")

        # Empty input
        trajectories = self.trajectories.subsample(0)
        SA = StateArray(trajectories, self.likelihood, self.params)
        self.check_plot_func(SA.plot_temporal_posterior_assignments, out_png="_test_out_0.png")       

    def test_plot_spatial_posterior_assignments(self):
        """ Test StateArray.plot_spatial_posterior_assignments """ 
        # Normal input 
        SA = StateArray(self.trajectories, self.likelihood, self.params)
        self.check_plot_func(SA.plot_spatial_posterior_assignments, out_png="_test_out_0.png")

        # Empty input
        trajectories = self.trajectories.subsample(0)
        SA = StateArray(trajectories, self.likelihood, self.params)
        self.check_plot_func(SA.plot_spatial_posterior_assignments, out_png="_test_out_0.png")       

