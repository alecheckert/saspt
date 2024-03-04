import os, unittest, numpy as np, pandas as pd

from saspt.constants import RBME
from saspt.lik import make_likelihood
from saspt.parameters import StateArrayParameters
from saspt.trajectory_group import TrajectoryGroup
from saspt.sa import StateArray
from saspt.dataset import StateArrayDataset, DEFAULT_CONDITION_COL

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(TEST_DIR, "fixtures")

class TestStateArrayDataset(unittest.TestCase):
    def setUp(self):
        # A couple of small SPT files
        self.paths = list(map(lambda p: os.path.join(FIXTURES, p), [
            "small_tracks_0.csv", "small_tracks_1.csv",
            "small_tracks_2.csv", "small_tracks_3.csv",
        ]))

        # A "paths" DataFrame appropriate for constructing a 
        # StateArrayDataset
        self.path_col = "filepath"
        self.condition_col = "condition"
        self.paths = pd.DataFrame({
            self.path_col:      self.paths,
            self.condition_col: ["A", "B", "A", "B"],
        })

        # Some parameters for the state array
        params = StateArrayParameters(
            pixel_size_um=0.16,
            frame_interval=0.01,
            focal_depth=0.7,
            splitsize=10,
            sample_size=1000,
            start_frame=0,
            max_iter=10,
            conc_param=1.0,
            progress_bar=False,
            num_workers=2,
        )
        self.params = params

        # RBME likelihood function
        self.likelihood = make_likelihood(
            likelihood_type=RBME, 
            pixel_size_um=params.pixel_size_um,
            frame_interval=params.frame_interval,
            focal_depth=params.focal_depth,
        )

    def check_plot_func(self, func, out_png, **kwargs):
        """ Check that a plotting function correctly produces an output file """
        if os.path.isfile(out_png):
            os.remove(out_png)
        func(out_png=out_png, **kwargs)
        assert os.path.isfile(out_png)
        os.remove(out_png)

    def test_basic(self):
        """ Basic test on sanitary input """
        D = StateArrayDataset(self.paths, self.likelihood,
            params=self.params, path_col=self.path_col,
            condition_col=self.condition_col)
        assert D.n_files == len(self.paths)

        # True number of nonsinglet trajectories per file
        expected_n_tracks = {
            "small_tracks_0.csv": 39,
            "small_tracks_1.csv": 43,
            "small_tracks_2.csv": 39,
            "small_tracks_3.csv": 38,
        }

        # Test StateArrayDataset._load_tracks on a single file
        for p in self.paths[self.path_col]:
            T = D._load_tracks(p)
            assert isinstance(T, TrajectoryGroup)
            assert T.n_tracks == expected_n_tracks[os.path.basename(p)]

            # Test StateArrayDataset._init_state_array
            SA = D._init_state_array(p)
            assert isinstance(SA, StateArray)
            assert SA.n_tracks == expected_n_tracks[os.path.basename(p)]

        # Test StateArrayDataset._load_tracks on multiple files
        T = D._load_tracks(*self.paths[self.path_col].tolist())
        assert isinstance(T, TrajectoryGroup)
        assert T.n_tracks == sum(expected_n_tracks.values())

    def test_marginal_naive_occs(self):
        """ Test validity of the StateArrayDataset.marginal_naive_occs
        property """
        D = StateArrayDataset(self.paths, self.likelihood,
            params=self.params, path_col=self.path_col,
            condition_col=self.condition_col)
        ML = D.marginal_naive_occs
        assert isinstance(ML, np.ndarray)
        assert ML.shape == (len(self.paths), len(self.likelihood.diff_coefs))
        assert (np.abs(ML.sum(axis=1) - D.jumps_per_file) < 1.0e-6).all()

        # Make sure StateArrayDataset.clear works
        D.clear()
        assert ~hasattr(D, "_marginal_naive_occs")

    def test_marginal_posterior_occs(self):
        """ Test validity of the StateArrayDataset.marginal_posterior_occs
        property """
        D = StateArrayDataset(self.paths, self.likelihood,
            params=self.params, path_col=self.path_col,
            condition_col=self.condition_col)
        occs = D.marginal_posterior_occs 
        assert isinstance(occs, np.ndarray)
        assert occs.shape == (len(self.paths), len(self.likelihood.diff_coefs))
        assert (np.abs(occs.sum(axis=1) - D.jumps_per_file) < 1.0e-6).all()

    def test_empty(self):
        """ Initialize a StateArrayDataset from empty input (no paths) """
        D = StateArrayDataset(self.paths[:0].copy(), self.likelihood,
            params=self.params, path_col=self.path_col,
            condition_col=self.condition_col)
        assert D.n_files == 0, D.n_files

        ML = D.marginal_naive_occs
        assert ML.shape == (0, len(self.likelihood.diff_coefs)), ML.shape

        for df in [D.processed_track_statistics, D.raw_track_statistics]:
            assert len(df) == 0
            assert D.path_col in df.columns 
            assert D.condition_col in df.columns
            assert (df[D.path_col] == D.paths[D.path_col]).all()
            assert (df[D.condition_col] == D.paths[D.condition_col]).all()

        assert D.marginal_posterior_occs_dataframe.empty
        assert D.raw_track_statistics.empty
        assert D.processed_track_statistics.empty

    def test_no_condition(self):
        """ Initialize a StateArrayDataset with no condition column """
        D = StateArrayDataset(self.paths, self.likelihood,
            params=self.params, path_col=self.path_col, condition_col=None)
        assert D.n_files == len(self.paths)
        assert D.paths[DEFAULT_CONDITION_COL].nunique() == 1

    def test_apply_by(self):
        paths = self.paths.copy()
        col = "some_filewise_condition"
        paths[col] = np.arange(len(paths)) % 3

        D = StateArrayDataset(paths, self.likelihood, params=self.params,
            path_col=self.path_col, condition_col=None)

        # Apply a simple function (len) by condition
        result, conditions = D.apply_by(col, len, is_variadic=False)
        assert len(conditions) == 3
        assert all(map(lambda x: x in conditions, paths[col].unique()))
        for i, c in zip(result, conditions):
            assert i == (paths[col] == c).sum()

        # Should raise a ValueError if grouping by a nonexistent column
        self.assertRaises(ValueError, lambda: D.apply_by("not_a_column",
            len, is_variadic=False))

    def test_infer_posterior_by_condition(self):
        D = StateArrayDataset.from_kwargs(self.paths, likelihood_type=RBME,
            pixel_size_um=0.16, frame_interval=0.00748, focal_depth=0.7, 
            path_col=self.path_col, condition_col=None, max_iter=1,
            sample_size=10, num_workers=2)

        result, conditions = D.infer_posterior_by_condition('condition', normalize=True)
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 2
        assert result.shape[0] == self.paths['condition'].nunique()
        assert result.shape[1] == D.n_diff_coefs
        assert (np.abs(result.sum(axis=1) - 1.0) < 1.0e-6).all()

    def test_marginal_posterior_occs_dataframe(self):
        with StateArrayDataset(self.paths, self.likelihood,
            params=self.params, path_col=self.path_col,
            condition_col=self.condition_col) as D:

            df = D.marginal_posterior_occs_dataframe 

            # Occupations are normalized
            assert (np.abs(df.groupby(self.path_col)["naive_occupation"].sum() - 1.0) < 1.0e-6).all()
            assert (np.abs(df.groupby(self.path_col)["posterior_occupation"].sum() - 1.0) < 1.0e-6).all()

            # Diffusion coefficients are correct
            diff_coefs = self.likelihood.diff_coefs
            for filepath, df_file in df.groupby(self.path_col):
                assert (np.abs(df_file['diff_coef'] - diff_coefs) < 1.0e-6).all()

            # Naive occupations are correct
            for i, (filepath, df_file) in enumerate(df.groupby(self.path_col)):
                occs_norm = D.marginal_naive_occs[i,:] / D.marginal_naive_occs[i,:].sum()
                assert (np.abs(df_file["naive_occupation"] - occs_norm) < 1.0e-6).all()

            # Posterior occupations are correct
            for i, (filepath, df_file) in enumerate(df.groupby(self.path_col)):
                occs_norm = D.marginal_posterior_occs[i,:] / D.marginal_posterior_occs[i,:].sum()
                assert (np.abs(df_file["posterior_occupation"] - occs_norm) < 1.0e-6).all()

    def test_posterior_heat_map(self):
        D = StateArrayDataset(self.paths, self.likelihood,
            params=self.params, path_col=self.path_col,
            condition_col=self.condition_col)
        self.check_plot_func(D.posterior_heat_map,
            "_out_test_posterior_heat_map.png")

    def test_posterior_line_plot(self):
        D = StateArrayDataset(self.paths, self.likelihood,
            params=self.params, path_col=self.path_col,
            condition_col=self.condition_col)
        self.check_plot_func(D.posterior_line_plot,
            "_out_test_posterior_line_plot.png")
    
    def test_subsampling(self):
        # New params with a smaller sample size
        sample_size = 10
        params = StateArrayParameters(
            pixel_size_um=0.16,
            frame_interval=0.01,
            focal_depth=0.7,
            splitsize=10,
            sample_size=sample_size,
            start_frame=0,
            max_iter=10,
            conc_param=1.0,
            progress_bar=False,
            num_workers=2,
        )
        self.params = params
        D = StateArrayDataset(self.paths, self.likelihood,
            params=self.params, path_col=self.path_col,
            condition_col=self.condition_col)
        
        # Check that jumps_per_file and implied jumps are correct
        assert np.allclose(D.jumps_per_file.astype(float), D.posterior_occs.sum(axis=(1,2)))
        assert np.allclose(D.jumps_per_file.astype(float), D.naive_occs.sum(axis=(1,2)))

        # Check that subsampling actually worked
        n_trajs = D.processed_track_statistics['n_tracks']
        assert (n_trajs <= sample_size).all()
