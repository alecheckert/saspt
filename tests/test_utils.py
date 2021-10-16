#!/usr/bin/env python

import os, unittest, numpy as np, pandas as pd

# Test targets
from saspt import utils

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(TEST_DIR, "fixtures")

class TestTrackUtilities(unittest.TestCase):
    def setUp(self):
        self.trajs = pd.DataFrame({
            'trajectory': [0, 1, 1, 10, 2, 2, 1, 3, 3],
            'frame':      [4, 2, 3,  2, 7, 8, 4, 6, 5], 
            'y':          [0, 1, 2, 3., 4, 5, 6, 7, 8],
            'x':          [0, 1, 2, 3., 4, 5, 6, 7, 8],
        })

        # Save some sample files
        self.test_loaddir = os.path.join(TEST_DIR, "_test_load_tracks")
        if not os.path.isdir(self.test_loaddir):
            os.makedirs(self.test_loaddir)
        self.out_paths = [
            os.path.join(self.test_loaddir, "_test_0_trajs.csv"),
            os.path.join(self.test_loaddir, "_test_1_trajs.csv"),
            os.path.join(self.test_loaddir, "_test_2_trajs.csv"),
        ]
        for o in self.out_paths:
            self.trajs.to_csv(o, index=False)


    def tearDown(self):
        if os.path.isdir(self.test_loaddir):
            for f in os.listdir(self.test_loaddir):
                os.remove(os.path.join(self.test_loaddir, f))
            os.rmdir(self.test_loaddir)


    def test_track_length(self):
        T = utils.track_length(self.trajs)
        assert "track_length" in T.columns, T.columns
        assert (T.groupby("trajectory")["track_length"].nunique() == 1).all(), T["track_length"]
        assert (T.groupby("trajectory")["track_length"].first() == \
            pd.Series(np.array([1, 3, 2, 2, 1]), index=[0, 1, 2, 3, 10])).all()


    def test_track_length_empty(self):
        T = utils.track_length(self.trajs[:0])
        assert isinstance(T, pd.DataFrame), type(T)
        assert T.empty, T


    def test_assign_index_in_track(self):
        T = utils.assign_index_in_track(self.trajs)
        assert "index_in_track" in T.columns, T.columns
        assert (T.groupby("trajectory")["index_in_track"].max() == (T.groupby("trajectory").size()-1)).all(), T 
        assert T["index_in_track"].sum() == 5, T 


    def test_concat_tracks(self):
        trajs_2 = self.trajs.copy()
        trajs_concat = utils.concat_tracks(self.trajs, trajs_2)
        assert len(trajs_concat) == len(self.trajs) + len(trajs_2), len(trajs_concat)
        assert trajs_concat['trajectory'].nunique() == self.trajs['trajectory'].nunique() * 2, trajs_concat


    def test_load_tracks(self):

        # Can load from multiple files
        trajs = utils.load_tracks(*self.out_paths)
        assert len(trajs) == len(self.trajs) * len(self.out_paths), trajs 
        assert trajs['trajectory'].nunique() == self.trajs['trajectory'].nunique() * len(self.out_paths), trajs 

        # Can load from a directory
        trajs_2 = utils.load_tracks(self.test_loaddir)
        assert len(trajs) == len(trajs_2), trajs_2
        assert trajs['trajectory'].sum() == trajs_2['trajectory'].sum(), trajs_2 

        # Can restrict to trajectories after a specific frame
        trajs = utils.load_tracks(*self.out_paths, start_frame=1)
        assert (trajs['frame'] > 0).all(), trajs

        # Can drop singlets
        trajs = utils.load_tracks(*self.out_paths, drop_singlets=True)
        assert (trajs.groupby("trajectory").size() > 1).all(), trajs 


    def test_tracks_to_jumps(self):
        jumps = utils.tracks_to_jumps(self.trajs, n_frames=1, start_frame=None, pixel_size_um=1.0)

        for track_idx, track in self.trajs.groupby("trajectory"):

            # Correct trajectory length
            assert (jumps[jumps[:,1] == track_idx, 0] == len(track)).all(), jumps 

            # Correct starting frame
            assert jumps[jumps[:,1] == track_idx, 2].sum() == sum(sorted(track['frame'].tolist())[:-1])

            # Correct squared jumps
            track = track.copy().sort_values(by='frame')
            if len(track) > 1:
                sum_sq_jumps = np.nansum(np.asarray(track[['y', 'x']].diff())**2)
                assert abs(jumps[jumps[:,1] == track_idx, 3].sum() - sum_sq_jumps) < 1.0e-8


    def test_sum_squared_jumps(self):
        jumps = utils.tracks_to_jumps(self.trajs, n_frames=1, start_frame=None, pixel_size_um=1.0)
        sum_jumps = utils.sum_squared_jumps(jumps, max_jumps_per_track=None)

        sum_jumps = sum_jumps.set_index("trajectory")

        for track_idx, track in self.trajs.groupby("trajectory"):

            if len(track) == 1:
                assert not track_idx in sum_jumps.index, track 
            else:
                assert sum_jumps.loc[track_idx, "n_jumps"] == len(track)-1, sum_jumps 
                sum_sq_jumps = np.nansum(np.asarray(track[['y', 'x']].diff())**2)
                assert sum_jumps.loc[track_idx, "sum_sq_jump"] == sum_sq_jumps, sum_jumps 


    def test_split_jumps(self):
        jumps = utils.tracks_to_jumps(self.trajs, n_frames=1, start_frame=None, pixel_size_um=1.0)
        orig_track_indices = jumps[:,1].copy()
        new_track_indices = utils.split_jumps(jumps, splitsize=1)
        assert len(np.unique(new_track_indices)) == jumps.shape[0], new_track_indices 
        new_track_indices = utils.split_jumps(jumps, splitsize=2)
        for t in np.unique(new_track_indices):
            assert (new_track_indices == t).sum() <= 2, new_track_indices

class TestArrayUtils(unittest.TestCase):
    def test_normalize_2d(self):
        np.random.seed(66666)
        A = np.random.random(size=(100, 6))
        for axis in [0, 1]:
            B = utils.normalize_2d(A, axis=axis)
            assert (np.abs(B.sum(axis=axis) - 1.0) <= 1.0e-6).all()

        # Tolerates empty axes
        A = np.random.random(size=(100, 0))
        for axis in [0, 1]:
            B = utils.normalize_2d(A, axis=axis)
            assert B.shape == A.shape

        A = np.random.random(size=(0, 6))
        for axis in [0, 1]:
            B = utils.normalize_2d(A, axis=axis)
            assert B.shape == A.shape

    def test_cartesian_product(self):
        A = np.array([1, 2, 3])
        B = np.array([4, 5])
        C = np.array([10, 11, 12])
        D = np.array([7])

        # Zero arrays (should raise ValueError)
        self.assertRaises(ValueError, lambda: utils.cartesian_product())

        # One array
        result = utils.cartesian_product(A)
        assert (result[:,0] == A).all()

        # Two arrays
        result = utils.cartesian_product(A, B)
        assert (result == np.array([
            [1, 4], [1, 5], [2, 4], [2, 5], [3, 4], [3, 5]
        ])).all()

        # Three arrays
        result = utils.cartesian_product(A, B, C)
        c = 0
        for Ai in A:
            for Bi in B:
                for Ci in C:
                    assert (result[c,:] == np.array([Ai, Bi, Ci])).all()
                    c += 1

        # Four arrays, one of which is a single element
        result = utils.cartesian_product(A, B, C, D)
        c = 0
        for Ai in A:
            for Bi in B:
                for Ci in C:
                    for Di in D:
                        assert (result[c,:] == np.array([Ai, Bi, Ci, Di])).all()
                        c += 1

        # Tolerates empty arrays
        result = utils.cartesian_product(np.array([]))
        assert result.shape == (0, 1)
        result = utils.cartesian_product(A, np.array([]))
        assert result.shape == (0, 2)
