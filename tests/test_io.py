import os, unittest, pandas as pd, numpy as np
from saspt.constants import TRACK, FRAME, PY, PX

from saspt import io as sio

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(TEST_DIR, "fixtures")

class TestTracksLoading(unittest.TestCase):
    def setUp(self):
        # Example set of trajectories
        self.sample_tracks = pd.DataFrame({
            TRACK:  [    0,  -1,   1,   0,   2,   4,   2,   2],
            FRAME:  [    0,   0,   0,   1,   1,   2,   2,   3],
            PY:     [  0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            PX:     [  0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        })

        # Save two identical copies to files for test
        self.sample_tracks_csv = os.path.join(TEST_DIR, "_test_io_sample_tracks.csv")
        self.sample_tracks_csv_2 = os.path.join(TEST_DIR, "_test_io_sample_tracks_2.csv")
        for fn in [self.sample_tracks_csv, self.sample_tracks_csv_2]:
            self.sample_tracks.to_csv(fn, index=False)

    def tearDown(self):
        for fn in [self.sample_tracks_csv, self.sample_tracks_csv_2]:
            if os.path.isfile(fn):
                os.remove(fn)

    def assert_tracks_equal(self, tracks_0, tracks_1, columns=[TRACK, FRAME, PY, PX],
        dtypes=[int, int, float, float]):
        assert all(map(lambda t: isinstance(t, pd.DataFrame), [tracks_0, tracks_1]))
        for col, dtype in zip(columns, dtypes):
            assert (tracks_0[col].astype(dtype) == tracks_1[col].astype(dtype)).all()

    def test_load_tracks_from_file(self):
        S = sio.load_detections_from_file(self.sample_tracks_csv)
        assert isinstance(S, pd.DataFrame)
        self.assert_tracks_equal(S, self.sample_tracks)
        self.assertRaises(ValueError, lambda: sio.load_detections_from_file("not_a_supported_extension.what"))

    def test_load_tracks(self):
        tracks = sio.load_detections(
            self.sample_tracks_csv,
            self.sample_tracks_csv_2
        )
        assert len(tracks) == len(self.sample_tracks) * 2, len(tracks)

        # Correctly increments trajectory indices
        assert (tracks[TRACK].astype(int) == \
            pd.Series([0, -1, 1, 0, 2, 4, 2, 2, 5, -1, 6, 5, 7, 9, 7, 7], dtype=int)
        ).all()

        # All other information is correctly preserved
        for col in [FRAME, PY, PX]:
            assert (np.asarray(tracks[:len(self.sample_tracks)][col]) \
                == self.sample_tracks[col]).all()
            assert (np.asarray(tracks[len(self.sample_tracks):][col]) \
                == self.sample_tracks[col]).all()

    def test_is_tracks(self):
        assert sio.is_detections(self.sample_tracks)
        for col in [TRACK, FRAME, PY, PX]:
            S = self.sample_tracks.copy().drop(col, axis=1)
            assert ~sio.is_detections(S)

    def test_empty_tracks(self):
        assert sio.is_detections(sio.empty_detections())

    def test_concat_tracks(self):
        T0, T1, T2 = map(lambda i: self.sample_tracks.copy(), range(3))

        # No trajectories passed
        assert sio.is_detections(sio.concat_detections())

        # One set of trajectories passed
        T = sio.concat_detections(T0)
        self.assert_tracks_equal(T, T0)
        
        # Multiple sets of trajectories passed
        T = sio.concat_detections(T0, T1, T2)
        assert len(T) == len(T0) * 3
        for j in range(3):
            T_slice = T[j*len(T0):(j+1)*len(T0)].copy().reset_index(drop=True)
            self.assert_tracks_equal(T_slice, T0, columns=[FRAME, PY, PX],
                dtypes=[int, float, float])
        assert (T[TRACK].astype(int) == pd.Series(
            [0, -1, 1, 0, 2, 4, 2, 2, 5, -1, 6, 5, 7, 9, 7, 7, 10, -1, 11, 10, 12, 14, 12, 12],
            dtype=int
        )).all()

        # First set of trajectories is all unassigned
        T0[TRACK] = -1
        T = sio.concat_detections(T0, T1, T2)
        assert len(T) == len(T0) * 3
        for j in range(3):
            T_slice = T[j*len(T0):(j+1)*len(T0)].copy().reset_index(drop=True)
            self.assert_tracks_equal(T_slice, T0, columns=[FRAME, PY, PX],
                dtypes=[int, float, float])
        assert (T[TRACK].astype(int) == pd.Series(
            [-1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 1, 0, 2, 4, 2, 2, 5, -1, 6, 5, 7, 9, 7, 7],
            dtype=int
        )).all()
       
        # All sets of trajectories are unassigned
        T1[TRACK] = -1; T2[TRACK] = -1
        T = sio.concat_detections(T0, T1, T2)
        assert (T[TRACK].astype(int) == -1).all()
        for j in range(3):
            T_slice = T[j*len(T0):(j+1)*len(T0)].copy().reset_index(drop=True)
            self.assert_tracks_equal(T_slice, T0, columns=[FRAME, PY, PX],
                dtypes=[int, float, float])
       
