import os, unittest, pandas as pd, numpy as np

from saspt.trajectory_group import TrajectoryGroup
from saspt.constants import TRACK, FRAME, PY, PX, TRACK_LENGTH, JUMPS_PER_TRACK, DFRAMES, DR2, DY, DX, RBME
from saspt.utils import track_length
from saspt.io import is_detections

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(TEST_DIR, "fixtures")

class TestTrajectoryGroup(unittest.TestCase):
    def setUp(self):
        # Simple set of trajectories
        self.sample_detections = pd.DataFrame({
            TRACK:  [     0,    1,    1,   -1,    3,    3,    3,    4,    4],
            FRAME:  [     0,    0,    1,    1,    1,    2,    3,    6,    7],
            PY:     [   0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8],
            PX:     [   0.8,  0.7,  0.6,  0.5,  0.4,  0.3,  0.2,  0.1,  0.0]
        })

        # More complex set of trajectories
        self.track_csv = os.path.join(FIXTURES, "sample_tracks.csv")

        # Sample set of TrajectoryGroup initialization kwargs
        self.init_kwargs = dict(pixel_size_um=0.160, frame_interval=0.00748,
            splitsize=10, start_frame=0)

    def tearDown(self):
        pass

    def test_split_tracks(self):
        # Test 1: Small set of trajectories with ground truth answer
        splitsize = 1
        old_indices = np.array([0, 0, 1, 3, 3, 3, 5, 5])
        new_indices = TrajectoryGroup.split_tracks(old_indices, splitsize)
        assert (new_indices == np.array([0, 0, 1, 2, 2, 3, 4, 4])).all(), new_indices

        # Test 2: Large set of trajectories
        splitsize = 4
        detections = pd.read_csv(self.track_csv).sort_values(by=[TRACK, FRAME]).reset_index(drop=True)
        old_indices = np.array(detections[TRACK])
        new_indices = TrajectoryGroup.split_tracks(old_indices, splitsize)
        assert (new_indices >= 0).all()

        # There are no "gaps" in trajectory indices
        T = np.unique(new_indices)
        T.sort()
        assert ((T[1:] - T[:-1]) == 1).all()

        # Each new trajectory contains detections from exactly one old trajectory
        T = pd.DataFrame({'old': old_indices, TRACK: new_indices, FRAME: detections[FRAME]})
        assert (T.groupby(TRACK)["old"].nunique() == 1).all()

        # No trajectories have more than *splitsize+1* detections
        assert (T.groupby(TRACK).size() <= splitsize+1).all()

        # Within each new trajectory, frame indices are monotonically increasing
        x = np.asarray(T[[TRACK, FRAME]])
        diff = x[1:,:] - x[:-1,:]
        assert (diff[diff[:,0]==0, 1] == 1).all()

        # Test 3: Empty set of detections
        new_indices = TrajectoryGroup.split_tracks(old_indices[:0], splitsize)
        assert isinstance(new_indices, np.ndarray)
        assert new_indices.shape[0] == 0

    def test_preprocess(self):
        # Large set of tracks
        detections = pd.read_csv(self.track_csv)

        # Preprocess
        splitsize = 4
        start_frame = 100
        processed = TrajectoryGroup.preprocess(detections.copy(), splitsize, start_frame)

        # Should not contain detections before the start frame
        assert (processed[FRAME] >= start_frame).all()

        # TRACK_LENGTH should be correct
        assert (processed[TRACK_LENGTH] == track_length(processed.copy())[TRACK_LENGTH]).all()

        # TRACK should be monotonically increasing
        assert processed[TRACK].is_monotonic_increasing

        # Should be no gaps in trajectory indices
        T = np.array(processed[TRACK].unique())
        T.sort()
        assert ((T[1:] - T[:-1]) == 1).all()

        # FRAME should be monotonically increasing in each trajectory
        T = np.asarray(processed[[TRACK, FRAME]])
        diff = T[1:,:] - T[:-1,:]
        assert (diff[diff[:,0]==0,1] == 1).all()

        # Should not contain unassigned detections or singlets
        assert (processed[TRACK] >= 0).all()
        assert (processed[TRACK_LENGTH] > 1).all()

        # The map to the original detections/trajectories should be correct
        DETECT_INDEX = TrajectoryGroup.DETECT_INDEX
        ORIG_TRACK = TrajectoryGroup.ORIG_TRACK
        for col in [PY, PX, FRAME]:
            assert (np.abs(
                processed[col] - processed[DETECT_INDEX].map(detections[col])
            ) <= 1.0e-6).all()
        assert (processed[ORIG_TRACK] == processed[DETECT_INDEX].map(detections[TRACK])).all()

        # Should be idempotent
        processed_again = TrajectoryGroup.preprocess(processed.copy(), splitsize, start_frame)
        assert len(processed_again) == len(processed)
        for col in [PY, PX, TRACK, FRAME]:
            assert (np.abs(processed_again[col] - processed[col]) <= 1.0e-6).all()

        # Should work on empty dataframes
        processed = TrajectoryGroup.preprocess(detections[:0], splitsize, start_frame)
        assert processed.empty
        assert all(map(lambda c: c in processed.columns, detections.columns))

        # Should work when all detections are before the start frame
        T = detections.copy()
        T[FRAME] -= 1000
        processed = TrajectoryGroup.preprocess(T, splitsize, start_frame)
        assert processed.empty
        assert all(map(lambda c: c in processed.columns, T.columns))

        # Should work on all singlets
        T = detections.groupby(TRACK, as_index=False).first()
        processed = TrajectoryGroup.preprocess(T, splitsize, start_frame)
        assert processed.empty
        assert all(map(lambda c: c in processed.columns, T.columns))

    def test_init(self):
        """ Test initialization of the TrajectoryGroup object. """
        T = TrajectoryGroup(pd.read_csv(self.track_csv), **self.init_kwargs)
        for k, v in self.init_kwargs.items():
            assert abs(getattr(T, k) - v) <= 1.0e-6, k

        # Set of trajectories is already preprocessed
        processed = TrajectoryGroup.preprocess(T.detections)
        assert len(processed) == len(T.detections)
        for col in [PY, PX, TRACK, FRAME]:
            assert (np.abs(processed[col] - T.detections[col]) <= 1.0e-6).all()

        # Attributes are correct
        assert T.n_detections == len(T.detections)
        assert T.n_jumps == len(T.jumps)
        assert T.n_tracks == T.detections[TRACK].nunique()

    def test_jumps(self):
        """ Check for correctness and internal consistency of TrajectoryGroup.jumps """
        T = TrajectoryGroup(pd.read_csv(self.track_csv), **self.init_kwargs)
        J = T.jumps 

        # There are no gaps in these trajectories...
        assert (J[DFRAMES] == 1).all()

        # ...so each trajectory should contribute a number of jumps equal
        # to its length in frames minus 1
        assert T.n_jumps == T.n_detections - T.n_tracks
        assert (J.groupby(TRACK).size() == T.detections.groupby(TRACK).size() - 1).all()
        assert (J.groupby(TRACK)[JUMPS_PER_TRACK].nunique() == 1).all()
        assert (J.groupby(TRACK)[JUMPS_PER_TRACK].first() == T.detections.groupby(TRACK).size() - 1).all()

        # Frame indices are correct
        assert (J.groupby(TRACK)[FRAME].sum() == \
            T.detections.groupby(TRACK)[FRAME].apply(lambda s: s[:-1].sum())).all()

        # These trajectories were tracked with search radius 1.0 micron, so DR2 should
        # never exceed the search radius squared
        assert (T.jumps[DR2] <= 1.0**2).all()

        # DR2 is correct, given these values for DY and DX
        assert (np.abs(T.jumps[DR2] - (T.jumps[DY]**2 + T.jumps[DX]**2)) <= 1.0e-6).all()

        # Checksums on DY, DX
        for p, d in zip([PY, PX], [DY, DX]):
            assert abs(
                np.nansum(T.detections.groupby(TRACK)[p].diff()) * self.init_kwargs.get('pixel_size_um') \
                - T.jumps[d].sum()
            ) <= 1.0e-6


    def test_empty(self):
        """ Test TrajectoryGroup instantiation on empty input """
        T = TrajectoryGroup(self.sample_detections[:0], **self.init_kwargs)
        assert T.n_detections == 0
        assert T.n_jumps == 0
        assert T.n_tracks == 0
        assert T.jumps.empty
        assert T.detections.empty
        assert is_detections(T.detections)
        assert all(map(lambda c: c in T.jumps.columns,
            [DFRAMES, TRACK, FRAME, DY, DX, DR2, JUMPS_PER_TRACK]))

    def test_singlets(self):
        """ Test TrajectoryGroup instantiation on a dataset consisting solely of singlets """
        tracks = self.sample_detections.groupby(TRACK, as_index=False).first()
        T = TrajectoryGroup(self.sample_detections[:0], **self.init_kwargs)
        assert T.n_detections == 0
        assert T.n_jumps == 0
        assert T.n_tracks == 0
        assert T.jumps.empty
        assert T.detections.empty
        assert is_detections(T.detections)
        assert all(map(lambda c: c in T.jumps.columns,
            [DFRAMES, TRACK, FRAME, DY, DX, DR2, JUMPS_PER_TRACK]))

    def test_get_track_vectors(self):
        T = TrajectoryGroup(pd.read_csv(self.track_csv), **self.init_kwargs)
        J = T.jumps 

        for n in range(1, self.init_kwargs.get("splitsize")+1):
            V, track_indices = T.get_track_vectors(n)
            assert isinstance(V, np.ndarray)
            assert isinstance(track_indices, np.ndarray)

            # Number of trajectories that should match
            assert V.shape[0] == J.loc[J[JUMPS_PER_TRACK] == n, TRACK].nunique()
            assert track_indices.shape[0] == V.shape[0]

            # Y/X index
            assert V.shape[1] == 2

            # Time index
            assert V.shape[2] == n

            # Track indices are correct (check that pandas.Series.unique preserves order)
            correct_track_indices = np.array(J.loc[J[JUMPS_PER_TRACK] == n].groupby(TRACK).apply(lambda j: j.name))
            assert (track_indices == correct_track_indices).all()

            # Track vectors are correct
            assert (np.abs(V[:,0,:].ravel() - np.asarray(J.loc[J[TRACK].isin(track_indices), DY])) <= 1.0e-6).all()
            assert (np.abs(V[:,1,:].ravel() - np.asarray(J.loc[J[TRACK].isin(track_indices), DX])) <= 1.0e-6).all()

        # Try with length-0 vectors
        V, track_indices = T.get_track_vectors(0)
        assert len(V.shape) == 3
        assert V.shape[0] == 0
        assert V.shape[1] == 2
        assert V.shape[2] == 0
        assert track_indices.shape[0] == 0

        # Try with jump vectors longer than the maximum tolerated number of jumps per track
        V, track_indices = T.get_track_vectors(self.init_kwargs.get("splitsize") + 1)
        assert len(V.shape) == 3
        assert V.shape[0] == 0
        assert V.shape[1] == 2
        assert V.shape[2] == self.init_kwargs.get("splitsize") + 1
        assert track_indices.shape[0] == 0

    def test_subsample(self):
        sample_size = 100
        T0 = TrajectoryGroup(pd.read_csv(self.track_csv), **self.init_kwargs)
        T1 = T0.subsample(sample_size)
        assert T1.n_tracks == sample_size 
        T2 = T0.subsample(0)
        assert T2.n_tracks == 0

    def test_track_statistics(self):
        # Small set of detections (test for correctness against ground truth)
        T = TrajectoryGroup(self.sample_detections, **self.init_kwargs)

        # Raw track statistics
        raw_stats = T.raw_track_statistics
        assert raw_stats.get("n_detections") == len(self.sample_detections), raw_stats
        assert raw_stats.get("n_jumps") == 4
        assert raw_stats.get("n_tracks") == 4
        assert abs(raw_stats.get("mean_track_length") - np.mean([1, 2, 3, 2])) < 1.0e-6
        assert raw_stats.get("max_track_length") == 3
        assert abs(raw_stats.get("fraction_singlets") - 0.25) < 1.0e-6, raw_stats
        assert abs(raw_stats.get("fraction_unassigned") - 1.0 / len(self.sample_detections)) < 1.0e-6, raw_stats
        assert abs(raw_stats.get("mean_jumps_per_track") - np.mean([0, 1, 2, 1])) < 1.0e-6
        assert abs(raw_stats.get("mean_detections_per_frame") - np.mean([2, 3, 1, 1, 0, 0, 1, 1])) < 1.0e-6
        assert raw_stats.get("max_detections_per_frame") == 3
        assert abs(raw_stats.get("fraction_of_frames_with_detections") - 0.75) < 1.0e-6

        # Processed track statistics
        processed_stats = T.processed_track_statistics
        assert processed_stats.get("n_detections") == 7
        assert processed_stats.get("n_jumps") == 4
        assert processed_stats.get("n_tracks") == 3
        assert abs(processed_stats.get("mean_track_length") - np.mean([2, 3, 2])) < 1.0e-6
        assert processed_stats.get("max_track_length") == 3
        assert abs(processed_stats.get("fraction_singlets") - 0.0) < 1.0e-6
        assert abs(processed_stats.get("fraction_unassigned") - 0.0) < 1.0e-6
        assert abs(processed_stats.get("mean_jumps_per_track") - np.mean([1, 2, 1])) < 1.0e-6
        assert abs(processed_stats.get("mean_detections_per_frame") - np.mean([1, 2, 1, 1, 0, 0, 1, 1])) < 1.0e-6
        assert processed_stats.get("max_detections_per_frame") == 2
        assert abs(processed_stats.get("fraction_of_frames_with_detections") - 0.75) < 1.0e-6

        # Works on empty dataframe
        T = TrajectoryGroup(self.sample_detections[:0], **self.init_kwargs)
        for stats in [T.processed_track_statistics, T.raw_track_statistics]:
            assert stats.get("n_detections") == 0
            assert stats.get("n_jumps") == 0
            assert stats.get("n_tracks") == 0
            assert pd.isnull(stats.get("mean_track_length"))
            assert stats.get("max_track_length") == 0
            assert pd.isnull(stats.get("fraction_singlets"))
            assert pd.isnull(stats.get("fraction_unassigned"))
            assert pd.isnull(stats.get("mean_jumps_per_track"))
            assert pd.isnull(stats.get("mean_detections_per_frame"))
            assert stats.get("max_detections_per_frame") == 0
            assert stats.get("fraction_of_frames_with_detections") == 0.0

        # Larger set of detections (check for stability)
        T = TrajectoryGroup(pd.read_csv(self.track_csv), **self.init_kwargs)
        for stats in [T.processed_track_statistics, T.raw_track_statistics]:
            for k, v in stats.items():
                assert ~pd.isnull(k)


