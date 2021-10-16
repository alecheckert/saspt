import os, numpy as np, pandas as pd 
from typing import List
from functools import reduce 

from .io import load_detections
from .utils import track_length
from .constants import (
    TRACK, FRAME, PY, PX, TRACK_LENGTH,
    DFRAMES, DY, DX, DR2, JUMPS_PER_TRACK,
    DEFAULT_MAX_ITER,
    DEFAULT_SPLITSIZE,
    DEFAULT_CONC_PARAM,
    DEFAULT_START_FRAME,
)
from .parameters import StateArrayParameters

class TrajectoryGroup:
    """ Encapsulates a set of trajectories to be analyzed by a StateArray,
    caching attributes required for downstream steps.

    init
    ----
        detections      :   pandas.DataFrame indexed by detection with at
                            minimum the columns TRACK, FRAME, PY, and PX,
                            representing a set of trajectories

        pixel_size_um   :   size of pixels in microns

        frame_interval  :   time between camera frames in seconds

        splitsize       :   maximum number of jumps per trajectory. Trajectories
                            longer than this are split into smaller pieces.

    """
    # Column in TrajectoryGroup.detections encoding the original trajectory
    # index before preprocessing
    ORIG_TRACK = "orig_trajectory"

    # Column in TrajectoryGroup.detections encoding the original detection
    # index before preprocessing
    DETECT_INDEX = "detect_index"

    # Names of statistics in TrajectoryGroup.raw_track_statistics and 
    # TrajectoryGroup.processed_track_statistics
    statistic_names = ['n_tracks', 'n_jumps', 'n_detections',
        'mean_track_length', 'max_track_length', 'fraction_singlets',
        'fraction_unassigned', 'mean_jumps_per_track', 'mean_detections_per_frame',
        'max_detections_per_frame', 'fraction_of_frames_with_detections']

    def __init__(self, detections: pd.DataFrame, pixel_size_um: float,
        frame_interval: float, splitsize: int=DEFAULT_SPLITSIZE,
        start_frame: int=DEFAULT_START_FRAME, **kwargs):

        # Store a copy of the _unpreprocessed_ detections. This is useful for
        # calculating dataset statistics and localization density.
        self._raw_detections = detections[[TRACK, FRAME, PY, PX]].sort_values(
            by=[TRACK, FRAME]).reset_index(drop=True)
        self._raw_detections = track_length(self._raw_detections)

        # Preprocess trajectories for state array operations. This includes removing
        # singlets, breaking long trajectories into pieces of maximum size *splitsize*,
        # and reindexing for contiguous trajectory indices
        self.detections = self.preprocess(detections, splitsize, start_frame)

        # Some dataset parameters
        self.pixel_size_um = pixel_size_um
        self.frame_interval = frame_interval
        self.splitsize = splitsize
        self.start_frame = start_frame

        # We cannot work with input that contains multiple detections in the same
        # frame assigned to the same trajectory
        if len(self.detections) > 0:
            G = self.detections.loc[self.detections[TRACK] >= 0].groupby(FRAME)
            if (np.asarray(G[TRACK].nunique()) != np.asarray(G.size())).all():
                raise ValueError("*detections* cannot contain multiple detections in the " \
                    "same frame that are assigned to the same trajectory")

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, traceback):
        return etype is None

    def __repr__(self):
        return "TrajectoryGroup:\n{}".format("\n".join([
            f"{attr}:\t{getattr(self, attr)}" for attr in ["n_detections", "n_jumps", "n_tracks"]
        ]))

    @classmethod
    def from_params(cls, detections: pd.DataFrame, params: StateArrayParameters):
        """ Alternative constructor; instantiate a TrajectoryGroup using a
        StateArrayParameters object """
        kwargs = {k: getattr(params, k) for k in 
            ["pixel_size_um", "frame_interval", "splitsize", "start_frame"]}
        return cls(detections, **kwargs)

    @classmethod
    def from_files(cls, detection_files: List[str], **kwargs):
        """ Alternative constructor; instantiate a TrajectoryGroup directly
        from files containing detections """
        return cls(load_detections(*detection_files), **kwargs)

    @classmethod
    def preprocess(cls, detections: pd.DataFrame, splitsize: int=DEFAULT_SPLITSIZE,
        start_frame: int=DEFAULT_START_FRAME):
        """ Preprocess some trajectories for analysis with state arrays.

        This involves three steps:
            1. We remove all singlets, unassigned detections, and detections
                before *start_frame*.

            2. Long trajectories are split into pieces with a maximum
                size of *splitsize*.

            3. Trajectories are reindexed so that trajectory indices are 
                contiguous, starting from 0 to n_tracks-1.

        args
        ----
            detections      :   pandas.DataFrame indexed by detection. Must
                                contain the columns FRAME, TRACK, PY, and PX.

            splitsize       :   maximum length of trajectories in jumps

            start_frame     :   disregard detections before this frame

        returns
        -------
            pandas.DataFrame, processed trajectories. The old trajectory
                index for each detection is recorded under the
                *TrajectoryGroup.ORIG_TRACK* column, while the new
                trajectory index is under the *TRACK* column.
        """
        # Assign each detection a unique index, to keep track of them after sorting
        detections[cls.DETECT_INDEX] = np.arange(len(detections), dtype=np.int64)

        # Discard singlets, unassigned detections, and detections before the start frame
        detections = track_length(detections)
        detections = detections[reduce(np.logical_and, [
            detections[TRACK] >= 0,
            detections[TRACK_LENGTH] > 1,
            detections[FRAME] >= start_frame
        ])].sort_values(by=[TRACK, FRAME]).reset_index(drop=True)

        # Break trajectories longer than *splitsize* into smaller pieces
        orig_indices = np.array(detections[TRACK])
        detections[cls.ORIG_TRACK] = orig_indices
        detections[TRACK] = cls.split_tracks(orig_indices, splitsize)

        # Drop singlets generated by the splitting process
        detections = track_length(detections)
        detections = detections[detections[TRACK_LENGTH] > 1].reset_index(drop=True)

        # Reindex so that trajectory index is contiguous
        M = {t: i for i, t in enumerate(detections[TRACK].unique())}
        detections[TRACK] = detections[TRACK].map(M)

        return detections

    @classmethod
    def split_tracks(cls, track_indices: np.ndarray, splitsize: int) -> np.ndarray:
        """ Split trajectories into smaller pieces that have a maximum of *splitsize*
        jumps. 

        args
        ----
            track_indices       :   1D np.ndarray with the current trajectory
                                    index of each detection
            splitsize           :   int, maximum number of jumps to tolerate

        returns
        -------
            1D np.ndarray, dtype int64, the new set of trajectory indices
        """
        if track_indices.shape[0] > 1 and \
            ((track_indices[1:]-track_indices[:-1])<0).any():
            raise ValueError("track_indices must be monotonically increasing")
        if track_indices.shape[0] == 0:
            return np.array([]).astype(np.int64)
        new_indices = np.zeros(len(track_indices), dtype=np.int64)
        c = 0
        L = 0
        prev_index = track_indices[0]
        for i, index in enumerate(track_indices):
            if index == prev_index and L > splitsize:
                L = 0
                c += 1
            elif index != prev_index:
                prev_index = index
                L = 0
                c += 1
            new_indices[i] = c
            L += 1
        return new_indices 

    ################
    ## PROPERTIES ##
    ################

    @property 
    def n_detections(self) -> int:
        if not hasattr(self, "_n_detections"):
            self._n_detections = len(self.detections)
        return self._n_detections

    @property 
    def n_jumps(self) -> int:
        if not hasattr(self, "_n_jumps"):
            self._n_jumps = len(self.jumps)
        return self._n_jumps

    @property 
    def n_tracks(self) -> int:
        if not hasattr(self, "_n_tracks"):
            self._n_tracks = self.detections[TRACK].nunique()
        return self._n_tracks

    @property 
    def jumps(self) -> pd.DataFrame:
        if not hasattr(self, "_jumps"):
            T = np.asarray(self.detections[[FRAME, FRAME, TRACK, PY, PX, PX]])
            T[:,3:5] *= self.pixel_size_um
            if T.shape[0] < 2:
                self._jumps = pd.DataFrame({FRAME: np.zeros(0, dtype=np.int64),
                    DFRAMES: np.zeros(0, dtype=np.int64), TRACK: np.zeros(0, dtype=np.int64),
                    DY: np.zeros(0, dtype=np.float64), DX: np.zeros(0, dtype=np.float64),
                    DR2: np.zeros(0, dtype=np.float64), JUMPS_PER_TRACK: np.zeros(0, dtype=np.int64)})
            else:
                # Diff
                D = T[1:,:] - T[:-1,:]
                # Map back trajectory and frame indices
                D[:,0] = T[:-1,0]
                # Detections that originate from the same trajectory
                same_track = D[:,2] == 0
                D[:,2] = T[:-1,2]
                D = D[same_track,:]
                # Calculate 2D radial jumps
                D[:,5] = (D[:,3:5]**2).sum(axis=1) / D[:,1]
                # Format as pandas.DataFrame
                self._jumps = pd.DataFrame(D, columns=[FRAME, DFRAMES, TRACK, DY, DX, DR2])
                # Enforce some types
                for c in [FRAME, DFRAMES, TRACK]:
                    self._jumps[c] = self._jumps[c].astype(np.int64)
                # Calculate number of jumps per track
                self._jumps = self._jumps.join(
                    self._jumps.groupby(TRACK).size().rename(JUMPS_PER_TRACK),
                    on=TRACK)
        return self._jumps

    @property
    def jumps_per_track(self) -> np.ndarray:
        """ The number of jumps per trajectory.

        returns
        -------
            1D numpy.ndarray of shape (n_tracks,), the number of jumps in each
                trajectory
        """
        if not hasattr(self, "_jumps_per_track"):
            self._jumps_per_track = np.array(self.jumps.groupby(TRACK)[JUMPS_PER_TRACK].first())
        return self._jumps_per_track

    @property 
    def raw_track_statistics(self) -> dict:
        """ Return some statistics on the raw unprocessed trajectories that 
        are wrapped by this TrajectoryGroup object.

        returns
        -------
            dict keyed by statistic name
        """
        if not hasattr(self, "_raw_track_statistics"):
            # Assigned trajectories
            T = self._raw_detections.loc[self._raw_detections[TRACK] >= 0]
            n_frames = 1 + self._raw_detections[FRAME].max() - self._raw_detections[FRAME].min()
            self._raw_track_statistics = {
                "n_tracks": T[TRACK].nunique(),
                "n_jumps": (T.groupby(TRACK).size() - 1).sum(),
                "n_detections": len(self._raw_detections),
                "mean_track_length": T.groupby(TRACK)[TRACK_LENGTH].first().mean(),
                "max_track_length": T.groupby(TRACK)[TRACK_LENGTH].first().max() \
                    if len(T)>0 else 0,
                "fraction_singlets": (T.groupby(TRACK)[TRACK_LENGTH].first() \
                    == 1).mean(),
                "fraction_unassigned": (self._raw_detections[TRACK] < 0).mean(),
                "mean_jumps_per_track": (T.groupby(TRACK).size() - 1).mean(),
                "mean_detections_per_frame": len(self._raw_detections) / n_frames,
                "max_detections_per_frame": self._raw_detections.groupby(FRAME).size().max() \
                    if len(self._raw_detections)>0 else 0,
                "fraction_of_frames_with_detections": self._raw_detections[FRAME].nunique() \
                    / n_frames if len(self._raw_detections)>0 else 0.0,
            }
        return self._raw_track_statistics

    @property 
    def processed_track_statistics(self) -> dict:
        """ Return some statistics on the preprocessed trajectories (i.e. after
        calling self.preprocess).

        self.preprocess does three things:
            - Removes singlets and detections unassigned to any trajectory
            - Splits big trajectories into smaller pieces (max size *splitsize*)
            - Reindexes trajectories for contiguous trajectory indices

        returns
        -------
            dict keyed by statistic name
        """
        if not hasattr(self, "_processed_track_statistics"):
            n_frames = 1 + self.detections[FRAME].max() - self.start_frame
            self._processed_track_statistics = {
                "n_tracks": self.n_tracks,
                "n_jumps": self.n_jumps,
                "n_detections": self.n_detections,
                "mean_track_length": self.detections.groupby(TRACK)[TRACK_LENGTH].first().mean(),
                "max_track_length": self.detections.groupby(TRACK)[TRACK_LENGTH].first().max() \
                    if len(self.detections)>0 else 0,
                "fraction_singlets": (self.detections.groupby(TRACK)[TRACK_LENGTH].first() \
                    == 1).mean(),
                "fraction_unassigned": (self.detections[TRACK] < 0).mean(),
                "mean_jumps_per_track": self.jumps.groupby(TRACK).size().mean(),
                "mean_detections_per_frame": len(self.detections) / n_frames,
                "max_detections_per_frame": self.detections.groupby(FRAME).size().max() \
                    if len(self.detections)>0 else 0,
                "fraction_of_frames_with_detections": self.detections[FRAME].nunique() \
                    / n_frames if len(self.detections)>0 else 0.0,
            }
        return self._processed_track_statistics

    #############
    ## METHODS ##
    #############

    def get_track_vectors(self, n: int) -> np.ndarray:
        """ Return the vectorial jumps corresponding to all trajectories with 
        exactly *n* jumps.

        args
        ----
            n       :   number of jumps per track

        returns
        -------
            (
                3D np.ndarray of shape (M, 2, n), the Y and X jumps
                    of each trajectory (where *M* is the number of trajectories
                    that have *n* jumps);

                1D np.ndarray of shape (M,), the trajectory indices of each
                    trajectory
            )
        """
        take = self.jumps[JUMPS_PER_TRACK] == n
        track_indices = np.array(self.jumps.loc[take, TRACK].unique())
        M = take.sum() // n if n>0 else 0
        V = np.empty((M, 2, n), dtype=np.float64)
        V[:,0,:] = self.jumps.loc[take, DY].to_numpy(copy=True).reshape((M, n))
        V[:,1,:] = self.jumps.loc[take, DX].to_numpy(copy=True).reshape((M, n))
        return V, track_indices

    def subsample(self, size: int):
        """ Generate another TrajectoryGroup object by subsampling the trajectories
        in the present object without replacement.

        args
        ----
            size        :   number of trajectories to sample

        returns
        -------
            new instance of TrajectryGroup
        """
        if size > self.n_tracks:
            size = self.n_tracks
        detections = self.detections[self.detections[TRACK].isin(
            np.random.choice(np.arange(self.n_tracks), size=size, replace=False)
        )].reset_index(drop=True)
        return TrajectoryGroup(detections, pixel_size_um=self.pixel_size_um,
            frame_interval=self.frame_interval, splitsize=self.splitsize, 
            start_frame=self.start_frame)
