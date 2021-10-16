#!/usr/bin/env python
"""
utils.py
"""
import os, warnings, numpy as np, pandas as pd 
from glob import glob 
from typing import List
from itertools import accumulate, chain, repeat
from .constants import FRAME, TRACK, TRACK_LENGTH, PY, PX

######################
## TRACKS UTILITIES ##
######################


def track_length(tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Add a new column to a trajectory dataframe with the trajectory
    length in frames.

    args
    ----
        tracks      :   pandas.DataFrame. Must have the column 
                        *TRACK*.

    returns
    -------
        pandas.DataFrame, with the "track_length" column. Overwritten
            if it already exists.

    """
    if TRACK_LENGTH in tracks.columns:
        tracks = tracks.drop(TRACK_LENGTH, axis=1)
    return tracks.join(tracks.groupby(TRACK).size().rename(TRACK_LENGTH), on=TRACK)


def assign_index_in_track(tracks):
    """
    Given a set of trajectories, determine the index of each localization in the
    context of its respective trajectory.

    args
    ----
        tracks      :   pandas.DataFrame, containing the "trajectory" and "frame"
                        columns

    returns
    -------
        pandas.DataFrame, the same dataframe with a new column, "index_in_track"

    """
    tracks = tracks.sort_values(by=[TRACK, FRAME])
    tracks["one"] =  1
    tracks["index_in_track"] = tracks.groupby(TRACK)["one"].cumsum() - 1
    tracks = tracks.drop("one", axis=1)
    return tracks 


def concat_tracks(*tracks):
    """
    Join some trajectory dataframes together into a larger dataframe,
    while preserving uniqe trajectory indices.

    args
    ----
        tracks      :   pandas.DataFrame with the "trajectory" column

    returns
    -------
        pandas.DataFrame, the concatenated trajectories

    """
    n = len(tracks)

    # Sort the tracks dataframes by their size. The only important thing
    # here is that if at least one of the tracks dataframes is nonempty,
    # we need to put that one first.
    df_lens = [len(t) for t in tracks]
    try:
        tracks = [t for _, t in sorted(zip(df_lens, tracks))][::-1]
    except ValueError:
        pass

    # Iteratively concatenate each dataframe to the first while 
    # incrementing the trajectory index as necessary
    out = tracks[0].assign(dataframe_index=0)
    c_idx = out[TRACK].max() + 1

    for t in range(1, n):

        # Get the next set of trajectories and keep track of the origin
        # dataframe
        new = tracks[t].assign(dataframe_index=t)

        # Ignore negative trajectory indices (facilitating a user filter)
        new.loc[new["trajectory"]>=0, "trajectory"] += c_idx 

        # Increment the total number of trajectories
        c_idx = new["trajectory"].max() + 1

        # Concatenate
        out = pd.concat([out, new], ignore_index=True, sort=False)

    return out


#############################
## TRACK LOADING UTILITIES ##
#############################


def load_tracks(*csv_paths, out_csv=None, start_frame=0,
    drop_singlets=False, suffix=".csv"):
    """
    Given a set of trajectories stored as CSVs, concatenate all
    of them, storing the paths to the original CSVs in the resulting
    dataframe, and optionally save the result to another CSV.
    If passed a directory instead of a set of CSV paths, find all 
    the CSVs in that directory that end with *suffix*, load the 
    trajectories, and concatenate them.

    args
    ----
        csv_paths       :   list of str, a set of trajectory CSVs.
                            Each must contain the "y", "x", "trajectory",
                            and "frame" columns
        out_csv         :   str, path to save to 
        start_frame     :   int, exclude any trajectories that begin before
                            this frame
        drop_singlets   :   bool, drop singlet localizations before
                            concatenating
        suffix          ;   str, suffix of CSVs if passing a directory

    returns
    -------
        pandas.DataFrame, the concatenated result

    """
    n = len(csv_paths)
    if n == 0:
        warnings.warn("no paths passed")
        return pd.DataFrame([], columns=["trajectory", "frame", "y", "x"], dtype=object)

    # If passed a directory instead of a set of file paths, just load all 
    # the CSVs from that directory
    if os.path.isdir(csv_paths[0]):
        return load_tracks_dir(csv_paths[0], start_frame=start_frame,
            drop_singlets=drop_singlets, suffix=suffix)

    if start_frame is None:
        start_frame = 0

    def drop_before_start_frame(tracks, start_frame):
        """
        Drop all trajectories that start before a specific frame.
        """
        if tracks.empty or (start_frame is None) or (start_frame <= tracks["frame"].min()):
            return tracks 

        tracks = tracks.join(
            (tracks.groupby("trajectory")["frame"].first() >= start_frame).rename("_take"),
            on="trajectory"
        )

        tracks = tracks[tracks["_take"]]
        tracks = tracks.drop("_take", axis=1)

        return tracks

    def drop_singlets_dataframe(tracks):
        """
        Drop all singlets and unassigned localizations from a 
        pandas.DataFrame with trajectory information.
        """
        if tracks.empty:
            return tracks 

        tracks = track_length(tracks)
        tracks = tracks[np.logical_and(tracks["track_length"]>1,
            tracks["trajectory"]>=0)]

        return tracks 

    def loader(path):
        tracks = pd.read_csv(path)

        if drop_singlets:
            tracks = drop_singlets_dataframe(tracks)

        tracks = drop_before_start_frame(tracks, start_frame)

        return tracks 

    # Load the trajectories into memory
    tracks = []
    for path in csv_paths:
        tracks.append(loader(path))

    # Concatenate 
    tracks = concat_tracks(*tracks)

    # Map the original path back to each file
    for i, path in enumerate(csv_paths):
        tracks.loc[tracks["dataframe_index"]==i, "source_file"] = \
            os.path.abspath(path)

    # Optionally save concatenated trajectories to a new CSV
    if not out_csv is None:
        tracks.to_csv(out_csv, index=False)

    return tracks 


def load_tracks_dir(dirname, suffix=".csv", start_frame=0,
    drop_singlets=False):
    """
    Load all of the trajectory CSVs in a target directory
    into a single pandas.DataFrame.

    args
    ----
        dirname         :   str, directory with the track CSVs
        suffix          :   str, extension for the track CSVs
        start_frame     :   int, exclude all tracks before this
                            frame
        drop_singlets   :   bool, don't include single-point 
                            trajectories
    returns
    -------
        pandas.DataFrame with an extra column, "origin_file",
            with the path to the CSV from which these 
            trajectories were taken

    """
    # Find target files
    if os.path.isdir(dirname):
        target_csvs = glob(os.path.join(dirname, "*{}".format(suffix)))
        if len(target_csvs) == 0:
            raise IOError("Could not find trajectory CSVs in directory {}".format(dirname))
    elif os.path.isfile(dirname):
        target_csvs = [dirname]

    # Concatenate trajectories
    tracks = [pd.read_csv(j) for j in target_csvs]
    tracks = concat_tracks(*tracks)

    # Exclude points before the start frame
    if isinstance(start_frame, int) and \
        (start_frame > tracks["frame"].min()) and \
        (not tracks.empty):

        tracks = tracks.join(
            (tracks.groupby("trajectory")["frame"].first() >= start_frame).rename("_take"),
            on="trajectory"
        )
        tracks = tracks[tracks["_take"]] 
        tracks = tracks.drop("_take", axis=1)

    # Exclude trajectories that are too short
    tracks = track_length(tracks)
    if drop_singlets:
        tracks = tracks[tracks["track_length"] > 1]

    return tracks 


####################
## JUMP COMPUTERS ##
####################


def tracks_to_jumps(tracks, n_frames=1, start_frame=None, pixel_size_um=0.16,
    pos_cols=["y", "x"]):
    """
    Convert trajectories in pandas.DataFrame format to an internal "jumps"
    format, specified in the *returns* section of this docstring.

    args
    ----
        tracks          :   pandas.DataFrame
        n_frames        :   int, the number of frames over which to compute
                            the jump. For instance, if n_frames = 1, then only
                            compute jumps between consecutive frames
        start_frame     :   int, disregard jumps before this frame
        pixel_size_um   :   float, size of pixels in microns
        pos_cols        :   list of str, the columns with the spatial 
                            coordinates of each point in pixels

    returns
    -------
        *jumps*, a 2D ndarray of shape (n_jumps, 6+). Each row corresponds
            to a single jump from the dataset. 
        The columns of *vecs* have the following meaning:
            jumps[:,0] -> length of the origin trajectory in frames
            jumps[:,1] -> index of the origin trajectory in *tracks*
            jumps[:,2] -> frame corresponding to the first point in 
                            the jump
            jumps[:,3] -> sum of squared jumps across all spatial
                            dimensions in squared microns
            jumps[:,4:] -> jumps in each Euclidean dimension in microns

    """
    def bail():
        return np.zeros((0, 6), dtype=np.float64)

    # If passed an empty dataframe, bail
    if tracks.empty: return bail()

    # Do not modify the original dataframe
    tracks = tracks.copy()

    # Calculate the original trajectory length and exclude
    # singlets and negative trajectory indices
    tracks = track_length(tracks)
    tracks = tracks[np.logical_and(
        tracks["trajectory"] >= 0,
        tracks["track_length"] > 1
    )]

    # Only consider trajectories after some start frame
    if not start_frame is None:
        tracks = tracks[tracks["frame"] >= start_frame]

    # If no trajectories remain, bail
    if tracks.empty: return bail()

    # Convert from pixels to um
    tracks[pos_cols] *= pixel_size_um 

    # Work with an ndarray, for speed
    tracks = tracks.sort_values(by=["trajectory", "frame"])
    T = np.asarray(tracks[["track_length", "trajectory", "frame", pos_cols[0]] + pos_cols])

    # Allowing for gaps, consider every possible comparison that 
    # leads to the correct frame interval
    target_jumps = []
    for j in range(1, n_frames+1):
        
        # Compute jumps
        jumps = T[j:,:] - T[:-j,:]

        # Only consider vectors between points originating 
        # from the same trajectory and from the target frame
        # interval
        same_track = jumps[:,1] == 0
        target_interval = jumps[:,2] == n_frames 
        take = np.logical_and(same_track, target_interval)

        # Map the corresponding track lengths, track indices,
        # and frame indices back to each jump
        jumps[:,:3] = T[:-j,:3]
        jumps = jumps[take, :]

        # Calculate the corresponding 2D squared jump and accumulate
        if jumps.shape[0] > 0:
            jumps[:,3] = (jumps[:,4:]**2).sum(axis=1)
            target_jumps.append(jumps)

    # Concatenate
    if len(target_jumps) > 0:
        return np.concatenate(target_jumps, axis=0)
    else:
        return bail()


def sum_squared_jumps(jumps, max_jumps_per_track=None, pos_cols=["y", "x"]):
    """
    For each trajectory in a dataset, calculate the sum of its squared
    jumps across all spatial dimensions.
    args
    ----
        jumps           :   2D ndarray, all jumps in the dataset as 
                            calculated by *tracks_to_jumps*
        max_jumps_per_track :   int, the maximum number of jumps 
                            to consider from any single trajectory
    returns
    -------
        pandas.DataFrame. Each row corresponds to a trajectory, with
            the following columns:
            "sum_sq_jump": the summed squared jumps of that trajectory
                           in microns
            "trajectory" : the index of the origin trajectory
            "frame"      : the first frame of the first jumps in the 
                           origin trajectory
            "n_jumps"    : the number of jumps used in *sum_sq_jump*
    """
    out_cols = ["sum_sq_jump", "trajectory", "frame", "n_jumps"]

    # If there are no jumps in this set of trajectories, bail
    if jumps.shape[0] == 0:
        return pd.DataFrame(index=[], columns=out_cols, dtype=object)

    # Format as a dataframe, indexed by jump
    cols = ["track_length", "trajectory", "frame", "sq_jump"] + list(pos_cols)
    jumps = pd.DataFrame(jumps, columns=cols)
    n_tracks = jumps["trajectory"].nunique()

    # Limit the number of jumps to consider per trajectory, if desired
    if not max_jumps_per_track is None:
        jumps = assign_index_in_track(jumps)
        tracks = jumps[jumps["index_in_track"] <= max_jumps_per_track]

    # Output dataframe, indexed by trajectory
    sum_jumps = pd.DataFrame(index=np.arange(n_tracks), columns=out_cols, dtype=object)

    # Calculate the sum of squared jumps for each trajectory
    sum_jumps["sum_sq_jump"] = np.asarray(jumps.groupby("trajectory")["sq_jump"].sum())

    # Calculate the number of jumps in each trajectory
    sum_jumps["n_jumps"] = np.asarray(jumps.groupby("trajectory").size())

    # Map back the indices of the origin trajectories
    sum_jumps["trajectory"] = np.asarray(jumps.groupby("trajectory").apply(lambda i: i.name)).astype(np.int64)

    # Map back the frame indices
    sum_jumps["frame"] = np.asarray(jumps.groupby("trajectory")["frame"].first()).astype(np.int64)

    return sum_jumps


def split_jumps(jumps, splitsize=8):
    """
    Split a set of long trajectories into shorter trajectories.

    Example 1
    ---------
        If we have a trajectory of 6 jumps and 
        splitsize = 3, then we split this trajectory into two 
        trajectories of 3 jumps, comprising the first and second halves
        of the original trajectory.

    Example 2
    ---------
        If we have a trajectory of 10 jumps and splitsize = 4,
        then we split this trajectory into 3 trajectories. The 
        first two are 4 jumps each, and the third is the last 2
        jumps of the original trajectory.

    args
    ----
        jumps           :   2D ndarray, a set of trajectory-indexed
                            jumps; output of *tracks_to_jumps*
        splitsize       :   int, the maximum size of a trajectory 
                            after splitting

    returns
    -------
        1D ndarray of shape (n_tracks), the indices of the 
            new trajectories. These start from 0 and go to the 
            highest new trajectory index; numerically they have 
            no relation to the original trajectory indices.
            
    """
    # If passed empty input, return empty output
    if jumps.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)

    # The original set of trajectory indices
    orig_indices = jumps[:,1].astype(np.int64)

    # The set of modified trajectory indices
    new_indices = np.zeros(orig_indices.shape[0], dtype=np.int64)

    # The current (new) trajectory index 
    c = 0

    # The length of the current trajectory in # of jumps
    L = 0

    # Iterate through the original set of trajectory indices
    prev_index = orig_indices[0]
    for i, index in enumerate(orig_indices):

        # Extend the existing trajectory
        L += 1

        # We're in the same original trajectory
        if index == prev_index:

            # Haven't exceeded the split trajectory size limit
            if L < splitsize:
                new_indices[i] = c 

            # Break into a new trajectory
            else:
                L = 0
                c += 1
                new_indices[i] = c

        # We've passed into a different original trajectory
        else:
            prev_index = index 
            L = 0
            c += 1
            new_indices[i] = c 

    return new_indices

#####################
## OTHER UTILITIES ##
#####################

def normalize_2d(arr: np.ndarray, axis: int):
    """ Normalize a 2D array over one of its axes.

    args
    ----
        arr         :   2D numpy.ndarray
        axis        :   int, axis to normalize over

    returns
    -------
        *arr* such as that arr.sum(axis=axis) == 1.0
    """
    if len(arr.shape) != 2:
        raise ValueError(f"arr has shape {arr.shape}; expected 2D")

    if axis == 1:
        S = arr.sum(axis=1)
        nonzero = S > 0
        arr[nonzero,:] = (arr[nonzero,:].T / S[nonzero]).T
    else:
        S = arr.sum(axis=0)
        nonzero = S > 0
        arr[:,nonzero] = arr[:,nonzero] / S[nonzero]
    return arr

def cartesian_product(*arrays: np.ndarray):
    """ Cartesian product of multiple 1D numpy.ndarrays.

    Source: https://stackoverflow.com/a/45378609

    args
    ----
        arrays  :   1D numpy.ndarray

    returns
    -------

    """
    la = len(arrays)
    L = *map(len, arrays), la
    dtype = np.result_type(*arrays)
    arr = np.empty(L, dtype=dtype)
    arrs = *accumulate(chain((arr,), repeat(0, la-1)), np.ndarray.__getitem__),
    idx = slice(None), *repeat(None, la-1)
    for i in range(la-1, 0, -1):
        arrs[i][..., i] = arrays[i][idx[:la-i]]
        arrs[i-1][1:] = arrs[i]
    arr[..., 0] = arrays[0][idx]
    return arr.reshape(-1, la)
