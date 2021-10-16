import os, warnings, numpy as np, pandas as pd
from functools import lru_cache
from .constants import FRAME, TRACK, PY, PX

def is_detections(df: pd.DataFrame) -> bool:
    """ Return True if the input pandas.DataFrame is a valid input 
    to the saspt module, meaning that it contains the *FRAME, 
    *TRACK*, *PY*, and *PX* columns.

    returns
    -------
        True if *df* is a valid set of trajectories, False otherwise.
    """
    return isinstance(df, pd.DataFrame) and \
        all(map(lambda c: c in df.columns, [FRAME, TRACK, PY, PX]))

def load_detections_from_file(filepath: str) -> pd.DataFrame:
    """ Read a set of trajectories from a file using any currently
    supported format.

    Currently, the only supported file format is a CSV indexed by 
    detection with columns *TRACK*, *FRAME*, *PY*, and *PX*.

    args
    ----
        filepath        :   path to the detections file

    returns
    -------
        pandas.DataFrame, set of detections
    """
    ext = os.path.splitext(filepath)[-1]
    if ext == ".csv":
        return pd.read_csv(filepath)
    else:
        raise ValueError(f"extension {ext} not recognized")

def load_detections(*filepaths: str) -> pd.DataFrame:
    """ Load detections from one or more files into a single
    pandas.DataFrame.

    args
    ----
        filepaths   :   one or more detection files

    returns
    -------
        pandas.DataFrame indexed by detection
    """
    def load(filepath: str) -> pd.DataFrame:
        return load_detections_from_file(filepath).assign(filepath=filepath)
    return concat_detections(*map(load, filepaths))

@lru_cache(1)
def empty_detections() -> pd.DataFrame:
    """ Return an empty set of detections. """
    return pd.DataFrame({
        TRACK: pd.Series([], dtype=np.int64, name=TRACK),
        FRAME: pd.Series([], dtype=np.int64, name=FRAME),
        PY: pd.Series([], dtype=np.float64, name=PY),
        PX: pd.Series([], dtype=np.float64, name=PX),
    })

def sample_detections() -> pd.DataFrame:
    """ Return a simple set of trajectories to be used for illustrations;
    e.g. in the documentation.

    returns
    -------
        pandas.DataFrame with columns FRAME, TRACK, PY, and PX
    """
    package_dir = os.path.split(os.path.abspath(__file__))[0]
    path = os.path.join(package_dir, "samples", "sample_tracks.csv")
    return load_detections_from_file(path)

def concat_detections(*detections: pd.DataFrame) -> pd.DataFrame:
    """ Join some detection DataFrames together into a larger DataFrame
    while preserving unique trajectory indices.

    args
    ----
        detections :   one or pandas.DataFrames with the *TRACK* column

    returns
    -------
        pandas.DataFrame, concatenated detections
    """
    if len(detections) == 0:
        return empty_detections()

    detections = [t.assign(dataframe_idx=i) for i, t in \
        enumerate(detections) if not t.empty]
    inc = 0
    for i, t in enumerate(detections):
        if (t[TRACK] >= 0).any():
            detections[i].loc[detections[i][TRACK] >= 0, TRACK] += inc
            inc = detections[i][TRACK].max()+1
    return pd.concat(detections, axis=0, ignore_index=True, sort=False) \
        if len(detections)>0 else pd.DataFrame(dtype=object)
