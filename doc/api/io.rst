File reading
============

.. py:function:: saspt.io.is_detections(df: pandas.DataFrame) -> bool

    Determine whether a `panda.DataFrame` is recognized as a viable set of SPT
    detections by the rest of `saSPT`. In particular, it must contain the following
    four columns:

        * **frame** (``saspt.constants.FRAME``): frame index for this detection
        * **trajectory** (``saspt.constants.TRACK``): trajectory index for this detection
        * **x** (``saspt.constants.PX``): position of the detection in *pixels*
        * **y** (``saspt.constants.PY``): position of the detection in *pixels*

    Example:

    .. code-block:: python

        >>> import pandas as pd
        >>> from saspt.constants import FRAME, TRACK, PY, PX
        >>> from saspt.io import is_detections
        >>> print(is_detections(pd.DataFrame(index=[], columns=[FRAME, TRACK, PY, PX], dtype=object)))
        True
        >>> print(is_detections(pd.DataFrame(index=[], columns=[FRAME, PY, PX], dtype=object)))
        False


    :param pandas.DataFrame df: each row corresponding to a detection

    :return: *bool*

.. py:function:: saspt.io.load_detections_from_file(filepath: str) -> pandas.DataFrame

    Load detections from a file in one of the currently recognized formats.

    At the moment, `saSPT` only recognizes a single file format for trajectories:
    a CSV where each row corresponds to a detection and the columns contain at 
    minimum **frame**, **trajectory**, **x**, and **y**.

    :param str filepath: path to the file containing detections

    :return: **detections** (*pandas.DataFrame*), the set of detections

.. py:function:: saspt.load_detections(*filepaths: str) -> pandas.DataFrame

    Load detections from one or more files and concatenate into a single
    `pandas.DataFrame`. Increments trajectory indices, so that indices between
    detections from different files do not collide.

    Example (using some files from the `saSPT` repo):

    .. code-block:: python

        >>> from saspt.io import load_detections
        >>> detections = load_detections(
        ...     'tests/fixtures/small_tracks_0.csv',
        ...     'tests/fixtures/small_tracks_1.csv'
        ... )
        >>> print(detections)
             trajectory  ...  dataframe_idx
        0             0  ...              0
        1             0  ...              0
        2             0  ...              0
        3             1  ...              0
        4             1  ...              0
        ..          ...  ...            ...
        449         107  ...              1
        450         107  ...              1
        451         107  ...              1
        452         107  ...              1
        453         107  ...              1

        [454 rows x 6 columns]

    :param str filepaths: one or more paths to files containing detections. Must be in a format recognized by `saspt.io.load_detections_from_file`.

    :return: **detections** (*pandas.DataFrame*), indexed by detection

.. py:function:: saspt.io.empty_detections() -> pandas.DataFrame

    Return an empty set of detections. Useful mostly for tests.

    :return: **empty_detections** (*pandas.DataFrame*)

.. py:function:: saspt.io.sample_detections() -> pandas.DataFrame

    Return a small, simple set of detections. Useful for illustrations and quick
    demos, especially in these docs.

    .. code-block:: python

        >>> from saspt import sample_detections
        >>> detections = sample_detections()
        >>> print(detections)
                      y           x  frame  trajectory
        0    575.730202   84.828673      0       13319
        1    538.416604  485.924667      0        1562
        2    107.647631   61.892363      0         363
        3    151.893969   63.246361      0         992
        4    538.737277  485.856905      1        1562
        ..          ...         ...    ...         ...
        491  365.801274   70.689108    296       14458
        492  409.236744   10.312949    296       14375
        493  366.475688   70.559735    297       14458
        494  363.350134   67.585339    298       14458
        495  360.006572   70.511980    299       14458

        [496 rows x 4 columns]

    :return: *pandas.DataFrame* with columns *frame*, *trajectory*, *y*, and *x*

.. py:function:: saspt.io.concat_detections(*detections: pandas.DataFrame) -> pandas.DataFrame

    Concatenate multiple detection DataFrames while incrementing trajectory indices
    to prevent index collisions.

    :param pandas.DataFrame detections: one or more DataFrames containing detections

    :return: **concatenated_detections** (*pandas.DataFrame*)