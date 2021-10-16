Utilities
=========

.. py:function:: saspt.utils.track_length(detections: pandas.DataFrame) -> pandas.DataFrame

    Add a new column to a detection-level DataFrame with the length of each 
    trajectory in frames.

    Example:

    .. code-block:: python

        >>> from saspt import sample_detections
        >>> detections = sample_detections()
        >>> from saspt.utils import track_length

        # Calculate length of each trajectory in frames
        >>> detections = track_length(detections)
        >>> print(detections)
                      y           x  frame  trajectory  track_length
        0    575.730202   84.828673      0       13319           247
        1    538.416604  485.924667      0        1562            13
        2    107.647631   61.892363      0         363             3
        3    151.893969   63.246361      0         992             8
        4    538.737277  485.856905      1        1562            13
        ..          ...         ...    ...         ...           ...
        491  365.801274   70.689108    296       14458             7
        492  409.236744   10.312949    296       14375             1
        493  366.475688   70.559735    297       14458             7
        494  363.350134   67.585339    298       14458             7
        495  360.006572   70.511980    299       14458             7

        [496 rows x 5 columns]

        # All detections in each trajectory have the same track length
        >>> print(detections.groupby('trajectory')['track_length'].nunique())
        trajectory
        363      1
        413      1
        439      1
        542      1
        580      1
                ..
        14174    1
        14324    1
        14360    1
        14375    1
        14458    1
        Name: track_length, Length: 100, dtype: int64

        # Mean trajectory length is ~5 frames
        >>> print(detections.groupby('trajectory')['track_length'].first().mean())
        4.96

    :param pandas.DataFrame detections:

    :return: *pandas.DataFrame*, the input DataFrame with a new column `track_length` (``saspt.constants.TRACK_LENGTH``)

.. py:function:: saspt.utils.assign_index_in_track(detections: pandas.DataFrame) -> pandas.DataFrame
    
    Given a set of detections, determine the index of each detection in
    its respective trajectory.

    Sorts the input.

    :param pandas.DataFrame detections: input set of detections

    :return: *pandas.DataFrame*, input with the *index_in_track* column

.. py:function:: saspt.utils.cartesian_product(*arrays: numpy.ndarray) -> numpy.ndarray

    Take the Cartesian product of multiple 1D arrays.

    :param numpy.ndarray arrays: one or more 1D arrays

    :return: **product** (*numpy.ndarray*), 2D array. Each corresponds to a unique combination of the elements of *arrays*.