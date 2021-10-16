TrajectoryGroup
===============

.. py:currentmodule:: saspt

.. py:class:: TrajectoryGroup(detections: pandas.DataFrame, pixel_size_um: float, frame_interval: float)

    A set of trajectories to be analyzed with state arrays.
    TrajectoryGroup takes a raw set of trajectories
    produced by a tracking algorithm, such as `quot <https://github.com/alecheckert/quot>`_,
    and performs some preprocessing steps to facilitate downstream
    calculations with state arrays. These include:

        * remove all singlets (trajectories with a single detection), unassigned detections, and detections before an arbitrary start frame
        * split long trajectories into smaller pieces, to minimize the effects of state transitions and tracking errors
        * reindex trajectories so that the trajectory indices are contiguous between 0 and *n_tracks-1*

    The TrajectoryGroup object also provides some methods to get some
    general information about the set of trajectories via
    the ``raw_track_statistics`` and ``processed_track_statistics`` attributes.
    An example:

    .. code-block:: python

        >>> import numpy as np, pandas as pd
        >>> from saspt import TrajectoryGroup

        # Simple set of three detections belonging to two trajectories
        >>> detections = pd.DataFrame({
        ...     'frame': [0, 0, 1],
        ...     'trajectory': [0, 1, 0],
        ...     'y': [1.1, 2.2, 3.3],
        ...     'x': [3.3, 2.2., 1.1]
        ... })

        # Imaging parameters
        >>> kwargs = dict(pixel_size_um = 0.16, frame_interval = 0.00748)

        # Make a TrajectoryGroup with these detections
        >>> with TrajectoryGroup(detections, **kwargs) as TG:
        ...     print(TG)
        TrajectoryGroup:
        n_detections:   2
        n_jumps:    1
        n_tracks:   1

        # Show some information about the raw trajectories
        ...     print(TG.raw_track_statistics)
        {'n_tracks': 2, 'n_jumps': 1, 'n_detections': 3, 'mean_track_length': 1.5,
        'max_track_length': 2, 'fraction_singlets': 0.5, 'fraction_unassigned': 0.0,
        'mean_jumps_per_track': 0.5, 'mean_detections_per_frame': 1.5,
        'max_detections_per_frame': 2, 'fraction_of_frames_with_detections': 1.0}

        # Show some information about the trajectories after preprocessing
        ...    print(TG.processed_track_statistics)
        {'n_tracks': 1, 'n_jumps': 1, 'n_detections': 2, 'mean_track_length': 2.0,
        'max_track_length': 2, 'fraction_singlets': 0.0, 'fraction_unassigned': 0.0,
        'mean_jumps_per_track': 1.0, 'mean_detections_per_frame': 1.0,
        'max_detections_per_frame': 1, 'fraction_of_frames_with_detections': 1.0}

    In this example, notice that trajectory 1 only has a single detection. As
    a result, it is filtered out by the preprocessing step, since it contributes
    no dynamic information to the result.

    .. py:method:: __init__(self, detections: pandas.DataFrame, pixel_size_um: float, frame_interval: float, splitsize: int=DEFAULT_SPLITSIZE, start_frame: int=DEFAULT_START_FRAME)

        Default constructor for the TrajectoryGroup object.

        :param pandas.DataFrame detections: raw detections/trajectories produced by a tracking algorithm. Each row of the DataFrame represents a single detections. Must contain at minimum the following four columns:

            #. ``y``: the y-coordinate of the detection in pixels
            #. ``x``: the x-coordinate of the detection in pixels
            #. ``frame``: the frame index of the detection
            #. ``trajectory``: the index of the trajectory to which the detection has been assigned by the tracking algorithm

        :param float pixel_size_um: size of camera pixels after magnification in microns
        :param float frame_interval: time between frames in seconds
        :param int splitsize: maximum trajectory length (in # of jumps) to consider. Trajectories longer than *splitsize* are broken into smaller pieces.
        :param int start_frame: disregard detections before this frame. Useful for restrict analysis to the later, lower-density parts of an SPT movie.

    :return: **new_instance** (*TrajectoryGroup*)


    .. py:property:: n_detections
        :type: int

        Total number of detections after preprocessing.


    .. py:property:: n_jumps
        :type: int

        Total number of *jumps* (particle-particle links) after preprocessing.


    .. py:property:: n_tracks
        :type: int

        Total number of trajectories (sequences of detections connected by links) in this dataset.


    .. py:property:: jumps
        :type: pandas.DataFrame

        The set of all *jumps* (particle-particle links) in these
        trajectories. Each row corresponds to a single jump. Contains
        the following columns:

            * **frame** (``saspt.constants.FRAME``): frame index of the first detection participating in this jump
            * **dframes** (``saspt.constants.DFRAMES``): difference in frames between the second and first detection in this jump. For instance, if ``dframes == 1``, then the jump is a link between detections in subsequent frames.
            * **trajectory** (``saspt.constants.TRACK``): index of the trajectory to which the detections in this jump have been assigned by the tracking algorithm.
            * **dy** (``saspt.constants.DY``): jump distance in the *y*-dimension (in microns)
            * **dx** (``saspt.constants.DX``): jump distance in the *x*-dimension (in microns)
            * **dr2** (``saspt.constants.DR2``): mean squared jump distance in the *xy* plane. Equivalent to ``(dy**2 + dx**2) / dframes``.
            * **jumps_per_track** (``saspt.constants.JUMPS_PER_TRACK``): total number of jumps in the trajectory to which this jump belongs

        Example:

        .. code-block:: python

            # Simple set of detections belonging to 3 trajectories
            >>> detections = pd.DataFrame({
            ...    'trajectory': [0, 0, 0, 1, 1, 1, 2, 2],
            ...    'frame':      [0, 1, 2, 0, 1, 2, 0, 1],
            ...    'y': [0., 1., 2., 0., 0., 0., 0., 3.],
            ...    'x': [0., 0., 0., 0., 2., 4., 0., 0.],
            ... })

            # Imaging parameters
            >>> kwargs = dict(pixel_size_um = 1.0, frame_interval = 0.00748)

            # Make a TrajectoryGroup
            >>> with TrajectoryGroup(detections, **kwargs) as TG:
            ...     print(TG.jumps)

               frame  dframes  trajectory   dy   dx  dr2  jumps_per_track
            0      0        1           0  1.0  0.0  1.0                2
            1      1        1           0  1.0  0.0  1.0                2
            2      0        1           1  0.0  2.0  4.0                2
            3      1        1           1  0.0  2.0  4.0                2
            4      0        1           2  3.0  0.0  9.0                1


    .. py:property:: jumps_per_track
        :type: numpy.ndarray, shape (n_tracks,)

        Number of jumps per trajectory


    .. py:property:: raw_track_statistics
        :type: dict

        Summary statistics on the *raw* set of trajectories (*i.e.* the set of trajectories
        passed when constructing this TrajectoryGroup object).

        These include:

            * **n_tracks**: total number of trajectories
            * **n_jumps**: total number of jumps
            * **n_detections**: total number of detections
            * **mean_track_length**: mean trajectory length in frames
            * **max_track_length**: length of the longest trajectory in frames
            * **fraction_singlets**: fraction of trajectories that have length 1 (in other words, they're just single detections)
            * **fraction_unassigned**: fraction of detections that are not assigned to any trajectory (have trajectory index <0). May not be relevant for all tracking algorithms.
            * **mean_jumps_per_track**: mean number of jumps per trajectory
            * **mean_detections_per_frame**: mean number of detections per frame
            * **max_detections_per_frame**: maximum number of detections per frame
            * **fraction_of_frames_with_detections**: fraction of all frames between the minimum and maximum frame indices that had detections. If 1.0, then all frames contained at least one detected spot.


    .. py:property:: processed_track_statistics
        :type: dict

        Summary statistics on the *processed* set of trajectories (*i.e.* the set of trajectories after calling ``TrajectoryGroup.preprocess`` on the raw set of trajectories).

        These are exactly the same as the metrics in ``raw_track_statistics``.


    .. py:property:: statistic_names
        :classmethod:
        :type: List[str]

        Names of each track summary statistic in ``TrajectoryGroup.raw_track_statistics`` and ``TrajectoryGroup.processed_track_statistics``.


    .. py:method:: get_track_vectors(self, n: int) -> Tuple[numpy.ndarray]

        Return the jumps of every trajectory with *n* jumps as a ``numpy.ndarray``.

        :param int n: number of jumps per trajectory

        :return: **V** (*numpy.ndarray*), **track_indices** (*numpy.ndarray*)

        **V** is a 3D ``numpy.ndarray`` with shape ``(n_tracks, 2, n)``. ``V[:,0,:]`` are the
        jumps along the `y`-axis, while ``V[:,1,:]`` are the jumps along the `x`-axis.

        **track_indices** is a 1D ``numpy.ndarray`` with shape ``(n_tracks,)`` and gives 
        the index of the trajectory corresponding to the first axis of ``V``.

        Using the example from above:

        .. code-block:: python

            # Simple set of detections belonging to 3 trajectories
            >>> detections = pd.DataFrame({
            ...    'trajectory': [0, 0, 0, 1, 1, 1, 2, 2],
            ...    'frame':      [0, 1, 2, 0, 1, 2, 0, 1],
            ...    'y': [0., 1., 2., 0., 0., 0., 0., 3.],
            ...    'x': [0., 0., 0., 0., 2., 4., 0., 0.],
            })

            # Make a TrajectoryGroup
            >>> TG = TrajectoryGroup(detections, pixel_size_um=1.0,
            ...     frame_interval=0.00748)
            >>> print(TG)
            TrajectoryGroup:
            n_detections:   8
            n_jumps:    5
            n_tracks:   3

            # Get the jump vectors for all trajectories with 2 jumps
            >>> V, track_indices = TG.get_jump_vectors(2)
            >>> print(V)
            [[[1. 1.]
              [0. 0.]]

             [[0. 0.]
              [2. 2.]]]

            >>> print(track_indices)
            [0 1]

            # Get the jump vectors for all trajectories with 1 jump
            >>> V, track_indices = TG.get_jump_vectors(1)
            >>> print(V)
            [[[3.]
              [0.]]]

            >>> print(track_indices)
            [2]


    .. py:method:: subsample(self, size: int) -> TrajectoryGroup

        Randomly subsample some number of trajectories from this TrajectoryGroup object
        to produce a new, smaller TrajectoryGroup object.

        :param int size: number of trajectories to subsample

        :return: **new_instance** (*TrajectoryGroup*)

        Example:

        .. code-block:: python

            # A TrajectoryGroup with 3 trajectories
            >>> print(TG)
            TrajectoryGroup:
            n_detections:   8
            n_jumps:    5
            n_tracks:   3

            # Randomly subsample 2 of these trajectories
            >>> TG2 = TG.subsample(2)
            >>> print(TG2)
            TrajectoryGroup:
            n_detections:   5
            n_jumps:    3
            n_tracks:   2

    .. py:method:: from_params(cls, detections: pandas.DataFrame, params: StateArrayParameters) -> TrajectoryGroup
        :classmethod:

        Alternative constructor that uses a `StateArrayParameters` object rather than
        a set of keyword arguments.

        :param pandas.DataFrame detections: the set of detections to use
        :param StateArrayParameters params: imaging and state array settings

        :return: **new_instance** (*TrajectoryGroup*)

        Example usage:

        .. code-block:: python

            >>> from saspt import StateArrayParameters, TrajectoryGroup
            >>> params = StateArrayParameters(
            ...     pixel_size_um = 0.16,
            ...     frame_interval = 0.00748
            ... )
            >>> TG = TrajectoryGroup.from_params(some_detections, params)

    .. py:method:: from_files(cls, filelist: List[str], **kwargs) -> TrajectoryGroup
        :classmethod:

        Alternative constructor. Create a `TrajectoryGroup`_ by loading and concatenating
        detections directly from one or more files. The files must be readable
        by `saspt.io.load_detections`.

        :param List[str] filelist: a list of paths to files containing detections
        :param kwargs: options to `TrajectoryGroup.__init__`

        :return: **new_instance** (*TrajectoryGroup*)

    .. py:method:: preprocess(cls, detections: pandas.DataFrame, splitsize: int=DEFAULT_SPLITSIZE, start_frame: int=DEFAULT_START_FRAME) -> pandas.DataFrame
        :classmethod:

        Preprocess some raw trajectories for state arrays. This involves:

            * remove all singlets (trajectories of length 1), unassigned detections, and detections before `start_frame`
            * break large trajectories into smaller pieces that have at most `splitsize` jumps
            * reindex trajectories so that the set of all trajectory indices is contiguous between 0 and `n_tracks-1`

        For most applications `preprocess` should not be called directly, and instead
        you should instantiate a `TrajectoryGroup`_ using one of the constructors.

        :param pandas.DataFrame detections: indexed by detection. Must be recognize by `saspt.io.is_detections`.

        :param int splitsize: maximum trajectory length in *jumps*

        :param int start_frame: disregard detections recorded before this frame. Useful to restrict attention to later frames with lower density.

        :return: **processed_detections** (*pandas.DataFrame*)