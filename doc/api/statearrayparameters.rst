.. _statearrayparameters_label:

StateArrayParameters
====================

.. py:currentmodule:: saspt

.. py:class:: StateArrayParameters(pixel_size_um: float, frame_interval: float)

    .. py:method:: __init__(pixel_size_um: float, frame_interval: float, focal_depth: float=np.inf, splitsize: int=DEFAULT_SPLITSIZE, sample_size: int=DEFAULT_SAMPLE_SIZE, start_frame: int=DEFAULT_START_FRAME, max_iter: int=DEFAULT_MAX_ITER, conc_param: float=DEFAULT_CONC_PARAM, progress_bar: bool=False, num_workers: int=1)

        :param float pixel_size_um: camera pixel size after magnification in microns
        :param float frame_interval: delay between frames in seconds
        :param int splitsize: maximum length of trajectories in frames. Trajectories longer than *splitsize* are split into smaller pieces.
        :param int sample_size: maximum number of trajectories to consider per state array. SPT experiments that exceed this number are subsampled.
        :param int start_frame: disregard detections before this frame. Useful to restrict analysis to later frames with lower detection density.
        :param int max_iter: maximum number of iterations of variational Bayesian inference to run when inferring the posterior distribution
        :param float conc_param: concentration parameter of the Dirichlet prior over state occupations. A `conc_param` of 1.0 is a naive prior; values less than 1.0 favor more states and values greater than 1.0 favor fewer states. Default value is 1.0.
        :param bool progress_bar: show progress and be a little verbose, where relevant
        :param int num_workers: number of parallel processes to use. Recommended not to set this higher than the number of CPUs.

        :return: new instance of StateArrayParameters

    .. py:property:: parameters
        :type: Tuple[str]

        Names of all parameters that directly impact the state array algorithm. Does not
        include parameters that determine implementation or display, such as *progress_bar*
        or *num_workers*

    .. py:property:: units
        :type: dict

        Physical units in which each parameter is defined

    .. py:method:: __eq__(self, other: StateArrayParameters) -> bool

        Check for equivalence of two StateArrayParameter objects

    .. py:method:: __repr__(self) -> str

        String representation of this StateArrayParameters object