Likelihood
==========

.. py:class:: Likelihood

    Abstract base class for likelihood functions, defining properties
    that all likelihood functions must implement.

    Each Likelihood subclass evaluates on a set of trajectories at each
    of a grid of parameter values.

    Instances of Likelihood subclasses should be generated with the 
    `saspt.make_likelihood` function. For example:

    .. code-block:: python

        >>> from saspt import make_likelihood, LIKELIHOOD_TYPES

        # Imaging parameters
        >>> kwargs = dict(
        ...     pixel_size_um = 0.16,
        ...     frame_interval = 0.00748,
        ...     focal_depth = 0.7
        ... )

        # Make a likelihood function for each of the available likelihood types,
        # and show the names of the parameters on which the likelihood function
        # evaluates
        >>> for likelihood_type in LIKELIHOOD_TYPES:
        ...     L = make_likelihood(likelihood_type, **kwargs)
        ...     print(f"Likelihood function '{L.name}' has parameters: {L.parameter_names}")
        Likelihood function 'rbme' has parameters: ('diff_coef', 'loc_error')
        Likelihood function 'rbme_marginal' has parameters: ('diff_coef',)
        Likelihood function 'gamma' has parameters: ('diff_coef',)
        Likelihood function 'fbme' has parameters: ('diff_coef', 'hurst_parameter')

    .. py:property:: name
        :abstractmethod:
        :type: str

        The name of the likelihood function.

    .. py:property:: shape
        :abstractmethod:
        :type: Tuple[int]

        The shape of the parameter grid on which the likelihood 
        function evaluates.

    .. py:property:: parameter_names
        :abstractmethod:
        :type: Tuple[str]

        Names of each parameter in the parameter grid.

    .. py:property:: parameter_grid
        :abstractmethod:
        :type: Tuple[numpy.ndarray]

        Values of each parameter in the parameter grid, given as a tuple of 1D ``numpy.ndarray``. The parameter grid is the Cartesian product of these arrays.

    .. py:property:: parameter_units
        :abstractmethod:
        :type: Tuple[str]

        Physical units for each parameter in the parameter grid.

    .. py:method:: __call__(self, trajectories: TrajectoryGroup) -> Tuple[numpy.ndarray]
        :abstractmethod:

        Evaluate the log likelihood function on each of a set of trajectories 
        at each point on the parameter grid.

        :param TrajectoryGroup trajectories: the trajectories on which to evaluate

        :return: **evaluated_log_likelihood** (*numpy.ndarray*), **jumps_per_track** (*numpy.ndarray*)

            - **evaluated_log_likelihood** is the value of the log likelihood function evaluated on each of the trajectories at each of the parameter values. It is a ``numpy.ndarray`` with shape ``(*self.shape, trajectories.n_tracks)``.

            - **jumps_per_track** is the number of jumps in each trajectory. It is a 1D ``numpy.ndarray`` with shape ``(trajectories.n_tracks,)``.

        Example usage:

        .. code-block:: python

            >>> from saspt import make_likelihood, RBME, TrajectoryGroup
            >>> kwargs = dict(pixel_size_um = 0.16, frame_interval = 0.00748)

            # Make an RBMELikelihood function
            >>> likelihood = make_likelihood(RBME, **kwargs)

            # Show the shape of the parameter grid on which this likelihood
            # function is defined
            >>> print(likelihood.shape)
            (101, 36)

            # Show the names of the parameters corresponding to each axis
            # on the parameter grid
            >>> print(likelihood.parameter_names)
            ('diff_coef', 'loc_error')

            # Load some trajectories (available in saspt/tests/fixtures)
            >>> tracks = TrajectoryGroup.from_files(["tests/fixtures/small_tracks_0.csv"], **kwargs)

            # Evaluate the log likelihood function on these trajectories
            >>> log_L, jumps_per_track = likelihood(tracks)

            # The log likelihood contains one element for each trajectory 
            # and each point in the parameter grid
            >>> print(log_L.shape)
            (101, 36, 39)

            >>> print(jumps_per_track.shape)
            (39,)

    .. py:method:: exp(self, log_L: numpy.ndarray) -> numpy.ndarray
        :abstractmethod:

        Take the exponent of a log likelihood function produced by :py:func:`Likelihood.__call__` in a numerically stable way.

        This function also normalizes the likelihood function so that 
        the values of the likelihood sum to 1 across all states in each 
        trajectories.

        We can take the exponent of the log likelihood function from 
        the example in `__call__`, above:

        .. code-block:: python

            from saspt import RBME

            # Make an RBME likelihood function
            >>> likelihood = make_likelihood(RBME, **kwargs)

            # Get the normalized likelihood
            >>> normed_L = likelihood.exp(log_L)

            # Likelihood is normalized across all states for each trajectory
            >>> print(normed_L.sum(axis=(0,1)))
            array([1., 1., 1., ..., 1., 1., 1.])

        :return: **L** (*numpy.ndarray*), shape *log_L.shape*, the normalized likelihood function for each trajectory-state assignment

    .. py:method:: correct_for_defocalization(self, occs: numpy.ndarray, normalize: bool) -> numpy..ndarray
        :abstractmethod:

        Correct a set of state occupations on this parameter grid for the effect of
        defocalization.

        :param numpy.ndarray occs: state occupations, with shape *self.shape*
        :param bool normalize: normalize the occupations after applying the correction

        :return: **corrected_occs** (*numpy.ndarray*, shape *self.shape*), corrected state occupations

    .. py:method:: marginalize_on_diff_coef(self, occs: numpy.ndarray) -> numpy.ndarray
        :abstractmethod:

        Given a set of state occupations, marginalize over all parameters except
        the diffusion coefficient.

        May raise ``NotImplementedError`` if the diffusion coefficient is not a 
        parameter supported by this likelihood function. (Although this is the 
        case for all Likelihood subclasses implemented to date!)

        :param numpy.ndarray occs: state occupations, shape *self.shape*

        :return: **marginal_occs** (*numpy.ndarray*), marginal state occupations. This will have a lower dimensionality than the input.

        For example, suppose we are using the RBME likelihood with a parameter grid of
        shape (10, 6). Since the parameters for the RBME likelihood are ``diff_coef``
        and ``loc_error``, this means that the parameter grid has 10 distinct diffusion
        coefficient values and 6 distinct localization error values. After applying
        ``marginalize_on_diff_coef``, the output has shape (10,) since the localization
        error is marginalized out.

        In code, this situation is:

        .. code-block:: python

            >>> import numpy as np
            >>> from saspt import make_likelihood, RBME

            # Define an RBME likelihood function on a grid of 10 diffusion 
            # coefficients and 6 localization errors
            >>> likelihood = make_likelihood(
            ...     RBME,
            ...     pixel_size_um = 0.16,
            ...     frame_interval = 0.00748,
            ...     focal_depth = 0.7,
            ...     diff_coefs = np.logspace(0.0, 1.0, 10),
            ...     loc_errors = np.linspace(0.0, 0.05, 6)
            ... )

            >>> print(likelihood.shape)
            (10, 6)

            # Some random state occupations
            >>> occs = np.random.dirichlet(np.ones(60)).reshape((10, 6))
            >>> print(occs.shape)
            (10, 6)

            # Marginalize on diffusion coefficient
            >>> marginal_occs = likelihood.marginalize_on_diff_coef(occs)
            >>> print(marginal_occs.shape)
            (10,)

            # Plot marginal occupations as a function of diffusion coefficient
            # (example)
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(likelihood.diff_coefs, marginal_occs)
            >>> plt.xlabel(f"Diff. coef. ({likelihood.parameter_units[0]})")
            >>> plt.ylabel("Marginal occupation")
            >>> plt.xscale('log'); plt.show(); plt.close()

RBMELikelihood
--------------

.. py:class:: RBMELikelihood(self, pixel_size_um: float, frame_interval: float, focal_depth: float=numpy.inf, diff_coefs: numpy.ndarray=DEFAULT_DIFF_COEFS, loc_errors: numpy.ndarray=DEFAULT_LOC_ERRORS, **kwargs)

    Subclass of `Likelihood`_ for the RBME (regular Brownian motion with localization
    error) likelihood function. Probably the most useful likelihood function in `saSPT`.

    Suppose we image an RBME with diffusion coefficient :math:`D`, localization error
    :math:`\sigma`, and frame interval :math:`\Delta t`. If there are :math:`n` 
    jumps in the trajectory, and if :math:`\mathbf{x}, \mathbf{y} \in \mathbb{R}^{n}` are its jumps along `y` and `x` axes respectively, then the likelihood function is 

    .. math::

        f(\mathbf{x}, \mathbf{y} | D, \sigma) = \frac{
            \exp \left( -\frac{1}{2} \left( \mathbf{x}^{T} \Gamma^{-1} \mathbf{x} + \mathbf{y}^{T} \Gamma^{-1} \mathbf{y} \right) \right)
        }{
            \left( 2 \pi \right)^{n} \text{det} \left( \Gamma \right)
        }

    where :math:`\Gamma \in \mathbb{R}^{n \times n}` is the covariance matrix defined by

    .. math::

        \Gamma_{ij} = \begin{cases}
            2 (D \Delta t + \sigma^{2}) &\text{if } i = j \\
            -\sigma^{2} &\text{if } |i - j| = 1 \\
            0 &\text{otherwise}
        \end{cases}

    The parameter grid for `RBMELikelihood` is a two-dimensional array with the
    first axis corresponding to the diffusion coefficient and the second 
    corresponding to localization error. By default, we use a set of logarithmically
    spaced diffusion coefficients between 0.01 and 100 :math:`\mu\text{m}^{2} \text{ sec}^{-1}` and linearly spaced localization errors between 0 and 0.08 :math:`\mu\text{m}`.

    In most situations, localization error is a nuisance parameter. When using the 
    RBME likelihood function with a state array run, we usually marginalize over the
    localization error afterward. As a result, the RBME likelihood is much more stable
    from day-to-day and microscope-to-microscope than likelihood functions that do not
    explicitly model the error, such as the `GammaLikelihood`.

    See `Likelihood`_ for a description of the class properties and methods.

    :param float pixel_size_um: camera pixel size after magnification in microns
    :param float frame_interval: time between frames in seconds
    :param float focal_depth: objective focal depth in microns. Used to calculate the effect of defocalization on apparent state occupations. If `numpy.inf`, no defocalization corrections are applied.
    :param numpy.ndarray diff_coefs: the set of diffusion coefficients to use for this likelihood function's parameter grid
    :param numpy.ndarray loc_errors: the set of localization errors to use for this likelihood function's parameter grid
    :param kwargs: ignored

RBMEMarginalLikelihood
----------------------

.. py:class:: RBMEMarginalLikelihood(self, pixel_size_um: float, frame_interval: float, focal_depth: float=numpy.inf, diff_coefs: numpy.ndarray=DEFAULT_DIFF_COEFS, loc_errors: numpy.ndarray=DEFAULT_LOC_ERRORS, **kwargs)

    The underlying model is identical to `RBMELikelihood`_. However, after evaluating
    the likelihood function on a 2D parameter grid of diffusion coefficient and 
    localization error, we marginalize over the localization error to produce a 1D
    grid over the diffusion coefficient. State arrays then evaluate the posterior
    distribution over this 1D grid, rather than the 2D grid in `RBMELikelihood`_.

    In short, the order of state inference and marginalization is switched:

    RBMELikelihood:
        #. Evaluate 2D likelihood function over diffusion coefficient and localization error
        #. Infer 2D posterior distribution over diffusion coefficient and localization error
        #. Marginalize over localization error to get 1D distribution over diffusion coefficient

    RBMEMarginalLikelihood:
        #. Evaluate 2D likelihood function over diffusion coefficient and localization error
        #. Marginalize over localization error to get 1D likelihood over diffusion coefficient
        #. Infer 1D posterior distribution over diffusion coefficient

    `RBMEMarginalLikelihood` generally is inferior to `RBMELikelihood` and is provided
    as a point of comparison.

    :param float pixel_size_um: camera pixel size after magnification in microns
    :param float frame_interval: time between frames in seconds
    :param float focal_depth: objective focal depth in microns. Used to calculate the effect of defocalization on apparent state occupations. If `numpy.inf`, no defocalization corrections are applied.
    :param numpy.ndarray diff_coefs: the set of diffusion coefficients to use for this likelihood function's parameter grid
    :param numpy.ndarray loc_errors: the set of localization errors to marginalize over when evaluating the likelihood function
    :param kwargs: ignored

GammaLikelihood
---------------

.. py:class:: GammaLikelihood(self, pixel_size_um: float, frame_interval: float, focal_depth: float=numpy.inf, diff_coefs: numpy.ndarray=DEFAULT_DIFF_COEFS, loc_error: float=0.035, mode: str="point, **kwargs)

    Subclass of `Likelihood`_ for the gamma approximation to the RBM (regular 
    Brownian motion) likelihood function.

    The gamma likelihood is obtained from the RBME likelihood by making two 
    approximations:

        * the localization error is treated as a constant
        * we neglect the off-diagonal terms of the covariance matrix :math:`\Gamma`

    In this case, the likelihood function simplifies to a gamma distribution.
    Suppose that :math:`\mathbf{x}, \mathbf{y} \in \mathbb{R}^{n}` are the `x`-
    and `y`-jumps of a trajectory with :math:`n` total jumps, and let :math:`S`
    be the sum of its squared jumps:

    .. math::

        S = \sum\limits_{i=1}^{n} \left( x_{i}^{2} + y_{i}^{2} \right)

    Then the likelihood function can be expressed

    .. math::

        f(S | D, \sigma^{2}) = \frac{
            S^{n-1} e^{-S / 4 D \Delta t}
        }{
            \Gamma (n) (4 (D \Delta t + \sigma^{2}))^{n}
        }

    Notice that in the term :math:`4 (D \delta t + \sigma^{2})`, the contributions of
    diffusion (:math:`D \Delta t`) and localization
    error (:math:`\sigma^{2}`) cannot be distinguished without introducing the 
    assumption that localization is constant. This is only approximately true, since
    the number of photons collectd per particle, the axial distance from the focus,
    and the motion of the particle will all influence localization error, creating
    variation within a single SPT movie. In particular, the gamma likelihood performs
    tolerably well when :math:`D \Delta t \gg \sigma^{2}`, but is highly inaccurate
    when :math:`D \Delta t \sim \sigma^{2}`. 

    The `GammaLikelihood` parameter grid is a simple 1D array of diffusion coefficients.
    By default, these are logarithmically spaced between 0.01 and 100 :math:`\mu\text{m}^{2} \text{ sec}^{-1}`.

    See `Likelihood`_ for a description of the class properties and methods.

    :param float pixel_size_um: camera pixel size after magnification in microns
    :param float frame_interval: time between frames in seconds
    :param float focal_depth: objective focal depth in microns. Used to calculate the effect of defocalization on apparent state occupations. If `numpy.inf`, no defocalization corrections are applied.
    :param numpy.ndarray diff_coefs: the set of diffusion coefficients to use for this likelihood function's parameter grid
    :param float loc_error: the 1D localization error (assumed constant), expressed as the root variance in microns
    :param str mode: deprecated; ignored
    :param kwargs: ignored

FBMELikelihood
--------------

.. py:class:: FBMELikelihood(self, pixel_size_um: float, frame_interval: float, focal_depth: float, diff_coefs: numpy.ndarray = DEFAULT_DIFF_COEFS, hurst_pars: numpy.ndarray = DEFAULT_HURST_PARS, loc_error: float=0.035, **kwargs)

    Likelihood function for fractional Brownian motion with localization error (FBME).
    This is similar to RBME but allows for temporal correlations (either positive or
    negative) between jumps in the same trajectory, depending on the value of the 
    Hurst parameter:

        * if :math:`H < \frac{1}{2}`, jumps in the same trajectory are anticorrelated (the trajectory tends to return to where it came from)
        * if :math:`H = \frac{1}{2}`, the motion is Brownian (no correlation between the jumps)
        * if :math:`H > \frac{1}{2}`, jumps in the same trajectory are positively correlated (the trajectory tends to keep moving in the same direction)

    In particular, if :math:`\mathbf{x}, \mathbf{y} \in \mathbb{R}^{n}` are the `x`- 
    and `y` components of the jumps of an FBME with :math:`n` jumps, diffusion coefficient
    :math:`D`, Hurst parameter :math:`H`, and localization error :math:`\sigma`, then 
    the likelihood function is defined

    .. math::

        f(\mathbf{x}, \mathbf{y} | D, H, \sigma^{2}) = \frac{
            \exp \left( -\frac{1}{2} \left( \mathbf{x}^{T} \Gamma^{-1} \mathbf{x} + \mathbf{y}^{T} \Gamma^{-1} \mathbf{y} \right) \right)
        }{
            \left( 2 \pi \right)^{n} \text{det} \left( \Gamma \right)
        }

    where :math:`\Gamma` is the covariance matrix:

    .. math::

        \Gamma_{ij} = \begin{cases}
            D \Delta t \left( |i - j + 1|^{2 H} + |i - j - 1|^{2 H} - 2 |i - j|^{2 H} \right) + 2 \sigma^{2} &\text{if } i = j \\
            D \Delta t \left( |i - j + 1|^{2 H} + |i - j - 1|^{2 H} - 2 |i - j|^{2 H} \right) - \sigma^{2} &\text{if } |i - j| = 1 \\
            D \Delta t \left( |i - j + 1|^{2 H} + |i - j - 1|^{2 H} - 2 |i - j|^{2 H} \right) &\text{otherwise}
        \end{cases}

    The parameter grid for `FBMELikelihood` is a 2D grid over diffusion coefficient
    and Hurst parameter, with localization error treated as a constant.

    :param float pixel_size_um: camera pixel size after magnification in microns
    :param float frame_interval: time between frames in seconds
    :param float focal_depth: objective focal depth in microns. Used to calculate the effect of defocalization on apparent state occupations. If `numpy.inf`, no defocalization corrections are applied.
    :param numpy.ndarray diff_coefs: the set of diffusion coefficients to use for this likelihood function's parameter grid
    :param numpy.ndarray hurst_pars: the set of Hurst parameters to use for this likelihood function's parameter grid
    :param float loc_error: the 1D localization error (assumed constant), expressed as a root variance in microns
    :param kwargs: ignored