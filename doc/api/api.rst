===
API
===

``saspt`` is an object-oriented Python 3 library. The classes perform discrete roles:

    * ``TrajectoryGroup`` represents a set of trajectories to be analyzed by state arrays
    * ``Likelihood`` represents a likelihood function, or physical model for the type of motion. Some examples of likelihood functions include Brownian motion with localization error (RBME), various approximations of RBME, and fractional Brownian motion (FBM). The ``Likelihood`` class also defines the grid of `state parameters` on which the state array is constructed.
    * ``StateArrayParameters`` is a struct with the settings for state array inference, including the pixel size, frame rate, and other imaging parameters.
    * ``StateArray`` implements the actual state array inference routines.
    * ``StateArrayDataset`` parallelizes state arrays across multiple target files, and provides some tools and visualizations for comparing between experimental conditions.

.. toctree:: 
    :maxdepth: 2
    
    trajectorygroup
    likelihood
    statearrayparameters
    statearray
    dataset
    io
    utils