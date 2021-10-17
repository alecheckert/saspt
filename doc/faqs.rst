.. _faqs_label:

====
FAQS
====

Q. Why doesn't ``saspt`` support input format :math:`X`?
========================================================

Because a table of detections is probably the simplest format that
exists to describe trajectories, so we added it first. We're happy to 
expand support for additional formats (within reason) - let us know with
a GitHub request.

Q. Why are the default diffusion coefficients log-spaced?
=========================================================

As the diffusion coefficient increases, our estimate of it becomes much
more error-prone. This makes it difficult for humans to compare the occupations
of states with widely varying diffusion coefficients. By plotting on a log
scale, we minimize these perceptual differences so that humans can accurately 
compare states across the full range of biologically observed diffusion 
coefficients.

To demonstrate this effect, consider the likelihood function for
the jumps of a 2D Brownian motion with no localization error, diffusion coefficient
:math:`D`, frame interval :math:`\Delta t`, and :math:`n` total jumps. The 
maximum likelihood estimator for the diffusion coefficient is the mean-squared
displacement (MSD):

.. math::

    \hat{D} = \frac{1}{4 n \Delta t} \sum\limits_{j=1}^{n} (\Delta x_{j}^{2} + \Delta y_{j}^{2})


where :math:`(\Delta x_{j}, \Delta y_{j})` is the :math:`j^{\text{th}}` jump
in the trajectory.

We can get the "best-case" error in this estimate using the Cramer-Rao
lower bound (CRLB), which provides the minimum variance our estimator
:math:`\hat{D}`:

.. math::

    \text{Var}(\hat{D}) \geq \text{CRLB}(\hat{D}) = \frac{D^{2}}{n}

So the error in the estimate of :math:`D` actually increases as the
*square* of :math:`D`. If we throw in localization error (represented 
as the 1D spatial measurement variance, :math:`\sigma^{2}`) and neglect the
off-diagonal terms in the covariance matrix, we get the approximation

.. math::

    \text{Var}(\hat{D}) \geq \text{CRLB}(\hat{D}) \approx \frac{(D \Delta t + \sigma^{2})^{2}}{n \Delta t^{2}}

Notice that this makes it even *harder* to estimate the diffusion coefficient,
especially when :math:`D \Delta t < \sigma^{2}`. 

Q. How does ``saspt`` estimate the posterior occupations, given the posterior distribution?
===========================================================================================

``saspt`` always uses the posterior mean. If :math:`\boldsymbol{\alpha}` is the parameter to the posterior Dirichlet distribution over state occupations, then the posterior mean :math:`\boldsymbol{\tau}` is simply the normalized Dirichlet parameter:

.. math::

    \boldsymbol{\tau} \sim \text{Dirichlet} \left( \boldsymbol{\alpha} \right) \\
    \mathbb{E} \left[ \boldsymbol{\tau} | \boldsymbol{\alpha} \right] = \frac{1}{\sum_{j=1}^{K} \alpha_{j}} \begin{bmatrix}
        \alpha_{1} \\
        ...        \\
        \alpha_{K}
    \end{bmatrix}

We prefer the posterior mean to max *a posteriori* (MAP) or other estimators because
it is very conservative and minimizes the occurrence of spurious features.

Q. I want to measure the fraction of particles in a particular state. How do I do that?
=======================================================================================

If you know the range of diffusion coefficients you're interested in,
you can directly integrate the mean posterior occupations. Say we want 
the fraction of particles with diffusion coefficients between 1 and 10
:math:`\mu\text{m}^{2}`/sec:

.. code-block:: python

    >>> occupations = SA.posterior_occs_dataframe
    >>> in_range = (occupations['diff_coef'] >= 1.0) & (occupations['diff_coef'] < 10.0)
    >>> print(occupations.loc[in_range, 'mean_posterior_occupation'].sum())

That being said, ``saspt`` does *not* provide any way to determine the 
endpoints for this range, and that is up to you or the methods you develop.
