.. _label_model:

=====================
The state array model
=====================

This section describes the underlying statistical model for state arrays.

Likelihood functions
====================

A physical model for the motion of particles in an SPT experiment is usually
expressed in the form of a probability distribution :math:`p(X|\boldsymbol{\theta})`, where
:math:`X` is the observed path of the trajectory and :math:`\boldsymbol{\theta}` represents
the *parameters* of the motion. Intuitively, this function tells us how
likely we are to see a specific trajectory given some kind of physical model.
We refer to the function :math:`p(X|\boldsymbol{\theta})` as
a *likelihood function*.

As an example, take the case of regular Brownian motion (RBM),
a model with a single parameter that characterizes each trajectory
(the diffusion coefficient :math:`D`). (For those familiar, this is
equivalent to a scaled Wiener process.) Its likelihood function is

.. math::

	p(X | D) = \prod\limits_{k=1}^{n} \frac{
		\exp \left( -\Delta r_{k}^{2} / 4 D \Delta t \right)
	}{
		\left( 4 \pi D \Delta t \right)^{\frac{d}{2}}
	}

where :math:`\Delta r_{k}` is the radial displacement of the 
:math:`k^{\text{th}}` jump in the trajectory, :math:`\Delta t` is the measurement interval, :math:`d` is the spatial dimension, and 
:math:`n` is the total number of jumps in the trajectory.

A common approach to recover the parameters of the motion is simply
to find the parameters that maximize :math:`p(X | \boldsymbol{\theta})`,
holding :math:`X` constant at its observed value.
In the case of RBM, this maximum
likelihood estimate has a closed form - the mean squared displacement,
or "MSD":

.. math::

	\hat{D}_{\text{mle}} = \frac{\sum\limits_{k=1}^{n} \Delta r_{k}^{2}}{2 d n \Delta t}

While this works well enough for one pure Brownian trajectory, this 
approach has several shortcomings when we try to generalize it:

	1. Closed-form maximum likelihood solutions only exist for the simplest physical models, like RBM. Even introducing measurement error, a ubiquitous feature of SPT-PALM experiments, is sufficient to eliminate any closed-form solution.
	2. The maximum likelihood estimate does not provide any intrinsic measure of confidence in the result. This becomes especially problematic for more complex models with multiple parameters, where a large range of parameter vectors may give near-equivalent results. In practice, this means that even when our maximum likelihood estimators work perfectly, they are highly instable from one experiment to the next.

An alternative to maximum likelihood inference is to treat both 
:math:`\mathbf{X}` and :math:`\boldsymbol{\theta}` as random 
variables, and evaluate the probability 
:math:`p(\boldsymbol{\theta} | \mathbf{X})`. For instance, we can estimate
:math:`\boldsymbol{\theta}` by taking the mean of 
:math:`p(\boldsymbol{\theta} | \mathbf{X})`. This is the Bayesian 
approach (and the one we use in ``saspt``).

The important part is that the likelihood function assigns a number 
to each trajectory based on (a) the observed path of the trajectory 
:math:`X` and (b) one or more *model parameters* :math:`\boldsymbol{\theta}`.

Mixture models
==============

Suppose we observe :math:`N` trajectories in an SPT experiment, which 
we represent as a vector :math:`\mathbf{X} = (X_{1}, ..., X_{N})`. If
all of the trajectories can be described by the same physical model, then the 
probability of seeing a set of trajectories :math:`\mathbf{X}` is just the product
of the distributions over each :math:`X_{i}`:

.. math::

	p(\mathbf{X}|\boldsymbol{\theta}) = \prod\limits_{i=1}^{N} p (X_{i} | \boldsymbol{\theta})

In reality, this only describes the simplest situations
because it assumes that the *same physical model governs all of the trajectories*.
Most of the time we cannot assume that all trajectories originate from
particles in the same physical state. Indeed, heterogeneity in
a particle's dynamical states is often one of the things we hope to
learn from an SPT experiment.

To deal with this complexity, we construct *mixture models*, which are exactly what they sound
like: mixtures of particles in different *states*. Each state is governed by a different physical model. Parameters of interest include
the model parameters characterizing each state, as well as the fraction of 
particles in each state (the state's *occupation*).

We formalize mixture models in the following way.
Suppose we have a mixture of :math:`K` states. 
Instead of a single vector of state parameters, we'll have one vector
for each state: 
:math:`\boldsymbol{\theta} = (\boldsymbol{\theta}_{1}, ..., \boldsymbol{\theta}_{K})`. And in addition to the state parameters, we'll
specify a set of *occupations* 
:math:`\boldsymbol{\tau} = (\tau_{1}, ..., \tau_{K})` that describe
the fraction of particles in each state. (These are also called 
*mixing probabilities*.)

With this formalism, the probability of seeing a single trajectory in state 
:math:`j` is :math:`\tau_{j}`. The probability of seeing two trajectories
in that state is :math:`\tau_{j}^{2}`. And the probability of seeing
:math:`n_{1}` trajectories in the first state, :math:`n_{2}` in the
second state, and so on is

.. math::

	\tau_{1}^{n_{1}} \tau_{2}^{n_{2}} \cdots \tau_{K}^{n_{K}}

Of course, usually we don't *know* which state a given trajectory
comes from. The more states we have, the more uncertainty there is.

The way to handle this in a Bayesian framework is to incorporate the
uncertainty explicitly into the model by introducing a new random variable
:math:`\mathbf{Z}` that we'll refer to as the *assignment matrix*.
:math:`\mathbf{Z}` is a :math:`N \times K` matrix composed solely of
0s and 1s such that

.. math::

	Z_{ij} = \begin{cases}
		1 &\text{if trajectory } i \text{ comes from a particle in state } j \\
		0 &\text{otherwise}
	\end{cases}

Notice that each row of :math:`\mathbf{Z}` contains a single 1 and the 
rest of the elements are 0. As an example, imagine we have three states and
two trajectories, with the first trajectory assigned to state 1 and the 
second assigned to state 3. Then the assignment matrix would be

.. math::

	\mathbf{Z} = \begin{bmatrix}
		1 & 0 & 0 \\
		0 & 0 & 1
	\end{bmatrix}

Given a particular state occupation vector :math:`\boldsymbol{\tau}`,
the probability of seeing a particular set of assignments :math:`\mathbf{Z}`
is

.. math::
	
	p(\mathbf{Z}|\boldsymbol{\tau}) = \prod\limits_{j=1}^{K} \prod\limits_{i=1}^{N} \tau_{j}^{Z_{ij}}

Notice that the probability and expected value for any given
:math:`Z_{ij}` are the same:

.. math::

	p(Z_{ij} | \boldsymbol{\tau}) = \mathbb{E} \left[ Z_{ij} | \boldsymbol{\tau} \right] = \tau_{j}

To review, we have four parameters that describe the mixture
model:

	* The state occupations :math:`\boldsymbol{\tau}`, which describe the fraction of particles in each state;
	* The state parameters :math:`\boldsymbol{\theta}`, which describe the type of motion produced by particles in each state;
	* The assignment matrix :math:`\mathbf{Z}`, which describes the underlying state for each observed trajectory;
	* The observed trajectories :math:`\mathbf{X}`

Bayesian mixture models
-----------------------

Of these four parameters, we only observe the trajectories
:math:`\mathbf{X}` in an SPT experiment. The Bayesian approach is to infer
the conditional distribution

.. math::

	p(\mathbf{Z}, \boldsymbol{\tau}, \boldsymbol{\theta} | \mathbf{X}) 

Using Bayes' theorem, we can rewrite this as 

.. math::
	p(\mathbf{Z}, \boldsymbol{\tau}, \boldsymbol{\theta} | \mathbf{X}) \propto p(\mathbf{X} | \mathbf{Z}, \boldsymbol{\tau}, \boldsymbol{\theta}) p (\mathbf{Z}, \boldsymbol{\tau}, \boldsymbol{\theta})

In order to proceed with this approach, it is necessary to 
specify the form of the last term, the *prior distribution*.
Actually, since :math:`\mathbf{Z}` only depends on :math:`\boldsymbol{\tau}` and not :math:`\boldsymbol{\theta}`, we can factor the prior as

.. math::

	p(\mathbf{Z}, \boldsymbol{\tau}, \boldsymbol{\theta}) = p(\mathbf{Z} | \boldsymbol{\tau}) p (\boldsymbol{\tau}) p(\boldsymbol{\theta})

We already saw the form of :math:`p(\mathbf{Z} | \boldsymbol{\tau})` earlier.
:math:`p(\boldsymbol{\theta})` is usually chosen so that it is conjugate
to the likelihood function (and, as we will see, it is irrelevant for
state arrays). For the prior :math:`p(\boldsymbol{\tau})`, we choose
a Dirichlet distribution with parameter 
:math:`\boldsymbol{\alpha}_{0} = (\alpha_{0}, ..., \alpha_{0}) \in \mathbb{R}^{K}`:

.. math::

	\boldsymbol{\tau} \sim \text{Dirichlet} \left( \boldsymbol{\alpha}_{0} \right) = 
	p(\boldsymbol{\tau}) = \frac{1}{B(\boldsymbol{\alpha}_{0})} \prod\limits_{j=1}^{K} \tau_{j}^{\alpha_{0}-1}

Each draw from this distribution is a possible set of state occupations
:math:`\boldsymbol{\tau}`, with the *mean* of these draws being a 
uniform distribution :math:`(\frac{1}{K}, ..., \frac{1}{K})`. The 
variability of these draws about their mean is governed by :math:`\alpha_{0}`,
with high values of :math:`\alpha_{0}` producing distributions that are
closer to a uniform distribution. (:math:`\alpha_{0}` is known as the
*concentration parameter*.)

Infinite mixture models and ARD
===============================

There are many approaches to estimate the posterior distribution 
:math:`p(\mathbf{Z}, \boldsymbol{\tau}, \boldsymbol{\theta} | \mathbf{Z})`, both numerical (Markov chain Monte Carlo) and 
approximative (variational Bayes with a factorable candidate posterior).

However, a fundamental problem is the choice of :math:`K`, the number of 
states. Nearly everything depends on it. 

As discussed in :ref:`description_label`, nonparametric Bayesian methods
developed in the 1970s through 1990s proceeded on the realization that, as
:math:`K \rightarrow \infty`, the number of states with nonzero occupation
in the posterior distribution approached a finite number. In effect, the 
these models "pruned" away superfluous features, leaving only the minimal
models required to explain observed data. (In the context of machine 
learning, this property of Bayesian inference is called *automatic relevance determination* (ARD).)

In math, these models replaced the separate priors 
:math:`p(\boldsymbol{\tau})` and :math:`p(\boldsymbol{\theta})` with
a single prior :math:`H(\boldsymbol{\theta})` defined on the
space of all possible parameters :math:`\boldsymbol{\Theta}`. The models
are known as *Dirichlet process mixture models* (DPMMs) because the 
priors are a kind of probability distribution called Dirichlet processes
(essentially the infinite-dimensional version of a regular Dirichlet 
distribution).

However, such models are unwieldy in practice. As MCMC methods, they are extremely computationally costly. This is particularly true for high-dimensional parameter vectors :math:`\boldsymbol{\theta}`, for which inference on any kind of practical timescale is basically impossible. 
So while they solve the problem of choosing :math:`K`, they introduce the
equally dire problem of impractical runtimes.

State arrays
============

State arrays are a finite-state approximation of DPMMs. Instead
of an infinite set of states, we choose a high but finite :math:`K` with
state parameters :math:`\theta_{j}` that are situated on a fixed 
"parameter grid". Then, we rely mostly on the automatic relevance 
determination property of variational Bayesian inference
to prune away the superfluous states. This leaves 
minimal models to describe observed trajectories. Because the states are
chosen with fixed parameters, they only require that we evaluate the 
likelihood function *once*, at the beginning of inference. This shaves
off an enormous amount of computational time relative to DPMMs.

In this section, we describe state arrays, landing at the actual algorithm
for posterior inference used in ``saspt``.

We choose a large set of :math:`K` different states
with *fixed* state parameters :math:`\boldsymbol{\theta}_{j}` that are 
situated on a grid. Because the state parameters are fixed, the 
values of the likelihood function are constant and can be represented
as a :math:`N \times K` matrix, :math:`\mathbf{R}`:

.. math::
	
	R_{ij} = f(X_{i} | Z_{ij} = 1, \boldsymbol{\theta}_{j})

The total probability function for the mixture model is then

.. math::

	p(\mathbf{X}, \mathbf{Z}, \boldsymbol{\tau}) = p(\mathbf{X} | \mathbf{Z}) p (\mathbf{Z} | \boldsymbol{\tau}) p (\boldsymbol{\tau})

where

.. math::

	p (\mathbf{X} | \mathbf{Z}) = \prod\limits_{i=1}^{N} \prod\limits_{j=1}^{K} R_{ij}^{Z_{ij}}

	p(\mathbf{Z} | \boldsymbol{\tau}) = \prod\limits_{i=1}^{N} \prod\limits_{j=1}^{K} \tau_{j}^{Z_{ij}}

	p(\boldsymbol{\tau}) = \text{Dirichlet} (\alpha_{0}, ..., \alpha_{0})

Following a variational approach, we seek an approximation to the posterior
:math:`q(\mathbf{Z}, \boldsymbol{\tau}) \approx p(\mathbf{Z}, \boldsymbol{\tau} | \mathbf{X})` that maximizes the variational lower bound

.. math::

	L[q] = \sum\limits_{\mathbf{Z}} \int\limits_{\boldsymbol{\tau}} q(\mathbf{Z}, \boldsymbol{\tau}) \log \left[ 
		\frac{p (\mathbf{X}, \mathbf{Z}, \boldsymbol{\tau})}
		{q(\mathbf{Z}, \boldsymbol{\tau})}
	\right] \: d \boldsymbol{\tau}
	
Under the assumption that :math:`q` factors as
:math:`q(\mathbf{Z}, \boldsymbol{\tau}) = q(\mathbf{Z}) q(\boldsymbol{\tau})`,
this criterion can be achieved via an expectation-maximization routine:
alternately evaluating the two equations

.. math::

	\log q(\mathbf{Z}) = \mathbb{E}_{\boldsymbol{\tau} \sim q(\boldsymbol{\tau})} \left[ \log p (\mathbf{X}, \mathbf{Z}, \boldsymbol{\tau}) \right] + \text{constant}

	\log q(\boldsymbol{\tau}) = \mathbb{E}_{\mathbf{Z} \sim q(\mathbf{Z})} \left[ \log p (\mathbf{X}, \mathbf{Z}, \boldsymbol{\tau}) \right] + \text{constant}

The constants are chosen so that the respective factors :math:`q(\mathbf{Z})` or :math:`q(\boldsymbol{\tau})` are normalized. These expectations are
just shorthand for 

.. math::
	
	\mathbb{E}_{\boldsymbol{\tau} \sim q(\boldsymbol{\tau})} \left[ \log p (\mathbf{X}, \mathbf{Z}, \boldsymbol{\tau}) \right] = \int \log p (\mathbf{X}, \mathbf{Z}, \boldsymbol{\tau}) q(\boldsymbol{\tau}) \: d \boldsymbol{\tau}

	\mathbb{E}_{\mathbf{Z} \sim q(\mathbf{Z})} \left[ \log p (\mathbf{X}, \mathbf{Z}, \boldsymbol{\tau}) \right] = \sum\limits_{\mathbf{Z}} \log p (\mathbf{X}, \mathbf{Z}, \boldsymbol{\tau}) q(\boldsymbol{\tau}) p(\mathbf{Z})

Evaluating the first of these factors (and ignoring terms that don't directly depend on :math:`\boldsymbol{\tau}`), we have

.. math::

	\log q(\boldsymbol{\tau}) = \sum\limits_{j=1}^{K} \left( \alpha_{0} - 1 + \sum\limits_{i=1}^{N} \mathbb{E} \left[ Z_{ij} \right] \right) \log \tau_{j} + \text{constant}

From this, we can see that :math:`q(\boldsymbol{\tau})` is a Dirichlet
distribution:

.. math::

	q(\boldsymbol{\tau}) = \text{Dirichlet} \left( 
		\alpha_{0} + \sum\limits_{i=1}^{N} \mathbb{E} \left[ Z_{i,0} \right], ..., 
		\alpha_{0} + \sum\limits_{i=1}^{N} \mathbb{E} \left[ Z_{i,K} \right]
	\right)

The distribution "counts" in terms of trajectories: each trajectory 
contributes one count (in the form of :math:`Z_{i}`) to the posterior.
This is not ideal: because SPT-PALM microscopes normally have a short
focal depth due to their high numerical aperture, fast-moving particles
contribute many short trajectories to the posterior while slow-moving
particles contribute a few long trajectories. As a result, if we count by
trajectories, we introduce strong *state biases* into the posterior. (This
is exactly the reason why the popular MSD histogram method, which also 
"counts by trajectories", affords such inaccurate measurements of state 
occupations in realistic simulations of SPT-PALM experiments.)

A better way is to count the contributions to each state by *jumps* rather 
than trajectories. Because fast-moving and slow-moving states with equal
occupation contribute the same number of *detections* within the focal volume, they contribute close to the same number of jumps (modulo
the increased fraction of jumps from the fast-moving particle that "land"
outside the focal volume). 

Modifying this factor to count by jumps rather than trajectories, we have

.. math::

	q(\boldsymbol{\tau}) = \text{Dirichlet} \left( \alpha_{0} + \alpha_{1}, ..., \alpha_{0} + \alpha_{K} \right)

	\alpha_{j} = \sum\limits_{i=1}^{N} n_{i} \mathbb{E} \left[ Z_{ij} \right]

where :math:`n_{i}` is the number of jumps observed for trajectory :math:`i`.

Next, we evaluate :math:`q(\mathbf{Z})`:

.. math::

	\log q(\mathbf{Z}) = \sum\limits_{j=1}^{K} \sum\limits_{i=1}^{N} \left( \log R_{ij} + \psi (\alpha_{0} + \alpha_{j}) \right) Z_{ij} + \text{const}

where we have used the result that if :math:`\boldsymbol{\tau} \sim \text{Dirichlet} \left( \boldsymbol{a} \right)`, then :math:`\mathbb{E} \left[ \tau_{j} \right] = \psi (a_{j}) - \psi (a_{1} + ... + a_{K} )`, where :math:`\psi` is the digamma function.

Normalizing over each trajectory :math:`i`, we have

.. math::

	q(\mathbf{Z}) = \prod\limits_{i=1}^{N} \prod\limits_{j=1}^{K} r_{ij}^{Z_{ij}}

	r_{ij} = \frac{
		R_{ij} e^{\psi (\tau_{j})}
	}{
		\sum\limits_{k=1}^{K} R_{ik} e^{\psi (\tau_{k})}
	}

Under this distribution, we have

.. math::

	\mathbb{E}_{\mathbf{Z} \sim q(\mathbf{Z})} \left[ Z_{ij} \right] = r_{ij}

To summarize, the joint posterior over :math:`\mathbf{Z}` and :math:`\boldsymbol{\tau}` is 

.. math::

	q(\mathbf{Z}) = \prod\limits_{i=1}^{N} \prod\limits_{j=1}^{K} r_{ij}^{Z_{ij}}

	q(\boldsymbol{\tau}) = \text{Dirichlet} \left( \alpha_{0} + \alpha_{1}, ..., \alpha_{0} + \alpha_{K} \right)

	r_{ij} = \frac{
		R_{ij} e^{\psi (\tau_{j})}
	}{
		\sum\limits_{k=1}^{K} R_{ik} e^{\psi (\tau_{k})}
	}

	\alpha_{j} = \sum\limits_{i=1}^{N} n_{i} r_{ij}

The two factors of :math:`q` are completely specified by the factors
:math:`\mathbf{r}` and :math:`\boldsymbol{\tau}`. The algorithm for refining
these factors is:

	* Evaluate the likelihood function for each trajectory-state pairing: :math:`R_{ij} = f(X_{i} | \boldsymbol{\theta}_{j})`.

	* Initialize :math:`\boldsymbol{\alpha}` and :math:`\mathbf{r}` such that

	.. math::

		\alpha_{j}^{(0)} = \alpha_{0}

		r_{ij}^{(0)} = \frac{R_{ij}}{\sum\limits_{k=1}^{K} R_{ik}}

	* At each iteration :math:`t = 1, 2, ...`:

		1. For each :math:`j = 1, ..., K`, set :math:`\alpha_{j} = \alpha_{0} + \sum\limits_{i=1}^{N} n_{i} r_{ij}^{(t-1)}`.
		2. For each :math:`i = 1, ..., N` and :math:`j = 1, ..., K`, set :math:`r_{ij}^{(*)} = R_{ij} e^{\psi (\alpha_{j}^{(t)})}`.
		3. Normalize :math:`\mathbf{r}` over all states for each trajectory :math:`r_{ij}^{(t)} = \frac{r_{ij}^{*}}{\sum\limits_{k=1}^{K} r_{ik}^{*}}`

This is the state array algorithm implemented in ``saspt``. After inference,
we can summarize the posterior using its mean:

.. math::

	\mathbb{E}_{q(\boldsymbol{\tau})} \left[ \tau_{j} \right] = \frac{\alpha_{j} + \alpha_{0}}{\sum\limits_{k=1}^{K} \alpha_{k} + \alpha_{0}}

	\mathbb{E}_{q(\mathbf{Z})} \left[ Z_{ij} \right] = r_{ij}

These are the values reported to the user as ``StateArray.posterior_occs`` and ``StateArray.posterior_assignment_probabilities``.

Accounting for defocalization
=============================


