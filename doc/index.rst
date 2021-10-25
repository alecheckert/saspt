.. saSPT documentation master file, created by
   sphinx-quickstart on Thu Oct 14 09:04:34 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to saSPT's documentation!
=================================

.. note:

    You're here early! The ``saspt`` documentation is under construction. We aim to finish it by the end of October, 2021.

``saspt`` is a Python tool for analyzing single particle tracking (SPT) experiments. It uses 
*state arrays*, a kind of variational Bayesian framework that learns intelligible models 
given raw trajectories from an SPT experiment.

There are a lot of great SPT analysis tools out there. We found it useful to
write ``saspt`` because:

    * it is simple and flexible enough to accommodate a wide variety of underlying stochastic models;
    * its outputs are familiar ``numpy`` and ``pandas`` objects;
    * it imposes no prior beliefs on the number of dynamic states;
    * it uses tried-and-true Bayesian methods (marginalization over nuisance parameters) to deal with measurement error in a natural way.

I originally wrote ``saspt`` to deal with live cell protein tracking experiments.
In the complex intracellular environment, a protein can occupy a large and unknown number of molecular
states with distinct dynamics. ``saspt`` provides a simple way to measure the number,
characteristics, and fractional occupations of these states. It is also "smart" enough to deal with 
situations where there may *not* be discrete states.

If you want to jump right into working with ``saspt``, see :ref:`quickstart_label`. If you want a 
more detailed explanation of why ``saspt`` exists, see :ref:`description_label`.
If you want to dig into the guts of the actual model and inference algorithm, see :ref:`label_model`.

(``saspt`` stands for "state arrays for single particle tracking".)

What does saSPT do?
===================

``saspt`` takes a set of trajectories from a tracking experiment, and identifies a mixture model to explain
them. It is designed to work natively with ``numpy`` and ``pandas`` objects.

What doesn't saSPT do?
======================

    1. ``saspt`` doesn't do tracking; it takes trajectories as input. (See: :ref:`faq_tracking_label`)
    2. ``saspt`` doesn't model *transitions between states*. For that purpose, we recommend the excellent `vbSPT package <https://doi.org/10.1038/nmeth.2367>`_.
    3. ``saspt`` doesn't check the quality of the input data.
    4. ``saspt`` expects you to know the parameters for your imaging experiment, including pixel size, frame rate, and focal depth.

Currently ``saspt`` only supports a small range of physical models. That may change as the package grows.

.. toctree::
    :maxdepth: 3
    :caption: Contents
    :glob:

    install
    getting_started
    description
    model
    api/api
    faqs
    reference

.. py:currentmodule:: saspt


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
