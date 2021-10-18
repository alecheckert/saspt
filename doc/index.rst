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

I've found ``saspt`` particularly useful when applied to fluorescent protein tracking in live cells.
In the complex intracellular environment, a protein may occupy a large and unknown number of molecular
states with distinct dynamics. ``saspt`` provides a simple way to measure the number,
characteristics, and fractional occupations of these states. It is also "smart" enough to deal with 
situations where there may *not* be discrete states.

If you want to jump right into working with ``saspt``, see :ref:`quickstart_label`. If you want a 
more detailed explanation of what ``saspt`` is and why it exists, see :ref:`description_label` 
(**currently under construction**). If you're feeling adventurous and want to see the guts of the 
actual model/inference algorithm, see :ref:`label_model`.

(``saspt`` stands for "state arrays for single particle tracking".)

.. toctree::
    :maxdepth: 3
    :caption: Contents:
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
