.. _install_label:

=======
Install
=======

``saspt`` has only been tested with Python 3.

Option 1: install with pip
==========================

``saspt`` is in active development. The test version is currently hosted on `TestPyPI <https://test.pypi.org/project/saspt-test/0.1.7/>`_ and can be installed with ``pip``:

.. code-block:: bash

    python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ saspt-test==0.1.7

Option 2: install from source
=============================

1. Clone the ``saspt`` repo:

.. code-block:: bash

    git clone https://github.com/alecheckert/saspt.git
    cd saspt

2. Build and switch to the ``saspt_env`` conda environment:
    
.. code-block:: bash

    conda env create -f example_env.yaml
    conda activate saspt_env

3. Install the ``saspt`` package:

.. code-block:: bash

    pip install .

We recommend running the testing suite after installing:

.. code-block:: bash

    pytest tests

Dependencies
============

    * ``numpy``
    * ``scipy``
    * ``dask``
    * ``pandas``
    * ``matplotlib``
    * ``tqdm`` (`pip <https://pypi.org/project/tqdm/>`_)
    * ``seaborn``

All dependencies are available via ``conda`` using either the ``defaults`` or ``conda-forge`` channels (`example environment spec <https://github.com/alecheckert/saspt/blob/main/example_env.yaml>`_).
