.. _installation:

Installation
============

``gwfast`` can be easily installed using pip, either from `PyPI <https://pypi.org>`_:

.. code-block:: console

  $ pip install --upgrade pip
  $ pip install gwfast

or from `GitHub <https://github.com>`_:

.. code-block:: console

  $ pip install --upgrade pip
  $ pip install --upgrade "jax[cpu]"
  $ pip install git+https://github.com/CosmoStatGW/gwfast


Otheriwse the git repository can be cloned:

.. code-block:: console

  $ git clone https://github.com/CosmoStatGW/gwfast

Then, to install the required packages, simply run

.. code-block:: console

  $ cd gwfast
  $ pip install --upgrade pip
  $ pip install -r requirements.txt

To install a ``JAX`` version for GPU or TPU proceed as explained in `https://github.com/google/jax#installation <https://github.com/google/jax#installation>`_.
