.. gwfast documentation master file, created by
   sphinx-quickstart on Tue Dec 20 12:18:43 2022.

Welcome to gwfast's documentation!
==================================

**gwfast** is a Python code for forecasting the *signal-to-noise ratios* and *parameter estimation capabilities* of networks of gravitational-wave detectors, based on the Fisher information matrix approximation.
It is designed for applications to third-generation gravitational-wave detectors.
It is based on *Automatic Differentiation*, which makes use of the library ``JAX``.
This allows numerical accuracy, and the possibility to vectorise the computation *even on a single CPU*, on top of the possibility to parallelise.
The code includes a module for *parallel computation on clusters*.

For further documentation refer to the release papers `arXiv:2207.02771 <https://arxiv.org/abs/2207.02771>`_ and `arXiv:2207.06910 <https://arxiv.org/abs/2207.06910>`_.

.. toctree::
   :maxdepth: 1
   :caption: Package description:

   installation
   create_data
   waveforms
   single_detector
   detector_networks
   fisher_tools
   additional_definitions

.. toctree::
  :maxdepth: 1
  :caption: Tutorials:

  notebooks/gwfast_tutorial
  notebooks/new_features_tutorial

.. toctree::
  :maxdepth: 1
  :caption: Available PSDs:

  psds_readme_link

.. toctree::
  :maxdepth: 1
  :caption: References:

  citation

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
