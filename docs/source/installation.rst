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

If willing to use numerical differentiation, a patch has to be applied to `numdifftools <https://pypi.org/project/numdifftools/>`_. This can be done by running the following command while being in the environment ``gwfast`` has been installed into

.. code-block:: console

  $ patch $(python -c "import site; print(site.getsitepackages()[0])")"/numdifftools/limits.py" $(python -c "import site; print(site.getsitepackages()[0])")"/gwfast/.patch/patch_ndt_complex_0-9-41.patch
