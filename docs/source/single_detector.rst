.. _signal_class:

Single detector analysis
========================

The :py:class:`signal` module allows to model the response of a single detector to a GW signal. We here present the core class, :py:class:`gwfast.signal.GWSignal`, and its methods.

Initialising a single detector
------------------------------

A single detector object can be initialised as follows

.. autoclass:: gwfast.signal.GWSignal

The locations and orientations of some detectors are already provided in :py:data:`gwfast.gwfastGlobals.detectors`

.. py:data:: gwfast.gwfastGlobals.detectors

  :type: dict(dict, dict, ...)

.. exec::
  import json
  from gwfast.gwfastGlobals import detectors
  import copy
  detsprint = copy.deepcopy(detectors)
  for key in detsprint.keys():
      for key2 in detsprint[key].keys():
          if type(detsprint[key][key2]) == float:
              detsprint[key][key2] = round(detsprint[key][key2], 3)
  json_obj = json.dumps(detsprint, sort_keys=False, indent=4)
  json_obj = json_obj[:-1] + "    }"
  print('.. code-block:: JavaScript\n\n    %s\n\n' % json_obj)

where ``"L1"`` and ``"H1"`` denote the two `LIGO detectors <https://www.ligo.caltech.edu>`_ in Livingston and Hanford, respectively, ``"Virgo"`` denotes the `Virgo detector <https://www.virgo-gw.eu>`_, ``"KAGRA"`` denotes the `KAGRA detector <https://gwcenter.icrr.u-tokyo.ac.jp/en/>`_, ``"LIGOI"`` denotes the `LIGO India detector <https://www.ligo-india.in>`_, ``"ETS"`` and ``"ETMR"`` denote the `ET detector <https://www.et-gw.eu>`_ in Sardinia and Meuse-Rhine, while ``"CE1Id"``, ``"CE2NM"`` and ``"CE2NSW"`` denote the `CE detector(s) <https://cosmicexplorer.org>`_ in Idaho, New Mexico and New South Wales, respectively.

Also many ASDs for different stages and desings of the detectors are included in the ``gwfast/psds/`` directory, see :ref:`PSDs_README`.

It is possible to initialise the ``seed`` for the duty cycle, i.e. the percentage of time the detector (each detector independently in the case of a triangular detector) is supposed to be operational, using

.. automethod:: gwfast.signal.GWSignal._update_seed

Projecting the signal onto the detector
---------------------------------------

We will now list the methods inside :py:class:`gwfast.signal.GWSignal` needed to project the signal onto a single detector. We refer to :ref:`parameters_names` for a complete list of the parameters conventions, units and ranges.

.. _pattern_function:

Pattern functions
"""""""""""""""""

The so-called detector pattern functions :math:`F_+ (\theta, \phi, t, \psi)` and :math:`F_{\times} (\theta, \phi, t, \psi)` as a function of the sky position of the event, time and polarisation can be computed as

.. automethod:: gwfast.signal.GWSignal._PatternFunction

Time delay from Earth center
""""""""""""""""""""""""""""

The time needed to go from the Earth center to the detector location, as a function of the sky position of the event(s) and time(s), can be computed as

.. automethod:: gwfast.signal.GWSignal._DeltLoc

Amplitude at the detector
"""""""""""""""""""""""""

The GW signal amplitudes :math:`A_+` and :math:`A_{\times}` at the detector, for a set of event(s) parameters and frequency(ies) can be computed as

.. automethod:: gwfast.signal.GWSignal.GWAmplitudes

Complete phase
""""""""""""""

The complete phase(s) of the signal(s) [i.e. including the coalescence phase :math:`\Phi_{c}` and time :math:`t_c` terms and the :math:`-\pi/4`], for a set of event(s) parameters and frequency(ies) can be computed as

.. automethod:: gwfast.signal.GWSignal.GWPhase

Complete strain
"""""""""""""""

The complete signal strain(s) at the detector for a set of event(s) parameters and frequency(ies) can be computed as

.. automethod:: gwfast.signal.GWSignal.GWstrain

.. _formalism_setup:

Formalism setup
---------------

We recall here the basics of the Fisher matrix formalism in GW parameter estimation. For comprehensive treatments see `arXiv:gr-qc/9402014 <https://arxiv.org/abs/gr-qc/9402014>`_, `arXiv:gr-qc/0703086 <https://arxiv.org/abs/gr-qc/0703086>`_ and `arXiv:1308.1397 <https://arxiv.org/abs/1308.1397>`_, which study also the limitations of the approach.
We assume that the time-domain signal in a GW detector can be written as the superposition of an expected signal :math:`h_0` and stationary, Gaussian noise :math:`n` with zero mean

.. math:: s(t) = h_0(t) + n(t)
  :label: signal_def

The statistical properties of the noise are encoded in the one-sided Power Spectral Density (PSD) :math:`S_n(f)`, defined by

.. math:: \langle \tilde{n}^*(f) \tilde{n}(f') \rangle = \dfrac{1}{2}\delta(f−f') S_n(f)

where the tilde denotes a temporal Fourier transform.

This determines an inner product for any two signals :math:`g(t)` and :math:`h(t)`

.. math:: \left( g \, | \, h \right) =  4 {\rm Re}{ \int_0^{\infty} {\rm d}f \, \frac{\tilde{g}^*(f) \, \tilde{h}(f) }{S_{n}(f)}}
  :label: innerprod_def

Using eq. :eq:`innerprod_def` we can express the signal–to–noise ratio (SNR) of the true signal as

.. math:: {\rm SNR} =  \left(h_0 \, | \, h_0\right)^{1/2}
  :label: SNR_def

Eq.s :eq:`signal_def` and eq. :eq:`innerprod_def` result in the following likelihood for a data realisation :math:`s` conditioned on the waveform parameters :math:`\overrightarrow{\theta}`

.. math:: \mathcal{L}(s \;|\; \overrightarrow{\theta}) \propto \exp\left\{-\frac{1}{2}\left( s -h(\overrightarrow{\theta})  \, | \, s -h(\overrightarrow{\theta}) \right) \right\}
  :label: lik_def

The Fisher Information Matrix (FIM) for the likelihood in eq. :eq:`lik_def` is defined as

.. math:: \Gamma_{ij} \equiv - \langle \partial_{i} \partial_{j} \log \mathcal{L}(s \,|\, \overrightarrow{\theta})  \rangle_n {\Big |}_{\overrightarrow{\theta}=\overrightarrow{\theta}_0} =\left(h_i \, | \, h_j\right)
  :label: Fisher_def

where :math:`h_i \equiv \partial_i h`, and the notation :math:`\langle\dots\rangle_n` denotes an average over noise realizations with fixed parameters.
Near a maximum of the likelihood, the latter is approximated by a multivariate Gaussian with covariance :math:`\Gamma_{ij}^{-1}`.

From the FIM it is then possible to have an estimation of the errors attainable on the parameters, without having to perform a full Bayesian parameter estimation, which is computationally very expensive.

SNR computation in a single detector
------------------------------------

To compute the SNR defined in eq. :eq:`SNR_def` in a single detector, for one or multiple events, it is sufficient to use the function

.. automethod:: gwfast.signal.GWSignal.SNRInteg

Signal derivatives
------------------

The fundamental ingredient to build the FIM in eq. :eq:`Fisher_def` is the computation of the signal derivatives with respect to the parameters.
In its pure Python implementation (when using the built-in :ref:`wf_models_py`), ``gwfast`` as a defualt uses a mixture of *automatic differentiation* through the `JAX <https://github.com/google/jax>`_ package and *analytical derivatives*.

Automatic differentiation is a technique to compute derivatives of any order in a *pseudo-analytic* way, iteratively applying the *chain rule* on a given function.
For this to work it is required that the function is written in a way the machine can understand, in our case pure Python.
For a review of automatic differentiation see `arXiv:1811.05031 <https://arxiv.org/abs/1811.05031>`_.
This technique ensures both **numerical accuracy** and **speed**.

``gwfast`` also offers the possibility to compute the derivatives using numerical differentiation (finite differences) through the `numdifftools <https://github.com/pbrod/numdifftools>`_ package.
This is the default option if using :ref:`wf_models_ext`.

The function to compute signal derivatives for one or multiple events is

.. automethod:: gwfast.signal.GWSignal._SignalDerivatives

Analytical derivatives can be computed for the parameters ``dL``, ``theta``, ``phi``, ``psi``, ``tcoal``, ``Phicoal`` and ``iota`` (the latter only for the fundamental mode in the non-precessing case).
These are cross-checked both with ``JAX`` derivatives and an independent `Wolfram Mathematica <https://www.wolfram.com/mathematica/>`_ code.
Computing analytical derivatives for these parameters considerably speeds-up the computation and further improves the accuracy.

The function that computes analytical derivatives for one or multiple events is

.. automethod:: gwfast.signal.GWSignal._AnalyticalDerivatives

Fisher matrix computation in a single detector
----------------------------------------------

To compute the FIM defined in eq. :eq:`Fisher_def` in a single detector, for one or multiple events, it is sufficient to use the function

.. automethod:: gwfast.signal.GWSignal.FisherMatr

.. note::
  We recall that the order (row/column numbers) in which the parameters appear in the FIM is stored in the :py:class:`gwfast.waveforms.WaveFormModel.ParNums` attribute of the :py:class:`gwfast.waveforms.WaveFormModel` object.

Optimal location for a single detector
--------------------------------------

``gwfast`` provides a method to compute the optimal location in the sky for a signal to be detected with the highest possible SNR at a given time. This is obtained maximising the :ref:`pattern_function` of the detector and the computation assumes :math:`\psi=0`.

To compute the optimal :math:`\theta` and :math:`\phi` sky coordinates for a single detector it is possible to use

.. automethod:: gwfast.signal.GWSignal.optimal_location

.. note::
  Even if considering Earth rotation, the highest SNR is still be obtained if the source is in the optimal location close to the merger.

.. _wf_overlap_single:

Waveform overlap for a single detector
--------------------------------------

.. versionadded:: 1.0.2

From the inner product in eq. :eq:`innerprod_def`, it is also possible to define the so-called *overlap* between two waveforms as

.. math:: {\rm overlap}(h_1, h_2) \equiv \frac{(h_1|h_2)}{\sqrt{(h_1|h_1)(h_2|h_2)}} = \frac{(h_1|h_2)}{\rm SNR_1 \, SNR_2}

``gwfast`` offers the possibility to compute this quantity for a single detector on two sets of events parameters (for one or multiple events at a time), using the function

.. automethod:: gwfast.signal.GWSignal.WFOverlap
