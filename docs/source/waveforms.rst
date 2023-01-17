.. _wf_models:

Waveform models
===============

Waveform models give the prediction for a GW signal emitted by a coalescing binary as a function of the :ref:`parameters of the source <parameters_names>`.

``gwfast`` uses Fourier domain waveform models. In particular, it provides a Python implementation for some selected state-of-the-art models, plus wrappers to use both the models present in the `LIGO Algorithm Library <https://wiki.ligo.org/Computing/LALSuite>`_, ``LAL``, and the `TEOBResumS <https://bitbucket.org/eob_ihes/teobresums/wiki/Home>`_ models.

We separately release also `WF4Py <https://github.com/CosmoStatGW/WF4Py>`_, a user-friendly package containing only the waveform models implemented in ``gwfast`` (avoiding some specific implementations needed for ``JAX`` derivatives).
Our Python implementation is fully vectorised, thus giving the possibility to evaluate waveforms for multiple events at a time.

.. _wf_model_class:

The WaveFormModel class
-----------------------

Waveforms in ``gwfast`` are built as classes, initialised as follows

.. autoclass:: gwfast.waveforms.WaveFormModel

Each waveform includes four fundamental methods, to compute, given the parameter of the source(s), the *phase* of the signal, :math:`\Phi (f)`, the *amplitude* of the signal, :math:`A (f)`, the *time to coalescence* as a function of the frequency, :math:`\tau^* (f)`, and the *cut frequency*.

.. automethod:: gwfast.waveforms.WaveFormModel.Phi

.. automethod:: gwfast.waveforms.WaveFormModel.Ampl

.. automethod:: gwfast.waveforms.WaveFormModel.tau_star

.. note::
  For :py:class:`gwfast.waveforms.NewtInspiral` we use the :math:`\tau^* (f)` expression in `M. Maggiore -- Gravitational Waves Vol. 1 <https://global.oup.com/academic/product/gravitational-waves-9780198570745?q=Michele%20Maggiore&lang=en&cc=it>`_ eq. (4.21).
  For the other models instead we use the :math:`\tau^* (f)` expression in `arXiv:0907.0700 <https://arxiv.org/abs/0907.0700>`_ eq. (3.8b), valid up to 3.5 PN.

.. automethod:: gwfast.waveforms.WaveFormModel.fcut

Some models also have a method denoted as :py:class:`hphc` to compute directly the :math:`\tilde{h}_+` and :math:`\tilde{h}_{\times}` polarisations of the gravitational wave, see below.

Each waveform further includes a dictionary storing the order of the parameters in the Fisher matrix

.. autoattribute:: gwfast.waveforms.WaveFormModel.ParNums

.. _wf_models_py:

Python waveform models
----------------------

We here report and briefly describe the waveform models implemented in pure Python in ``gwfast``. We carefully checked that our Python implementation accurately reproduces the original ``LAL`` waveforms.

.. note::
  Only when using these models it will be possible to compute the derivatives using Automatic Differentiation through ``JAX``.

Leading order inspiral
""""""""""""""""""""""

This model includes only the leading order term in the inspiral.

.. autoclass:: gwfast.waveforms.NewtInspiral

.. _TF2:

TaylorF2
""""""""

.. versionadded:: 1.0.2
  Added the possibility to terminate the waveform at the ISCO frequency of a remnant Kerr BH.
  Added the spin-induced quadrupole due to tidal effects.

This model includes contributions to the phase up to 3.5 order in the *Post Newtonian*, PN, expansion, and can thus be used to describe the inspiral. The amplitude is the same as in Newtonian approximation.
Our implementation can include both the tidal terms at 5 and 6 PN (see `arXiv:1402.5156 <https://arxiv.org/abs/1402.5156>`_) and a moderate eccentricity in the orbit :math:`e_0\lesssim 0.1` up to 3 PN (see `arXiv:1605.00304 <https://arxiv.org/abs/1605.00304>`_).
There is no limitation in the parameters range, but, being an inspiral-only model, :py:class:`gwfast.waveforms.TaylorF2_RestrictedPN` is better suited for BNS systems.

.. autoclass:: gwfast.waveforms.TaylorF2_RestrictedPN

.. _IMRPhenomD:

IMRPhenomD
""""""""""

This is a full inspiral–merger-ringdown model tuned with NR simulations, which can be used to simulate signals coming from BBH mergers, with non–precessing spins up to :math:`|\chi_z|\sim 0.85` and mass ratios up to :math:`q = m_1/m_2 \sim 18`.

.. autoclass:: gwfast.waveforms.IMRPhenomD

.. _IMRPhenomD_NRTidalv2:

IMRPhenomD_NRTidalv2
""""""""""""""""""""

This is a full inspiral-merger-ringdown model tuned with NR simulations, which extends the :ref:`IMRPhenomD` model to include tidal effects, and can thus be used to accurately describe signals coming from BNS mergers. The validity has been assessed for masses between :math:`1\,{\rm M}_{\odot}` and :math:`3\,{\rm M}_{\odot}`, spins up to :math:`|\chi_z|\sim 0.6` and tidal deformabilities up to :math:`\Lambda_i\sim 5000`.

.. autoclass:: gwfast.waveforms.IMRPhenomD_NRTidalv2

The model includes a Planck taper filter to terminate the waveform after merger, we thus the cut frequency slightly before the end of the filter, for numerical stability.

.. _IMRPhenomHM:

IMRPhenomHM
"""""""""""

This is a full inspiral–merger-ringdown model tuned with NR simulations, which takes into account not only the :math:`(2,2)` quadrupole of the signal, but also the sub–dominant multipoles :math:`(l,m) = (2,1),\, (3,2),\, (3,3),\, (4,3)`, and :math:`(4,4)`, that can be particularly relevant to better describe the signal coming from BBH systems. The calibration range is the same of the :ref:`IMRPhenomD` model.

.. autoclass:: gwfast.waveforms.IMRPhenomHM

To avoid for loops and recomputing coefficients (given that in this case the amplitude cannot be computed separately from the phase), this model features a :py:class:`gwfast.waveforms.IMRPhenomHM.hphc` method, to compute directly :math:`\tilde{h}_+` and :math:`\tilde{h}_{\times}`

.. automethod:: gwfast.waveforms.IMRPhenomHM.hphc

Also, in this case the :py:class:`gwfast.waveforms.IMRPhenomHM.Phi` and :py:class:`gwfast.waveforms.IMRPhenomHM.Ampl` methods return dictionaries containing the phases and amplitudes of the various modes, respectively. The dictionaries have keys ``'21'``, ``'22'``, ``'32'``, ``'33'``, ``'43'`` and ``'44'``.

.. automethod:: gwfast.waveforms.IMRPhenomHM.Phi

.. automethod:: gwfast.waveforms.IMRPhenomHM.Ampl

To combine the various modes and obtain the full amplitude and phase from these outputs, it is possible to use the function

.. autofunction:: gwfast.gwfastUtils.Add_Higher_Modes

.. _IMRPhenomNSBH:

IMRPhenomNSBH
"""""""""""""

This is a full inspiral–merger-ringdown model tuned with NR simulations, which can describe the signal coming from the merger of a NS and a BH, with mass ratios up to :math:`q = m_1/m_2 \sim 100`, also taking into account tidal effects and the impact of the possible tidal disruption of the NS.

.. autoclass:: gwfast.waveforms.IMRPhenomNSBH

.. note::

  In ``LAL``, to compute the parameter :math:`\xi_{\rm tide}` in `arXiv:1509.00512 <https://arxiv.org/abs/1509.00512>`_ eq. (8), the roots are extracted.
  In Python this would break the possibility to vectorise so, to circumvent the issue, we compute a grid of :math:`\xi_{\rm tide}` as a function of the compactness, mass ratio and BH spin, and then use a 3D interpolator.
  The first time the code runs, if this interpolator is not already present, it will be computed.
  The base resolution of the grid is 200 pts per parameter, that we find sufficient to reproduce the ``LAL`` implementation with good precision, given the smooth behaviour of the function, but this can be raised if needed.
  In this case, it is necessary to change the name of the file assigned to :py:data:`self.path_xiTide_tab` and the ``res`` input passed to the function that loads the grid.

  .. automethod:: gwfast.waveforms.IMRPhenomNSBH._make_xiTide_interpolator

.. _wf_models_ext:

Wrappers to external waveform models
------------------------------------

``gwfast`` features wrappers to use the waveform models implemented in other libraries.

.. note::
  When using these models the derivatives are computed using Numerical Differentiation (finite differences) through the `numdifftools <https://github.com/pbrod/numdifftools>`_ package.

LAL wrapper
"""""""""""

This is a wrapper to use the waveform models present in the `LIGO Algorithm Library <https://wiki.ligo.org/Computing/LALSuite>`_

.. autoclass:: gwfast.waveforms.LAL_WF

.. note::
  As a defualt, we use the ``LAL`` function :py:class:`SimInspiralChooseFDWaveformSequence`, which computes the waveform on a given frequency grid. However, this shows numerical issues with some waveform models (e.g. ``IMRPhenomXHM``), we thus also give the possibility to use the function :py:class:`SimInspiralChooseFDWaveform` which appears more stable. This performs the computation on a ``LAL`` defined grid, which then has to be interpolated, resulting in less accurate evaluation at low frequencies and slower execution time.
  This can be chosen with the boolean ``compute_sequence``: setting it to ``True`` means that the function will perform the computation directly on the user grid, setting it to False it will let ``LAL`` choose the grid and then extrapolate.

.. warning::
  In this case it is **necessary** to specify if the chosen waveform includes tidal effects, higher order modes, precessing spins or orbital eccentricity through the ``is_tidal``, ``is_HigherModes``, ``is_Precessing`` and ``is_eccentric`` arguments, respectively.

Given that ``LAL`` outputs directly :math:`\tilde{h}_+` and :math:`\tilde{h}_{\times}`, this class has a :py:class:`gwfast.waveforms.LAL_WF.hphc` method

.. automethod:: gwfast.waveforms.LAL_WF.hphc

TEOBResumS wrapper
""""""""""""""""""

.. versionadded:: 1.0.2

This is a wrapper to use the ``TEOBResumS`` waveform models, available `here <https://bitbucket.org/eob_ihes/teobresums/wiki/Home>`_, in their Fourier domain version ``TEOBResumSPA``

.. autoclass:: gwfast.waveforms.TEOBResumSPA_WF

.. warning::
  In this case it is **necessary** to specify if the waveform has to include tidal effects or precessing spins through the ``is_tidal`` and ``is_Precessing`` arguments, respectively.

Given that ``TEOBResumS`` can output directly :math:`\tilde{h}_+` and :math:`\tilde{h}_{\times}`, this class has a :py:class:`gwfast.waveforms.TEOBResumSPA_WF.hphc` method

.. automethod:: gwfast.waveforms.TEOBResumSPA_WF.hphc
