.. _det_net:

Detector networks
=================

The :py:class:`network` module allows to build a detector network, made up of single :py:class:`gwfast.signal.GWSignal` objects. We here present the core class, :py:class:`gwfast.network.DetNet`, and its methods.

Initialising a detector network
-------------------------------

A detector network object can be initialised as follows from multiple :py:class:`gwfast.signal.GWSignal` objects

.. autoclass:: gwfast.network.DetNet

It is possible to initialise all the ``seed`` s for the duty cycle, i.e. the percentage of time each detector is supposed to be operational, using

.. automethod:: gwfast.network.DetNet._update_all_seeds

Also, to store a detector configuration in a ``json`` file, it is possible to use the function

.. autofunction:: gwfast.gwfastUtils.save_detectors

Definition of SNR and Fisher for a detector network
---------------------------------------------------

In :ref:`formalism_setup` we introduced how the SNRs and FIM for a single detector are defined. We here generalise the definition to a detector network.

When considering a detector network, the *network SNR* can be simply computed as the sum in quadrature of the SNRs in each individual detector

.. math:: {\rm SNR} = \sqrt{\sum_{\rm d\in dets} {\rm SNR}_d^2}
  :label: Net_SNR_def

Similarly, when considering a detector network, the *total FIM* can be simply computed as the sum of the FIMs in each individual detector

.. math:: \Gamma_{ij} = \sum_{\rm d\in dets} \Gamma_{ij}^{(d)}
  :label: Net_Fisher_def

SNR computation in a detector network
-------------------------------------

To compute the network SNR defined in eq. :eq:`Net_SNR_def`, for one or multiple events, it is sufficient to use the function

.. automethod:: gwfast.network.DetNet.SNR

Fisher matrix computation in a detector network
-----------------------------------------------

To compute the total network FIM defined in eq. :eq:`Net_Fisher_def`, for one or multiple events, it is sufficient to use the function

.. automethod:: gwfast.network.DetNet.FisherMatr

.. note::
  We recall that the order (row/column numbers) in which the parameters appear in the FIM is stored in the :py:class:`gwfast.waveforms.WaveFormModel.ParNums` attribute of the :py:class:`gwfast.waveforms.WaveFormModel` object.

Optimal location for a detector network
---------------------------------------

``gwfast`` provides a method to compute the optimal location in the sky for a signal to be detected with the highest possible SNR at a given time. This is obtained maximising the sum in quadrature of the :ref:`pattern_function` of the individual detectors, and the computation assumes :math:`\psi=0`.

To compute the optimal :math:`\theta` and :math:`\phi` sky coordinates for a detector network it is possible to use

.. automethod:: gwfast.network.DetNet.optimal_location

.. warning::
  The estimate provided by this function works only if the detectors in the network have comparable characteristics, i.e. PSDs and shape. See the `gwfast code paper <https://arxiv.org/abs/2207.06910>`_ for discussion.

.. note::
  Even if considering Earth rotation, the highest SNR is still be obtained if the source is in the optimal location close to the merger.

Waveform overlap for a detector network
---------------------------------------

.. versionadded:: 1.0.2

In :ref:`wf_overlap_single` we introduced the overlap between two waveform models in a single detector.
Similarly to the SNR and FIM, also this definition can be generalised to a network of detectors, as

.. math:: {\rm overlap}(h_1, h_2) = \frac{\sum_{d\in {\rm dets}}(h_1|h_2)_d}{\sqrt{\sum_{d\in {\rm dets}}(h_1|h_1)_d(h_2|h_2)_d}}

``gwfast`` offers the possibility to compute this quantity for a detector network on two sets of events parameters (for one or multiple events at a time), using the function

.. automethod:: gwfast.network.DetNet.WFOverlap

Relative orientation and distance between detectors
---------------------------------------------------

.. versionadded:: 1.1.0

Relative orientation
""""""""""""""""""""

The relative orientation of two detectors with respect to the *great circle* that joins them can be computed using the function

.. autofunction:: gwfast.gwfastUtils.ang_btw_dets_GC

Detector distance
"""""""""""""""""

- The great circle distance between two detectors can be computed using the function

  .. autofunction:: gwfast.gwfastUtils.dist_btw_dets_GC

- The great circle chord length between two detectors can be computed using the function

  .. autofunction:: gwfast.gwfastUtils.dist_btw_dets_Chord
