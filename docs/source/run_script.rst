
Running script
==============

``gwfast`` includes the executable :py:class:`calculate_forecasts_from_catalog.py` that implements parallelisation routines, on top of the possibility to vectorise the calculation of the FIM, and is ready to use both on single machines and on clusters.
This is ideal to use ``gwfast`` for forecasting parameter estimation capabilities for large catalogs of sources, which is the main purpose of the code.

To use this script it is necessary to store a catalog in ``h5`` format in the folder ``data/``. This can be done using the function :py:class:`gwfast.gwfastUtils.save_data`.

After creating a folder to store the results, e.g.

.. code-block:: console

  $ mkdir my_results

it is possible to run the executable :py:class:`calculate_forecasts_from_catalog.py` as

.. code-block::

  usage: calculate_forecasts_from_catalog.py [-h] --fname_obs FNAME_OBS
                                             --fout FOUT
                                             [--wf_model WF_MODEL]
                                             [--batch_size BATCH_SIZE]
                                             [--npools NPOOLS] [--snr_th SNR_TH]
                                             [--idx_in IDX_IN] [--idx_f IDX_F]
                                             [--fmin FMIN] [--fmax FMAX]
                                             [--compute_fisher COMPUTE_FISHER]
                                             [--net NET [NET ...]]
                                             [--netfile NETFILE]
                                             [--psds PSDS [PSDS ...]]
                                             [--mpi MPI]
                                             [--duty_factor DUTY_FACTOR]
                                             [--concatenate CONCATENATE]
                                             [--params_fix PARAMS_FIX [PARAMS_FIX ...]]
                                             [--rot ROT]
                                             [--lalargs LALARGS [LALARGS ...]]
                                             [--return_all RETURN_ALL]
                                             [--seeds SEEDS [SEEDS ...]]

Named Arguments
---------------

--fname_obs

  Name of the file containing the catalog, without the extension ``h5``.

  *Default*: ``''``

--fout

  Path to output folder, which has to exist before the script is launched.

  *Default*: ``'test_gwfast'``

--wf_model

  Name of the waveform model.

  The names corresponding to the various waveform models described in :ref:`wf_models` are:

      - ``'tf2'`` for :py:class:`gwfast.waveforms.TaylorF2_RestrictedPN`;
      - ``'tf2_tidal'`` for :py:class:`gwfast.waveforms.TaylorF2_RestrictedPN` including tidal effects;
      - ``'tf2_ecc'`` for :py:class:`gwfast.waveforms.TaylorF2_RestrictedPN` including orbital eccentricity;
      - ``'IMRPhenomD'`` for :py:class:`gwfast.waveforms.IMRPhenomD`;
      - ``'IMRPhenomD_NRTidalv2'`` for :py:class:`gwfast.waveforms.IMRPhenomD_NRTidalv2`;
      - ``'IMRPhenomHM'`` for :py:class:`gwfast.waveforms.IMRPhenomHM`;
      - ``'IMRPhenomNSBH'`` for :py:class:`gwfast.waveforms.IMRPhenomNSBH`;
      - ``'LAL-wfname'`` for :py:class:`gwfast.waveforms.LAL_WF` , with ``wfname`` being the name of the desired approximant present in ``LAL`` (e.g. ``'LAL-IMRPhenomXPHM'``);

        .. note::
          In this case it is **necessary** to specify the waveform characteristics (``'tidal'``, ``'HM'``, ``'precessing'`` and ``'eccentric'``) through the **--\ --lalargs** argument.

      - ``'TEOBResumSPA'`` for :py:class:`gwfast.waveforms.TEOBResumSPA_WF` ;
      - ``'TEOBResumSPA_tidal'`` for :py:class:`gwfast.waveforms.TEOBResumSPA_WF` including tidal effects;

  *Default*: ``'tf2'``

--batch_size

  Size of the batch to be computed in vectorised form on each process.

  *Default*: ``1``

--npools

  Number of parallel processes.

  *Default*: ``1``

--snr_th

  Threshold value for the SNR to consider the event detectable. FIMs are computed only for events with SNR exceeding this value.

  *Default*: ``12.0``

--idx_in

  Index of the event in the catalog from which to start the calculation.

  *Default*: ``0``

--idx_f

  Index of the event in the catalog from which to end the calculation.

  *Default*: ``None``

--fmin

  Minimum frequency of the grid, in :math:`\rm Hz`.

  *Default*: ``2.0``

--fmax

  Maximum frequency of the grid, in :math:`\rm Hz`. If not specified, this coincides with the cut frequency of the waveform.

  *Default*: ``None``

--compute_fisher

  Int specifying if the FIMs have to be computed (``1``) or not (``0``).

  *Default*: ``1``

--net

  The network of detectors to be used, separated by *single spacing*.

  It is possible to use the locations and orientations of the detectors are already provided in :py:data:`gwfast.gwfastGlobals.detectors`, passing the corresponding keys.

  *Default*: ``['ETS']``

--netfile

  ``json`` file containing the detector configuration, alternative to **--\ --net** and **--\ --psds**.

  The ``json`` files can be created using the :py:class:`gwfast.gwfastUtils.save_detectors` function.

  *Default*: ``None``

--psds

  The paths to PSDs of each detector in the network inside the folder ``psds/``, separated by *single spacing*.

  For a list and description of the available PSDs and ASDs see :ref:`PSDs_README`.

  *Default*: ``['ET-0000A-18.txt']``

--mpi

  Int specifying if the code has to parallelise using `multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`_ (``0``), or using `MPI <https://mpi4py.readthedocs.io/en/stable/>`_ (``1``), suitable for clusters.

  *Default*: ``0``

--duty_factor

  Duty factor of the detectors (the same is used for all detectors in a network).

  *Default*: ``1.0``

--concatenate

  Int specifying if the results of the individual batches have to be concatenated (``1``) or not (``0``).

  *Default*: ``1``

--params_fix

  List of parameters to fix to the fiducial values, i.e. to eliminate from the FIMs, separated by *single spacing*. The names have to be the same as in :ref:`parameters_names`.

  *Default*: ``[]``

--rot

  Int specifying if the effect of the rotation of the Earth has to be included in the analysis (``1``) or not (``0``).

  *Default*: ``1``

--lalargs

  Specifications of the waveform when using LAL interface, separated by *single spacing*.

  The specifications can be:

    - ``'tidal'`` if the waveform includes tidal effects;
    - ``'HM'`` if the waveform includes the contribution of sub-dominant (higher-order) modes;
    - ``'precessing'`` if the waveform includes precessing spins;
    - ``'eccentric'`` if the waveform includes orbital eccentricity.

  *Default*: ``[]``

--return_all

  Int specifying if, in case a network of detectors is used, the SNRs and Fisher matrices of the individual detector have to be stored (``1``) or not (``0``).

  Default: ``0``

--seeds

  List of seeds to set for the duty factors in individual detectors, to help reproducibility, separated by *single spacing*.

  Default: ``[]``


Script outputs
--------------

The script will produce the following files in the output folder (in alphabetical order):

  - ``cond_numbers_idxs.txt``: File containing the condition numbers of the FIMs of the detected events (i.e. having SNR > **--\ --snr_th**);
  - ``covs_idxs.npy``: File containing the covariance matrices of the detected events (i.e. having SNR > **--\ --snr_th**). The order of the parameters is the one given in :py:class:`gwfast.waveforms.WaveFormModel.ParNums` (with the exception of the parameters that have been fixed through **--\ --params_fix**);
  - ``detectors.json``: File containing the detector configuration used, as produced by the :py:class:`gwfast.gwfastUtils.save_detectors` function;
  - ``errors_idxs.txt``: File containing the errors on the parameters for the detected events (i.e. having SNR > **--\ --snr_th**). The order of the parameters is the one given in :py:class:`gwfast.waveforms.WaveFormModel.ParNums` (with the exception of the parameters that have been fixed through **--\ --params_fix**);
  - ``events_detected_idxs.hdf5``: File containing the parameters of the detected events (i.e. having SNR > **--\ --snr_th**). This is a dictionary as :py:data:`events`, saved through :py:class:`gwfast.gwfastUtils.save_data` , that can be loaded through :py:class:`gwfast.gwfastUtils.load_population`;
  - ``fishers_idxs.npy``: File containing the FIMs of the detected events (i.e. having SNR > **--\ --snr_th**). The order of the parameters is the one given in :py:class:`gwfast.waveforms.WaveFormModel.ParNums` (with the exception of the parameters that have been fixed through **--\ --params_fix**);
  - ``idxs_det_idxs.txt``: File containing the indices of the detected events (i.e. having SNR > **--\ --snr_th**) in the original catalog;
  - ``inversion_errors_idxs.txt``: File containing the inversion errors of the FIMs of the detected events (i.e. having SNR > **--\ --snr_th**);
  - ``sky_area_idxs.txt``: File containing the 90\% sky localisation areas of the detected events (i.e. having SNR > **--\ --snr_th**) in :math:`\rm deg^2`;
  - ``snrs_idxs.txt``: File containing the SNRs of all the events in the original catalog;

  .. note::
    All the above files refer to quantities evaluated by the full detector network.

.. note::
  The suffix ``idxs`` present in all file names refers to the initial and final indices of the events used in the original catalog. As an example, if using the events from ``0`` to ``5000`` in the original catalog, the suffix of all files will be ``0_to_5000``, e.g. ``snrs_0_to_5000.txt``.

.. note::
  The ``npy`` files can be loaded using the :py:class:`numpy.load` function.

Also, if **--\ --return_all** = 1, the following files will be produced:

  - ``all_fishers_idxs.hdf5`` : File containing a dictionary with the FIMs of the detected events (i.e. having SNR > **--\ --snr_th**), both for the full network and for the single detectors. The order of the parameters is the one given in :py:class:`gwfast.waveforms.WaveFormModel.ParNums` (with the exception of the parameters that have been fixed through **--\ --params_fix**);
  - ``all_snrs_idxs.hdf5`` : File containing a dictionary with the SNRs of all the events in the original catalog, both for the full network and for the single detectors.
