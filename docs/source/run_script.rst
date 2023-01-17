
Running script
==============

``gwfast`` includes the executable :py:class:`calculate_forecasts_from_catalog.py` that implements parallelization routines, on top of the possibility to vectorize the calculation of the FIM, and is ready to use both on single machines and on clusters.
This is ideal to use ``gwfast`` for forecasting parameter estimation capabilities for large catalogs of sources, which is the main purpose of the code.

To use this script it is necessary to store a catalog in ``h5`` format in the folder ``data/``. This can be done using the function :py:class:`gwfast.gwfastUtils.save_data`.

After creating a folder to store the results, e.g.

.. code-block:: console

  $ mkdir my_results

it is possible to run the executable :py:class:`calculate_forecasts_from_catalog.py` as

.. argparse::
  :filename: ../run/calculate_forecasts_from_catalog.py
  :func: parser
  :prog: calculate_forecasts_from_catalog.py
  :nodescription:

  --wf_model : @after
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

        - ``'TEOBResumSPA'`` for :py:class:`gwfast.waveforms.TEOBResumSPA_WF`;
        - ``'TEOBResumSPA_tidal'`` for :py:class:`gwfast.waveforms.TEOBResumSPA_WF` including tidal effects.

  --fmin : @replace
    Minimum frequency of the grid, in :math:`\rm Hz`.

    Default: 2.0

  --fmax : @replace
    Maximum frequency of the grid, in :math:`\rm Hz`. If not specified, this coincides with the cut frequency of the waveform.

  --net : @after
    It is possible to use the locations and orientations of the detectors are already provided in :py:data:`gwfast.gwfastGlobals.detectors`, passing the corresponding keys.

  --netfile : @after
    The ``json`` files can be created using the :py:class:`gwfast.gwfastUtils.save_detectors` function.

  --psds : @after
    For a list and description of the available PSDs and ASDs see :ref:`PSDs_README`.

  --mpi : @replace
    Int specifying if the code has to parallelize using `multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`_ (``0``), or using `MPI <https://mpi4py.readthedocs.io/en/stable/>`_ (``1``), suitable for clusters.

  --params_fix : @replace
    List of parameters to fix to the fiducial values, i.e. to eliminate from the FIMs, separated by *single spacing*. The names have to be the same as in :ref:`parameters_names`.

    Default: []

  --lalargs : @after
    The specifications can be:

      - ``'tidal'`` if the waveform includes tidal effects;
      - ``'HM'`` if the waveform includes the contribution of sub-dominant (higher-order) modes;
      - ``'precessing'`` if the waveform includes precessing spins;
      - ``'eccentric'`` if the waveform includes orbital eccentricity.

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
  The suffix ``idxs`` present in all file names refers to the initial and final index of the events used in the original catalog. As an example, if using the events from ``0`` to ``5000`` in the original catalog, the suffix of all files will be ``0_to_5000``, e.g. ``snrs_0_to_5000.txt``.

.. note::
  The ``npy`` files can be loaded using the :py:class:`numpy.load` function.

Also, if **--\ --return_all** = 1, the following files will be produced:

  - ``all_fishers_idxs.hdf5`` : File containing a dictionary with the FIMs of the detected events (i.e. having SNR > **--\ --snr_th**), both for the full network and for the single detectors. The order of the parameters is the one given in :py:class:`gwfast.waveforms.WaveFormModel.ParNums` (with the exception of the parameters that have been fixed through **--\ --params_fix**);
  - ``all_snrs_idxs.hdf5`` : File containing a dictionary with the SNRs of all the events in the original catalog, both for the full network and for the single detectors.
