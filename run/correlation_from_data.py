#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time
import argparse
import configparser
import numpy as np
import multiprocessing
import copy
import h5py
import subprocess
# Needed for mpipool not to stall when trying to write on a file (do not ask me why)
multiprocessing.set_start_method("spawn",force=True)
# commit_info = subprocess.check_output(["git", "describe", "--always"]).strip().decode()
last_commit_info = subprocess.check_output(["git", "log"]).decode("utf-8").split("\n")[0:6]

PACKAGE_PARENT = '../gwfast'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd())))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR,PACKAGE_PARENT )))

import gwfast.gwfastGlobals as glob
import gwfast.gwfastUtils as utils
from gwfast.waveforms import TaylorF2_RestrictedPN, IMRPhenomD, IMRPhenomHM, IMRPhenomD_NRTidalv2, IMRPhenomNSBH
from gwfast.signal import GWSignal
from gwfast.stochastic import stochasticTools as st

try:
    import lal
    from gwfast.waveforms import LAL_WF
except ModuleNotFoundError:
    print('LSC Algorithm Library (LAL) is not installed, only the GWFAST waveform models are available, namely: TaylorF2, IMRPhenomD, IMRPhenomD_NRTidalv2, IMRPhenomHM and IMRPhenomNSBH')
TEOBResumS_installed = False
try:
    import EOBRun_module
    from gwfast.waveforms import TEOBResumSPA_WF
    TEOBResumS_installed = True
except ModuleNotFoundError:
    print('TEOBResumS is not installed, only the GWFAST waveform models are available, namely: TaylorF2, IMRPhenomD, IMRPhenomD_NRTidalv2, IMRPhenomHM and IMRPhenomNSBH')

#####################################################################################
# GLOBALS
#####################################################################################

RESPATH = os.path.join(SCRIPT_DIR, PACKAGE_PARENT, 'results' )

# shortcuts for wf model names; have to be used in input
wf_models_dict = {'IMRPhenomD':IMRPhenomD(),
                  'IMRPhenomHM':IMRPhenomHM(),
                  'tf2':TaylorF2_RestrictedPN(is_tidal=False, use_3p5PN_SpinHO=True),
                  'IMRPhenomD_NRTidalv2': IMRPhenomD_NRTidalv2(),
                  'tf2_tidal':TaylorF2_RestrictedPN(is_tidal=True, use_3p5PN_SpinHO=True),
                  'IMRPhenomNSBH':IMRPhenomNSBH(verbose=False),
                  'tf2_ecc':TaylorF2_RestrictedPN(is_tidal=False, use_3p5PN_SpinHO=True, is_eccentric=True),
                  }
if TEOBResumS_installed:
    wf_models_dict['TEOBResumSPA'] = TEOBResumSPA_WF()
    wf_models_dict['TEOBResumSPA_tidal'] = TEOBResumSPA_WF(is_tidal=True)

def get_pool(mpi=False, threads=None):
    """ Always returns a pool object with a `map()` method. By default,
        returns a `SerialPool()` -- `SerialPool.map()` just calls the built-in
        Python function `map()`. If `mpi=True`, will attempt to import the 
        `MPIPool` implementation from `emcee`. If `threads` is set to a 
        number > 1, it will return a Python multiprocessing pool.
        Parameters
        ----------
        mpi : bool (optional)
            Use MPI or not. If specified, ignores the threads kwarg.
        threads : int (optional)
            If mpi is False and threads is specified, use a Python
            multiprocessing pool with the specified number of threads.
    """

    if mpi:
        from schwimmbad import MPIPool
        print('Using MPI...')
        # Initialize the MPI pool
        pool = MPIPool()

        # Make sure the thread we're running on is the master
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    elif threads > 1:
        print('Using multiprocessing with %s processes...' %(threads-1))
        pool = multiprocessing.Pool(threads-1)

    else:
        raise ValueError('Called get pool with threads=1. No need')
        #pool = SerialPool()

    return pool

def get_indexes(npool, n_events, n_noise_realizations, batch_size):
    # We assume n_noise_realizations>=batch_size & n_noise_realizations%batch_size=0

    nseeds_per_ev = np.ceil(n_noise_realizations/batch_size).astype(int)

    idx_list = np.array_split(np.arange(n_events).astype(int), n_events/n_noise_realizations)
    pools_idx = {str(p):[idx_list[i].tolist() for i in range(len(idx_list)) if i % npool == p] for p in range(npool)}    
    pools_idx_flatten = {k: [item for sublist in v for item in sublist] for k, v in pools_idx.items()}

    idx_ev_list = np.arange(n_events/n_noise_realizations).astype(int)
    evs_in_pool = {str(p):[idx_ev_list[i].tolist() for i in range(len(idx_ev_list)) if i % npool == p] for p in range(npool)}
    noise_idx_pool = {str(p):[np.arange(nseeds_per_ev)+i*nseeds_per_ev for i in evs_in_pool[str(p)]] for p in range(npool)}
    noise_idx_pool_flatten = {k: [item for sublist in v for item in sublist] for k, v in noise_idx_pool.items()}

    return pools_idx_flatten, noise_idx_pool_flatten

def get_net(FLAGS):
    Net= {k: copy.deepcopy(glob.detectors).pop(k) for k in FLAGS.net} 
    for i,psd in enumerate(FLAGS.psds): 
        Net[FLAGS.net[i]]['psd_path'] = os.path.join(glob.detPath, psd)

    return Net

def generate_noise_from_asd(asd_freq, asd_val, freqs, seed=None):

    if seed is not None:
        np.random.seed(seed)

    strainGrids = np.interp(freqs, asd_freq, asd_val, left=1., right=1.)
    scale = 0.5 * strainGrids/np.sqrt(freqs[1] - freqs[0])

    nre = np.random.normal(0., scale)
    nco = np.random.normal(0., scale)

    return nre + 1j*nco

def signal_handle_grid(noiseor, fgrids, asd_freq, asd_val, nptsMax=16368, fmax=2048., df=1./8.):

    if fgrids.shape[0] > nptsMax:
        retnoise = noiseor[:nptsMax+1]
    elif fgrids.shape[0] == nptsMax:
        retnoise = noiseor
    else:
        missingfgrid = np.arange(np.amax(fgrids)+df, fmax+df, df)
        missingfgrids = np.array([missingfgrid for i in range(fgrids.shape[1])]).T
        newnoise = generate_noise_from_asd(asd_freq, asd_val, missingfgrids)
        retnoise = np.concatenate((noiseor, newnoise), axis=0)

    return retnoise

def save_results(correlator, fgrid, fname):

    with h5py.File(fname, 'w') as f:
        pg = f.create_group('correlator')
        for pn in correlator.keys():
            pg.create_dataset(pn, data=correlator[pn], compression='gzip', shuffle=False)

        pg = f.create_group('frequencies')
        pg.create_dataset('fgrid', data=fgrid, compression='gzip', shuffle=False)

def save_results_single_terms(first_term, second_term, third_term, fgrid, n_noise_realizations, fname):

    with h5py.File(fname, 'w') as f:

        f.attrs["n_noise_realizations"] = n_noise_realizations

        pg = f.create_group('term_1')
        for pn in first_term.keys():
            pg.create_dataset(pn, data=first_term[pn]/n_noise_realizations, compression='gzip', shuffle=False)

        pg = f.create_group('term_2')
        for pn in second_term.keys():
            pg.create_dataset(pn, data=second_term[pn]/n_noise_realizations, compression='gzip', shuffle=False)

        pg = f.create_group('term_3')
        for pn in third_term.keys():
            pg.create_dataset(pn, data=third_term[pn], compression='gzip', shuffle=False)

        pg = f.create_group('frequencies')
        pg.create_dataset('fgrid', data=fgrid, compression='gzip', shuffle=False)

def save_results_pool(data, save_dir, pool_idx, suff='true_minus_noise_avg'):

    with h5py.File(os.path.join(RESPATH, save_dir, 'correlator_'+suff+'_'+pool_idx+'.h5'), 'w') as f:
        pg = f.create_group('res')
        for pn in data.keys():
            pg.create_dataset(pn, data=data[pn], compression='gzip', shuffle=False)

def load_results_pool(save_dir, pool_idx, suff='true_minus_noise_avg'):
    data = {}
    with h5py.File(os.path.join(RESPATH, save_dir, 'correlator_'+suff+'_'+pool_idx+'.h5'), 'r') as f:
        for key in f['res'].keys():
            data[key] = np.array(f['res'][key])
    return data

def main(idx, FLAGS):

    evs_cat_in_pool = FLAGS.total_cat_splitted[str(idx)]
    evs_obs_in_pool = FLAGS.obs_cat_splitted[str(idx)]
    snrs_true_in_pool = FLAGS.snrs_true_splitted[str(idx)]
    seeds_list = FLAGS.seeds_list_splitted[str(idx)]
    snr_obs_splitted = FLAGS.snr_obs_splitted[str(idx)]

    indx_list_per_batch = [np.arange(i*FLAGS.batch_size, (i+1)*FLAGS.batch_size)[snrs_true_in_pool[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size]>FLAGS.snr_true_th].tolist() for i in range(int(np.floor(len(snrs_true_in_pool)/FLAGS.batch_size)))]
    if len(snrs_true_in_pool)%FLAGS.batch_size != 0:
        indx_list_per_batch.append(np.arange(int(np.floor(len(snrs_true_in_pool)/FLAGS.batch_size))*FLAGS.batch_size, len(snrs_true_in_pool))[snrs_true_in_pool[int(np.floor(len(snrs_true_in_pool)/FLAGS.batch_size))*FLAGS.batch_size:]>FLAGS.snr_true_th].tolist())

    indx_list_per_batch_unres = [np.arange(i*FLAGS.batch_size, (i+1)*FLAGS.batch_size)[snrs_true_in_pool[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size]<FLAGS.snr_true_th].tolist() for i in range(int(np.floor(len(snrs_true_in_pool)/FLAGS.batch_size)))]
    if len(snrs_true_in_pool)%FLAGS.batch_size != 0:
        indx_list_per_batch_unres.append(np.arange(int(np.floor(len(snrs_true_in_pool)/FLAGS.batch_size))*FLAGS.batch_size, len(snrs_true_in_pool))[snrs_true_in_pool[int(np.floor(len(snrs_true_in_pool)/FLAGS.batch_size))*FLAGS.batch_size:]<FLAGS.snr_true_th].tolist())

    if FLAGS.wf_model.split('-')[0] !=  'LAL':
            wf_model = wf_models_dict[ FLAGS.wf_model]
    else:
        is_tidal, is_prec, is_HM, is_ecc = False, False, False, False
        if 'tidal' in FLAGS.lalargs:
            is_tidal = True
        if 'precessing' in FLAGS.lalargs:
            is_prec = True
        if 'HM' in FLAGS.lalargs:
            is_HM = True
        if 'eccentric' in FLAGS.lalargs:
            is_ecc = True
        wf_model = LAL_WF(FLAGS.wf_model.split('-')[1], is_tidal=is_tidal, is_HigherModes=is_HM, is_Precessing=is_prec, is_eccentric=is_ecc)

    if hasattr(wf_model, 'fRef'):
        wf_model.fRef = FLAGS.fmin

    Net = get_net(FLAGS)

    mySignals = {}

    for d in Net.keys():

        mySignals[d] = GWSignal( wf_model,
                psd_path= Net[d]['psd_path'],
                detector_shape = Net[d]['shape'],
                det_lat= Net[d]['lat'],
                det_long=Net[d]['long'],
                det_xax=Net[d]['xax'],
                verbose=False,
                useEarthMotion = FLAGS.rot,
                fmin=FLAGS.fmin, fmax=FLAGS.fmax)

    fgrid = FLAGS.fgrid_target

    neventsused = 0
    batchesused = 1

    true_minus_noise_avg_signalobs  = {d:np.zeros_like(fgrid).astype(np.complex128) for d in FLAGS.all_dets}
    noise_avg_signalobs_singlev     = {d:np.zeros_like(fgrid).astype(np.complex128) for d in FLAGS.all_dets}
    noise_avg_signalobs             = {d:np.zeros((fgrid.shape[0], int(len(snrs_true_in_pool)/FLAGS.n_noise_realizations))).astype(np.complex128) for d in FLAGS.all_dets}
    noise_avg_mixedterm             = {key:np.zeros_like(fgrid).astype(np.complex128) for key in FLAGS.det_pairs_plus_self}
    noise   = {}
    tmpnoise   = {}
    evFinished_nnoisecut = False

    n_noise_realizations = FLAGS.n_noise_realizations
    n_noise_max          = FLAGS.n_noise_max

    df = FLAGS.fres
    nptsGridmax = np.floor((FLAGS.fmax - FLAGS.fmin)/df).astype(int)

    ievdone = 0

    inTime = time.time()
    for i in range(int(np.ceil(len(snrs_true_in_pool)/FLAGS.batch_size))):

        np.random.seed(seeds_list[i])

        evsdetinbatch = len(indx_list_per_batch[i])
        snrs_obs_tmp = snr_obs_splitted[neventsused:neventsused+evsdetinbatch]
        evs_true_det = {key: evs_cat_in_pool[key][i*FLAGS.batch_size:(i+1)*FLAGS.batch_size] for key in evs_cat_in_pool.keys()}
        indx_list_per_batch_use = (np.array(indx_list_per_batch[i])-i*FLAGS.batch_size).tolist()

        if (evFinished_nnoisecut) & (batchesused*FLAGS.batch_size < n_noise_realizations):
            batchesused+=1
            if not evsdetinbatch == 0:
                neventsused+=evsdetinbatch
            continue
        elif (evFinished_nnoisecut) & (batchesused*FLAGS.batch_size >= n_noise_realizations):
            if not evsdetinbatch == 0:
                neventsused+=evsdetinbatch
            batchesused = 1
            evidxinorcat = idx + int(FLAGS.npools*ievdone)
            evFinished_nnoisecut=False
            ievdone+=1
            print('Done for event %d on CPU %s...'%(evidxinorcat, idx))
            continue

        if evsdetinbatch == 0:
            noDetection = True
            noDetection_obs = True
        elif (snrs_obs_tmp<FLAGS.snr_th).all():
            noDetection = False
            noDetection_obs = True
            neventsused+=evsdetinbatch
        else:
            noDetection = False
            noDetection_obs = False
            evs_obs_det  = {key: evs_obs_in_pool[key][neventsused:neventsused+evsdetinbatch][snrs_obs_tmp>FLAGS.snr_th] for key in evs_obs_in_pool.keys()}
            neventsused+=evsdetinbatch

        if not noDetection:
            fcut = np.array(wf_model.fcut(**evs_true_det))[indx_list_per_batch_use]
            fgrid_single  = np.arange(FLAGS.fmin, max(fcut)+df, df)
            fgrids = np.array([fgrid_single for i in range(fcut.shape[0])]).T

        fgrids_target_cat_single = FLAGS.fgrid_target[:,np.newaxis]

        if not noDetection_obs:
            fgrids_target_obs = np.array([FLAGS.fgrid_target for i in range(evs_obs_det['dL'].shape[0])]).T

        if wf_model.is_Precessing:
            chi1xUse, chi2xUse, chi1yUse, chi2yUse = evs_true_det['chi1x'], evs_true_det['chi2x'], evs_true_det['chi1y'], evs_true_det['chi2y']
            if not noDetection_obs:
                chi1xUse_obs, chi2xUse_obs, chi1yUse_obs, chi2yUse_obs = evs_obs_det['chi1x'], evs_obs_det['chi2x'], evs_obs_det['chi1y'], evs_obs_det['chi2y']
        else:
            chi1xUse, chi2xUse, chi1yUse, chi2yUse = evs_true_det['Mc']*0., evs_true_det['Mc']*0., evs_true_det['Mc']*0., evs_true_det['Mc']*0.
            if not noDetection_obs:
                chi1xUse_obs, chi2xUse_obs, chi1yUse_obs, chi2yUse_obs = evs_obs_det['Mc']*0., evs_obs_det['Mc']*0., evs_obs_det['Mc']*0., evs_obs_det['Mc']*0.

        if wf_model.is_tidal:
            Lambda1, Lambda2 = evs_true_det['Lambda1'], evs_true_det['Lambda2']
            if not noDetection_obs:
                Lambda1_obs, Lambda2_obs = evs_obs_det['Lambda1'], evs_obs_det['Lambda2']
        else:
            Lambda1, Lambda2 = evs_true_det['Mc']*0., evs_true_det['Mc']*0.
            if not noDetection_obs:
                Lambda1_obs, Lambda2_obs = evs_true_det['Mc']*0., evs_true_det['Mc']*0.

        if wf_model.is_eccentric:
            ecc = evs_true_det['ecc']
            if not noDetection_obs:
                ecc_obs = evs_obs_det['ecc']
        else:
            ecc = evs_true_det['Mc']*0.
            if not noDetection_obs:
                ecc_obs = evs_true_det['Mc']*0.

        single_ev_true = {key: evs_true_det[key][0] for key in evs_true_det.keys()}
        indx_list_per_batch_use_snr_obs_ov_th = (np.arange(len(indx_list_per_batch_use))[snrs_obs_tmp>FLAGS.snr_th]).tolist()

        if not noDetection:
            tmpnoise = {d: generate_noise_from_asd(mySignals[d.split("_")[0]].strainFreq, np.sqrt(mySignals[d.split("_")[0]].noiseCurve), fgrids) for d in FLAGS.all_dets}

        dict_of_zeros = {d: np.zeros((FLAGS.fgrid_target.shape[0], len(evs_true_det['Mc']))).astype(np.complex128) for d in FLAGS.all_dets}
        noise = copy.deepcopy(dict_of_zeros)
        signalsObs = copy.deepcopy(dict_of_zeros)
        signalsTrue = copy.deepcopy(dict_of_zeros)

        for d in FLAGS.all_dets:
            d_T = d.split("_")[0]
            if (not noDetection) & (not noDetection_obs):
                tmpnoise[d] = signal_handle_grid(tmpnoise[d], fgrids, mySignals[d_T].strainFreq, np.sqrt(mySignals[d_T].noiseCurve), nptsMax=nptsGridmax, fmax=FLAGS.fmax, df=df)
                noise[d][:,indx_list_per_batch_use_snr_obs_ov_th] = tmpnoise[d][:,snrs_obs_tmp>FLAGS.snr_th]

            elif (not noDetection) & (noDetection_obs):
                tmpnoise[d] = signal_handle_grid(tmpnoise[d], fgrids, mySignals[d_T].strainFreq, np.sqrt(mySignals[d_T].noiseCurve), nptsMax=nptsGridmax, fmax=FLAGS.fmax, df=df)

            j = 0 if len(d.split("_")) == 1 else int(d.split("_")[-1])
            signalsTrue_single = mySignals[d_T].GWstrain(fgrids_target_cat_single, single_ev_true['Mc'], single_ev_true['eta'], single_ev_true['dL'], single_ev_true['theta'],
                                                          single_ev_true['phi'], single_ev_true['iota'], single_ev_true['psi'], single_ev_true['tcoal'],
                                                          single_ev_true['Phicoal'], single_ev_true['chi1z'], single_ev_true['chi2z'], np.array([chi1xUse[0]]),
                                                          np.array([chi2xUse[0]]), np.array([chi1yUse[0]]), np.array([chi2yUse[0]]), np.array([Lambda1[0]]),
                                                          np.array([Lambda2[0]]), np.array([ecc[0]]), is_m1m2=False,
                                                          is_chi1chi2=True, is_prec_ang=False, return_single_comp=None,
                                                          is_Lam1Lam2=False, rot=j*60.)
            signalsTrue[d] = np.array([np.squeeze(signalsTrue_single) for i in range(len(evs_true_det['Mc']))]).T
            if not noDetection_obs:
                Lambda1_obs = np.where(Lambda1_obs <= 0., Lambda1[0], Lambda1_obs)
                Lambda2_obs = np.where(Lambda2_obs <= 0., Lambda2[0], Lambda2_obs)

                signalsObs[d][:,indx_list_per_batch_use_snr_obs_ov_th]  = mySignals[d_T].GWstrain(fgrids_target_obs, evs_obs_det['Mc'], evs_obs_det['eta'], evs_obs_det['dL'], evs_obs_det['theta'],
                                                                                             evs_obs_det['phi'], evs_obs_det['iota'], evs_obs_det['psi'], evs_obs_det['tcoal'],
                                                                                             evs_obs_det['Phicoal'], evs_obs_det['chi1z'], evs_obs_det['chi2z'], chi1xUse_obs,
                                                                                             chi2xUse_obs, chi1yUse_obs, chi2yUse_obs, Lambda1_obs, Lambda2_obs, ecc_obs, is_m1m2=False,
                                                                                             is_chi1chi2=True, is_prec_ang=False, return_single_comp=None,
                                                                                             is_Lam1Lam2=True, rot=j*60.)

            true_minus_noise_avg_signalobs[d] += signalsTrue[d].sum(axis=1) - signalsObs[d].sum(axis=1)
            noise_avg_signalobs_singlev[d] += signalsObs[d].sum(axis=1)

        for pair in FLAGS.det_pairs_plus_self:
            det1, det2 = pair.split('-')

            noise_avg_mixedterm[pair] += (np.conjugate(noise[det1]-signalsObs[det1])*(noise[det2]-signalsObs[det2]) - np.conjugate(noise[det1])*noise[det2]).sum(axis=1)

        if (n_noise_max>=n_noise_realizations) & (batchesused*FLAGS.batch_size == n_noise_realizations):
            batchesused = 1

            for d in FLAGS.all_dets:
                noise_avg_signalobs[d][:,ievdone] = noise_avg_signalobs_singlev[d]/n_noise_realizations
                noise_avg_signalobs_singlev[d] = np.zeros_like(fgrid).astype(np.complex128)

            evidxinorcat = idx + int(FLAGS.npools*ievdone)
            ievdone+=1
            print('Done for event %d on CPU %s...'%(evidxinorcat, idx))

        elif (n_noise_max<n_noise_realizations) & (batchesused*FLAGS.batch_size == n_noise_max):

            for d in FLAGS.all_dets:
                noise_avg_signalobs[d][:,ievdone] = noise_avg_signalobs_singlev[d]/n_noise_max
                noise_avg_signalobs_singlev[d] = np.zeros_like(fgrid).astype(np.complex128)

            evFinished_nnoisecut=True
            batchesused+=1
        else:
            batchesused+=1

    save_results_pool(true_minus_noise_avg_signalobs, FLAGS.fout, str(idx), suff='true_minus_noise_avg')
    save_results_pool(noise_avg_signalobs, FLAGS.fout, str(idx), suff='noise_avg_signalobs')
    save_results_pool(noise_avg_mixedterm, FLAGS.fout, str(idx), suff='noise_avg_mixedterm')

    print('Done for CPU %s in %.2f seconds.'%(idx, time.time() - inTime))


parser = argparse.ArgumentParser(prog = 'correlation_from_data.py', description='Compute the correlated output of a network given an observed GW catalog.')
parser.add_argument("--fname_cat", default='', type=str, required=True, help='Name of the file containing the catalog, without the extension ``h5``.')
parser.add_argument("--fname_obs", default='', type=str, required=True, help='Name of the file containing the catalog with ML parameters, without the extension ``h5``.')
parser.add_argument("--fout", default='test_correlation', type=str, required=True, help='Name of the output directory.')
parser.add_argument("--wf_model",  default='tf2', type=str, required=False, help='Name of the waveform model.')
parser.add_argument("--batch_size", default=1, type=int, required=False, help='Size of the batch to be computed in vectorized form on each process.')
parser.add_argument("--snr_th", default=12., type=float, required=False, help='Threshold value for the detection SNR to consider the event detectable.')
parser.add_argument("--snr_true_th", default=12., type=float, required=False, help='Threshold value for the true SNR to consider the event detectable.')
parser.add_argument("--fmin", default=2., type=float, required=False, help='Minimum frequency of the grid, in Hz.')
parser.add_argument("--fmax", default=2048., type=float, required=False, help='Maximum frequency of the grid, in Hz. If not specified, this coincides with the cut frequency of the waveform.')
parser.add_argument("--fres", default=0.125, type=float, required=False, help='Frequency resolution (spacing) of the grid, in Hz. If not specified, the default is 1/8.')
parser.add_argument("--snr_obs", default=None, type=str, required=False, help='Path to the file containing the events detection SNRs.')
parser.add_argument("--snr_true", default=None, type=str, required=False, help='Path to the file containing the events true SNRs.')
parser.add_argument("--net", nargs='+', default=['ETS', ], type=str, required=False, help='The network of detectors to be used, separated by *single spacing*.')
parser.add_argument("--psds", nargs='+', default=['ET-0000A-18.txt', ], type=str, required=False, help='The paths to PSDs of each detector in the network inside the folder ``psds/``, separated by *single spacing*.')
parser.add_argument("--lalargs", nargs='+', default=[ ], type=str, required=False, help='Specifications of the waveform when using ``LAL`` interface, separated by *single spacing*.')
parser.add_argument("--t_obs", default=1., type=float, required=False, help='Observation time for subtraction methods.')
parser.add_argument("--noise_parent_seed", default=42, type=int, required=False, help='Original seed to generate noise seeds.')
parser.add_argument("--rot", default=1, type=int, required=False, help='Int specifying if the effect of the rotation of the Earth has to be included in the analysis (``1``) or not (``0``).')
parser.add_argument("--triangle_arm_length", default=10., type=float, required=False, help='Int specifying if the effect of the rotation of the Earth has to be included in the analysis (``1``) or not (``0``).')
parser.add_argument("--npools", default=1, type=int, required=False, help='Number of parallel processes.')
parser.add_argument("--n_noise_max", default=None, type=int, required=False, help='Number of noise realisations to use. This has to be a multiple of ``batch_size``.')
parser.add_argument("--save_single_terms", default=0, type=int, required=False, help='Int specifying if the single terms of the correlator have to be stored separately.')

if __name__ =='__main__':

    print(*last_commit_info, sep="\n")

    if ".ini" in sys.argv[-1]:

        config = configparser.ConfigParser()
        config_file_path = sys.argv[-1]
        config.read(config_file_path)
        subconfig = config[__file__.split("/")[-1]]
        FLAGS = utils.config_conversion(subconfig)
        with open(os.path.join(RESPATH, FLAGS.fout, "config_file.ini"), "w") as config_file:
            config.write(config_file)
        print("Using config file.")

    else:

        FLAGS = parser.parse_args()
        print("Using arg parser.")

    inTime = time.time()
    print('Loading starting events from %s...' %FLAGS.fname_cat)
    events_loaded_cat = utils.load_population(FLAGS.fname_cat)

    keylist=list(events_loaded_cat.keys())
    nevents_total_cat = len(events_loaded_cat[keylist[0]])
    print('This starting catalog has %s events.' %nevents_total_cat)
    FLAGS.nevents_total_cat = nevents_total_cat

    print('Loading observed events from %s...' %FLAGS.fname_obs)
    events_loaded_obs = utils.load_population(FLAGS.fname_obs)

    keylist_obs=list(events_loaded_obs.keys())
    nevents_total_obs = len(events_loaded_obs[keylist[0]])
    print('This observed catalog has %s events.' %nevents_total_obs)

    nseeds = int(np.ceil(nevents_total_cat/FLAGS.batch_size))
    np.random.seed(FLAGS.noise_parent_seed)
    seeds_list = np.random.randint(low=1, high=(2**32-1), size=nseeds)

    snrs_true = np.loadtxt(FLAGS.snr_true)
    with h5py.File(FLAGS.snr_obs, 'r') as f:
        snrs_obs = np.array(f['snr']['net'])

    n_noise_realizations = np.where(events_loaded_cat['dL'] == events_loaded_cat['dL'][0])[0].shape[0]
    idx_det = np.arange(nevents_total_cat)[snrs_true>FLAGS.snr_true_th]

    FLAGS.n_noise_realizations = n_noise_realizations
    if FLAGS.n_noise_max is None:
        FLAGS.n_noise_max = n_noise_realizations
    elif FLAGS.n_noise_max > n_noise_realizations:
        print('WARNING: the number of noise realizations requested is larger than the number of noise realizations available. Using the maximum number of noise realizations available.')
        FLAGS.n_noise_max = n_noise_realizations

    df = FLAGS.fres
    nptsGridmax = np.floor((FLAGS.fmax - FLAGS.fmin)/df).astype(int)
    fgrid = np.arange(FLAGS.fmin,FLAGS.fmax+df,df)

    FLAGS.fgrid_target = fgrid

    ORFs = {}
    Net = get_net(FLAGS)
    netcombinations = [det1+'-'+det2 for det1 in FLAGS.net for det2 in FLAGS.net if FLAGS.net.index(det1) < FLAGS.net.index(det2)]
    for comb in netcombinations:
        det1, det2 = comb.split('-')
        tmpORFs = st.overlap_reduction_function(fgrid, Net[det1], Net[det2], det1, det2, arm_length_1=FLAGS.triangle_arm_length, arm_length_2=FLAGS.triangle_arm_length)
        for key in tmpORFs.keys():
            if key not in ORFs.keys():
                ORFs[key] = tmpORFs[key]

    FLAGS.det_pairs = list(ORFs.keys())
    all_dets_names = []
    for d in FLAGS.net:
        if Net[d]['shape'] == 'L':
            all_dets_names.append(d)
        elif Net[d]['shape'] == 'T':
            for j in range(3):
                all_dets_names.append(d+'_%s'%j)
    FLAGS.all_dets = all_dets_names
    FLAGS.det_pairs_plus_self = FLAGS.det_pairs + [det+"-"+det for det in FLAGS.all_dets]

    idx_per_pool, noise_idx_per_pool = get_indexes(FLAGS.npools, nevents_total_cat, n_noise_realizations, FLAGS.batch_size)

    idx_obs = np.arange(nevents_total_obs)
    idx_obs_tot = np.arange(nevents_total_cat)[snrs_true > FLAGS.snr_true_th]

    total_cat_splitted = {}
    obs_cat_splitted = {}
    snrs_true_splitted = {}
    seeds_list_splitted = {}
    snr_obs_splitted = {}
    for pool in idx_per_pool.keys():
        total_cat_splitted[pool] = {key: events_loaded_cat[key][idx_per_pool[pool]] for key in events_loaded_cat.keys()}
        snrs_true_splitted[pool] = snrs_true[idx_per_pool[pool]]
        obs_cat_splitted[pool]   = {key: events_loaded_obs[key][idx_obs[np.isin(idx_obs_tot, idx_per_pool[pool])]] for key in events_loaded_obs.keys()}
        seeds_list_splitted[pool] = seeds_list[noise_idx_per_pool[pool]]
        snr_obs_splitted[pool] = snrs_obs[idx_obs[np.isin(idx_obs_tot, idx_per_pool[pool])]]

    FLAGS.total_cat_splitted = total_cat_splitted
    FLAGS.snrs_true_splitted = snrs_true_splitted
    FLAGS.obs_cat_splitted = obs_cat_splitted
    FLAGS.seeds_list_splitted = seeds_list_splitted
    FLAGS.snr_obs_splitted = snr_obs_splitted

    if FLAGS.npools>1:
        print('Parallelizing on %s CPUs ' %FLAGS.npools)
        print('Total available CPUs: %s' %str(multiprocessing.cpu_count()) )
        pool =  get_pool(mpi=False, threads=FLAGS.npools+1)
        pool.starmap( main, [ ( i, FLAGS ) for i in range(FLAGS.npools)] )
        pool.close()
    else:
        (main(0, FLAGS), )

    # Load results
    print('Concatenating results...')
    for idx in range(FLAGS.npools):
        true_minus_noise_avg_pool = load_results_pool(FLAGS.fout, str(idx), suff='true_minus_noise_avg')
        noise_avg_signalobs_pool = load_results_pool(FLAGS.fout, str(idx), suff='noise_avg_signalobs')
        noise_avg_mixedterm_pool = load_results_pool(FLAGS.fout, str(idx), suff='noise_avg_mixedterm')

        if idx == 0:
            true_minus_noise_avg = true_minus_noise_avg_pool
            noise_avg_signalobs = noise_avg_signalobs_pool
            noise_avg_mixedterm = noise_avg_mixedterm_pool

        else:
            for key in true_minus_noise_avg.keys():
                true_minus_noise_avg[key] += true_minus_noise_avg_pool[key]
                noise_avg_signalobs[key] = np.concatenate((noise_avg_signalobs[key], noise_avg_signalobs_pool[key]), axis=1)

            for pair in FLAGS.det_pairs_plus_self:
                noise_avg_mixedterm[pair] += noise_avg_mixedterm_pool[pair]

        os.remove(os.path.join(RESPATH, FLAGS.fout, 'correlator_true_minus_noise_avg_'+str(idx)+'.h5'))
        os.remove(os.path.join(RESPATH, FLAGS.fout, 'correlator_noise_avg_signalobs_'+str(idx)+'.h5'))
        os.remove(os.path.join(RESPATH, FLAGS.fout, 'correlator_noise_avg_mixedterm_'+str(idx)+'.h5'))

    final_correlator = {p:np.zeros_like(fgrid).astype(np.complex128) for p in FLAGS.det_pairs_plus_self}
    third_term = {}

    for pair in FLAGS.det_pairs_plus_self:
        det1, det2 = pair.split('-')
        third_term[pair] = (np.conjugate(noise_avg_signalobs[det1])*noise_avg_signalobs[det2]).sum(axis=1)
        final_correlator[pair] = np.conjugate(true_minus_noise_avg[det1])*true_minus_noise_avg[det2]/(FLAGS.n_noise_max**2.) + noise_avg_mixedterm[pair]/FLAGS.n_noise_max - third_term[pair]

    save_results(final_correlator, fgrid, os.path.join(RESPATH, FLAGS.fout, 'subtraction_correlation_tot.h5'))

    if FLAGS.save_single_terms:
        save_results_single_terms(true_minus_noise_avg, noise_avg_mixedterm, third_term, fgrid, FLAGS.n_noise_max, os.path.join(RESPATH, FLAGS.fout, 'single_terms_correlation.h5'))

    print('Done.')
    print('Total execution time: %.2f seconds.' %(time.time() - inTime))
