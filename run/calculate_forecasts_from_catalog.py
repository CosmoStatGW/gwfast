#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
#    Copyright (c) 2022 Francesco Iacovelli <francesco.iacovelli@unige.ch>, Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time
import multiprocessing
# Needed for mpipool not to stall when trying to write on a file (do not ask me why)
multiprocessing.set_start_method("spawn",force=True)


PACKAGE_PARENT = '../gwfast'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd())))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR,PACKAGE_PARENT )))
import copy
import numpy as onp
import argparse
import h5py

from jax import jit

import gwfast.gwfastGlobals as glob
from gwfast.gwfastGlobals import detectors as base_dets
from gwfast.waveforms import TaylorF2_RestrictedPN, IMRPhenomD, IMRPhenomHM, IMRPhenomD_NRTidalv2, IMRPhenomNSBH
from gwfast.signal import GWSignal
from gwfast.network import DetNet
from gwfast.fisherTools import compute_localization_region, fixParams, CheckFisher, CovMatr, compute_inversion_error
from gwfast.gwfastUtils import  get_events_subset, save_detectors, load_population, save_data

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
                  'IMRPhenomNSBH':IMRPhenomNSBH(),
                  'tf2_ecc':TaylorF2_RestrictedPN(is_tidal=False, use_3p5PN_SpinHO=True, is_eccentric=True),
                  }
if TEOBResumS_installed:
    wf_models_dict['TEOBResumSPA'] = TEOBResumSPA_WF()
    wf_models_dict['TEOBResumSPA_tidal'] = TEOBResumSPA_WF(is_tidal=True)

#####################################################################################
# input/output logic
#####################################################################################

def get_out_dir(wf_model, net, create=False):
    
        wf_model_name =  type(wf_model).__name__
        
        
        netname =  ('').join(list(net.keys()))
            
        fname_obs, fout_base = get_fout_base_name(FLAGS.fname_obs, wf_model_name, FLAGS.snr_th, netname)
        fout_base+='_'+FLAGS.fout_suff
        out_path = os.path.join(RESPATH, fout_base)
        if not os.path.exists(out_path):
                #if idx==0:
                if create:
                    os.makedirs(out_path)
                    print('Creating directory %s' %out_path)
                dname=fout_base
                
        else:   
                exists=True
                i=1
                while exists:
                    dname=fout_base+'_'+str(i)
                    out_path = os.path.join(RESPATH, dname)
                    exists =  os.path.exists(out_path)
                    print('Directory %s exists! Trying %s' %(fout_base, dname))
                    i+=1
                #if idx==0:
                if create:
                    os.makedirs(out_path)
                    print('Creating directory %s' %out_path)
                #print('Using directory %s' %out_path)
        print('\n------------ Results directory name:  ------------\n%s' %dname)
        print('------------------------\n')
        
        return out_path, fname_obs


def get_fout_base_name(fname_obs_in, wf_model_name, snr_th, netname):
    my_fname_obs = os.path.join('../../data', fname_obs_in+'.h5')
    my_fout_base = 'Fisher'+'_'+fname_obs_in+'_'+netname+'_'+wf_model_name+'_snr_th-'+str(snr_th)
    return my_fname_obs, my_fout_base




# Writes output both on std output and on log file
class Logger(object):
    
    def __init__(self, fname):
        self.terminal = sys.__stdout__
        self.log = open(fname, "w+")
        self.log.write('--------- LOG FILE ---------\n')
        print('Logger created log file: %s' %fname)
        #self.write('Logger')
       
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

    def close(self):
        self.log.close
        sys.stdout = sys.__stdout__
        
    def isatty(self):
        return False



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



def get_indexes(p, all_n_per_pool):
    
    if p==0:
        pin = 0
        pf = all_n_per_pool[p]
    else: 
        pin = sum(all_n_per_pool[:p])
        pf = pin+all_n_per_pool[p]
    return pin, pf

    
    

def get_net(FLAGS):
    if FLAGS.netfile is not None:
            print('Custom detector file passed. Loading network specifications from %s...' %FLAGS.netfile)
            import json
            with open(FLAGS.netfile, 'r') as j:
                Net = json.loads(j.read())
    else:
            Net= { k: copy.deepcopy(base_dets).pop(k) for k in FLAGS.net} 
            for i,psd in enumerate(FLAGS.psds): 
                Net[FLAGS.net[i]]['psd_path'] = os.path.join(glob.detPath, psd)
    return Net



#####################################################################################
# load and save results
#####################################################################################




def to_file(snrs, fishers, eps_dL, covs, sky_area, errors, idxs, out_path, suff='', cond_numbers=None):
    
    if onp.ndim(snrs)==0 or onp.isscalar(snrs):
        snrs = onp.array([snrs,])
        sky_area = onp.array([sky_area,])
        eps_dL = onp.array([eps_dL,])
        
        
        
    
    if cond_numbers is None:
        _, _, cond_numbers = CheckFisher(fishers, use_mpmath=True)
    
    print('Saving all to files. Names of snrs: %s, names of fishers: %s' %('snrs'+suff+'.txt', 'fishers'+suff+'.npy'))
    onp.savetxt(os.path.join(out_path, 'snrs'+suff+'.txt'), snrs)
    onp.save(os.path.join(out_path, 'fishers'+suff+'.npy'), fishers)
    onp.save(os.path.join(out_path, 'covs'+suff+'.npy'), covs)
    onp.savetxt(os.path.join(out_path, 'sky_area'+suff+'.txt'), sky_area)
    onp.savetxt(os.path.join(out_path, 'errors'+suff+'.txt'), errors)
    onp.savetxt(os.path.join(out_path, 'inversion_errors'+suff+'.txt'), eps_dL)
    onp.savetxt(os.path.join(out_path, 'cond_numbers'+suff+'.txt'), cond_numbers)
    onp.savetxt(os.path.join(out_path, 'idxs_det'+suff+'.txt'), idxs)
    
    print('Saving successful.')
    
    
def from_file(out_path, suff=''):
       
    
        
    print('Loading files. Names of snrs: %s, names of fishers: %s' %('snrs'+suff+'.txt', 'fishers'+suff+'.npy'))
    snrs = onp.loadtxt(os.path.join(out_path, 'snrs'+suff+'.txt'), )
    if onp.ndim(snrs)==0:
        snrs = onp.array([snrs,])
    
    
    if os.path.exists(os.path.join(out_path, 'fishers'+suff+'.npy')):
        fishers = onp.load(os.path.join(out_path, 'fishers'+suff+'.npy'), )
        covs = onp.load(os.path.join(out_path, 'covs'+suff+'.npy'), )
        sky_area = onp.loadtxt(os.path.join(out_path, 'sky_area'+suff+'.txt'), )
        errors = onp.loadtxt(os.path.join(out_path, 'errors'+suff+'.txt'), )
        eps_dL = onp.loadtxt(os.path.join(out_path, 'inversion_errors'+suff+'.txt'), )
        cond_numbers = onp.loadtxt(os.path.join(out_path, 'cond_numbers'+suff+'.txt'), )
        idxs = onp.loadtxt(os.path.join(out_path, 'idxs_det'+suff+'.txt'), )
    
        print('Loaded.')
    
    
        # handle the case where the batch has only one element
        if onp.ndim(snrs)==0:
            #snrs = onp.array([snrs,])
            sky_area = onp.array([sky_area,])
            errors = onp.array([errors,]).T
            eps_dL = onp.array([eps_dL,])
            cond_numbers = onp.array([cond_numbers,])
            idxs = onp.array([idxs,])
        if onp.ndim(errors)==1:
            errors = errors[:,onp.newaxis]
            
        return [snrs, fishers, eps_dL, covs, sky_area, errors, cond_numbers, idxs]
    
    else:
        return [snrs, ]

        


def delete_files(idxin, idxf, out_path):
    
    print('Removing unnecessary files from %s to %s... ' %(idxin, idxf))
    suff = '_'+str(idxin)+'_to_'+str(idxf)
    os.remove(os.path.join(out_path, 'snrs'+suff+'.txt'))
    try:
        os.remove(os.path.join(out_path, 'all_snrs'+suff+'.hdf5'))
    except Exception as e:
        print(e)
    try:
        os.remove(os.path.join(out_path, 'fishers'+suff+'.npy'))
        os.remove(os.path.join(out_path, 'covs'+suff+'.npy'))
        os.remove(os.path.join(out_path, 'sky_area'+suff+'.txt'))
        os.remove(os.path.join(out_path, 'errors'+suff+'.txt'))
        os.remove(os.path.join(out_path, 'inversion_errors'+suff+'.txt'))
        os.remove(os.path.join(out_path, 'cond_numbers'+suff+'.txt'))
        os.remove(os.path.join(out_path, 'idxs_det'+suff+'.txt'))
        try:
            os.remove(os.path.join(out_path, 'all_fishers'+suff+'.hdf5'))
        except Exception as e:
            print(e)
    except Exception as e:
        print(e)
        
        



def load_snrs_all(out_path, suff=''):
    fname=os.path.join(out_path, 'all_snrs'+suff+'.hdf5') #'all_snrs'+suff+'.hdf5'
    print('Loading all snrs files. Names of snrs: %s,' %(fname))
    allsnrs={}
    with h5py.File(fname, 'r') as phi:
        for det in phi['snr'].keys():
            allsnrs[det] = onp.array(phi['snr'][det])
    return allsnrs
    


def load_fishers_all(out_path, suff=''):
    fname=os.path.join(out_path, 'all_fishers'+suff+'.hdf5') 
    print('Loading all fishers files. Names: %s,' %(fname))
    allfishers={}
    with h5py.File(fname, 'r') as phi:
        for det in phi['fisher'].keys():
            allfishers[det] = onp.array(phi['fisher'][det])
    return allfishers


def to_file_all(snrs, fishers, out_path, suff='', ):
    
    
    fname_out_snr = os.path.join(out_path, 'all_snrs'+suff+'.hdf5') 
    fname_out_fish = os.path.join(out_path, 'all_fishers'+suff+'.hdf5') 
    print('Saving all snrs to files. Names of snrs: %s' %(fname_out_snr,))
    with h5py.File(fname_out_snr, 'a') as f:
        try:
            del f['snr']
        except:
            pass
        pg = f.create_group('snr')
        
        for i,pn in enumerate(snrs.keys()):
            print('Saving %s at position %s...' %(pn, i))
            if onp.isscalar(snrs[pn]) or onp.ndim(snrs[pn])==0 :
                snrs[pn]=onp.array([snrs[pn],])
            try:
                pg.create_dataset(pn, data=snrs[pn], compression='gzip', shuffle=False, )
            except TypeError:
                pg.create_dataset(pn, data=onp.array([snrs[pn],]), compression='gzip', shuffle=False, )
     
    if fishers is not None:
        print('Saving all fishers to files. Names of fishers: %s' %(fname_out_fish))
        with h5py.File(fname_out_fish, 'a') as f:
            try:
                del f['fisher']
            except:
                pass
            pg = f.create_group('fisher')
            
            for i,pn in enumerate(fishers.keys()):
                print('Saving %s at position %s...' %(pn, i))
                pg.create_dataset(pn, data=fishers[pn], compression='gzip', shuffle=False, )

    

#####################################################################################
# ACTUAL COMPUTATIONS OF FISHERS AND ERRORS
#####################################################################################


def compute_errs(events, net, FLAGS, i_in, i_f):
    
    # Computes snrs, fishers, covs, errors, sky areas
    # for a single batch of events
    
    nevents = len(events[list(events.keys())[0]])
    
    

    print('Computing snrs...')
    tsnrinit=  time.time()
    snrs_all_df1 = net.SNR(events, return_all=True) #FLAGS.return_all)
    
    # Impose duty
    if (FLAGS.duty_factor is None) or (FLAGS.duty_factor>=1):
        snrs_all = copy.deepcopy(snrs_all_df1)
    else:
        print('Imposing duty factor...')
        snrs_all = copy.deepcopy(snrs_all_df1)
        is_duty_applied = {}
        
        net_snr = onp.zeros(nevents)
        for i,key in enumerate(net.signals):
            
            if key!='net':
                onp.random.seed(FLAGS.seeds[i])
                if net.signals[key].detector_shape=='L':
                    #onp.random.seed(FLAGS.seeds[i])
                    excl = onp.random.choice([0,1], nevents, p=[1.-FLAGS.duty_factor,FLAGS.duty_factor])
                    is_duty_applied[key] = excl
                    snrs_all[key] = snrs_all[key]*excl
                    net_snr += snrs_all[key]**2
                    
                elif net.signals[key].detector_shape=='T':
                    for i_arm in range(3):
                        excl = onp.random.choice([0,1], nevents, p=[1.-FLAGS.duty_factor,FLAGS.duty_factor])
                        is_duty_applied[key+'_%s'%i_arm] = excl
                        snrs_all[key+'_%s'%i_arm] = snrs_all[key+'_%s'%i_arm]*excl
                        net_snr += snrs_all[key+'_%s'%i_arm]**2
                
        snrs_all['net'] = onp.sqrt(net_snr)
    
    snrs=snrs_all['net']
    snrs_df1=snrs_all_df1['net']

    
    tsnrend=time.time()
    print('%s snrs computed in %s sec' %(nevents, str(tsnrend-tsnrinit)))
    print('... which is, %s seconds/snr' %(str( (tsnrend-tsnrinit)/nevents ) ))

    detected = snrs>FLAGS.snr_th
    if onp.isscalar(detected):
        detected=onp.array([detected,])
    
    is_duty_applied_det = {}
    if not ((FLAGS.duty_factor is None) or (FLAGS.duty_factor>=1)):
        for key in is_duty_applied.keys():
            is_duty_applied_det[key] = is_duty_applied[key][detected]
    
    
    detected_all = snrs_df1>FLAGS.snr_th
    if onp.isscalar(detected_all):
        detected_all=onp.array([detected_all,])
        
    print('%s events have snr>%s' %( detected.sum(), FLAGS.snr_th))
    print('%s events have snr>%s with duty factor 1' %( detected_all.sum(), FLAGS.snr_th))
    idxs_detected = onp.arange(i_in, i_f)[onp.argwhere(detected)]
    
    events_det = get_events_subset(events, detected)
    nevents_det = len(events_det[list(events.keys())[0]])

    assert nevents_det==detected.sum()
    
    npar = net.signals[list(net.signals.keys())[0]].wf_model.nParams
    
    
    final_fisher_shape = ( npar-len(FLAGS.params_fix), npar-len(FLAGS.params_fix), nevents_det)
    

    compute_fisher_ = FLAGS.compute_fisher
    if detected.sum()==0:
        compute_fisher_ =False

    if not compute_fisher_:
        Cov_dL = onp.full( final_fisher_shape, onp.nan)
        eps_dL = onp.full(nevents, onp.nan)
        my_sky_area_90 = onp.full(nevents, onp.nan)
        cond_numbers = onp.full(nevents, onp.nan)
        totF = onp.full( final_fisher_shape, onp.nan)
        Fres = onp.full( final_fisher_shape, onp.nan)
        
        
        
    else:
        # Fisher
        tFinit=  time.time()
        
        Fres_df1_ = net.FisherMatr( events_det, 
                                              res=1000, 
                                              df=None, 
                                              spacing='geom', 
                                              use_chi1chi2=True, 
                                              computeAnalyticalDeriv=True, 
                                              return_all=True ) #FLAGS.return_all)
        
        
        if (FLAGS.duty_factor is None) or (FLAGS.duty_factor>=1):
            Fres_ = copy.deepcopy(Fres_df1_)
        else:
            print('Imposing duty factor...')
            Fres_ = copy.deepcopy(Fres_df1_)
            
            net_F = onp.zeros((npar, npar, nevents_det))
            for i,key in enumerate(net.signals):
                
                if key!='net':
                    if net.signals[key].detector_shape=='L':
                        
                        Fres_[key] = Fres_[key]*is_duty_applied_det[key]
                        net_F += Fres_[key]
                        
                    elif net.signals[key].detector_shape=='T':
                        for i_arm in range(3):
                            #print('For %s, is_duty_applied_det=%s' %(key+'_%s'%i_arm, str(is_duty_applied_det[key+'_%s'%i_arm])))
                            Fres_[key+'_%s'%i_arm] = Fres_[key+'_%s'%i_arm]*is_duty_applied_det[key+'_%s'%i_arm]
                            net_F += Fres_[key+'_%s'%i_arm]
                    
            Fres_['net'] = net_F
        
        totF_ = Fres_['net']
   
        # Fix parameters if required
        ParNums_inp = net.signals[list(net.signals.keys())[0]].wf_model.ParNums
        if len(FLAGS.params_fix)>0:
            print('In the Fisher matrix, we fix the following parameters: %s' %str(FLAGS.params_fix))
            
            if FLAGS.return_all:
                Fres = {}
                for k in Fres_.keys():
                    res_fix = fixParams(Fres_[k], ParNums_inp, FLAGS.params_fix)
                    Fres[k] = res_fix[0]
                    if k=='net':
                        totF, parNums =  res_fix[0], res_fix[1]
                        
            else:
                # fres is just one fisher
                Fres, parNums = fixParams(totF_, ParNums_inp, FLAGS.params_fix)
                totF = Fres
            
                    
        else:
            totF = totF_
            parNums = ParNums_inp
            Fres=Fres_
            
        
        
        tFend=time.time()
        print('%s fishers computed in %s sec' %(nevents_det, str(tFend-tFinit)))
        print('... which is, %s seconds/fisher' %(str( (tFend-tFinit)/nevents_det ) ))

        
        _, _, cond_numbers = CheckFisher(totF, use_mpmath=True)
        
        npar = totF.shape[0]
        
        Cov_dL = onp.full( totF.shape, onp.nan)
        eps_dL = onp.full( totF.shape[-1], onp.nan)
        my_sky_area_90 = onp.full( totF.shape[-1], onp.nan)
        
        try:
            Cov_dL, eps_dL = CovMatr( totF,
                                                               invMethodIn='cho', 
                                                               condNumbMax=1e50, 
                                                               svals_thresh=1e-15, 
                                                               truncate=False, 
                                                               verbose=False
                                                               )
            
            #eps_dL = compute_inversion_error(totF, Cov_dL)
            print('Computing localization region...')
            my_sky_area_90 = compute_localization_region(Cov_dL, parNums, events_det["theta"], perc_level=90, units='SqDeg')
        except Exception as e:
            print(e)
            print()
            Cov_dL = onp.full( totF.shape, onp.nan)
            eps_dL = onp.full(totF.shape[-1], onp.nan)
            my_sky_area_90 = onp.full(totF.shape[-1], onp.nan)
            cond_numbers = onp.full(totF.shape[-1], onp.nan)
    

    return snrs_all, Fres, eps_dL, Cov_dL, my_sky_area_90, cond_numbers, idxs_detected





def main(idx, FLAGS):
        
        # Computes snrs, fishers, covs, errors, sky areas
        # for all batches assigned to a single thread (one cpu)
        # Saves each batch to separate file
        

        idx=idx-1        

        ti=  time.time()

        Net = get_net(FLAGS)
        
        if FLAGS.wf_model.split('-')[0] !=  'LAL':
            wf_model = wf_models_dict[ FLAGS.wf_model]
            wf_model_name =  type(wf_model).__name__
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
            wf_model_name = FLAGS.wf_model
        
        dname = FLAGS.fout.split('/')[-1]
        if dname=='':
            dname = FLAGS.fout.split('/')[-2]
        
        print('\n------------ Results directory name:  ------------\n%s' %dname)
        print('------------------------\n')
        
        
        
        logidx = '_'+str(FLAGS.idx_in)+'_to_'+str(FLAGS.idx_f) 
        logfile = os.path.join(FLAGS.fout, 'logfile'+logidx+'_'+str(idx)+'.txt') #out_path+'logfile.txt'
        myLog = Logger(logfile)
        sys.stdout = myLog
        sys.stderr = myLog
        
        
        print('\n------------ Network used:  ------------\n%s' %str(Net))
        if FLAGS.netfile is not None:
            print('(Custom detector file was passed. Loaded network specifications from %s.)' %FLAGS.netfile)
        print('------------------------\n')
    
        print('------ Waveform:------\n%s' %wf_model_name)
        print('------\n')
        
        if (FLAGS.jit_Fisher) and (not wf_model.is_LAL):
            jitCompileDerivs=True
        else:
            jitCompileDerivs=False
        
        mySignals = {}

        for d in Net.keys():

            mySignals[d] = GWSignal( wf_model, 
                    psd_path= Net[d]['psd_path'],
                    detector_shape = Net[d]['shape'],
                    det_lat= Net[d]['lat'],
                    det_long=Net[d]['long'],
                    det_xax=Net[d]['xax'], 
                    verbose=True,
                    useEarthMotion = FLAGS.rot,
                    fmin=FLAGS.fmin, fmax=FLAGS.fmax,
                    IntTablePath=None, 
                    DutyFactor=None,
                    jitCompileDerivs=jitCompileDerivs) #FLAGS.duty_factor) 
            

        myNet = DetNet(mySignals) 
        
        #if FLAGS.seed is not None:
        myNet._update_all_seeds(seeds=FLAGS.seeds, verbose=True)
        
        if idx==0:
            fname_det_new = os.path.join(FLAGS.fout, 'detectors.json')
            save_detectors(fname_det_new, Net)
        
        ti_evs=  time.time()
        for it in range( FLAGS.all_n_it_pools[idx] ):
            
            idxin=FLAGS.idxs_lists[str(idx)][it][0] 
            idxf=FLAGS.idxs_lists[str(idx)][it][1]  
            
            ev_chunk = FLAGS.events_lists[str(idx)][it] 
            nevents_chunk = len(ev_chunk['dL'])
            
            i_in = idxin+FLAGS.idx_in
            i_f = idxf+FLAGS.idx_in

            
            print('\nIn this chunk we have %s events, from %s to %s' %(nevents_chunk, i_in,  i_f  ))
            
         
            snrs_all, F_all, eps_dL, Cov_dL, my_sky_area_90, condition_numbers, idxs_detected = compute_errs(ev_chunk, myNet, FLAGS, i_in, i_f)                      
            suffstr = '_'+str(idxin+FLAGS.idx_in)+'_to_'+str(idxf+FLAGS.idx_in)
        
            if FLAGS.return_all:
                
                snrs = snrs_all['net']
                
                if type(F_all)!=dict: #
                    if onp.all(onp.isnan(F_all)):
                        print('Nothing to save')
                        save=False
                        F_all=None
                    else:
                        raise ValueError()
                else:
                    totF = F_all['net']
                    save=True
                to_file_all(snrs_all, F_all, FLAGS.fout, suff=suffstr)
            else:
                snrs = snrs_all['net']
                if type(F_all)==dict:
                    totF = F_all['net']
                else:
                    totF=F_all
                if onp.all(onp.isnan(totF)):
                    print('Nothing to save')
                    save=False
                else:
                    save=True
              
            if save:
                errs = onp.array([onp.sqrt(Cov_dL[i, i]) for i in range(totF.shape[0])])  
                to_file(snrs, totF, eps_dL, Cov_dL, my_sky_area_90, errs, idxs_detected, FLAGS.fout, suff=suffstr, cond_numbers=condition_numbers)
            else:
                if onp.ndim(snrs)==0 or onp.isscalar(snrs):
                    snrs = onp.array([snrs,])
                print('Saving only snrs to files. Names of snrs: %s' %('snrs'+suffstr+'.txt',))
                onp.savetxt(os.path.join(FLAGS.fout, 'snrs'+suffstr+'.txt'), snrs)
    
        te=time.time()
        print('------')
        print('------ Time to compute events: %s sec.\n\n' %( str((te-ti_evs))))
        print('------ Total execution time: %s sec.\n\n' %( str((te-ti))))
        print('------ Done for %s. ' %(wf_model_name ))
        myLog.close()
        
  
def mainMPI(i):
    # Just a workaround to make mpi work without starmap
    return main(i, FLAGS)


parser = argparse.ArgumentParser(prog = 'calculate_forecasts_from_catalog.py', description='Executable to run ``gwfast`` on a catalog of events, with the possibility to parallelize over multiple CPUs, ready to use both on single machines and on clusters.')
parser.add_argument("--fname_obs", default='', type=str, required=True, help='Name of the file containing the catalog, without the extension ``h5``.')
parser.add_argument("--fout", default='test_gwfast', type=str, required=True, help='Path to output folder, which has to exist before the script is launched.')
parser.add_argument("--wf_model",  default='tf2', type=str, required=False, help='Name of the waveform model.')
parser.add_argument("--batch_size", default=1, type=int, required=False, help='Size of the batch to be computed in vectorized form on each process.')
parser.add_argument("--npools", default=1, type=int, required=False, help='Number of parallel processes.')
parser.add_argument("--snr_th", default=12., type=float, required=False, help='Threshold value for the SNR to consider the event detectable. FIMs are computed only for events with SNR exceeding this value.')
parser.add_argument("--idx_in", default=0, type=int, required=False, help='Index of the event in the catalog from which to start the calculation.')
parser.add_argument("--idx_f", default=None, type=int, required=False, help='Index of the event in the catalog from which to end the calculation.')
parser.add_argument("--fmin", default=2., type=float, required=False, help='Minimum frequency of the grid, in Hz.')
parser.add_argument("--fmax", default=None, type=float, required=False, help='Maximum frequency of the grid, in Hz. If not specified, this coincides with the cut frequency of the waveform.')
parser.add_argument("--compute_fisher", default=1, type=int, required=False, help='Int specifying if the FIMs have to be computed (``1``) or not (``0``).')
parser.add_argument("--net", nargs='+', default=['ETS', ], type=str, required=False, help='The network of detectors to be used, separated by *single spacing*.')
parser.add_argument("--netfile", default=None, type=str, required=False, help='``json`` file containing the detector configuration, alternative to **--net** and **--psds**.')
parser.add_argument("--psds", nargs='+', default=['ET-0000A-18.txt', ], type=str, required=False, help='The paths to PSDs of each detector in the network inside the folder ``psds/``, separated by *single spacing*.')
parser.add_argument("--mpi", default=0, type=int, required=False, help='Int specifying if the code has to parallelize using multiprocessing (``0``), or using MPI (``1``), suitable for clusters.')
parser.add_argument("--duty_factor", default=1., type=float, required=False, help='Duty factor of the detectors (the same is used for all detectors in a network).')
parser.add_argument("--concatenate", default=1, type=int, required=False, help='Int specifying if the results of the individual batches have to be concatenated (``1``) or not (``0``).')
parser.add_argument("--params_fix", nargs='+', default=[ ], type=str, required=False, help='List of parameters to fix to the fiducial values, i.e. to eliminate from the FIMs, separated by *single spacing*.')
parser.add_argument("--rot", default=1, type=int, required=False, help='Int specifying if the effect of the rotation of the Earth has to be included in the analysis (``1``) or not (``0``).')
parser.add_argument("--lalargs", nargs='+', default=[ ], type=str, required=False, help='Specifications of the waveform when using ``LAL`` interface, separated by *single spacing*.')
parser.add_argument("--return_all", default=0, type=int, required=False, help='Int specifying if, in case a network of detectors is used, the SNRs and Fishher matrices of the individual detector have to be stored (``1``) or not (``0``).')
parser.add_argument("--seeds", nargs='+', default=[ ], type=int, required=False, help='List of seeds to set for the duty factors in individual detectors, to help reproducibility, separated by *single spacing*.') # This should be one per detector (one per arm for triangular shapes)
parser.add_argument("--jit_Fisher", default=0, type=int, required=False, help='Int specifying if the Fisher function has to be jit compiled (``1``) or not (``0``). This works only if computing derivatives using JAX.')

if __name__ =='__main__':

    FLAGS = parser.parse_args()

    if FLAGS.duty_factor == 1.:
        FLAGS.duty_factor=None
    
    print('Input arguments: %s' %str(FLAGS))
    
    ti =  time.time()
    
    if len(FLAGS.seeds) == 0:
        tmpNet = get_net(FLAGS)
        onp.random.seed(None)
        for i in range(len(tmpNet.keys())):
            FLAGS.seeds = FLAGS.seeds + [onp.random.randint(2**32 - 1, size=1)[0]]
        
    #####################################################################################
    # LOAD EVENTS
    #####################################################################################
    
    fname_obs = os.path.join(  FLAGS.fname_obs)
    #fname_obs = os.path.join('../data', FLAGS.fname_obs+'.h5')
    if not os.path.exists(fname_obs):
        raise ValueError('Path to catalog does not exist. Value entered: %s' %fname_obs)
    
    print('Loading events from %s...' %fname_obs)
    events_loaded = load_population(fname_obs)
    
    keylist=list(events_loaded.keys())
    nevents_total = len(events_loaded[keylist[0]])
    print('This catalog has %s events.' %nevents_total)
    
    
    if FLAGS.idx_f is None:
        events_loaded = {k: events_loaded[k][FLAGS.idx_in:] for k in events_loaded.keys()}
    else:
        events_loaded = {k: events_loaded[k][FLAGS.idx_in:FLAGS.idx_f] for k in events_loaded.keys()}
    nevents_total = len(events_loaded[keylist[0]])
    print('Using events between %s and %s, total %s events' %(FLAGS.idx_in, FLAGS.idx_f, nevents_total) )
    
    
    #####################################################################################
    # SPLIT EVENTS BETWEEN PROCESSES ACCORDING TO BATCH SIZE AND NUMBER OF PROCESSES REQUIRED
    #####################################################################################
    
    
    batch_size = FLAGS.batch_size
    npools = FLAGS.npools
    snr_th=FLAGS.snr_th
    
    n_per_it = batch_size*FLAGS.npools # total events computed simultaneously on all cores
    if n_per_it>nevents_total:
        raise ValueError('The chosen batch size and number of pools are too large (it would take <1 iteration to cover all data). Choose smaller values.')
    
    n_it_per_pool = nevents_total//n_per_it # iterations done on every core
    nev_covered = n_per_it*n_it_per_pool # events done on every core
    #if nev_covered<nevents_total:
        
    nev_miss = nevents_total-nev_covered
    # compute how many pools will have to do one iteration more  
    diff = nevents_total-nev_covered
    
    if diff==0: # all cores do same number of iterations

        all_n_it_pools = [ n_it_per_pool for _ in range(npools) ]
        all_n_per_pool =  [ n_it_per_pool*batch_size for _ in range(npools)]
        full_last_chunk_sizes =  [ batch_size for _ in range(npools) ]
        #last_chunk_sizes = 
        cores_to_add=0
        n_it_per_pool_extra=0
        res=0
        int_part=0
        last_chunk_sizes=[]
    else:
        if diff<batch_size: 
            cores_to_add=1 # only 1 core needs to do one more iteration but with n of events < batch size
            res = 1
            int_part = 0
            n_it_per_pool_extra = 1
        elif diff==batch_size: # only one core needs to do one more iteration with n of events = batch size
            cores_to_add=1 
            res=0
            int_part = 1
            n_it_per_pool_extra = 1
        else:
            evsres = diff%batch_size # number of events left after cores did extra iteration 
            int_part = int((diff-evsres)/batch_size) # number of cores doing exactly one batch more
            if evsres>0:
                res=1
            else: res=0
                
            cores_to_add =  int(int_part+res)
            #if 
            n_it_per_pool_extra = 1
        print('Cores which do one extra iteration: %s' %cores_to_add)
        print('N of cores doing less than one batch more: %s' %res)
        print('N of cores doing exatcly one batch more: %s' %int_part)
            
        last_chunk_sizes = [ batch_size for _ in range(int_part) ]
        if int(nev_miss-batch_size*int_part)>0:
            last_chunk_sizes+=[ int(nev_miss-batch_size*int_part) for _ in range(1)] 
        print('Last chunk size will be %s on last %s cores' %(str(last_chunk_sizes), cores_to_add))
        all_n_it_pools = [ n_it_per_pool for _ in range(npools-cores_to_add) ]+[ n_it_per_pool+n_it_per_pool_extra for _ in range(cores_to_add)]
        all_n_per_pool = [ n_it_per_pool*batch_size for _ in range(npools-cores_to_add)]+[ (n_it_per_pool*batch_size)+last_chunk_sizes[i] for i in range(cores_to_add)] # events computed on every core in total 
        full_last_chunk_sizes = onp.hstack( [onp.zeros(npools-len(last_chunk_sizes)), last_chunk_sizes] ) #.T
    
    
    print('Number of iterations on every core: %s' %str(all_n_it_pools))     
    print('%s iterations will be used to cover all the %s events.' %(max(all_n_it_pools), nevents_total))
    print('For each iteration, we will parallise on %s cores with max %s events/core.' %(npools, batch_size))
    print('Number of events computed on every core in total: %s' %str(all_n_per_pool))
    
    all_n_it_pools = onp.array(all_n_it_pools)
    
    all_batch_sizes = onp.zeros((npools, max(all_n_it_pools) )).T
    for i in range(npools):
        for j in range(max(all_n_it_pools)):
            #print(i, j)
            is_last = j==max(onp.array(all_n_it_pools)-1)
            #print(is_last)
            if not is_last or onp.all(full_last_chunk_sizes==0):
                all_batch_sizes[j, i] = batch_size
            else:
                all_batch_sizes[j, i] = full_last_chunk_sizes[i]
    
    all_batch_sizes = all_batch_sizes.astype('int')
    print('All batch sizes, last three iterations (shape nbatches x npools ): ')
    print(all_batch_sizes[-3:, :])
    
    assert all_batch_sizes.sum()==nevents_total
    assert onp.array(all_n_per_pool).sum()==nevents_total
      
    events_lists = {str(i):[] for i in range(npools)}
    idxs_lists = {str(i):[] for i in range(npools)}
      
    pin=0
    for it in range(all_batch_sizes.shape[0]): # iterations
        for p in range(all_batch_sizes.shape[-1]): # pools
            pf =  pin+all_batch_sizes[it, p]
            if pf>pin:
                events_lists[str(p)].append({k: events_loaded[k][pin:pf] for k in events_loaded.keys()})
                idxs_lists[str(p)].append( (pin, pf) )
                nevents_chunk = len(events_lists[str(p)][-1]['dL'])
                assert nevents_chunk == all_batch_sizes[it, p]
                pin = pf
    ncheck=0       
    for evl in  events_lists.values():
        for evs in evl:
            ncheck+=len(evs['dL'])
        
    assert ncheck==nevents_total
    
    
    FLAGS.all_n_it_pools = all_n_it_pools
    FLAGS.events_lists = events_lists
    FLAGS.idxs_lists = idxs_lists
    
    
    ############################################################################
    # Run processes in parallel
    ############################################################################
    
    if FLAGS.npools>1:
        print('Parallelizing on %s CPUs ' %FLAGS.npools)    
        print('Total available CPUs: %s' %str(multiprocessing.cpu_count()) )
        pool =  get_pool(mpi=FLAGS.mpi, threads=FLAGS.npools+1)  
        if FLAGS.mpi:
            pool.map( mainMPI, [ i for i in range(1, FLAGS.npools+1)] ) 
        else:
            pool.starmap( main, [ ( i, FLAGS ) for i in range(1, FLAGS.npools+1)] )
        pool.close()
    else:
        (main(1, FLAGS), )
  
    
    ############################################################################
    # Concatenate results and clean
    ############################################################################
    
    
    if FLAGS.concatenate:
               
        print('\nSaving final version to file...')
        if FLAGS.idx_f is None:
                idxf = str(nevents_total)
        else: idxf = FLAGS.idx_f
            
        suffstr = '_'+str(FLAGS.idx_in)+'_to_'+str(idxf)      
            
    
        pin=FLAGS.idx_in
        res = []
        ndet_tot = 0
        fishers_initialised = False
        for it in range(all_batch_sizes.shape[0]): # iterations
            for p in range(all_batch_sizes.shape[-1]): # pools
                pf =  pin+all_batch_sizes[it, p]
                if pf>pin:
                    print('Concatenating files from %s to %s' %(pin, pf))
                    suff_batch = '_'+str(pin)+'_to_'+str(pf)
                    res_ = from_file( FLAGS.fout, suff=suff_batch)
                    for i in range(len(res_)):
                        if onp.ndim(res_[i])==0:
                            res_[i]=onp.array([res_[i],])
                        
                    res.append(res_)
                    #print(len(res))
                    #print(len(res_))
                    # snrs, fishers, eps_dL, covs, sky_area, errors, cond_numbers
                    snrs = onp.concatenate([res[i][0] for i in range(len(res))], axis=-1)
                    
                    snrs_ = res_[0]
                    ndet = (snrs_>FLAGS.snr_th).sum()
                    print('Number of detections in %s: %s' %(suff_batch, ndet))
                    ndet_tot += ndet
                    
                    not_load_fishers = (ndet==0) or (FLAGS.compute_fisher==False)
                    load_fishers = not not_load_fishers
                    
                    if not_load_fishers:
                        #print('No fisher file for %s' %suff_batch)
                        if ndet==0:
                            print('No detections so far')
                        elif FLAGS.compute_fisher==False:
                            print('Fishers not computed')
                    else: 
                        fishers = onp.concatenate([res[i][1] for i in range(len(res)) if len(res[i]) > 1 ], axis=-1)
                        eps_dL = onp.concatenate([res[i][2] for i in range(len(res)) if len(res[i]) > 1 ], axis=-1)
                        sky_area = onp.concatenate([res[i][4] for i in range(len(res)) if len(res[i]) > 1 ], axis=-1)
                        covs = onp.concatenate([res[i][3] for i in range(len(res)) if len(res[i]) > 1 ], axis=-1)
                        errors = onp.concatenate([res[i][5] for i in range(len(res)) if len(res[i]) > 1 ], axis=-1)
                        cond_numbers = onp.concatenate([res[i][6] for i in range(len(res)) if len(res[i]) > 1 ], axis=-1)
                        idxs_det = onp.concatenate([res[i][7] for i in range(len(res)) if len(res[i]) > 1 ], axis=-1)
                        print('Shape of Fishers: %s\n' %str(fishers.shape))
                    
                    if FLAGS.return_all:
                        snrs_all_ = load_snrs_all(FLAGS.fout, suff=suff_batch)
                        if load_fishers:
                            fishers_all_ = load_fishers_all(FLAGS.fout, suff=suff_batch)
                        if it==0 and p==0:
                            snrs_all = snrs_all_
                            if load_fishers:
                                fishers_all = fishers_all_
                                fishers_initialised = True
                        else:
                            if (not fishers_initialised) and (load_fishers):
                                fishers_all = fishers_all_
                                fishers_initialised = True
                                load_fishers=False
                            for k in snrs_all_.keys():
                                snrs_all[k] = onp.append(snrs_all[k], snrs_all_[k])
                                if (load_fishers) and (fishers_initialised):
                                    fishers_all[k] = onp.append(fishers_all[k], fishers_all_[k], axis=-1)
                    
                    pin = pf
    
        if (ndet_tot==0) or (FLAGS.compute_fisher==False):
            onp.savetxt(os.path.join(FLAGS.fout, 'snrs'+suffstr+'.txt'), snrs)    
        else:
            to_file(snrs, fishers, eps_dL, covs, sky_area, errors, idxs_det, FLAGS.fout, suff=suffstr, cond_numbers=cond_numbers)
        
        
        if FLAGS.return_all:
            if (ndet_tot==0) or (FLAGS.compute_fisher==False): 
                to_file_all(snrs_all, None, FLAGS.fout, suff=suffstr)
            else:
                to_file_all(snrs_all, fishers_all, FLAGS.fout, suff=suffstr)
        
        if ndet_tot>0:        
            print('Saving catalog of detected events...')
            
            if FLAGS.compute_fisher==False:
                idxs_det = onp.arange(0, idxf-FLAGS.idx_in)[ onp.argwhere(snrs>FLAGS.snr_th) ]
                onp.savetxt(os.path.join(FLAGS.fout, 'idxs_det'+suffstr+'.txt'), idxs_det)    
                
            
            events_detected = {k: events_loaded[k][idxs_det.astype('int')] for k in events_loaded.keys()}
            save_data(os.path.join(FLAGS.fout, 'events_detected'+suffstr+'.hdf5'), events_detected, )
        
        if (FLAGS.npools>1) or (FLAGS.npools==1 and all_n_it_pools[0]>1): 
            print('Cleaning...')
            pin=FLAGS.idx_in
            res = []
            for it in range(all_batch_sizes.shape[0]): # iterations
                for p in range(all_batch_sizes.shape[-1]): # pools
                    pf =  pin+all_batch_sizes[it, p]
                    if pf>pin:
                        delete_files(pin, pf, FLAGS.fout)
                        pin = pf
    

        
    
    te=time.time()
    print('------ Done for all. Total execution time: %s sec.\n\n' %(str((te-ti))))
    
