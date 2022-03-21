#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 17:22:56 2022

@author: Michi
"""

import numpy as onp
import fisherUtils as utils
import fisherTools


class DetNet(object):
    
    def __init__(self, signals):
        
        # signals is a dictionary of the form
        # {'detector_name': GWSignal object }
        
        self.signals = signals
    
    def _clear_cache(self):
        for d in self.signals.keys():
            self.signals[d]._clear_cache()
        
    
    def _update_all_seeds(self, verbose=True):
        for d in self.signals.keys():
            self.signals[d]._update_seed()
            if verbose:
                print('\nSeed for detector %s is %s'%(d,self.signals[d].seedUse))
    
    def SNR(self, evParams, res=1000):
        self.snrs = {}
        utils.check_evparams(evParams)
        for d in self.signals.keys():
            self.snrs[d] =  self.signals[d].SNRInteg(evParams, res=res)**2 
        return onp.sqrt(sum(self.snrs.values()))
        
    
    def FisherMatr(self, evParams, **kwargs):
        #self.Fishers = {}
        nparams = self.signals[list(self.signals.keys())[0]].wf_model.nParams
        nevents = len(evParams[list(evParams.keys())[0]])
        totF = onp.zeros((nparams,nparams,nevents))
        utils.check_evparams(evParams)
        for d in self.signals.keys():
            print('\nComputing Fisher for %s...' %d)
            totF +=  self.signals[d].FisherMatr(evParams, **kwargs) 
        print('\nComputing total Fisher ...')
        #totF = sum(self.Fishers.values())
        #_, _, _ = fisherTools.CheckFisher(totF)
        print('Done.')
        return totF

    
    
    def CovMatr(self, evParams, 
                    FisherM=None, 
                   get_individual = True, 
                    **kwargs):
        
        utils.check_evparams(evParams)
        
        if FisherM is None:
            # Compute total Fisher (which also computes individial Fishers)
            FisherM = self.FisherMatr(evParams, **kwargs)
            
        CovMatrices=None
        if get_individual:
            CovMatrices = {}
            for d in self.signals.keys():
                FisherM_ = self.Fishers[d]
                if FisherM is None:
                    FisherM_ = None
                print('\nComputing Covariance for %s...' %d)
                CovMatrices[d],  _ =  fisherTools.CovMatr(FisherM_, evParams, **kwargs ) 

        print('\nComputing total covariance ...')
        Cov, _ = fisherTools.CovMatr(FisherM, evParams, **kwargs )
        
        print('Done.')
        #print(Cov.shape)
        eps=None
        try:
            return_inv_err=kwargs['return_inv_err']
        except:
            return_inv_err=False
        try:
            return_dL_derivs=kwargs['return_dL_derivs']
        except:
            return_dL_derivs=True
        
        if return_inv_err:
            if return_dL_derivs:
                print('Converting Fisher to dL to check inversion error...')
                FisherM_check = fisherTools.log_dL_to_dL_derivative_fish(FisherM, {'dL':2}, evParams)
            else: 
                FisherM_check = FisherM
            # TODO : vectorize this
            eps = fisherTools.compute_inversion_error(FisherM_check, Cov)
            
        return Cov, eps, CovMatrices
        
