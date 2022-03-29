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
            print('Computing Fisher for %s...' %d)
            totF +=  self.signals[d].FisherMatr(evParams, **kwargs) 
        print('Computing total Fisher ...')
        #totF = sum(self.Fishers.values())
        #_, _, _ = fisherTools.CheckFisher(totF)
        print('Done.')
        return totF

    
    