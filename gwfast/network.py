#
#    Copyright (c) 2022 Francesco Iacovelli <francesco.iacovelli@unige.ch>, Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import numpy as onp
from gwfast import gwfastUtils as utils
from scipy.optimize import minimize, basinhopping


class DetNet(object):
    """
    Class to build a network of multiple detectors.
    
    The functions defined within this class allow to get e.g. the *network* SNR and Fisher matrix.
    
    :param dict(GWSignal, ...) signals: Dictionary containing one or multiple individual detector objects.
    :param bool, optional verbose: Boolean specifying if the code has to print additional details during execution.
    
    """
    def __init__(self, signals, verbose=True):
        """
        Constructor method
        """
        # signals is a dictionary of the form
        # {'detector_name': GWSignal object }
        
        self.signals = signals
        self.verbose=verbose
    

    def _clear_cache(self):
        for d in self.signals.keys():
            self.signals[d]._clear_cache()
        
    
    def _update_all_seeds(self, seeds=[], verbose=True):
        """
        Update the seeds for the duty cycle in each individual detector with user input values.
        
        :param list(int) seeds: User input values for the seeds.
        :param bool, optional verbose: Boolean specifying if the code has to print additional details during execution.
        
        """
        if seeds==[]:
            seeds = [ None for _ in range(len(list(self.signals.keys())))]
        for i,d in enumerate(list(self.signals.keys())):
            self.signals[d]._update_seed(seed=seeds[i])
            if verbose:
                print('\nSeed for detector %s is %s'%(d,self.signals[d].seedUse))
    
    def SNR(self, evParams, res=1000, return_all=False):
        """
        Compute the *network signal-to-noise-ratio*, SNR, as a function of the parameters of the event(s).
        
        :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
        :param int res: The resolution of the frequency grid to use.
        :param bool, optional return_all: Boolean specifying if the SNRs of the individual detectors have to be returned separately, together with the network SNR(s). In this case the return type is *dict(array, array, ...)*.
        
        :return: Network SNR(s) as a function of the parameters of the event(s). The shape is :math:`(N_{\\rm events})`.
        :rtype: 1-D array
        
        """
        snrs = {}
        utils.check_evparams(evParams)
        for d in self.signals.keys():
            snr_ = self.signals[d].SNRInteg(evParams, res=res, return_all=return_all)
            if self.signals[d].detector_shape=='T' and return_all:
                for i in range(3):
                   snrs[d+'_%s'%i] = snr_[i]
            else:
                snrs[d] = snr_
        
        net_snr = onp.sqrt(onp.array([ snrs[k]**2 for k in snrs.keys() ]).sum(axis=0))
        if return_all:
            snrs['net'] = net_snr
            #onp.squeeze(onp.sqrt(sum( onp.array( list(snrs.values()),dtype=object)**2)))
            return snrs
        else:
            return net_snr #onp.squeeze(onp.sqrt(sum( onp.array(list(snrs.values()),dtype=object)**2)))
        
    
    def FisherMatr(self, evParams, return_all=False, **kwargs):
        #nparams = self.signals[list(self.signals.keys())[0]].wf_model.nParams
        #nevents = len(evParams[list(evParams.keys())[0]])
        #totF = onp.zeros((nparams,nparams,nevents))
        """
        Compute the *total Fisher information matrix*, FIM, as a function of the parameters of the event(s).
        
        :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
        :param bool, optional return_all: Boolean specifying if the FIMs of the individual detectors have to be returned separately, together with the network FIM(s). In this case the return type is *dict(array, array, ...)*.
        :param kwargs: Optional arguments to be passed to :py:class:`gwfast.signal.GWSignal.FisherMatr`, such as ``res`` or ``use_m1m2``.
        :return: FIM(s) as a function of the parameters of the event(s). The shape is :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
        :rtype: 3-D array
        
        """
        allF={}
        utils.check_evparams(evParams)
        for d in self.signals.keys():
            if self.verbose:
                print('Computing Fisher for %s...' %d)
            F_ = self.signals[d].FisherMatr(evParams, return_all=return_all, **kwargs) 
            #totF +=  self.signals[d].FisherMatr(evParams, **kwargs) 
            if self.signals[d].detector_shape=='T' and return_all:
                for i in range(3):
                   allF[d+'_%s'%i] = F_[i]
            elif self.signals[d].detector_shape=='L' and return_all:
                allF[d] = F_[0]
            else:
                allF[d] = F_
        if self.verbose:
            print('Done.')
        totF = onp.array([allF[k] for k in allF.keys()]).sum(axis=0)
        if return_all:
            allF['net'] = totF #sum(allF.values())
            return allF
        else:
            return totF #sum(allF.values())
        
        

    def optimal_location(self, tcoal, is_tGPS=False):
        """
        Compute the optimal sky position for a signal to be seen by the detector at a given time.
        
        The computation assumes :math:`\psi = 0`.
        
        :param float tcoal: The time at which to compute the optimal location, as GMST in days.
        :param bool, optional is_tGPS: Boolean specifying if the provided time is a GPS time (in seconds) rather than a GMST.
    
        :return: Optimal :math:`\\theta` and :math:`\phi` sky coordinates, in :math:`\\rm rad`.
        :rtype: array(float, float)
        
        """
        # WARNING: The estimate provided by this function works only if the detectors in the network have comparable characteristics, see the paper for discussion.
        # Function to compute the optimal theta and phi for a signal to be seen by the detector network at a given GMST. The boolean is_tGPS can be used to specify whether the provided time is a GPS time rather than a GMST, so that it will be converted.
        # Even if considering Earth rotation, the highest SNR will still be obtained if the source is in the optimal location close to the merger.
        
        if is_tGPS:
            tc = utils.GPSt_to_LMST(tcoal, lat=0., long=0.)
        else:
            tc = tcoal
        
        def pattern_fixedtpsi(pars, tc=tc):
            theta, phi = pars
            tmpsum = 0.
            for d in self.signals.keys():
                # compute the total by adding the squares of the signals and then taking the square root
                if self.signals[d].detector_shape=='T':
                    for i in range(3):
                        Fp, Fc = self.signals[d]._PatternFunction(theta, phi, t=tc, psi=0, rot=i*60.)
                        tmpsum = tmpsum + (Fp**2 + Fc**2)
                else:
                    Fp, Fc = self.signals[d]._PatternFunction(theta, phi, t=tc, psi=0)
                    tmpsum = tmpsum + (Fp**2 + Fc**2)
            return -onp.sqrt(tmpsum)
        # we actually minimize the total pattern function times -1, which is the same as maximizing it
        if len(self.signals.keys())==1:
            # For a single detector all the maxima will correspond to the same value, i.e. 1 for an L and 1.5 for a triangle, thus a single minimization is sufficient
            return minimize(pattern_fixedtpsi, [1.,1.], bounds=((0.,onp.pi), (0.,2.*onp.pi))).x
        else:
            # For a network the minima can be different, thus to find the global one we use the basin-hopping method. Given the small interval, we find 50 iterations sufficient, but this can be easily changed
            return basinhopping(pattern_fixedtpsi, [1.,1.], niter=50, minimizer_kwargs={'bounds':((0.,onp.pi), (0.,2.*onp.pi))}).x
    
    def WFOverlap(self, WF1, WF2, evParams1, evParams2, res=1000, **kwargs):
        """
        Compute the *overlap* of two waveforms in a detector network on two sets of parameters, for one or multiple events.
        
        :param WaveFormModel WF1: Object containing the first waveform model to analyse.
        :param WaveFormModel WF2: Object containing the second waveform model to analyse.
        :param dict(array, array, ...) evParams1: Dictionary containing the parameters of the event(s) for the first waveform model, as in :py:data:`events`.
        :param dict(array, array, ...) evParams2: Dictionary containing the parameters of the event(s) for the second waveform model, as in :py:data:`events`.
        :param int res: The resolution of the frequency grid to use.
        :param unused kwargs: Optional arguments.
        
        :return: Overlap(s) of the two waveforms. The shape is :math:`(N_{\\rm events})`.
        :rtype: 1-D array
        
        """
        utils.check_evparams(evParams1)
        utils.check_evparams(evParams2)
        
        overlap_all = onp.zeros_like(evParams1['Mc'])
        SNR1_all    = onp.zeros_like(evParams1['Mc'])
        SNR2_all    = onp.zeros_like(evParams1['Mc'])
        
        for d in self.signals.keys():
            overlap_int, SNR1, SNR2 = self.signals[d].WFOverlap(WF1, WF2, evParams1, evParams2, res=res, return_separate=True)
            
            overlap_all += overlap_int
            SNR1_all    += SNR1**2
            SNR2_all    += SNR2**2
            
        return overlap_all/onp.sqrt(SNR1_all*SNR2_all)
        
