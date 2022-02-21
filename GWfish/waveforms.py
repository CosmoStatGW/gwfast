#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 17:11:30 2022

@author: Michi
"""

from jax.config import config
config.update("jax_enable_x64", True)

# We use both the original numpy, denoted as onp, and the JAX implementation of numpy, denoted as np
import numpy as onp
#import numpy as np
import jax.numpy as np

from abc import ABC, abstractmethod
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
#sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
sys.path.append(SCRIPT_DIR)


import Globals as glob


class WaveFormModel(ABC):
    '''
    Abstract class to compute waveforms
    '''
    
    def __init__(self, objType, fcutNum, is_newtonian=False):
        # The kind of system the wf model is made for, can be 'BBH' or 'BNS'
        self.objType = objType 
        # The cut frequency factor of the waveform, in Hz, to be divided by Mtot (in units of Msun)
        self.fcutNum = fcutNum
        
        self.ParNums = {'Mc':0, 'dL':1, 'theta':2, 'phi':3, 'iota':4, 'psi':5, 'tcoal':6, 'eta':7, 'Phicoal':8}
        self.is_newtonian=is_newtonian
        
        if is_newtonian:
            # In the Newtonian case eta is not included in the Fisher, since it does not enter the signal
            self.ParNums['Phicoal']=7
            self.ParNums.pop('eta', None)

    
    @abstractmethod    
    def Phi(self, f, **kwargs): 
        # The frequency of the GW, as a function of frequency
        # With reference to the book M. Maggiore - Gravitational Waves Vol. 1, with Phi we mean only
        # the GW frequency, not the full phase of the signal, given by 
        # Psi+(f) = 2 pi f (t_c + r/c) - Phi0 - pi/4 - Phi(f)  
        pass
    
    @abstractmethod
    def ddot_Phi(self, f, **kwargs):
        # The second derivative of the GW frequency, needed to compute its amplitude in the 
        # Stationary Phase Approximation, see M. Maggiore - Gravitational Waves Vol. 1 problem 4.1
        pass
    
    def ampl_mod_fac(self, f, **kwargs):
        # Modifications of the amplitude arising from wf models other than 1st order and restricted PN
        # As default we use the 1st order and restricted PN, where no modification arise
        if not np.isscalar(kwargs['Mc']):
            res = np.ones(kwargs['Mc'].shape)
        else:
            res = 1.
        
        return res
    def tau_star(self, f, **kwargs):
        # The relation among the time to coalescence (in seconds) and the frequency (in Hz). We use as default 
        # the expression in M. Maggiore - Gravitational Waves Vol. 1 eq. (4.21), valid in Newtonian and restricted PN approximation
        return 2.18567 * ((1.21/kwargs['Mc'])**(5./3.)) * ((100/f)**(8./3.))
        




class NewtInspiralBNS(WaveFormModel):
    '''
    Leading order (inspiral only) WF for BNS
    '''
    
    def __init__(self): 
        # From T. Dietrich et al. Phys. Rev. D 99, 024029, 2019, below eq. (4) (also look at Fig. 1)
        super().__init__('BNS', 0.04/(2.*np.pi*glob.GMsun_over_c3), is_newtonian=True)
    
    def Phi(self, f, **kwargs):
        return -3.*0.25*(glob.GMsun_over_c3*kwargs['Mc']*8.*np.pi*f)**(-5./3.)  
    
    def ddot_Phi(self, f, **kwargs):
        return (192./5.) * (glob.GMsun_over_c3*kwargs['Mc'])**(5./3.) * (np.pi*f)**(11./3.)



    
class NewtInspiralBBH(WaveFormModel):
    '''
    Leading order (inspiral only) WF for BBH
    '''
    
    def __init__(self):
        # From M. Maggiore - Gravitational Waves Vol. 2 eq. (14.106)
        super().__init__('BBH', 4400., is_newtonian=True)
    
    def Phi(self, f, **kwargs):
        return -3.*0.25*(glob.GMsun_over_c3*kwargs['Mc']*8.*np.pi*f)**(-5./3.)  
    
    def ddot_Phi(self, f, **kwargs):
        return (192./5.) * (glob.GMsun_over_c3*kwargs['Mc'])**(5./3.) * (np.pi*f)**(11./3.)



class ReducedPN_TaylorF2_BNS(WaveFormModel):
    '''
    taylorF2 restricted PN
    '''
    
    # This waveform model is restricted PN (the amplitude stays as in Newtonian approximation) up to 3.5 PN
    def __init__(self, fHigh=None): 
        
        if fHigh is None:
            fHigh = 0.04/(2.*np.pi*glob.GMsun_over_c3)
            # From T. Dietrich et al. Phys. Rev. D 99, 024029, 2019, below eq. (4) (also look at Fig. 1)
        super().__init__('BNS', fHigh)
    
    def Phi(self, f, **kwargs):
        # From A. Buonanno, B. Iyer, E. Ochsner, Y. Pan, B.S. Sathyaprakash - arXiv:0907.0700 - eq. (3.18)
        Mtot = kwargs['Mc']/(kwargs['eta']**(3./5.))
        v = (np.pi*Mtot*f)**(1./3.)
        nu = kwargs['eta']
        # The frequency at last stable orbit is given, in our description, by fcutNum/Mtot, so
        # vlso = (pi*fcutNum)**1/3
        vlso = (np.pi*self.fcutNum)**(1./3.)
        # The number next to the various terms denotes the order in v, 
        Phi03 = 1. + (20./9.)*((743./336) + (11./4.)*nu)*(v**2) - 16.*np.pi*(v**3)  
        Phi4 = 10.*((3058673./1016064.) + (5429./1008.)*nu + (617./144.)*(nu**2))*(v**4)
        Phi5 = np.pi*((38645./756.) - (65./9.)*nu)*(1.+3.*np.log(v/vlso))*(v**5)
        Phi6 = ((11583231236531./4694215680.) - (640./3.)*np.pi - (6848./21.)*(np.euler_gamma + np.log(4.*v)) + ((15737765635./3048192.) + (2255.*(np.pi**2)/12.))*nu + (76055./1728.)*(nu**2) + (127825./1296.)*(nu**3))*(v**6)
        Phi7 = np.pi*((77096675./254016.) + (378515./1512.)*nu - (74045./756)*(nu**2))*(v**7)
        
        return (3./128.)/(nu*(v**5))*(Phi03 + Phi4 + Phi5 + Phi6 + Phi7)
    
    
    def ddot_Phi(self, f, **kwargs):
        # In the restricted PN approach the amplitude is the same as for the Newtonian approximation, so
        # this term is equivalent
        return (192./5.) * (glob.GMsun_over_c3*kwargs['Mc'])**(5./3.) * (np.pi*f)**(11./3.)
    
