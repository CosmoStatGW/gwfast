#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,sys,jax

jax.config.update("jax_enable_x64", True)


import jax.numpy as jnp
import numpy as np
from jax import jacrev, vmap, jit, hessian

from abc import ABC, abstractmethod

from gwfast.population import POPutils as utils
#import POPutils as utils
from scipy.stats import powerlaw as plawscpy
from scipy.stats import truncnorm as truncnormscpy

class MassDistribution(ABC):
    '''
    Abstract class to compute mass distributions.

    :param list parameters: List containing the parameters of the mass model.
    :param dict, optional hyperparameters: Dictionary containing the hyperparameters of the mass model as keys and their fiducial value as entry.
    :param dict, optional priorlims_parameters: Dictionary containing the parameters of the mass model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.
    '''
    
    def __init__(self, object_type='BBH'):
        self.par_list = []
        self.hyperpar_dict = {}
        self.priorlims_dict = {}
        self.object_type = object_type

    def set_parameters(self, parameters):
        '''
        Setter method for the parameters of the mass model.

        :param list parameters: List containing the parameters of the mass model.
        '''
        self.par_list = parameters

    def set_hyperparameters(self, hyperparameters):
        '''
        Setter method for the hyperparameters of the mass model.

        :param dict new_hyperparameters: Dictionary containing the hyperparameters of the mass model as keys and their new value as entry.
        '''
        self.hyperpar_dict = hyperparameters

    def update_hyperparameters(self, new_hyperparameters):
        '''
        Method to update the hyperparameters of the mass model.

        :param dict new_hyperparameters: Dictionary containing the new hyperparameters of the mass model as keys and their new value as entry.
        '''
        for key in new_hyperparameters.keys():
            if key in self.hyperpar_dict.keys():
                self.hyperpar_dict[key] = new_hyperparameters[key]
            else:
                raise ValueError('The hyperparameter '+key+' is not present in the hyperparameter dictionary.')

    def set_priorlimits(self, limits):
        '''
        Setter method for the prior limits on the parameters of the mass model.

        :param dict limits: Dictionary containing the parameters of the mass model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.
        '''
        self.priorlims_dict = limits

    def update_priorlimits(self, new_limits):
        '''
        Method to update the prior limits on the parameters of the mass model.
        
        :param dict new_limits: Dictionary containing the new prior limits on the parameters of the mass model as keys and their new value as entry, given as a tuple :math:`(l, h)`.
        '''
        for key in new_limits.keys():
            if key in self.priorlims_dict.keys():
                self.priorlims_dict[key] = new_limits[key]
            else:
                raise ValueError('The parameter '+key+' is not present in the prior limits dictionary.')

    def _isin_prior_range(self, par,  val):
        '''
        Function to check if a value is in the prior range of a parameter.

        :param str par: Parameter name.
        :param float val: Parameter value.
        :return: Boolean value.
        :rtype: bool
        '''

        return (val >= self.priorlims_dict[par][0]) & (val <= self.priorlims_dict[par][1])
    
    @abstractmethod
    def mass_function(self,):
        '''
        Mass function of the model.

        :return: Mass function value.
        :rtype: float
        '''
        pass

    @abstractmethod
    def sample_population(self, size):
        '''
        Function to sample the mass distribution.

        :param int size: number of samples
        '''
        pass

    @abstractmethod
    def mass_function_derivative(self,):
        '''
        Derivative of the mass function of the mass distribution.

        :return: Derivative of the mass function value.
        :rtype: float
        '''
        pass


class TruncatedPowerLaw_MassDistribution(MassDistribution):
    '''
    Truncated mass model

    Parameters:
        * alpha_m: Spectral index for the power-law of the primary mass distribution
        * beta_q: Spectral index for the power-law of the mass ratio distribution
        * m_min: Minimum mass of the power-law component of the mass distribution.
        * m_max: Maximum mass of the power-law component of the mass distribution.
    
    :param list parameters: List containing the parameters of the mass model.
    :param dict, optional hyperparameters: Dictionary containing the hyperparameters of the mass model as keys and their fiducial value as entry.
    :param dict, optional priorlims_parameters: Dictionary containing the parameters of the mass model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.

    '''

    def __init__(self, hyperparameters=None, priorlims_parameters=None):

        self.expected_hyperpars = ['alpha_m', 'beta_q', 'm_min', 'm_max']
        super().__init__()

        self.set_parameters(['m1_src', 'm2_src'])

        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)
        else:
            print('Expected hyperparameters: ', self.expected_hyperpars)
            basevalues = {'alpha_m':0.75, 'beta_q':0.1, 'm_min':5., 'm_max':45.}
            print('Base values are: ', list(basevalues.items()))
            self.set_hyperparameters(basevalues)
        
        if priorlims_parameters is not None:
            self.set_priorlimits(priorlims_parameters)
        else:
            # Define the prior limits for 'm1_src' and 'q'
            self.set_priorlimits({'m1_src': (5., 100.), 'm2_src': (5., 100.)})
        
        self.derivative_par_nums = {'alpha_m':0, 'beta_q':1, 'm_min':2, 'm_max':3}

    def _mass1_function(self, m1_src, alpha_m=None, mmin=None, mmax=None):
        '''
        Mass distribution of the primary object.
        
        :param array m1_src: Primary mass.
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional mmin: Minimum mass of the power-law component of the mass distribution.
        :param float, optional mmax: Maximum mass of the power-law component of the mass distribution.
        
        :return: Mass function value at the input masses.
        :rtype: array
        '''

        alpha = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        mmin  = self.hyperpar_dict['m_min'] if mmin is None else mmin
        mmax  = self.hyperpar_dict['m_max'] if mmax is None else mmax

        where_compute   = (m1_src>=mmin) & (m1_src<=mmax)
        pm1             = (m1_src**(-alpha))
        return jnp.where(where_compute, pm1, 0.)
    
    def _mass2_function(self, m2_src, beta_q=None, m_min=None):
        '''
        Mass distribution of the secondary object.
        
        :param array m2_src: Secondary mass.
        :param array m1_src: Primary mass.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        
        :return: Mass function value at the input masses.
        :rtype: array
        '''

        beta  = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin  = self.hyperpar_dict['m_min'] if m_min is None else m_min

        where_compute = m2_src>=mmin
        pm2           = (m2_src)**(beta)#/norm
        return jnp.where(where_compute, pm2, 0.)
    
    def _Cnorm(self, m1_src, beta_q=None, m_min=None):
        '''
        Inverse integral of p(m1, m2) dm2 (i.e. C(m1) in the LVC notation).
        
        :param array m1_src: Primary mass (i.e. the upper bound of the integrals).
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        
        :return: Inverse integral of p(m1, m2) dm2.
        :rtype: array
        '''
        beta  = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin  = self.hyperpar_dict['m_min'] if m_min is None else m_min

        if beta==1:
            C=jnp.log(m1_src/mmin)
        else:
            C=(m1_src**(1+beta)-mmin**(1+beta))/(1+beta)
        
        return 1./C

    def _norm(self, alpha_m=None, m_min=None, m_max=None, sigma_l=None, sigma_h=None):
        '''
        Normalization of the primary mass function.
        
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional sigma_l: Width of the lower filter of the mass distributions.
        :param float, optional sigma_h: Width of the upper filter of the mass distributions.
        
        :return: Normalization of the primary mass function.
        :rtype: float
        '''

        alpha = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        mmin  = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax  = self.hyperpar_dict['m_max'] if m_max is None else m_max

        # Now this is just the normalization of p(m1)! Because we have normalized the rest already
        if  alpha==1:
            NN=jnp.log(mmax/mmin)
        else:
            NN= (mmax**(1-alpha)-mmin**(1-alpha))/(1-alpha)
        return NN
    


    def mass_function(self, m1_src, m2_src, alpha_m=None, beta_q=None, m_min=None, m_max=None, uselog=False):
        '''
        Mass distribution of the objects.
        
        :param array m1_src: Primary mass.
        :param array m2_src: Secondary mass.
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param bool, optional uselog: Boolean specifying whether to return the probability or log-probability, defaults to False.
        
        :return: Mass function value at the input masses.
        :rtype: array
        '''

        alpha = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        beta  = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin  = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax  = self.hyperpar_dict['m_max'] if m_max is None else m_max

        pm1=self._mass1_function(m1_src, alpha, mmin, mmax)
        pm2=self._mass2_function(m2_src, beta, mmin)
        C=self._Cnorm(m1_src, beta, mmin)
        NN=self._norm(alpha, mmin, mmax)
        where_compute = (m2_src < m1_src) & (mmin < m2_src) & (m1_src < mmax )
        pdf = pm1*pm2*C/NN
        
        if not uselog:
            return jnp.where(where_compute,pdf,0)
        else:
            return jnp.where(where_compute,jnp.log(pdf),  -jnp.inf)
        
    def sample_population(self, size, m_min=None, m_max=None):

        '''
        Function to sample the mass model.
        
        :param int size: Size of the masses sample.

        :return: Sampled masses.
        :rtype: dict(array, array)
        '''
        mmin  = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax  = self.hyperpar_dict['m_max'] if m_max is None else m_max

        m1_src = utils.inverse_cdf_sampling(self._mass1_function, size, [mmin,mmax])
        m2_src = utils.inverse_cdf_sampling_uppercond(self._mass2_function, mmin, m1_src)

        return {'m1_src':m1_src, 'm2_src':m2_src}
    
    def mass_function_derivative(self, m1_src, m2_src, alpha_m=None, beta_q=None, m_min=None, m_max=None, uselog=False):
        '''
        Derivative with respect to the hyperparameters of the mass function.
        
        :param array m1_src: Primary mass.
        :param array m2_src: Secondary mass.
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the derivatives of the mass function.
        :rtype: array
        '''

        alpha = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        beta  = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin  = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax  = self.hyperpar_dict['m_max'] if m_max is None else m_max

        funder = lambda m1_src, m2_src, alpha, beta, mmin, mmax: self.mass_function(m1_src, m2_src, alpha, beta, mmin, mmax, uselog=uselog)
        
        derivs_all = np.squeeze(np.asarray(jacrev(funder, argnums=(2,3,4,5))(m1_src, m2_src, jnp.array([alpha]), jnp.array([beta]), jnp.array([mmin]), jnp.array([mmax]))))
        
        return derivs_all

    def mass_function_hessian(self, m1_src, m2_src, alpha_m=None, beta_q=None, m_min=None, m_max=None, uselog=False):
        '''
        Hessian with respect to the hyperparameters of the mass function.
        
        :param array m1_src: Primary mass.
        :param array m2_src: Secondary mass.
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the Hessians of the mass function.
        :rtype: array
        '''

        alpha = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        beta  = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin  = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax  = self.hyperpar_dict['m_max'] if m_max is None else m_max
        
        funder = lambda m1_src, m2_src, alpha, beta, mmin, mmax: self.mass_function(m1_src, m2_src, alpha, beta, mmin, mmax, uselog=uselog)

        derivs_all = np.squeeze(np.asarray(hessian(funder, argnums=(2,3,4,5))(m1_src, m2_src, jnp.array([alpha]), jnp.array([beta]), jnp.array([mmin]), jnp.array([mmax]))))
        
        return derivs_all
    
class BrokenPowerLaw_MassDistribution(MassDistribution):
    '''
    Broken Power-Law model.

    Parameters:
        * alpha_m_1: Spectral index for the power-law of the primary mass distribution below m_break.
        * alpha_m_2: Spectral index for the power-law of the primary mass distribution above m_break.
        * beta_q: Spectral index for the power-law of the mass ratio distribution.
        * m_min: Minimum mass of the mass distribution.
        * m_max: Maximum mass of the mass distribution.
        * b_m: The fraction of the way between m_min and m_max at which the primary mass distribution breaks.
        * delta_m: Width of the smoothing component.
    
    :param list parameters: List containing the parameters of the mass model.
    :param dict, optional hyperparameters: Dictionary containing the hyperparameters of the mass model as keys and their fiducial value as entry.
    :param dict, optional priorlims_parameters: Dictionary containing the parameters of the mass model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.

    '''

    def __init__(self, hyperparameters=None, priorlims_parameters=None):

        self.expected_hyperpars = ['alpha_m_1', 'alpha_m_2', 'beta_q', 'm_min', 'm_max', 'b_m', 'delta_m']
        super().__init__()

        self.set_parameters(['m1_src', 'm2_src'])

        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)
        else:
            print('Expected hyperparameters: ', self.expected_hyperpars)
            basevalues = {'alpha_m_1':1.58, 'alpha_m_2':5.59, 'beta_q':1.40, 'm_min':3.96, 'm_max':87.14, 'b_m':0.43, 'delta_m':4.83}
            print('Base values are: ', list(basevalues.items()))
            self.set_hyperparameters(basevalues)
        
        if priorlims_parameters is not None:
            self.set_priorlimits(priorlims_parameters)
        else:
            # Define the prior limits for 'm1_src' and 'q'
            self.set_priorlimits({'m1_src': (3.96, 87.14), 'm2_src': (3.96, 87.14)})
        
        self.derivative_par_nums = {'alpha_m_1':0, 'alpha_m_2':1, 'beta_q':2, 'm_min':3, 'm_max':4, 'b_m':5, 'delta_m':6}

    
    def _mass1_function(self, m1_src, alpha_m_1=None, alpha_m_2=None, mmin=None, mmax=None, b_m=None, delta_m=None):
        '''
        Mass distribution of the primary object.
        
        :param array m1_src: Primary mass.
        :param float, optional alpha_m_1: Spectral index for the power-law of the primary mass distribution below m_break.
        :param float, optional alpha_m_2: Spectral index for the power-law of the primary mass distribution above m_break.
        :param float, optional mmin: Minimum mass of the power-law component of the mass distribution.
        :param float, optional mmax: Maximum mass of the power-law component of the mass distribution.
        :param float, optional b_m: The fraction of the way between m_min and m_max at which the primary mass distribution breaks.
        :param float, optional delta_m: Width of the smoothing component.
        
        :return: Mass function value at the input masses.
        :rtype: array
        '''

        alpha_1      = self.hyperpar_dict['alpha_m_1'] if alpha_m_1 is None else alpha_m_1
        alpha_2      = self.hyperpar_dict['alpha_m_2'] if alpha_m_2 is None else alpha_m_2
        mmin         = self.hyperpar_dict['m_min'] if mmin is None else mmin
        mmax         = self.hyperpar_dict['m_max'] if mmax is None else mmax
        bm           = self.hyperpar_dict['b_m'] if b_m is None else b_m
        delta_m      = self.hyperpar_dict['delta_m'] if delta_m is None else delta_m

        mbreak         = bm*(mmax-mmin) + mmin
        PTcomp         = jnp.nan_to_num(utils.planck_taper(m1_src, mmin, delta_m))# smoothing
        where_compute1 = (m1_src>=mmin) & (m1_src<mbreak) 
        PLcomp1        = m1_src**(-alpha_1)*PTcomp
        where_compute2 = (m1_src>=mbreak) & (m1_src<=mmax)
        PLcomp2        = (mbreak**(alpha_2-alpha_1))*m1_src**(-alpha_2)*PTcomp
        
        BPL = jnp.where(where_compute1, PLcomp1, jnp.where(where_compute2, PLcomp2, 0.))
        return BPL
    
    def _mass2_function(self, m2_src, beta_q=None, m_min=None, delta_m=None):
        '''
        Mass distribution of the secondary object.
        
        :param array m2_src: Secondary mass.
        :param array m1_src: Primary mass.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional delta_m: Width of the smoothing component.
        
        return: Mass function value at the input masses.
        :rtype: array
        '''

        beta = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin = self.hyperpar_dict['m_min'] if m_min is None else m_min
        delta_m = self.hyperpar_dict['delta_m'] if delta_m is None else delta_m
        
        where_compute = (mmin<=m2_src)
        PLcomp = jnp.where(where_compute,m2_src**(beta), 0.)
        PTcomp = jnp.nan_to_num(utils.planck_taper(m2_src, mmin, delta_m))

        return PLcomp * PTcomp
    

    def _Cnorm(self, m1_src, beta_q=None, m_min=None,m_max=None, delta_m=None):
        '''
        Inverse integral of p(m1, m2) dm2 (i.e. C(m1) in the LVC notation).
        
        :param array m1_src: Primary mass (i.e. the upper bound of the integrals).
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional delta_m: Width of the lower filter of the mass distributions.
        
        :return: Inverse integral of p(m1, m2) dm2.
        :rtype: array
        '''
        beta    = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin    = self.hyperpar_dict['m_min'] if m_min is None else m_min
        deltam  = self.hyperpar_dict['delta_m'] if delta_m is None else delta_m

        # Create an adaptive grid with finer resolution near m_min to accurately integrate p(m2)
        xlow    =jnp.linspace(mmin, mmin+deltam+deltam/10, 200)
        xup     =jnp.linspace(mmin+deltam+deltam/10+deltam/100, m1_src.max(), 200)
        xx      =jnp.sort(jnp.concatenate([xlow,xup], ))

        pm2     = self._mass2_function(xx, beta, mmin, deltam)
        cdf     = utils.cumtrapz_JAX(pm2, xx)
        C       = jnp.interp( m1_src, xx[1:], cdf)
        return jnp.nan_to_num(1./C,posinf=0.)

    def _norm(self, alpha_m_1=None, alpha_m_2=None, mmin=None, mmax=None, b_m=None, delta_m=None):
        '''
        Normalization of the primary mass function.
        
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional lambda_peak: Ratio of the power-law to Gaussian component.
        :param float, optional sigma_l: Width of the lower filter of the mass distributions.
        :param float, optional sigma_h: Width of the upper filter of the mass distributions.
        :param float, optional mu_m: Mean of the Gaussian component.
        :param float, optional sigma_m: Standard deviation of the Gaussian component.
        
        :return: Normalization of the primary mass function.
        :rtype: float
        '''

        alpha_1      = self.hyperpar_dict['alpha_m_1'] if alpha_m_1 is None else alpha_m_1
        alpha_2      = self.hyperpar_dict['alpha_m_2'] if alpha_m_2 is None else alpha_m_2
        mmin         = self.hyperpar_dict['m_min'] if mmin is None else mmin
        mmax         = self.hyperpar_dict['m_max'] if mmax is None else mmax
        bm           = self.hyperpar_dict['b_m'] if b_m is None else b_m
        deltam      = self.hyperpar_dict['delta_m'] if delta_m is None else delta_m


        # Now this is just the normalization of p(m1)! Because we have normalized the rest already

        mbreak  = bm*(mmax-mmin) + mmin
        ms1     = jnp.linspace(1., mmin+deltam+deltam/10, 200)
        ms2     = jnp.linspace( mmin+deltam+deltam/10+1e-01, mbreak-mbreak/10, 100 )
        ms3     = jnp.linspace( mbreak-mbreak/10+1e-01, mbreak+mbreak/10, 50 )
        ms4     = jnp.linspace(mbreak+mbreak/10+1e-01, mmax+mmax/10, 200 )
        
        ms      = jnp.sort(jnp.concatenate([ms1,ms2, ms3, ms4], ))
        
        pm1     = self._mass1_function(ms,alpha_1, alpha_2, mmin, mmax, bm, deltam)
        norm    = jnp.trapz(pm1,ms)
        return norm


    def mass_function(self, m1_src, m2_src, alpha_m_1=None, alpha_m_2=None, beta_q=None, m_min=None, m_max=None, b_m=None, delta_m=None, uselog=False):
        '''
        Mass distribution of the objects.
        
        :param array m1_src: Primary mass.
        :param array m2_src: Secondary mass.
        :param float, optional alpha_m_1: Spectral index for the power-law of the primary mass distribution below m_break.
        :param float, optional alpha_m_2: Spectral index for the power-law of the primary mass distribution above m_break.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional b_m: The fraction of the way between m_min and m_max at which the primary mass distribution breaks.
        :param float, optional delta_m: Width of the smoothing component.
        :param bool, optional uselog: Boolean specifying whether to return the probability or log-probability, defaults to False.
        
        :return: Mass function value at the input masses.
        :rtype: array
        '''

        alpha_1      = self.hyperpar_dict['alpha_m_1'] if alpha_m_1 is None else alpha_m_1
        alpha_2      = self.hyperpar_dict['alpha_m_2'] if alpha_m_2 is None else alpha_m_2
        beta         = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin         = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax         = self.hyperpar_dict['m_max'] if m_max is None else m_max
        bm           = self.hyperpar_dict['b_m'] if b_m is None else b_m
        delta_m      = self.hyperpar_dict['delta_m'] if delta_m is None else delta_m

        # First, p(m1) and p(m2), un-normalized
        pm1= self._mass1_function(m1_src, alpha_1, alpha_2, mmin, mmax, bm, delta_m)
        pm2= self._mass2_function(m2_src, beta, mmin, delta_m)

        # This term normalizes the marginal distribution \int p(m1,m2) dm2 to be p(m1) 
        C= self._Cnorm(m1_src, beta, mmin, mmax, delta_m)

        # And finally this normalizes p(m1) to unity 
        NN= self._norm(alpha_1, alpha_2, mmin, mmax, bm, delta_m)

        where_compute = (m2_src < m1_src) & (mmin< m2_src) & (m1_src < mmax )
        
        pdf = pm1*pm2*C/NN
        if not uselog:
            return jnp.where(where_compute, pdf, 0.)
        else:
            return jnp.where(where_compute, jnp.log(pdf),-jnp.inf)
            
                
    def sample_population(self, size,m_min=None, m_max=None):
        '''
        Function to sample the mass model.
        
        :param int size: Size of the masses sample.

        :return: Sampled masses.
        :rtype: dict(array, array)
        '''
        mmin  = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax  = self.hyperpar_dict['m_max'] if m_max is None else m_max

        m1_src = utils.inverse_cdf_sampling(self._mass1_function, size, [mmin,mmax])
        m2_src = utils.inverse_cdf_sampling_uppercond(self._mass2_function, mmin, m1_src)

        return {'m1_src':m1_src, 'm2_src':m2_src}
    
    def mass_function_derivative(self, m1_src, m2_src, alpha_m_1=None, alpha_m_2=None, beta_q=None, m_min=None, m_max=None, b_m=None, delta_m=None, uselog=False):
        '''
        Derivative with respect to the hyperparameters of the mass function.
        
        :param array m1_src: Primary mass.
        :param array m2_src: Secondary mass.
        :param float, optional alpha_m_1: Spectral index for the power-law of the primary mass distribution below m_break.
        :param float, optional alpha_m_2: Spectral index for the power-law of the primary mass distribution above m_break.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional b_m: The fraction of the way between m_min and m_max at which the primary mass distribution breaks.
        :param float, optional delta_m: Width of the smoothing component.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the derivatives of the mass function.
        :rtype: array
        '''

        alpha_1      = self.hyperpar_dict['alpha_m_1'] if alpha_m_1 is None else alpha_m_1
        alpha_2      = self.hyperpar_dict['alpha_m_2'] if alpha_m_2 is None else alpha_m_2
        beta         = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin         = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax         = self.hyperpar_dict['m_max'] if m_max is None else m_max
        bm           = self.hyperpar_dict['b_m'] if b_m is None else b_m
        delta_m      = self.hyperpar_dict['delta_m'] if delta_m is None else delta_m
        
        funder = lambda m1_src, m2_src, alpha_1, alpha_2, beta, mmin, mmax, bm, delta_m: self.mass_function(m1_src, m2_src, alpha_1, alpha_2, beta, mmin, mmax, bm, delta_m, uselog=uselog)
        
        derivs_all = np.squeeze(np.asarray(jacrev(funder, argnums=(2,3,4,5,6,7,8))(m1_src, m2_src, jnp.array([alpha_1]), jnp.array([alpha_2]), jnp.array([beta]), jnp.array([mmin]), jnp.array([mmax]), jnp.array([bm]), jnp.array([delta_m]))))
        
        return derivs_all
    
    def mass_function_hessian(self, m1_src, m2_src, alpha_m_1=None, alpha_m_2=None, beta_q=None, m_min=None, m_max=None, b_m=None, delta_m=None, uselog=False):
        '''
        Hessian with respect to the hyperparameters of the mass function.
        
        :param array m1_src: Primary mass.
        :param array m2_src: Secondary mass.
        :param float, optional alpha_m_1: Spectral index for the power-law of the primary mass distribution below m_break.
        :param float, optional alpha_m_2: Spectral index for the power-law of the primary mass distribution above m_break.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional b_m: The fraction of the way between m_min and m_max at which the primary mass distribution breaks.
        :param float, optional delta_m: Width of the smoothing component.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the Hessians of the mass function.
        :rtype: array
        '''

        alpha_1      = self.hyperpar_dict['alpha_m_1'] if alpha_m_1 is None else alpha_m_1
        alpha_2      = self.hyperpar_dict['alpha_m_2'] if alpha_m_2 is None else alpha_m_2
        beta         = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin         = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax         = self.hyperpar_dict['m_max'] if m_max is None else m_max
        bm           = self.hyperpar_dict['b_m'] if b_m is None else b_m
        delta_m      = self.hyperpar_dict['delta_m'] if delta_m is None else delta_m

        funder = lambda m1_src, m2_src, alpha_1, alpha_2, beta, mmin, mmax, bm, delta_m: self.mass_function(m1_src, m2_src, alpha_1, alpha_2, beta, mmin, mmax, bm, delta_m, uselog=uselog)
        
        derivs_all = np.squeeze(np.asarray(hessian(funder, argnums=(2,3,4,5,6,7,8))(m1_src, m2_src, jnp.array([alpha_1]), jnp.array([alpha_2]), jnp.array([beta]), jnp.array([mmin]), jnp.array([mmax]), jnp.array([bm]), jnp.array([delta_m]))))
        
        return derivs_all


    
class PowerLawPlusPeak_MassDistribution(MassDistribution):
    '''
    Power-Law+Peak model.

    Parameters:
        * alpha_m: Spectral index for the power-law of the primary mass distribution
        * beta_q: Spectral index for the power-law of the mass ratio distribution
        * m_min: Minimum mass of the power-law component of the mass distribution.
        * m_max: Maximum mass of the power-law component of the mass distribution.
        * lambda_peak: Ratio of the power-law to Gaussian component.
        * delta_m: Width of the smoothing component.
        * mu_m: Mean of the Gaussian component.
        * sigma_m: Standard deviation of the Gaussian component.
    
    :param list parameters: List containing the parameters of the mass model.
    :param dict, optional hyperparameters: Dictionary containing the hyperparameters of the mass model as keys and their fiducial value as entry.
    :param dict, optional priorlims_parameters: Dictionary containing the parameters of the mass model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.

    '''

    def __init__(self, hyperparameters=None, priorlims_parameters=None):

        self.expected_hyperpars = ['alpha_m', 'beta_q', 'm_min', 'm_max', 'lambda_peak', 'delta_m', 'mu_m', 'sigma_m']
        super().__init__()

        self.set_parameters(['m1_src', 'm2_src'])

        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)
        else:
            print('Expected hyperparameters: ', self.expected_hyperpars)
            basevalues = {'alpha_m':3.4, 'beta_q':1.1, 'm_min':5.1, 'm_max':87., 'lambda_peak':0.039, 'delta_m':4.8, 'mu_m':34., 'sigma_m':3.6}
            print('Base values are: ', list(basevalues.items()))
            self.set_hyperparameters(basevalues)
        
        if priorlims_parameters is not None:
            self.set_priorlimits(priorlims_parameters)
        else:
            # Define the prior limits for 'm1_src' and 'q'
            self.set_priorlimits({'m1_src': (5., 87.), 'm2_src': (5., 87.)})
        
        self.derivative_par_nums = {'alpha_m':0, 'beta_q':1, 'm_min':2, 'm_max':3, 'lambda_peak':4, 'delta_m':5, 'mu_m':6, 'sigma_m':7}


    
    def _mass1_function(self, m1_src, alpha_m=None, m_min=None, m_max=None, lambda_peak=None, delta_m=None, mu_m=None, sigma_m=None):
        '''
        Mass distribution of the primary object.
        
        :param array m1_src: Primary mass.
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional mmin: Minimum mass of the power-law component of the mass distribution.
        :param float, optional mmax: Maximum mass of the power-law component of the mass distribution.
        :param float, optional lambda_peak: Ratio of the power-law to Gaussian component.
        :param float, optional delta_m: Width of the smoothing component.
        :param float, optional mu_m: Mean of the Gaussian component.
        :param float, optional sigma_m: Standard deviation of the Gaussian component.
        
        :return: Mass function value at the input masses.
        :rtype: array
        '''

        alpha        = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        mmin         = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax         = self.hyperpar_dict['m_max'] if m_max is None else m_max
        lambda_peak  = self.hyperpar_dict['lambda_peak'] if lambda_peak is None else lambda_peak
        delta_m      = self.hyperpar_dict['delta_m'] if delta_m is None else delta_m
        mu_m         = self.hyperpar_dict['mu_m'] if mu_m is None else mu_m
        sigma_m      = self.hyperpar_dict['sigma_m'] if sigma_m is None else sigma_m

        where_compute   = (m1_src>=mmin) & (m1_src<=mmax)
        PLcomp = jnp.where(where_compute, utils.inversepowerlaw(m1_src, alpha, (mmin,mmax)), 0.)
        Gcomp  = utils.gaussian_norm(m1_src, mu_m, sigma_m)        
        PTcomp = jnp.nan_to_num(utils.planck_taper(m1_src, mmin, delta_m))
        PLP    = ((1.-lambda_peak)*PLcomp + lambda_peak*Gcomp)*PTcomp
        return PLP
        
    
    def _mass2_function(self, m2_src, beta_q=None, m_min=None, delta_m=None):
        '''
        Mass distribution of the secondary object.
        
        :param array m2_src: Secondary mass.
        :param array m1_src: Primary mass.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional delta_m: Width of the smoothing component.
        
        '''
        beta = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin = self.hyperpar_dict['m_min'] if m_min is None else m_min
        delta_m = self.hyperpar_dict['delta_m'] if delta_m is None else delta_m

        PLcomp = m2_src**(beta)
        PTcomp = jnp.nan_to_num(utils.planck_taper(m2_src, mmin, delta_m))

        return PLcomp*PTcomp
    
    def _Cnorm(self, m1_src, beta_q=None, m_min=None, sigma_l=None):
        '''
        Inverse integral of  p(m1, m2) dm2 (i.e. C(m1) in the LVC notation).
        
        :param array m1_src: Primary mass (i.e. the upper bound of the integrals).
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional sigma_l: Width of the lower filter of the mass distributions.
        :param float, optional sigma_h: Width of the upper filter of the mass distributions.
        
        :return: Inverse integral of p(m1, m2) dm2.
        :rtype: array
        '''
        
        beta  = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin  = self.hyperpar_dict['m_min'] if m_min is None else m_min
        deltam = self.hyperpar_dict['delta_m'] if sigma_l is None else sigma_l

        
        # lower edge
        ms1 = jnp.linspace(mmin, mmin+deltam+deltam/10, 200)
        # in between 
        ms2 = jnp.linspace(mmin+deltam+deltam/10+1e-01, m1_src.max(), 200)
        xx= jnp.sort(jnp.unique(jnp.concatenate([ms1,ms2] )))
        
        pm2 = self._mass2_function(xx, beta, mmin, deltam)
        cdf = utils.cumtrapz_JAX(pm2, xx) #cumtrapz(pm2, xx) #utils.cumtrapz_JAX(pm2, xx)
        C = jnp.interp( m1_src, xx[1:], cdf)
        return jnp.nan_to_num(1./C,posinf=0.)
    
    def _norm(self, alpha_m=None, m_min=None, m_max=None, lambda_peak=None, delta_m=None, mu_m=None, sigma_m=None):
        '''
        Normalization of the primary mass function.
        
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional lambda_peak: Ratio of the power-law to Gaussian component.
        :param float, optional sigma_l: Width of the lower filter of the mass distributions.
        :param float, optional sigma_h: Width of the upper filter of the mass distributions.
        :param float, optional mu_m: Mean of the Gaussian component.
        :param float, optional sigma_m: Standard deviation of the Gaussian component.
        
        :return: Normalization of the primary mass function.
        :rtype: float
        '''

        alpha        = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        mmin         = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax         = self.hyperpar_dict['m_max'] if m_max is None else m_max
        lambda_peak  = self.hyperpar_dict['lambda_peak'] if lambda_peak is None else lambda_peak
        deltam        = self.hyperpar_dict['delta_m'] if delta_m is None else delta_m
        mu_m         = self.hyperpar_dict['mu_m'] if mu_m is None else mu_m
        sigma_m      = self.hyperpar_dict['sigma_m'] if sigma_m is None else sigma_m

        # Now this is just the normalization of p(m1)! Because we have normalized the rest already
        # lower edge
        ms1 = np.linspace(1., mmin+deltam+deltam/10, 200)
            
        # before gaussian peak
        ms2 = np.linspace( mmin+deltam+deltam/10+1e-01, mu_m-5*sigma_m, 100 )
            
        # around gaussian peak
        ms3= np.linspace( mu_m-5*sigma_m+1e-01, mu_m+5*sigma_m, 100 )
            
        # after gaussian peak
        max_compute = max(mmax, mu_m+10*sigma_m)
        ms4 = np.linspace(mu_m+5*sigma_m+1e-01, max_compute+max_compute/2, 100 )

        ms  = jnp.sort(jnp.unique(jnp.concatenate([ms1,ms2, ms3,ms4],)))
        
        pm1= self._mass1_function(ms, alpha_m=alpha, m_min=mmin, m_max=mmax, lambda_peak=lambda_peak, delta_m=delta_m, mu_m=mu_m, sigma_m=sigma_m)
        norm=jnp.trapz(pm1,ms)
        return norm


    
    def mass_function(self, m1_src, m2_src, alpha_m=None, beta_q=None, m_min=None, m_max=None, lambda_peak=None, delta_m=None, mu_m=None, sigma_m=None, uselog=False):
        '''
        Mass distribution of the objects.
        
        :param array m1_src: Primary mass.
        :param array m2_src: Secondary mass.
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional lambda_peak: Ratio of the power-law to Gaussian component.
        :param float, optional delta_m: Width of the smoothing component.
        :param float, optional mu_m: Mean of the Gaussian component.
        :param float, optional sigma_m: Standard deviation of the Gaussian component.
        :param bool, optional uselog: Boolean specifying whether to return the probability or log-probability, defaults to False.
        
        :return: Mass function value at the input masses.
        :rtype: array
        '''

        alpha        = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        beta         = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin         = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax         = self.hyperpar_dict['m_max'] if m_max is None else m_max
        lambda_peak  = self.hyperpar_dict['lambda_peak'] if lambda_peak is None else lambda_peak
        delta_m      = self.hyperpar_dict['delta_m'] if delta_m is None else delta_m
        mu_m         = self.hyperpar_dict['mu_m'] if mu_m is None else mu_m
        sigma_m      = self.hyperpar_dict['sigma_m'] if sigma_m is None else sigma_m

        # First, p(m1) and p(m2), un-normalized
        pm1= self._mass1_function(m1_src, alpha, mmin, mmax, lambda_peak, delta_m, mu_m, sigma_m)
        pm2= self._mass2_function(m2_src, beta, mmin, delta_m)

        # This term normalizes the marginal distribution \int p(m1,m2) dm2 to be p(m1)
        C= self._Cnorm(m1_src, beta, mmin, delta_m)

        # And finally this normalizes p(m1) to unity
        NN= self._norm(alpha, mmin, mmax, lambda_peak, delta_m, mu_m, sigma_m)

        # compute normalized pdf
        where_compute = (m2_src < m1_src) & (mmin< m2_src) & (m1_src < mmax )
        pdf = pm1*pm2*C/NN

        if not uselog:
            return jnp.where(where_compute, pdf, 0.)
        else:
            return jnp.where(where_compute, jnp.log(pdf), -jnp.inf)


    def sample_population(self, size, m_min=None, m_max=None):
        '''
        Function to sample the mass model.
        
        :param int size: Size of the masses sample.

        :return: Sampled masses.
        :rtype: dict(array, array)
        '''
        mmin  = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax  = self.hyperpar_dict['m_max'] if m_max is None else m_max
        m1_src = utils.inverse_cdf_sampling(self._mass1_function, size, [mmin,mmax])
        m2_src = utils.inverse_cdf_sampling_uppercond(self._mass2_function, mmin, m1_src)


        return {'m1_src':m1_src, 'm2_src':m2_src}
    
    def mass_function_derivative(self, m1_src, m2_src, alpha_m=None, beta_q=None, m_min=None, m_max=None, lambda_peak=None, delta_m=None, mu_m=None, sigma_m=None, uselog=False):
        '''
        Derivative with respect to the hyperparameters of the mass function.
        
        :param array m1_src: Primary mass.
        :param array m2_src: Secondary mass.
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional lambda_peak: Ratio of the power-law to Gaussian component.
        :param float, optional delta_m: Width of the smoothing component.
        :param float, optional mu_m: Mean of the Gaussian component.
        :param float, optional sigma_m: Standard deviation of the Gaussian component.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the derivatives of the mass function.
        :rtype: array
        '''
        
        alpha        = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        beta         = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin         = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax         = self.hyperpar_dict['m_max'] if m_max is None else m_max
        lambda_peak  = self.hyperpar_dict['lambda_peak'] if lambda_peak is None else lambda_peak
        delta_m      = self.hyperpar_dict['delta_m'] if delta_m is None else delta_m
        mu_m         = self.hyperpar_dict['mu_m'] if mu_m is None else mu_m
        sigma_m      = self.hyperpar_dict['sigma_m'] if sigma_m is None else sigma_m
        
        funder = lambda m1_src, m2_src, alpha, beta, mmin, mmax, lambda_peak, delta_m, mu_m, sigma_m: self.mass_function(m1_src, m2_src, alpha, beta, mmin, mmax, lambda_peak, delta_m, mu_m, sigma_m, uselog=uselog)
        
        derivs_all = np.squeeze(np.asarray(jacrev(funder, argnums=(2,3,4,5,6,7,8,9))(m1_src, m2_src, jnp.array([alpha]), jnp.array([beta]), jnp.array([mmin]), jnp.array([mmax]), jnp.array([lambda_peak]), jnp.array([delta_m]), jnp.array([mu_m]), jnp.array([sigma_m]))))
        
        return derivs_all
    
    def mass_function_hessian(self, m1_src, m2_src, alpha_m=None, beta_q=None, m_min=None, m_max=None, lambda_peak=None, delta_m=None, mu_m=None, sigma_m=None, uselog=False):
        '''
        Hessian with respect to the hyperparameters of the mass function.
        
        :param array m1_src: Primary mass.
        :param array m2_src: Secondary mass.
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional lambda_peak: Ratio of the power-law to Gaussian component.
        :param float, optional delta_m: Width of the smoothing component.
        :param float, optional mu_m: Mean of the Gaussian component.
        :param float, optional sigma_m: Standard deviation of the Gaussian component.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the Hessians of the mass function.
        :rtype: array
        '''

        alpha        = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        beta         = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin         = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax         = self.hyperpar_dict['m_max'] if m_max is None else m_max
        lambda_peak  = self.hyperpar_dict['lambda_peak'] if lambda_peak is None else lambda_peak
        delta_m      = self.hyperpar_dict['delta_m'] if delta_m is None else delta_m
        mu_m         = self.hyperpar_dict['mu_m'] if mu_m is None else mu_m
        sigma_m      = self.hyperpar_dict['sigma_m'] if sigma_m is None else sigma_m
        
        funder = lambda m1_src, m2_src, alpha, beta, mmin, mmax, lambda_peak, delta_m, mu_m, sigma_m: self.mass_function(m1_src, m2_src, alpha, beta, mmin, mmax, lambda_peak, delta_m, mu_m, sigma_m, uselog=uselog)
        
        derivs_all = np.squeeze(np.asarray(hessian(funder, argnums=(2,3,4,5,6,7,8,9))(m1_src, m2_src, jnp.array([alpha]), jnp.array([beta]), jnp.array([mmin]), jnp.array([mmax]), jnp.array([lambda_peak]), jnp.array([delta_m]), jnp.array([mu_m]), jnp.array([sigma_m]))))
        
        return derivs_all
    

class MultiPeak_MassDistribution(MassDistribution):
    '''
    Multi Peak model.

    Parameters:
        * alpha_m: Spectral index for the power-law of the primary mass distribution
        * beta_q: Spectral index for the power-law of the mass ratio distribution
        * m_min: Minimum mass of the power-law component of the mass distribution.
        * m_max: Maximum mass of the power-law component of the mass distribution.
        * lambda_m: Fraction of systems in the Gaussian components.
        * lambda_1_m: Fraction of systems in the Gaussian components belonging to the lower-mass component.
        * delta_m: Width of the smoothing component.
        * mu_m_1: Mean of the lower-mass Gaussian component.
        * sigma_m_1: Standard deviation of the lower-mass Gaussian component.
        * mu_m_2: Mean of the higher-mass Gaussian component.
        * sigma_m_2: Standard deviation of the higher-mass Gaussian component.
    
    :param list parameters: List containing the parameters of the mass model.
    :param dict, optional hyperparameters: Dictionary containing the hyperparameters of the mass model as keys and their fiducial value as entry.
    :param dict, optional priorlims_parameters: Dictionary containing the parameters of the mass model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.

    '''

    def __init__(self, hyperparameters=None, priorlims_parameters=None):

        self.expected_hyperpars = ['alpha_m', 'beta_q', 'm_min', 'm_max', 'lambda_m', 'lambda_1_m', 'delta_m', 'mu_m_1', 'sigma_m_1', 'mu_m_2', 'sigma_m_2']
        super().__init__()

        self.set_parameters(['m1_src', 'm2_src'])

        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)
        else:
            print('Expected hyperparameters: ', self.expected_hyperpars)
            basevalues = {'alpha_m':2.9, 'beta_q':0.9, 'm_min':4.6, 'm_max':65., 'lambda_m':0.09, 'lambda_1_m':0.92, 'delta_m':4.8, 'mu_m_1':33.4, 'sigma_m_1':5.38, 'mu_m_2':68., 'sigma_m_2':6.76}
            print('Base values are: ', list(basevalues.items()))
            self.set_hyperparameters(basevalues)
        
        if priorlims_parameters is not None:
            self.set_priorlimits(priorlims_parameters)
        else:
            # Define the prior limits for 'm1_src' and 'q'
            self.set_priorlimits({'m1_src': (4.6, 65.), 'm2_src': (4.6, 65.)})
        
        self.derivative_par_nums = {'alpha_m':0, 'beta_q':1, 'm_min':2, 'm_max':3, 'lambda_m':4, 'lambda_1_m':5, 'delta_m':6, 'mu_m_1':7, 'sigma_m_1':8, 'mu_m_1':9, 'sigma_m_1':10}


    def _mass1_function(self, m1_src, alpha_m=None, m_min=None, m_max=None, lambda_m=None, lambda_1_m=None, delta_m=None, mu_m_1=None, sigma_m_1=None, mu_m_2=None, sigma_m_2=None):
        '''
        Mass distribution of the primary object.
        
        :param array m1_src: Primary mass.
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional mmin: Minimum mass of the power-law component of the mass distribution.
        :param float, optional mmax: Maximum mass of the power-law component of the mass distribution.
        :param float, optional lambda_m: Fraction of systems in the Gaussian components.
        :param float, optional lambda_1_m: Fraction of systems in the Gaussian components belonging to the lower-mass component.
        :param float, optional delta_m: Width of the smoothing component.
        :param float, optional mu_m_1: Mean of the lower-mass Gaussian component.
        :param float, optional sigma_m_1: Standard deviation of the lower-mass Gaussian component.
        :param float, optional mu_m_2: Mean of the higher-mass Gaussian component.
        :param float, optional sigma_m_2: Standard deviation of the higher-mass Gaussian component.
        
        :return: Mass function value at the input masses.
        :rtype: array
        '''

        alpha        = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        mmin         = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax         = self.hyperpar_dict['m_max'] if m_max is None else m_max
        lambda_m     = self.hyperpar_dict['lambda_m'] if lambda_m is None else lambda_m
        lambda_1_m   = self.hyperpar_dict['lambda_1_m'] if lambda_1_m is None else lambda_1_m
        delta_m      = self.hyperpar_dict['delta_m'] if delta_m is None else delta_m
        mu_m_1       = self.hyperpar_dict['mu_m_1'] if mu_m_1 is None else mu_m_1
        sigma_m_1    = self.hyperpar_dict['sigma_m_1'] if sigma_m_1 is None else sigma_m_1
        mu_m_2       = self.hyperpar_dict['mu_m_2'] if mu_m_2 is None else mu_m_2
        sigma_m_2    = self.hyperpar_dict['sigma_m_2'] if sigma_m_2 is None else sigma_m_2


        where_compute   = (m1_src>=mmin) & (m1_src<=mmax)
        PLcomp = jnp.where(where_compute, utils.inversepowerlaw(m1_src, alpha, (mmin,mmax)), 0.)
        Gcomp_1 = utils.gaussian_norm(m1_src, mu_m_1, sigma_m_1) 
        Gcomp_2 = utils.gaussian_norm(m1_src, mu_m_2, sigma_m_2)        
        PTcomp  = jnp.nan_to_num(utils.planck_taper(m1_src, mmin, delta_m))
        MP=((1.-lambda_m)*PLcomp + lambda_m*lambda_1_m*Gcomp_1 + lambda_m*(1.-lambda_1_m)*Gcomp_2)*PTcomp
        return MP

    
    def _mass2_function(self, m2_src, beta_q=None, m_min=None, delta_m=None):
        '''
        Mass distribution of the secondary object.
        
        :param array m2_src: Secondary mass.
        :param array m1_src: Primary mass.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional delta_m: Width of the smoothing component.
        
        :return: Mass function value at the input masses.
        :rtype: array
        '''

        beta = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin = self.hyperpar_dict['m_min'] if m_min is None else m_min
        deltam = self.hyperpar_dict['delta_m'] if delta_m is None else delta_m
        
        
        PLcomp = m2_src**(beta)
        PTcomp = jnp.nan_to_num(utils.planck_taper(m2_src, mmin, deltam))
        return PLcomp*PTcomp
    
    def mass_function(self, m1_src, m2_src, alpha_m=None, beta_q=None, m_min=None, m_max=None, lambda_m=None, lambda_1_m=None, delta_m=None, mu_m_1=None, sigma_m_1=None, mu_m_2=None, sigma_m_2=None, uselog=False):
        '''
        Mass distribution of the objects.
        
        :param array m1_src: Primary mass.
        :param array m2_src: Secondary mass.
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional lambda_m: Fraction of systems in the Gaussian components.
        :param float, optional lambda_1_m: Fraction of systems in the Gaussian components belonging to the lower-mass component.
        :param float, optional delta_m: Width of the smoothing component.
        :param float, optional mu_m_1: Mean of the lower-mass Gaussian component.
        :param float, optional sigma_m_1: Standard deviation of the lower-mass Gaussian component.
        :param float, optional mu_m_2: Mean of the higher-mass Gaussian component.
        :param float, optional sigma_m_2: Standard deviation of the higher-mass Gaussian component.
        :param bool, optional uselog: Boolean specifying whether to return the probability or log-probability, defaults to False.
        
        :return: Mass function value at the input masses.
        :rtype: array
        '''

        alpha        = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        beta         = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin         = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax         = self.hyperpar_dict['m_max'] if m_max is None else m_max
        lambda_m     = self.hyperpar_dict['lambda_m'] if lambda_m is None else lambda_m
        lambda_1_m   = self.hyperpar_dict['lambda_1_m'] if lambda_1_m is None else lambda_1_m
        delta_m      = self.hyperpar_dict['delta_m'] if delta_m is None else delta_m
        mu_m_1       = self.hyperpar_dict['mu_m_1'] if mu_m_1 is None else mu_m_1
        sigma_m_1    = self.hyperpar_dict['sigma_m_1'] if sigma_m_1 is None else sigma_m_1
        mu_m_2       = self.hyperpar_dict['mu_m_2'] if mu_m_2 is None else mu_m_2
        sigma_m_2    = self.hyperpar_dict['sigma_m_2'] if sigma_m_2 is None else sigma_m_2


        # First, p(m1) and p(m2), un-normalized
        pm1= self._mass1_function(m1_src, alpha, mmin, mmax, lambda_m, lambda_1_m, delta_m, mu_m_1, sigma_m_1, mu_m_2, sigma_m_2)
        pm2= self._mass2_function(m2_src, beta, mmin, delta_m)

        # This term normalizes the marginal distribution \int p(m1,m2) dm2 to be p(m1)
        C= self._Cnorm(m1_src, beta, mmin, delta_m)

        # And finally this normalizes p(m1) to unity
        NN= self._norm(alpha, mmin, mmax, lambda_m, lambda_1_m, delta_m, mu_m_1, sigma_m_1, mu_m_2, sigma_m_2)

        # compute normalized pdf
        where_compute = (m2_src < m1_src) & (mmin< m2_src) & (m1_src < mmax )
        pdf = pm1*pm2*C/NN

        if not uselog:
            return jnp.where(where_compute, pdf, 0.)
        else:
            return jnp.where(where_compute, jnp.log(pdf), -jnp.inf)
    
    def _Cnorm(self, m1_src, beta_q=None, m_min=None, delta_m=None):
        '''
        Inverse integral of  p(m1, m2) dm2 (i.e. C(m1) in the LVC notation).
        
        :param array m1_src: Primary mass (i.e. the upper bound of the integrals).
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional sigma_l: Width of the lower filter of the mass distributions.
        :param float, optional sigma_h: Width of the upper filter of the mass distributions.
        
        :return: Inverse integral of p(m1, m2) dm2.
        :rtype: array
        '''
        
        beta  = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin  = self.hyperpar_dict['m_min'] if m_min is None else m_min
        deltam = self.hyperpar_dict['delta_m'] if delta_m is None else delta_m

        
        # lower edge
        ms1 = jnp.linspace(mmin, mmin+deltam+deltam/10, 200)

        # in between 
        ms2 = jnp.linspace(mmin+deltam+deltam/10+1e-01, m1_src.max(), 200)

        xx= jnp.sort(jnp.unique(jnp.concatenate([ms1,ms2] )))

        
        pm2 = self._mass2_function(xx, beta, mmin, deltam)  

        cdf = utils.cumtrapz_JAX(pm2, xx) #cumtrapz(pm2, xx) #utils.cumtrapz_JAX(pm2, xx)

        C = jnp.interp( m1_src, xx[1:], cdf) 

        return jnp.nan_to_num(1./C,posinf=0.)
    
    def _norm(self,alpha_m=None, m_min=None, m_max=None, lambda_m=None, lambda_1_m=None, delta_m=None, mu_m_1=None, sigma_m_1=None, mu_m_2=None, sigma_m_2=None):
        '''
        Normalization of the primary mass function.
        
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional lambda_peak: Ratio of the power-law to Gaussian component.
        :param float, optional sigma_l: Width of the lower filter of the mass distributions.
        :param float, optional sigma_h: Width of the upper filter of the mass distributions.
        :param float, optional mu_m: Mean of the Gaussian component.
        :param float, optional sigma_m: Standard deviation of the Gaussian component.
        
        :return: Normalization of the primary mass function.
        :rtype: float
        '''

        alpha        = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        mmin         = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax         = self.hyperpar_dict['m_max'] if m_max is None else m_max
        lambda_m     = self.hyperpar_dict['lambda_m'] if lambda_m is None else lambda_m
        lambda_1_m   = self.hyperpar_dict['lambda_1_m'] if lambda_1_m is None else lambda_1_m
        deltam      = self.hyperpar_dict['delta_m'] if delta_m is None else delta_m
        mu1       = self.hyperpar_dict['mu_m_1'] if mu_m_1 is None else mu_m_1
        sigma1    = self.hyperpar_dict['sigma_m_1'] if sigma_m_1 is None else sigma_m_1
        mu2       = self.hyperpar_dict['mu_m_2'] if mu_m_2 is None else mu_m_2
        sigma2    = self.hyperpar_dict['sigma_m_2'] if sigma_m_2 is None else sigma_m_2

        # Now this is just the normalization of p(m1)! Because we have normalized the rest already
        max_compute = max(mmax, mu2+10*sigma2)
            
        # lower edge
        ms1 = np.linspace(1., mmin+deltam+deltam/10, 200)
            
        # before first gaussian peak
        ms2 = np.linspace( mmin+deltam+deltam/10+1e-01, mu1-5*sigma1, 200)
            
        # around first gaussian peak
        ms3 = np.linspace( mu1-5*sigma1+1e-01, mu1+5*sigma1, 200)
            
        # after first gaussian peak, before second gaussian peak
        ms4 = np.linspace( mu1+5*sigma1+1e-01, mu2-5*sigma2, 200 )
            
        # around second gaussian peak
        ms5 = np.linspace( mu2-5*sigma2+1e-01, mu2+5*sigma2, 200 )
            
        # after second gaussian peak
        ms6 = np.linspace(mu2+5*sigma2+1e-01, max_compute+max_compute/2, 200 )
            
        ms=np.sort(np.concatenate([ms1,ms2, ms3, ms4, ms5, ms6], ))
        
        pm1 = self._mass1_function(ms, alpha_m=alpha, m_min=mmin, m_max=mmax, lambda_m=lambda_m, lambda_1_m=lambda_1_m, delta_m=deltam, mu_m_1=mu1, sigma_m_1=sigma1, mu_m_2=mu2, sigma_m_2=sigma2)
        
        norm=jnp.trapz(pm1,ms)
        return norm

        
    



    def sample_population(self, size, m_min=None, m_max=None):
        '''
        Function to sample the mass model.
        
        :param int size: Size of the masses sample.

        :return: Sampled masses.
        :rtype: dict(array, array)
        '''

        mmin  = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax  = self.hyperpar_dict['m_max'] if m_max is None else m_max
        m1_src = utils.inverse_cdf_sampling(self._mass1_function, size, [mmin,mmax])
        m2_src = utils.inverse_cdf_sampling_uppercond(self._mass2_function, mmin, m1_src)
    
        return {'m1_src':m1_src, 'm2_src':m2_src}
    
    def mass_function_derivative(self, m1_src, m2_src, alpha_m=None, beta_q=None, m_min=None, m_max=None, lambda_m=None, lambda_1_m=None, delta_m=None, mu_m_1=None, sigma_m_1=None, mu_m_2=None, sigma_m_2=None, uselog=False):
        '''
        Derivative with respect to the hyperparameters of the mass function.
        
        :param array m1_src: Primary mass.
        :param array m2_src: Secondary mass.
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional lambda_m: Fraction of systems in the Gaussian components.
        :param float, optional lambda_1_m: Fraction of systems in the Gaussian components belonging to the lower-mass component.
        :param float, optional delta_m: Width of the smoothing component.
        :param float, optional mu_m_1: Mean of the lower-mass Gaussian component.
        :param float, optional sigma_m_1: Standard deviation of the lower-mass Gaussian component.
        :param float, optional mu_m_2: Mean of the higher-mass Gaussian component.
        :param float, optional sigma_m_2: Standard deviation of the higher-mass Gaussian component.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the derivatives of the mass function.
        :rtype: array
        '''

        alpha        = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        beta         = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin         = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax         = self.hyperpar_dict['m_max'] if m_max is None else m_max
        lambda_m     = self.hyperpar_dict['lambda_m'] if lambda_m is None else lambda_m
        lambda_1_m   = self.hyperpar_dict['lambda_1_m'] if lambda_1_m is None else lambda_1_m
        delta_m      = self.hyperpar_dict['delta_m'] if delta_m is None else delta_m
        mu_m_1       = self.hyperpar_dict['mu_m_1'] if mu_m_1 is None else mu_m_1
        sigma_m_1    = self.hyperpar_dict['sigma_m_1'] if sigma_m_1 is None else sigma_m_1
        mu_m_2       = self.hyperpar_dict['mu_m_2'] if mu_m_2 is None else mu_m_2
        sigma_m_2    = self.hyperpar_dict['sigma_m_2'] if sigma_m_2 is None else sigma_m_2
        
        funder = lambda m1_src, m2_src, alpha, beta, mmin, mmax, lambda_m, lambda_1_m, delta_m, mu_m_1, sigma_m_1, mu_m_2, sigma_m_2: self.mass_function(m1_src, m2_src, alpha, beta, mmin, mmax, lambda_m, lambda_1_m, delta_m, mu_m_1, sigma_m_1, mu_m_2, sigma_m_2, uselog=uselog)
        
        derivs_all = np.squeeze(np.asarray(jacrev(funder, argnums=(2,3,4,5,6,7,8,9,10,11,12))(m1_src, m2_src, jnp.array([alpha]), jnp.array([beta]), jnp.array([mmin]), jnp.array([mmax]), jnp.array([lambda_m]), jnp.array([lambda_1_m]), jnp.array([delta_m]), jnp.array([mu_m_1]), jnp.array([sigma_m_1]), jnp.array([mu_m_2]), jnp.array([sigma_m_2]))))
        
        return derivs_all
    
    def mass_function_hessian(self, m1_src, m2_src, alpha_m=None, beta_q=None, m_min=None, m_max=None, lambda_m=None, lambda_1_m=None, delta_m=None, mu_m_1=None, sigma_m_1=None, mu_m_2=None, sigma_m_2=None, uselog=False):
        '''
        Hessian with respect to the hyperparameters of the mass function.
        
        :param array m1_src: Primary mass.
        :param array m2_src: Secondary mass.
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional lambda_m: Fraction of systems in the Gaussian components.
        :param float, optional lambda_1_m: Fraction of systems in the Gaussian components belonging to the lower-mass component.
        :param float, optional delta_m: Width of the smoothing component.
        :param float, optional mu_m_1: Mean of the lower-mass Gaussian component.
        :param float, optional sigma_m_1: Standard deviation of the lower-mass Gaussian component.
        :param float, optional mu_m_2: Mean of the higher-mass Gaussian component.
        :param float, optional sigma_m_2: Standard deviation of the higher-mass Gaussian component.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the Hessians of the mass function.
        :rtype: array
        '''
        
        alpha        = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        beta         = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin         = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax         = self.hyperpar_dict['m_max'] if m_max is None else m_max
        lambda_m     = self.hyperpar_dict['lambda_m'] if lambda_m is None else lambda_m
        lambda_1_m   = self.hyperpar_dict['lambda_1_m'] if lambda_1_m is None else lambda_1_m
        delta_m      = self.hyperpar_dict['delta_m'] if delta_m is None else delta_m
        mu_m_1       = self.hyperpar_dict['mu_m_1'] if mu_m_1 is None else mu_m_1
        sigma_m_1    = self.hyperpar_dict['sigma_m_1'] if sigma_m_1 is None else sigma_m_1
        mu_m_2       = self.hyperpar_dict['mu_m_2'] if mu_m_2 is None else mu_m_2
        sigma_m_2    = self.hyperpar_dict['sigma_m_2'] if sigma_m_2 is None else sigma_m_2
        
        funder = lambda m1_src, m2_src, alpha, beta, mmin, mmax, lambda_m, lambda_1_m, delta_m, mu_m_1, sigma_m_1, mu_m_2, sigma_m_2: self.mass_function(m1_src, m2_src, alpha, beta, mmin, mmax, lambda_m, lambda_1_m, delta_m, mu_m_1, sigma_m_1, mu_m_2, sigma_m_2, uselog=uselog)
        
        derivs_all = np.squeeze(np.asarray(hessian(funder, argnums=(2,3,4,5,6,7,8,9,10,11,12))(m1_src, m2_src, jnp.array([alpha]), jnp.array([beta]), jnp.array([mmin]), jnp.array([mmax]), jnp.array([lambda_m]), jnp.array([lambda_1_m]), jnp.array([delta_m]), jnp.array([mu_m_1]), jnp.array([sigma_m_1]), jnp.array([mu_m_2]), jnp.array([sigma_m_2]))))
        
        return derivs_all


###################################################################################################
# Functions with a reformulated version of the smoothing function


class TruncatedPowerLaw_modsmooth_MassDistribution(MassDistribution):
    '''
    Truncated mass model

    Parameters:
        * alpha_m: Spectral index for the power-law of the primary mass distribution.
        * beta_q: Spectral index for the power-law of the mass ratio distribution.
        * m_min: Minimum mass of the power-law component of the mass distribution.
        * m_max: Maximum mass of the power-law component of the mass distribution.
        * sigma_l: Width of the lower filter of the mass distributions.
        * sigma_h: Width of the upper filter of the mass distributions.
    
    :param list parameters: List containing the parameters of the mass model.
    :param dict, optional hyperparameters: Dictionary containing the hyperparameters of the mass model as keys and their fiducial value as entry.
    :param dict, optional priorlims_parameters: Dictionary containing the parameters of the mass model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.

    '''

    def __init__(self, hyperparameters=None, priorlims_parameters=None):

        self.expected_hyperpars = ['alpha_m', 'beta_q', 'm_min', 'm_max', 'sigma_l', 'sigma_h']
        super().__init__()

        self.set_parameters(['m1_src', 'm2_src'])

        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)
        else:
            print('Expected hyperparameters: ', self.expected_hyperpars)
            basevalues = {'alpha_m':0.75, 'beta_q':0.1, 'm_min':5.5, 'm_max':45., 'sigma_l':.5, 'sigma_h':0.5}
            print('Base values are: ', list(basevalues.items()))
            self.set_hyperparameters(basevalues)
        
        if priorlims_parameters is not None:
            self.set_priorlimits(priorlims_parameters)
        else:
            # Define the prior limits for 'm1_src' and 'q'
            self.set_priorlimits({'m1_src': (5., 45.), 'm2_src': (5., 45.)})
        
        self.derivative_par_nums = {'alpha_m':0, 'beta_q':1, 'm_min':2, 'm_max':3, 'sigma_l':4, 'sigma_h':5}

    def _mass1_function(self, m1_src, alpha_m=None, m_min=None, m_max=None, sigma_l=None, sigma_h=None):
        '''
        Power-law mass function of the primary mass.
        
        :param array m1_src: Primary mass.
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional sigma_l: Width of the lower filter of the mass distributions.
        :param float, optional sigma_h: Width of the upper filter of the mass distributions.
        
        :return: Mass function value at the input masses.
        :rtype: array
        '''

        # In general, p(m1) need not to be normalized in the implementation
        # Note that the sampling algorithm does not rely on normalization

        alpha = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        mmin  = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax  = self.hyperpar_dict['m_max'] if m_max is None else m_max
        sig_l = self.hyperpar_dict['sigma_l'] if sigma_l is None else sigma_l
        sig_h = self.hyperpar_dict['sigma_h'] if sigma_h is None else sigma_h


        return (m1_src**(-alpha))*utils.polynomial_filter_hl(m1_src, mmin, sig_l, mmax, sig_h)#utils.normCDF_filter_hl(m1_src, mmin, sig_l, mmax, sig_h)


    def _mass2_function(self, m2_src, beta_q=None, m_min=None, m_max=None, sigma_l=None, sigma_h=None):
        '''
        Secondary mass function.
        
        :param array m2_src: Secondary mass.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional sigma_l: Width of the lower filter of the mass distributions.
        :param float, optional sigma_h: Width of the upper filter of the mass distributions.
        
        :return: Mass function value at the input masses.
        :rtype: array
        '''

        # I define  p(m2) directly, not p(m2/m1) . All dependence on m1 is inside p(m1, m2) 
        # Again, normalization is not needed here, only in p(m1, m2)

        beta  = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax  = self.hyperpar_dict['m_max'] if m_max is None else m_max
        sig_l = self.hyperpar_dict['sigma_l'] if sigma_l is None else sigma_l
        sig_h = self.hyperpar_dict['sigma_h'] if sigma_h is None else sigma_h


        return m2_src**(beta)*utils.polynomial_filter_hl(m2_src, mmin, sig_l, mmax, sig_h)#utils.normCDF_filter_hl(m2_src, mmin, sig_l, mmax, sig_h)

    def _Cnorm(self, m1_src, beta_q=None, m_min=None, m_max=None, sigma_l=None, sigma_h=None):
        '''
        Inverse integral of p(m1, m2) dm2 (i.e. C(m1) in the LVC notation).
        
        :param array m1_src: Primary mass (i.e. the upper bound of the integrals).
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional sigma_l: Width of the lower filter of the mass distributions.
        :param float, optional sigma_h: Width of the upper filter of the mass distributions.
        
        :return: Inverse integral of p(m1, m2) dm2.
        :rtype: array
        '''

        beta  = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin  = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax  = self.hyperpar_dict['m_max'] if m_max is None else m_max
        sig_l = self.hyperpar_dict['sigma_l'] if sigma_l is None else sigma_l
        sig_h = self.hyperpar_dict['sigma_h'] if sigma_h is None else sigma_h

        C = utils.polynomial_filter_hl_invpowerlaw_integral_uptox(m1_src, mmin, sig_l, mmax, sig_h, -beta)

        return jnp.nan_to_num(1./C,posinf=0.)

    def _norm(self, alpha_m=None, m_min=None, m_max=None, sigma_l=None, sigma_h=None):
        '''
        Normalization of the primary mass function.
        
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional sigma_l: Width of the lower filter of the mass distributions.
        :param float, optional sigma_h: Width of the upper filter of the mass distributions.
        
        :return: Normalization of the primary mass function.
        :rtype: float
        '''

        alpha = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        mmin  = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax  = self.hyperpar_dict['m_max'] if m_max is None else m_max
        sig_l = self.hyperpar_dict['sigma_l'] if sigma_l is None else sigma_l
        sig_h = self.hyperpar_dict['sigma_h'] if sigma_h is None else sigma_h

        # Now this is just the normalization of p(m1)! Because we have normalized the rest already

        return utils.polynomial_filter_hl_invpowerlaw_integral(mmin, sig_l, mmax, sig_h, alpha)
        
    def mass_function(self, m1_src, m2_src, alpha_m=None, beta_q=None, m_min=None, m_max=None, sigma_l=None, sigma_h=None, uselog=False):
        '''
        Mass distribution of the objects.
        
        :param array m1_src: Primary mass.
        :param array m2_src: Secondary mass.
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional sigma_l: Width of the lower filter of the mass distributions.
        :param float, optional sigma_h: Width of the upper filter of the mass distributions.
        :param bool, optional uselog: Boolean specifying whether to return the probability or log-probability, defaults to False.
        
        :return: Mass function value at the input masses.
        :rtype: array
        '''

        # Now, this has to contain the good normalization factors

        alpha = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        beta  = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin  = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax  = self.hyperpar_dict['m_max'] if m_max is None else m_max
        sig_l = self.hyperpar_dict['sigma_l'] if sigma_l is None else sigma_l
        sig_h = self.hyperpar_dict['sigma_h'] if sigma_h is None else sigma_h

        # First, p(m1) and p(m2), un-normalized
        pm1 = self._mass1_function(m1_src, alpha_m=alpha, m_min=mmin, m_max=mmax, sigma_l=sig_l, sigma_h=sig_h)
        pm2 = self._mass2_function(m2_src, beta_q=beta, m_min=mmin, m_max=mmax, sigma_l=sig_l, sigma_h=sig_h)

        # This term normalizes the marginal distribution \int p(m1,m2) dm2 to be p(m1) 

        C = self._Cnorm(m1_src, beta_q=beta, m_min=mmin, m_max=mmax, sigma_l=sig_l, sigma_h=sig_h)

        # And finally this normalizes p(m1) to unity 

        NN = self._norm(alpha_m=alpha, m_min=mmin, m_max=mmax, sigma_l=sig_l, sigma_h=sig_h)

        pdf = pm1*pm2*C/NN
        
        if not uselog:
            return pdf
        else:
            return jnp.log(pdf)
        
    def sample_population(self, size, m_min=None, m_max=None, sigma_l=None, sigma_h=None,):
        '''
        Function to sample the mass model.
        
        :param int size: Size of the masses sample.
        :param float, optional m_min: Minimum mass of the power-law component of the primary mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the primary mass distribution.
        :param float, optional sigma_l: Width of the lower filter of the mass distributions.
        :param float, optional sigma_h: Width of the upper filter of the mass distributions.

        :return: Sampled masses.
        :rtype: dict(array, array)
        '''

        mmin  = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax  = self.hyperpar_dict['m_max'] if m_max is None else m_max
        sig_l = self.hyperpar_dict['sigma_l'] if sigma_l is None else sigma_l
        sig_h = self.hyperpar_dict['sigma_h'] if sigma_h is None else sigma_h

        m1_src = utils.inverse_cdf_sampling(self._mass1_function, size, [mmin-sig_l, mmax+sig_h])
        m2_src = utils.inverse_cdf_sampling_uppercond(self._mass2_function, mmin-sig_l, m1_src)

        return {'m1_src':m1_src, 'm2_src':m2_src}
    
    def mass_function_derivative(self, m1_src, m2_src, alpha_m=None, beta_q=None, m_min=None, m_max=None, sigma_l=None, sigma_h=None, uselog=False):
        '''
        Derivative with respect to the hyperparameters of the mass function.
        
        :param array m1_src: Primary mass.
        :param array m2_src: Secondary mass.
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional sigma_l: Width of the lower filter of the mass distributions.
        :param float, optional sigma_h: Width of the upper filter of the mass distributions.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the derivatives of the mass function.
        :rtype: array
        '''

        alpha = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        beta  = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin  = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax  = self.hyperpar_dict['m_max'] if m_max is None else m_max
        sig_l = self.hyperpar_dict['sigma_l'] if sigma_l is None else sigma_l
        sig_h = self.hyperpar_dict['sigma_h'] if sigma_h is None else sigma_h

        funder = lambda m1_src, m2_src, alpha, beta, mmin, mmax, sig_l, sig_h: self.mass_function(m1_src, m2_src, alpha, beta, mmin, mmax, sig_l, sig_h, uselog=uselog)
        
        derivs_all = np.squeeze(np.asarray(jacrev(funder, argnums=(2,3,4,5,6,7))(m1_src, m2_src, jnp.array([alpha]), jnp.array([beta]), jnp.array([mmin]), jnp.array([mmax]), jnp.array([sig_l]), jnp.array([sig_h]))))
        
        return derivs_all

    def mass_function_hessian(self, m1_src, m2_src, alpha_m=None, beta_q=None, m_min=None, m_max=None, sigma_l=None, sigma_h=None, uselog=False):
        '''
        Hessian with respect to the hyperparameters of the mass function.
        
        :param array m1_src: Primary mass.
        :param array m2_src: Secondary mass.
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional sigma_l: Width of the lower filter of the mass distributions.
        :param float, optional sigma_h: Width of the upper filter of the mass distributions.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the Hessians of the mass function.
        :rtype: array
        '''

        alpha = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        beta  = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin  = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax  = self.hyperpar_dict['m_max'] if m_max is None else m_max
        sig_l = self.hyperpar_dict['sigma_l'] if sigma_l is None else sigma_l
        sig_h = self.hyperpar_dict['sigma_h'] if sigma_h is None else sigma_h
        
        funder = lambda m1_src, m2_src, alpha, beta, mmin, mmax, sig_l, sig_h: self.mass_function(m1_src, m2_src, alpha, beta, mmin, mmax, sig_l, sig_h, uselog=uselog)

        derivs_all = np.squeeze(np.asarray(hessian(funder, argnums=(2,3,4,5,6,7))(m1_src, m2_src, jnp.array([alpha]), jnp.array([beta]), jnp.array([mmin]), jnp.array([mmax]), jnp.array([sig_l]), jnp.array([sig_h]))))
        
        return derivs_all

class PowerLawPlusPeak_modsmooth_MassDistribution(MassDistribution):
    '''
    Power-Law+Peak model.

    Parameters:
        * alpha_m: Spectral index for the power-law of the primary mass distribution
        * beta_q: Spectral index for the power-law of the mass ratio distribution
        * m_min: Minimum mass of the power-law component of the mass distribution.
        * m_max: Maximum mass of the power-law component of the mass distribution.
        * lambda_peak: Ratio of the power-law to Gaussian component.
        * sigma_l: Width of the lower filter of the mass distributions.
        * sigma_h: Width of the upper filter of the mass distributions.
        * mu_m: Mean of the Gaussian component.
        * sigma_m: Standard deviation of the Gaussian component.
    
    :param list parameters: List containing the parameters of the mass model.
    :param dict, optional hyperparameters: Dictionary containing the hyperparameters of the mass model as keys and their fiducial value as entry.
    :param dict, optional priorlims_parameters: Dictionary containing the parameters of the mass model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.

    '''

    def __init__(self, hyperparameters=None, priorlims_parameters=None):

        self.expected_hyperpars = ['alpha_m', 'beta_q', 'm_min', 'm_max', 'lambda_peak', 'sigma_l', 'sigma_h', 'mu_m', 'sigma_m']
        super().__init__()

        self.set_parameters(['m1_src', 'm2_src'])

        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)
        else:
            print('Expected hyperparameters: ', self.expected_hyperpars)
            basevalues = {'alpha_m':3.4, 'beta_q':1.1, 'm_min':9.1, 'm_max':87., 'lambda_peak':0.039, 'sigma_l':4., 'sigma_h':0.5, 'mu_m':34., 'sigma_m':3.6}
            print('Base values are: ', list(basevalues.items()))
            self.set_hyperparameters(basevalues)
        
        if priorlims_parameters is not None:
            self.set_priorlimits(priorlims_parameters)
        else:
            # Define the prior limits for 'm1_src' and 'q'
            self.set_priorlimits({'m1_src': (0., 100.), 'm2_src': (0., 100.)})
        
        self.derivative_par_nums = {'alpha_m':0, 'beta_q':1, 'm_min':2, 'm_max':3, 'lambda_peak':4, 'sigma_l':5, 'sigma_h':6, 'mu_m':7, 'sigma_m':8}
    
    def _mass1_function(self, m1_src, alpha_m=None, m_min=None, m_max=None, lambda_peak=None, sigma_l=None, sigma_h=None, mu_m=None, sigma_m=None):
        '''
        Mass function of the primary object.
        
        :param array m1_src: Primary mass.
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional lambda_peak: Ratio of the power-law to Gaussian component.
        :param float, optional sigma_l: Width of the lower filter of the mass distributions.
        :param float, optional sigma_h: Width of the upper filter of the mass distributions.
        :param float, optional mu_m: Mean of the Gaussian component.
        :param float, optional sigma_m: Standard deviation of the Gaussian component.
        
        :return: Primary mass function value at the input masses.
        :rtype: array
        '''

        # In general, p(m1) need not to be normalized in the implementation
        # Note that the sampling algorithm does not rely on normalization

        alpha        = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        mmin         = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax         = self.hyperpar_dict['m_max'] if m_max is None else m_max
        lambda_peak  = self.hyperpar_dict['lambda_peak'] if lambda_peak is None else lambda_peak
        sig_l        = self.hyperpar_dict['sigma_l'] if sigma_l is None else sigma_l
        sig_h        = self.hyperpar_dict['sigma_h'] if sigma_h is None else sigma_h
        mu_m         = self.hyperpar_dict['mu_m'] if mu_m is None else mu_m
        sigma_m      = self.hyperpar_dict['sigma_m'] if sigma_m is None else sigma_m

        return  ((1.-lambda_peak)*utils.inversepowerlaw(m1_src, alpha, (mmin-sig_l, mmax+sig_h)) + lambda_peak*utils.gaussian_norm(m1_src, mu_m, sigma_m))*utils.polynomial_filter_hl(m1_src, mmin, sig_l, mmax, sig_h)#utils.normCDF_filter_hl(m1_src, mmin, sig_l, mmax, sig_h)
        
    def _mass2_function(self, m2_src, beta_q=None, m_min=None, m_max=None, sigma_l=None, sigma_h=None):
        '''
        Secondary mass function.
        
        :param array m2_src: Secondary mass.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional sigma_l: Width of the lower filter of the mass distributions.
        :param float, optional sigma_h: Width of the upper filter of the mass distributions.
        
        :return: Secondary mass function value at the input masses.
        :rtype: array
        '''

        # I define  p(m2) directly, not p(m2/m1) . All dependence on m1 is inside p(m1, m2) 
        # Again, normalization is not needed here, only in p(m1, m2)

        beta  = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin  = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax  = self.hyperpar_dict['m_max'] if m_max is None else m_max
        sig_l = self.hyperpar_dict['sigma_l'] if sigma_l is None else sigma_l
        sig_h = self.hyperpar_dict['sigma_h'] if sigma_h is None else sigma_h

        return m2_src**(beta)*utils.polynomial_filter_hl(m2_src, mmin, sig_l, mmax, sig_h)#utils.normCDF_filter_hl(m1_src, mmin, sig_l, mmax, sig_h)
    
    def _Cnorm(self, m1_src, beta_q=None, m_min=None, m_max=None, sigma_l=None, sigma_h=None):
        '''
        Inverse integral of  p(m1, m2) dm2 (i.e. C(m1) in the LVC notation).
        
        :param array m1_src: Primary mass (i.e. the upper bound of the integrals).
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional sigma_l: Width of the lower filter of the mass distributions.
        :param float, optional sigma_h: Width of the upper filter of the mass distributions.
        
        :return: Inverse integral of p(m1, m2) dm2.
        :rtype: array
        '''
        
        beta  = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin  = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax  = self.hyperpar_dict['m_max'] if m_max is None else m_max
        sig_l = self.hyperpar_dict['sigma_l'] if sigma_l is None else sigma_l
        sig_h = self.hyperpar_dict['sigma_h'] if sigma_h is None else sigma_h

        C = utils.polynomial_filter_hl_invpowerlaw_integral_uptox(m1_src, mmin, sig_l, mmax, sig_h, -beta)
        return jnp.nan_to_num(1./C,posinf=0.)


    def _norm(self, alpha_m=None, m_min=None, m_max=None, lambda_peak=None, sigma_l=None, sigma_h=None, mu_m=None, sigma_m=None):
        '''
        Normalization of the primary mass function.
        
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional lambda_peak: Ratio of the power-law to Gaussian component.
        :param float, optional sigma_l: Width of the lower filter of the mass distributions.
        :param float, optional sigma_h: Width of the upper filter of the mass distributions.
        :param float, optional mu_m: Mean of the Gaussian component.
        :param float, optional sigma_m: Standard deviation of the Gaussian component.
        
        :return: Normalization of the primary mass function.
        :rtype: float
        '''

        alpha        = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        mmin         = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax         = self.hyperpar_dict['m_max'] if m_max is None else m_max
        lambda_peak  = self.hyperpar_dict['lambda_peak'] if lambda_peak is None else lambda_peak
        sig_l        = self.hyperpar_dict['sigma_l'] if sigma_l is None else sigma_l
        sig_h        = self.hyperpar_dict['sigma_h'] if sigma_h is None else sigma_h
        mu_m         = self.hyperpar_dict['mu_m'] if mu_m is None else mu_m
        sigma_m      = self.hyperpar_dict['sigma_m'] if sigma_m is None else sigma_m

        # Now this is just the normalization of p(m1)! Because we have normalized the rest already

        return (1.-lambda_peak)*utils.polynomial_filter_hl_invpowerlawnorm_integral(mmin, sig_l, mmax, sig_h, alpha) + lambda_peak*utils.polynomial_filter_hl_gaussian_integral(mmin, sig_l, mmax, sig_h, mu_m, sigma_m)

    def mass_function(self, m1_src, m2_src, alpha_m=None, beta_q=None, m_min=None, m_max=None, lambda_peak=None, sigma_l=None, sigma_h=None, mu_m=None, sigma_m=None, uselog=False):
        '''
        Mass distribution of the objects.
        
        :param array m1_src: Primary mass.
        :param array m2_src: Secondary mass.
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional lambda_peak: Ratio of the power-law to Gaussian component.
        :param float, optional sigma_l: Width of the lower filter of the mass distributions.
        :param float, optional sigma_h: Width of the upper filter of the mass distributions.
        :param float, optional mu_m: Mean of the Gaussian component.
        :param float, optional sigma_m: Standard deviation of the Gaussian component.
        :param bool, optional uselog: Boolean specifying whether to return the probability or log-probability, defaults to False.
        
        :return: Mass function value at the input masses.
        :rtype: array
        '''
        # Now, this has to contain the good normalization factors

        alpha        = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        beta         = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin         = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax         = self.hyperpar_dict['m_max'] if m_max is None else m_max
        lambda_peak  = self.hyperpar_dict['lambda_peak'] if lambda_peak is None else lambda_peak
        sig_l        = self.hyperpar_dict['sigma_l'] if sigma_l is None else sigma_l
        sig_h        = self.hyperpar_dict['sigma_h'] if sigma_h is None else sigma_h
        mu_m         = self.hyperpar_dict['mu_m'] if mu_m is None else mu_m
        sigma_m      = self.hyperpar_dict['sigma_m'] if sigma_m is None else sigma_m

        # First, p(m1) and p(m2), un-normalized
        pm1 = self._mass1_function(m1_src, alpha_m=alpha, m_min=mmin, m_max=mmax, lambda_peak=lambda_peak, sigma_l=sig_l, sigma_h=sig_h, mu_m=mu_m, sigma_m=sigma_m)
        pm2 = self._mass2_function(m2_src, beta_q=beta, m_min=mmin, m_max=mmax, sigma_l=sig_l, sigma_h=sig_h)
        
        # This term normalizes the marginal distribution \int p(m1,m2) dm2 to be p(m1) 
        C = self._Cnorm(m1_src, beta_q=beta, m_min=mmin, m_max=mmax, sigma_l=sig_l, sigma_h=sig_h)

        # And finally this normalizes p(m1) to unity 
        NN = self._norm(alpha_m=alpha, m_min=mmin, m_max=mmax, lambda_peak=lambda_peak, sigma_l=sig_l, sigma_h=sig_h, mu_m=mu_m, sigma_m=sigma_m)
    
        pdf = pm1*pm2*C/NN
        
        if not uselog:
            return pdf
        else:
            return jnp.log(pdf)
    
    def sample_population(self, size, m_min=None, m_max=None, sigma_l=None, sigma_h=None):
        '''
        Function to sample the mass model.
        
        :param int size: Size of the masses sample.
        :param float, optional m_min: Minimum mass of the power-law component of the primary mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the primary mass distribution.
        :param float, optional sigma_l: Width of the lower filter of the mass distributions.
        :param float, optional sigma_h: Width of the upper filter of the mass distributions.

        :return: Sampled masses.
        :rtype: dict(array, array)
        '''

        mmin  = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax  = self.hyperpar_dict['m_max'] if m_max is None else m_max
        sig_l = self.hyperpar_dict['sigma_l'] if sigma_l is None else sigma_l
        sig_h = self.hyperpar_dict['sigma_h'] if sigma_h is None else sigma_h

        m1_src = utils.inverse_cdf_sampling(self._mass1_function, size, [mmin-sig_l, mmax+sig_h])
        m2_src = utils.inverse_cdf_sampling_uppercond(self._mass2_function, mmin-sig_l, m1_src)

        return {'m1_src':m1_src, 'm2_src':m2_src}
    
    def mass_function_derivative(self, m1_src, m2_src, alpha_m=None, beta_q=None, m_min=None, m_max=None, lambda_peak=None, sigma_l=None, sigma_h=None, mu_m=None, sigma_m=None, uselog=False):
        '''
        Derivative with respect to the hyperparameters of the mass function.
        
        :param array m1_src: Primary mass.
        :param array m2_src: Secondary mass.
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional lambda_peak: Ratio of the power-law to Gaussian component.
        :param float, optional sigma_l: Width of the lower filter of the mass distributions.
        :param float, optional sigma_h: Width of the upper filter of the mass distributions.
        :param float, optional mu_m: Mean of the Gaussian component.
        :param float, optional sigma_m: Standard deviation of the Gaussian component.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the derivatives of the mass function.
        :rtype: array
        '''
        
        alpha        = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        beta         = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin         = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax         = self.hyperpar_dict['m_max'] if m_max is None else m_max
        lambda_peak  = self.hyperpar_dict['lambda_peak'] if lambda_peak is None else lambda_peak
        sig_l        = self.hyperpar_dict['sigma_l'] if sigma_l is None else sigma_l
        sig_h        = self.hyperpar_dict['sigma_h'] if sigma_h is None else sigma_h
        mu_m         = self.hyperpar_dict['mu_m'] if mu_m is None else mu_m
        sigma_m      = self.hyperpar_dict['sigma_m'] if sigma_m is None else sigma_m

        funder = lambda m1_src, m2_src, alpha, beta, mmin, mmax, lambda_peak, sigma_l, sigma_h, mu_m, sigma_m: self.mass_function(m1_src, m2_src, alpha, beta, mmin, mmax, lambda_peak, sigma_l, sigma_h, mu_m, sigma_m, uselog=uselog)
        
        derivs_all = np.squeeze(np.asarray(jacrev(funder, argnums=(2,3,4,5,6,7,8,9,10))(m1_src, m2_src, jnp.array([alpha]), jnp.array([beta]), jnp.array([mmin]), jnp.array([mmax]), jnp.array([lambda_peak]), jnp.array([sig_l]), jnp.array([sig_h]), jnp.array([mu_m]), jnp.array([sigma_m]))))
        
        return derivs_all
    
    def mass_function_hessian(self, m1_src, m2_src, alpha_m=None, beta_q=None, m_min=None, m_max=None, lambda_peak=None, sigma_l=None, sigma_h=None, mu_m=None, sigma_m=None, uselog=False):
        '''
        Hessian with respect to the hyperparameters of the mass function.
        
        :param array m1_src: Primary mass.
        :param array m2_src: Secondary mass.
        :param float, optional alpha_m: Spectral index for the power-law of the primary mass distribution.
        :param float, optional beta_q: Spectral index for the power-law of the mass ratio distribution.
        :param float, optional m_min: Minimum mass of the power-law component of the mass distribution.
        :param float, optional m_max: Maximum mass of the power-law component of the mass distribution.
        :param float, optional lambda_peak: Ratio of the power-law to Gaussian component.
        :param float, optional sigma_l: Width of the lower filter of the mass distributions.
        :param float, optional sigma_h: Width of the upper filter of the mass distributions.
        :param float, optional mu_m: Mean of the Gaussian component.
        :param float, optional sigma_m: Standard deviation of the Gaussian component.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the Hessians of the mass function.
        :rtype: array
        '''
        
        alpha        = self.hyperpar_dict['alpha_m'] if alpha_m is None else alpha_m
        beta         = self.hyperpar_dict['beta_q'] if beta_q is None else beta_q
        mmin         = self.hyperpar_dict['m_min'] if m_min is None else m_min
        mmax         = self.hyperpar_dict['m_max'] if m_max is None else m_max
        lambda_peak  = self.hyperpar_dict['lambda_peak'] if lambda_peak is None else lambda_peak
        sig_l        = self.hyperpar_dict['sigma_l'] if sigma_l is None else sigma_l
        sig_h        = self.hyperpar_dict['sigma_h'] if sigma_h is None else sigma_h
        mu_m         = self.hyperpar_dict['mu_m'] if mu_m is None else mu_m
        sigma_m      = self.hyperpar_dict['sigma_m'] if sigma_m is None else sigma_m
        
        funder = lambda m1_src, m2_src, alpha, beta, mmin, mmax, lambda_peak, sigma_l, sigma_h, mu_m, sigma_m: self.mass_function(m1_src, m2_src, alpha, beta, mmin, mmax, lambda_peak, sigma_l, sigma_h, mu_m, sigma_m, uselog=uselog)
        
        derivs_all = np.squeeze(np.asarray(hessian(funder, argnums=(2,3,4,5,6,7,8,9,10))(m1_src, m2_src, jnp.array([alpha]), jnp.array([beta]), jnp.array([mmin]), jnp.array([mmax]), jnp.array([lambda_peak]), jnp.array([sig_l]), jnp.array([sig_h]), jnp.array([mu_m]), jnp.array([sigma_m]))))
        
        return derivs_all