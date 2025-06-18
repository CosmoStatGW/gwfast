#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,sys,jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax import jacrev, vmap, jit, hessian
from astropy.cosmology import Planck18 as cosmo

from abc import ABC, abstractmethod

import POPutils as utils
import Globals as glob
#import POPutils as utils

import h5py
from scipy.optimize import root

# This is to avoid calling astropy during minimization, improves speed
zGridGLOB = np.logspace(start=-10, stop=3, base=10, num=1000)
dLGridGLOB = cosmo.luminosity_distance(zGridGLOB).value/1e3 # in Gpc
dVcomGridGLOB = 4.*np.pi*cosmo.differential_comoving_volume(zGridGLOB).value/1e9 # in Gpc^3
lbtGridGLOB = cosmo.lookback_time(zGridGLOB).value*1000. # Lookback time, Myr

class RateDistribution(ABC):
    '''
    Abstract class to compute redshift distributions.

    :param dict, optional hyperparameters: Dictionary containing the hyperparameters of the redshift model as keys and their fiducial value as entry.
    :param dict, optional priorlims_parameters: Dictionary containing the parameters of the redshift model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.
    
    '''

    def __init__(self,):
        
        self.par_list = ['z']
        self.hyperpar_dict = {}
        self.priorlims_dict = {}
    
    def set_hyperparameters(self, hyperparameters):
        '''
        Setter method for the hyperparameters of the redshift model.

        :param dict new_hyperparameters: Dictionary containing the hyperparameters of the redshift model as keys and their new value as entry.
        '''
        self.hyperpar_dict = hyperparameters

    def update_hyperparameters(self, new_hyperparameters):
        '''
        Method to update the hyperparameters of the redshift model.

        :param dict new_hyperparameters: Dictionary containing the new hyperparameters of the redshift model as keys and their new value as entry.
        '''
        for key in new_hyperparameters.keys():
            if key in self.hyperpar_dict.keys():
                self.hyperpar_dict[key] = new_hyperparameters[key]
            else:
                raise ValueError('The hyperparameter '+key+' is not present in the hyperparameter dictionary.')
    def set_priorlimits(self, limits):
        '''
        Setter method for the prior limits on the parameters of the redshift model.

        :param dict limits: Dictionary containing the parameters of the redshift model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.
        '''
        # This is to avoid numerical issues with the rate function when interpolating the differential comoving volume
        eps = 1e-6
        if limits['z'][0] < eps:
            limits['z'] = (eps, limits['z'][1])
        self.priorlims_dict = limits
        
    def update_priorlimits(self, new_limits):
        '''
        Method to update the prior limits on the parameters of the redshift model.
        
        :param dict new_limits: Dictionary containing the new prior limits on the parameters of the redshift model as keys and their new value as entry, given as a tuple :math:`(l, h)`.
        '''
        eps = 1e-6
        for key in new_limits.keys():
            if key in self.priorlims_dict.keys():
                if (key == 'z') & (new_limits['z'][0] < eps):
                    new_limits['z'] = (eps, new_limits['z'][1])
                self.priorlims_dict[key] = new_limits[key]
            else:
                raise ValueError('The parameter '+key+' is not present in the prior limits dictionary.')

    def _isin_prior_range(self, par,  val):
        """
        Function to check if a value is in the prior range of a parameter.

        :param str par: Parameter name.
        :param float val: Parameter value.
        :return: Boolean value.
        :rtype: bool
        """

        return (val >= self.priorlims_dict[par][0]) & (val <= self.priorlims_dict[par][1])
    @abstractmethod
    def rate_function(self, z):
        pass

    @abstractmethod
    def sample_population(self, size):
        pass
    
    @abstractmethod
    def rate_function_derivative(self,):
        pass
    
    @abstractmethod
    def N_per_yr(self,):
        pass
    
class PowerLaw_RateDistribution(RateDistribution):
    '''
    Power-law redshift distribution of the form :math:`(1+z)^{k_z-1}`.
    
    Parameters:
        * k_z: Power-law index of the redshift evolution of the merger rate.
    
    :param dict, optional hyperparameters: Dictionary containing the hyperparameters of the mass model as keys and their fiducial value as entry.
    :param dict, optional priorlims_parameters: Dictionary containing the parameters of the mass model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.
    
    '''

    def __init__(self, hyperparameters=None, priorlims_parameters=None):
        
        self.expected_hyperpars = ['k_z']
        super().__init__()

        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)
        else:
            print('Expected hyperparameters: ', self.expected_hyperpars)
            basevalues = {'k_z':2.7}
            print('Base values are: ', list(basevalues.items()))
            self.set_hyperparameters(basevalues)
        
        if priorlims_parameters is not None:
            self.set_priorlimits(priorlims_parameters)
        else:
            self.set_priorlimits({'z': (0., 2.)})
            
        self.derivative_par_nums = {'k_z':0}
    
    def N_per_yr(self, k_z=None, R0=1.):
        """
        Compute the number of mergers per year.

        :param float, optional k_z: Power law index of the redshift evolution of the merger rate.
        :param float, optional R0: Normalization of the local merger rate, in :math:`{\\rm Gpc}^{-3}\,{\\rm yr}^{-1}`.

        :return: Number of mergers per year.
        :rtype: float
        """
        k = self.hyperpar_dict['k_z'] if k_z is None else k_z

        zgrid = jnp.geomspace(self.priorlims_dict['z'][0], self.priorlims_dict['z'][1], 1000)

        res = np.trapezoid(R0*((1.+zgrid)**(k-1))*jnp.interp(zgrid, zGridGLOB, dVcomGridGLOB), zgrid)

        return np.ceil(res).astype(int)

    def rate_function(self, z, k_z=None, uselog=False):
        '''
        Redshift distribution function.
        
        :param array z: Redshift.
        :param float, optional k_z: Power-law index of the redshift evolution of the merger rate.
        :param bool, optional uselog: Boolean specifying whether to return the probability or log-probability, defaults to False.
        
        :return: Redshift distribution value at the input redshifts.
        :rtype: array
        '''
        
        k = self.hyperpar_dict['k_z'] if k_z is None else k_z
        
        goodsamples = self._isin_prior_range('z', z)
        zgrid = jnp.geomspace(self.priorlims_dict['z'][0], self.priorlims_dict['z'][1], 1000)

        distr = ((1.+z)**(k-1))*jnp.interp(z, zGridGLOB, dVcomGridGLOB)
        norm  = jnp.trapezoid(((1.+zgrid)**(k-1))*jnp.interp(zgrid, zGridGLOB, dVcomGridGLOB), zgrid)

        if not uselog:
            return jnp.where(goodsamples, distr/norm, 0.)
        else:
            return jnp.where(goodsamples, jnp.log(distr/norm), -jnp.inf)#np.NINF
    
    def sample_population(self, size):
        '''
        Function to sample the redshift model.
        
        :param int size: Size of the redshift sample.

        :return: Sampled redshifts and corresponding luminosity distances in :math:`{\\rm Gpc}`.
        :rtype: dict(array, array)
        '''
        
        z = utils.inverse_cdf_sampling(self.rate_function, size, self.priorlims_dict['z'])
        
        dL = np.interp(z, zGridGLOB, dLGridGLOB)
        
        return {'z':z, 'dL':dL}
        
    def rate_function_derivative(self, z, k_z=None, uselog=False):
        '''
        First derivative with respect to the hyperparameters of the redshift function.
        
        :param array z: Redshift.
        :param float, optional k_z: Power-law index of the redshift evolution of the merger rate.
        :param bool uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the derivatives of the redshift distribution.
        :rtype: array
        '''
    
        k = self.hyperpar_dict['k_z'] if k_z is None else k_z

        funder = lambda z, k: self.rate_function(z, k, uselog=uselog)

        derivs_all = np.squeeze(np.asarray((jacrev(funder, argnums=(1)))(z, jnp.array([k]))))

        return derivs_all[np.newaxis,:]
    
    def rate_function_hessian(self, z, k_z=None, uselog=False):
        '''
        Hessian with respect to the hyperparameters of the redshift function.
        
        :param array z: Redshift.
        :param float, optional k_z: Power-law index of the redshift evolution of the merger rate.
        :param bool uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the Hessians of the redshift distribution.
        :rtype: array
        '''
    
        k = self.hyperpar_dict['k_z'] if k_z is None else k_z
        
        funder = lambda z, k: self.rate_function(z, k, uselog=uselog)

        derivs_all = np.squeeze(np.asarray(hessian(funder, argnums=(1))(z, jnp.array([k]))))
        
        return derivs_all[np.newaxis,np.newaxis,:]

class MadauDickinson_RateDistribution(RateDistribution):
    '''
    Redshift distribution assuming a Madau-Dickinson profile for the star formation rate.
    
    Parameters:
        * alpha_z: Power-law index governing the rise of the star formation rate at low redshift.
        * beta_z: Power-law index governing the decline of the star formation rate at high redshift.
        * zp: Redshift at which the star formation rate peaks.
    
    :param dict, optional hyperparameters: Dictionary containing the hyperparameters of the mass model as keys and their fiducial value as entry.
    :param dict, optional priorlims_parameters: Dictionary containing the parameters of the mass model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.
    
    '''

    def __init__(self, hyperparameters=None, priorlims_parameters=None):
        
        self.expected_hyperpars = ['alpha_z', 'beta_z', 'zp']
        super().__init__()

        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)
        else:
            print('Expected hyperparameters: ', self.expected_hyperpars)
            basevalues = {'alpha_z':2.7, 'beta_z':3., 'zp':2.}
            print('Base values are: ', list(basevalues.items()))
            self.set_hyperparameters(basevalues)
        
        if priorlims_parameters is not None:
            self.set_priorlimits(priorlims_parameters)
        else:
            self.set_priorlimits({'z': (0., 10.)})

        self.derivative_par_nums = {'alpha_z':0, 'beta_z':1, 'zp':2}

    def _MadauDickinson_profile_z0norm(self, z, alpha_z=None, beta_z=None, zp=None, R0=1.):
        '''
        Madau-Dickinson profile normalized at z=0.
        
        :param array z: Redshift.
        :param float alpha_z: First power law index of the redshift evolution of the star formation rate.
        :param float beta_z: Second power law index of the redshift evolution of the star formation rate.
        :param float zp: Redshift at which the star formation rate peaks.
        :param float R0: Normalization of the merger rate, in :math:`{\\rm Gpc}^{-3}\,{\\rm yr}^{-1}`.
        
        :return: Normalized Madau-Dickinson profile at the input redshifts.
        :rtype: array
        '''
        
        alpha = self.hyperpar_dict['alpha_z'] if alpha_z is None else alpha_z
        beta  = self.hyperpar_dict['beta_z'] if beta_z is None else beta_z
        zp    = self.hyperpar_dict['zp'] if zp is None else zp

        return R0*((1.+zp)**(alpha+beta) + 1.) * (1.+z)**(alpha) / ((1.+zp)**(alpha+beta) + (1.+z)**(alpha+beta))
    
    def N_per_yr(self, alpha_z=None, beta_z=None, zp=None, R0=1.):
        """
        Compute the number of mergers per year.

        :param float alpha_z: First power law index of the redshift evolution of the star formation rate.
        :param float beta_z: Second power law index of the redshift evolution of the star formation rate.
        :param float zp: Redshift at which the star formation rate peaks.
        :param float R0: Normalization of the merger rate, in :math:`{\\rm Gpc}^{-3}\,{\\rm yr}^{-1}`.

        :return: Number of mergers per year.
        :rtype: float
        """
        alpha = self.hyperpar_dict['alpha_z'] if alpha_z is None else alpha_z
        beta  = self.hyperpar_dict['beta_z'] if beta_z is None else beta_z
        zp    = self.hyperpar_dict['zp'] if zp is None else zp

        zgrid = jnp.geomspace(self.priorlims_dict['z'][0], self.priorlims_dict['z'][1], 1000)

        res = np.trapezoid(self._MadauDickinson_profile_z0norm(zgrid, alpha, beta, zp, R0)*jnp.interp(zgrid, zGridGLOB, dVcomGridGLOB)/(1.+zgrid), zgrid) 

        return np.ceil(res).astype(int)
    
    def rate_function(self, z, alpha_z=None, beta_z=None, zp=None, uselog=False):
        '''
        Redshift distribution function.
        
        :param array z: Redshift.
        :param float alpha_z: First power law index of the redshift evolution of the merger rate.
        :param float beta_z: Second power law index of the redshift evolution of the merger rate.
        :param float zp: Redshift at which the star formation rate peaks.
        :param bool uselog: Boolean specifying whether to return the probability or log-probability, defaults to False.
        
        :return: Redshift distribution value at the input redshifts.
        :rtype: array
        '''
        
        alpha = self.hyperpar_dict['alpha_z'] if alpha_z is None else alpha_z
        beta  = self.hyperpar_dict['beta_z'] if beta_z is None else beta_z
        zp    = self.hyperpar_dict['zp'] if zp is None else zp
        
        goodsamples = self._isin_prior_range('z', z)
        zgrid = jnp.geomspace(self.priorlims_dict['z'][0], self.priorlims_dict['z'][1], 1000)

        distr = self._MadauDickinson_profile_z0norm(z, alpha, beta, zp, 1.)*jnp.interp(z, zGridGLOB, dVcomGridGLOB)/(1.+z)
        norm  = jnp.trapezoid(self._MadauDickinson_profile_z0norm(zgrid, alpha, beta, zp, 1.)*jnp.interp(zgrid, zGridGLOB, dVcomGridGLOB)/(1.+zgrid), zgrid)    
        
        if not uselog:
            return jnp.where(goodsamples, distr/norm, 0.)
        else:
            return jnp.where(goodsamples, jnp.log(distr/norm), -jnp.inf)#np.NINF
    
    def sample_population(self, size):
        '''
        Function to sample the redshift model.
        
        :param int size: Size of the redshift sample.

        :return: Sampled redshifts and corresponding luminosity distances in :math:`{\\rm Gpc}`.
        :rtype: dict(array, array)
        '''
        
        z = utils.inverse_cdf_sampling(self.rate_function, size, self.priorlims_dict['z'])
        
        dL = np.interp(z, zGridGLOB, dLGridGLOB)
        
        return {'z':z, 'dL':dL}
    
    def rate_function_derivative(self, z, alpha_z=None, beta_z=None, zp=None, uselog=False):
        '''
        Derivative with respect to the hyperparameters of the redshift function.
        
        :param array z: Redshift.
        :param float alpha_z: First power law index of the redshift evolution of the merger rate.
        :param float beta_z: Second power law index of the redshift evolution of the merger rate.
        :param float zp: Redshift at which the star formation rate peaks.
        :param bool uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the derivatives of the redshift distribution.
        :rtype: array
        '''
    
        alpha = self.hyperpar_dict['alpha_z'] if alpha_z is None else alpha_z
        beta  = self.hyperpar_dict['beta_z'] if beta_z is None else beta_z
        zp    = self.hyperpar_dict['zp'] if zp is None else zp

        funder = lambda z, alpha, beta, zp: self.rate_function(z, alpha, beta, zp, uselog=uselog)

        derivs_all = np.squeeze(np.asarray(jacrev(funder, argnums=(1,2,3))(z, jnp.array([alpha]), jnp.array([beta]), jnp.array([zp]))))
        
        return derivs_all
    
    def rate_function_hessian(self, z, alpha_z=None, beta_z=None, zp=None, uselog=False):
        '''
        Hessians with respect to the hyperparameters of the redshift function.
        
        :param array z: Redshift.
        :param float alpha_z: First power law index of the redshift evolution of the merger rate.
        :param float beta_z: Second power law index of the redshift evolution of the merger rate.
        :param float zp: Redshift at which the star formation rate peaks.
        :param bool uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the Hessians of the redshift distribution.
        :rtype: array
        '''
    
        alpha = self.hyperpar_dict['alpha_z'] if alpha_z is None else alpha_z
        beta  = self.hyperpar_dict['beta_z'] if beta_z is None else beta_z
        zp    = self.hyperpar_dict['zp'] if zp is None else zp

        funder = lambda z, alpha, beta, zp: self.rate_function(z, alpha, beta, zp, uselog=uselog)

        derivs_all = np.squeeze(np.asarray(hessian(funder, argnums=(1,2,3))(z, jnp.array([alpha]), jnp.array([beta]), jnp.array([zp]))))
    
        return derivs_all

class MadauDickinsonPLTimeDelta_RateDistribution(RateDistribution):
    '''
    Redshift distribution assuming a Madau-Dickinson profile for the star formation rate and a time delay between formation and merger with :math:`P(t_d)\propto t_d^{\alpha_{\tau}}`.
    
    Parameters:
        * alpha_z: Power-law index governing the rise of the star formation rate at low redshift.
        * beta_z: Power-law index governing the decline of the star formation rate at high redshift.
        * zp: Redshift at which the star formation rate peaks.
        * alpha_tau: Power-law index governing the time delay distribution.
        * tau_min: Minimum time delay between formation and merger, in :math:`{\\rm Myr}`.
    
    :param dict, optional hyperparameters: Dictionary containing the hyperparameters of the mass model as keys and their fiducial value as entry.
    :param dict, optional priorlims_parameters: Dictionary containing the parameters of the mass model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.
    :bool, optional verbose: Boolean specifying if the code has to print additional details during execution.
    
    '''

    def __init__(self, hyperparameters=None, priorlims_parameters=None, verbose=False):
        
        self.expected_hyperpars = ['alpha_z', 'beta_z', 'zp', 'alpha_tau', 'tau_min']
        super().__init__()

        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)
        else:
            print('Expected hyperparameters: ', self.expected_hyperpars)
            basevalues = {'alpha_z':2.7, 'beta_z':3., 'zp':2., 'alpha_tau':-1., 'tau_min':10.}
            print('Base values are: ', list(basevalues.items()))
            self.set_hyperparameters(basevalues)
        
        if priorlims_parameters is not None:
            self.set_priorlimits(priorlims_parameters)
        else:
            self.set_priorlimits({'z': (0., 10.)})

        self.derivative_par_nums = {'alpha_z':0, 'beta_z':1, 'zp':2, 'alpha_tau':3, 'tau_min':4}

        self.verbose = verbose

        self.path_zform_tab = os.path.join(glob.AuxFilesPath, 'zform_Table_200.h5')
        self._make_zform_interpolator(res=200)

    def _MadauDickinson_profile_z0norm(self, z, alpha_z=None, beta_z=None, zp=None, R0=1.):
        '''
        Madau-Dickinson profile normalized at z=0.
        
        :param array z: Redshift.
        :param float alpha_z: First power law index of the redshift evolution of the star formation rate.
        :param float beta_z: Second power law index of the redshift evolution of the star formation rate.
        :param float zp: Redshift at which the star formation rate peaks.
        :param float R0: Normalization of the merger rate, in :math:`{\\rm Gpc}^{-3}\,{\\rm yr}^{-1}`.
        
        :return: Normalized Madau-Dickinson profile at the input redshifts.
        :rtype: array
        '''
        
        alpha = self.hyperpar_dict['alpha_z'] if alpha_z is None else alpha_z
        beta  = self.hyperpar_dict['beta_z'] if beta_z is None else beta_z
        zp    = self.hyperpar_dict['zp'] if zp is None else zp

        return R0*((1.+zp)**(alpha+beta) + 1.) * (1.+z)**(alpha) / ((1.+zp)**(alpha+beta) + (1.+z)**(alpha+beta))
        
    def _convolved_MadauDickinson_profile_z0norm(self, z, alpha_z=None, beta_z=None, zp=None, alpha_tau=None, tau_min=None, R0=1., norm_z0=False):
        '''
        Madau-Dickinson profile normalized at z=0 convolved with a time-delay distribution :math:`P(t_d)\propto t_d^{\alpha_{\tau}}`.
        
        :param array z: Redshift.
        :param float alpha_z: First power law index of the redshift evolution of the star formation rate.
        :param float beta_z: Second power law index of the redshift evolution of the star formation rate.
        :param float zp: Redshift at which the star formation rate peaks.
        :param float alpha_tau: Power-law index governing the time delay distribution.
        :param float tau_min: Minimum time delay between formation and merger, in :math:`{\\rm Myr}`.
        :param float R0: Normalization of the merger rate, in :math:`{\\rm Gpc}^{-3}\,{\\rm yr}^{-1}`.
        :param bool norm_z0: Boolean specifying whether to normalize the convolved redshift distribution at z=0, defaults to False.
        
        :return: Convolved Madau-Dickinson profile at the input redshifts.
        :rtype: array
        '''
        
        alpha     = self.hyperpar_dict['alpha_z'] if alpha_z is None else alpha_z
        beta      = self.hyperpar_dict['beta_z'] if beta_z is None else beta_z
        zp        = self.hyperpar_dict['zp'] if zp is None else zp
        alpha_tau = self.hyperpar_dict['alpha_tau'] if alpha_tau is None else alpha_tau
        tau_min   = self.hyperpar_dict['tau_min'] if tau_min is None else tau_min

        tdmax=5000.
        tdgrid = jnp.squeeze(jnp.geomspace(tau_min, tdmax, 200))
        tdgrids = jnp.squeeze(jnp.array([tdgrid for i in range(len(z))]).T)
        zz = jnp.array([z for i in range(len(tdgrid))])

        zf = self.zform_interp(jnp.asarray((zz, tdgrids)).T).T
        res = jnp.trapezoid(self._MadauDickinson_profile_z0norm(zf, alpha, beta, zp, R0=R0)*(tdgrids**alpha_tau)/(1.+zf), tdgrids, axis=0)

        zeroval = 1.
        if norm_z0:
            zf0 = self.zform_interp(jnp.asarray((jnp.zeros_like(tdgrid), tdgrid)).T).T
            zeroval = jnp.trapezoid(self._MadauDickinson_profile_z0norm(zf0, alpha, beta, zp, R0=R0)*(tdgrid**alpha_tau)/(1.+zf0), tdgrid, axis=0)

        return R0*res/zeroval
    
    def N_per_yr(self, alpha_z=None, beta_z=None, zp=None, alpha_tau=None, tau_min=None, R0=1.):
        """
        Compute the number of mergers per year.

        :param float alpha_z: First power law index of the redshift evolution of the merger rate.
        :param float beta_z: Second power law index of the redshift evolution of the merger rate.
        :param float zp: Redshift at which the star formation rate peaks.
        :param float alpha_tau: Power-law index governing the time delay distribution.
        :param float tau_min: Minimum time delay between formation and merger, in :math:`{\\rm Myr}`.
        :param float R0: Normalization of the merger rate, in :math:`{\\rm Gpc}^{-3}\,{\\rm yr}^{-1}`.

        :return: Number of mergers per year.
        :rtype: float
        """
        alpha     = self.hyperpar_dict['alpha_z'] if alpha_z is None else alpha_z
        beta      = self.hyperpar_dict['beta_z'] if beta_z is None else beta_z
        zp        = self.hyperpar_dict['zp'] if zp is None else zp
        alpha_tau = self.hyperpar_dict['alpha_tau'] if alpha_tau is None else alpha_tau
        tau_min   = self.hyperpar_dict['tau_min'] if tau_min is None else tau_min

        zgrid = jnp.geomspace(self.priorlims_dict['z'][0], self.priorlims_dict['z'][1], 1000)

        res = np.trapezoid(self._convolved_MadauDickinson_profile_z0norm(zgrid, alpha, beta, zp, alpha_tau, tau_min, R0, norm_z0=True)*jnp.interp(zgrid, zGridGLOB, dVcomGridGLOB), zgrid) 

        return np.ceil(res).astype(int)
    
    def rate_function(self, z, alpha_z=None, beta_z=None, zp=None, alpha_tau=None, tau_min=None, uselog=False):
        '''
        Redshift distribution function.
        
        :param array z: Redshift.
        :param float alpha_z: First power law index of the redshift evolution of the merger rate.
        :param float beta_z: Second power law index of the redshift evolution of the merger rate.
        :param float zp: Redshift at which the star formation rate peaks.
        :param float alpha_tau: Power-law index governing the time delay distribution.
        :param float tau_min: Minimum time delay between formation and merger, in :math:`{\\rm Myr}`.
        :param bool uselog: Boolean specifying whether to return the probability or log-probability, defaults to False.
        
        :return: Redshift distribution value at the input redshifts.
        :rtype: array
        '''
        
        alpha     = self.hyperpar_dict['alpha_z'] if alpha_z is None else alpha_z
        beta      = self.hyperpar_dict['beta_z'] if beta_z is None else beta_z
        zp        = self.hyperpar_dict['zp'] if zp is None else zp
        alpha_tau = self.hyperpar_dict['alpha_tau'] if alpha_tau is None else alpha_tau
        tau_min   = self.hyperpar_dict['tau_min'] if tau_min is None else tau_min
        
        goodsamples = self._isin_prior_range('z', z)
        zgrid = jnp.geomspace(self.priorlims_dict['z'][0], self.priorlims_dict['z'][1], 1000)

        distr = self._convolved_MadauDickinson_profile_z0norm(z, alpha, beta, zp, alpha_tau, tau_min, 1.)*jnp.interp(z, zGridGLOB, dVcomGridGLOB) # In this case the 1/(1+z) is already accounted for!
        norm  = jnp.trapezoid(self._convolved_MadauDickinson_profile_z0norm(zgrid, alpha, beta, zp, alpha_tau, tau_min, 1.)*jnp.interp(zgrid, zGridGLOB, dVcomGridGLOB), zgrid)    
        
        if not uselog:
            return jnp.where(goodsamples, distr/norm, 0.)
        else:
            return jnp.where(goodsamples, jnp.log(distr/norm), -jnp.inf)#np.NINF
    
    def sample_population(self, size):
        '''
        Function to sample the redshift model.
        
        :param int size: Size of the redshift sample.

        :return: Sampled redshifts and corresponding luminosity distances in :math:`{\\rm Gpc}`.
        :rtype: dict(array, array)
        '''
        
        z = utils.inverse_cdf_sampling(self.rate_function, size, self.priorlims_dict['z'], res=1000)
        
        dL = np.interp(z, zGridGLOB, dLGridGLOB)
        
        return {'z':z, 'dL':dL}
    
    def rate_function_derivative(self, z, alpha_z=None, beta_z=None, zp=None, alpha_tau=None, tau_min=None, uselog=False):
        '''
        Derivative with respect to the hyperparameters of the redshift function.
        
        :param array z: Redshift.
        :param float alpha_z: First power law index of the redshift evolution of the merger rate.
        :param float beta_z: Second power law index of the redshift evolution of the merger rate.
        :param float zp: Redshift at which the star formation rate peaks.
        :param float alpha_tau: Power-law index governing the time delay distribution.
        :param float tau_min: Minimum time delay between formation and merger, in :math:`{\\rm Myr}`.
        :param bool uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the derivatives of the redshift distribution.
        :rtype: array
        '''
    
        alpha     = self.hyperpar_dict['alpha_z'] if alpha_z is None else alpha_z
        beta      = self.hyperpar_dict['beta_z'] if beta_z is None else beta_z
        zp        = self.hyperpar_dict['zp'] if zp is None else zp
        alpha_tau = self.hyperpar_dict['alpha_tau'] if alpha_tau is None else alpha_tau
        tau_min   = self.hyperpar_dict['tau_min'] if tau_min is None else tau_min

        funder = lambda z, alpha, beta, zp, alpha_tau, tau_min: self.rate_function(z, alpha, beta, zp, alpha_tau, tau_min, uselog=uselog)

        derivs_all = np.squeeze(np.asarray((jacrev(funder, argnums=(1,2,3,4,5)))(z, jnp.array([alpha]), jnp.array([beta]), jnp.array([zp]), jnp.array([alpha_tau]), jnp.array([tau_min]))))
        
        return derivs_all
    
    def rate_function_hessian(self, z, alpha_z=None, beta_z=None, zp=None, alpha_tau=None, tau_min=None, uselog=False):
        '''
        Hessians with respect to the hyperparameters of the redshift function.
        
        :param array z: Redshift.
        :param float alpha_z: First power law index of the redshift evolution of the merger rate.
        :param float beta_z: Second power law index of the redshift evolution of the merger rate.
        :param float zp: Redshift at which the star formation rate peaks.
        :param float alpha_tau: Power-law index governing the time delay distribution.
        :param float tau_min: Minimum time delay between formation and merger, in :math:`{\\rm Myr}`.
        :param bool uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the Hessians of the redshift distribution.
        :rtype: array
        '''
    
        alpha     = self.hyperpar_dict['alpha_z'] if alpha_z is None else alpha_z
        beta      = self.hyperpar_dict['beta_z'] if beta_z is None else beta_z
        zp        = self.hyperpar_dict['zp'] if zp is None else zp
        alpha_tau = self.hyperpar_dict['alpha_tau'] if alpha_tau is None else alpha_tau
        tau_min   = self.hyperpar_dict['tau_min'] if tau_min is None else tau_min

        funder = lambda z, alpha, beta, zp, alpha_tau, tau_min: self.rate_function(z, alpha, beta, zp, alpha_tau, tau_min, uselog=uselog)

        derivs_all = np.squeeze(np.asarray(hessian(funder, argnums=(1,2,3,4,5))(z, jnp.array([alpha]), jnp.array([beta]), jnp.array([zp]), jnp.array([alpha_tau]), jnp.array([tau_min]))))
    
        return derivs_all
    
    def _tabulate_zform(self, H0=cosmo.H(0).value, tdmin=5., zmax=20., res=200, store=True):
        '''
        Function to compute the redshift of formation as a function of redshift and time delay on a regular grid.
        
        :param float H0: Hubble constant at redshift zero, in :math:`{\\rm km}\,{\\rm s}^{-1}\,{\\rm Mpc}^{-1}`.
        :param float tdmin: Minimum time delay between formation and merger, in :math:`{\\rm Myr}`.
        :param float zmax: Maximum redshift to consider.
        :param int res: Number of points in the grid.
        :param bool store: Boolean specifying whether to store the computed grid, defaults to True.
        
        :return: Redshift grid, time delay grid, and redshift of formation grid.
        :rtype: tuple(array, array, array)
        '''

        zgrid = np.linspace(0., zmax, res)
        tdgrid = np.geomspace(tdmin, 5000., res)

        def zform(z,td,H0=cosmo.H(0).value):
            func = lambda zfo : np.where(zfo>z, np.interp(zfo, zGridGLOB, lbtGridGLOB) - np.interp(z, zGridGLOB, lbtGridGLOB), 0.)*(cosmo.H(0).value/H0) - td 
            zest = root(func, z, method='hybr')
            return zest.x[0]

        zfgrid = np.zeros((res,res))
        for i,z in enumerate(zgrid):
            for j,td in enumerate(tdgrid):
                zfgrid[i,j] = zform(z, td, H0=H0)

        if store:
            print('Saving result...')
            with h5py.File(os.path.join(glob.AuxFilesPath, 'zform_Table_'+str(res)+'.h5'), 'w') as out:
                out.create_dataset('z', data=zgrid, compression='gzip', shuffle=True)
                out.create_dataset('td', data=tdgrid, compression='gzip', shuffle=True)
                out.create_dataset('zform', data=zfgrid, compression='gzip', shuffle=True)
                out.attrs['npoints'] = res
                out.attrs['td_min'] = tdmin
                out.attrs['z_max'] = zmax
            print('Done...')

        return zgrid, tdgrid, zfgrid
    
    def _make_zform_interpolator(self, res=200):
        '''
        Build the interpolator for the redshift of formation as a function of redshift and time delay. The output is stored in the attribute `zform_interp`.
        
        :param int res: Number of points in the grid.
        
        '''

        if self.path_zform_tab is not None:
            if os.path.exists(self.path_zform_tab):
                if self.verbose:
                    print('Pre-computed z_form grid is present. Loading...')
                with h5py.File(os.path.join(glob.AuxFilesPath, 'zform_Table_'+str(res)+'.h5'), 'r') as inp:
                    zgrid = inp['z'][:]
                    tdgrid = inp['td'][:]
                    zfgrid = inp['zform'][:]
                    if self.verbose:
                        print('Attributes of pre-computed grid: ')
                        print([(k, inp.attrs[k]) for k in inp.attrs.keys()])
                        self.verbose=False
            else:
                print('Tabulating z_form...')
                zgrid, tdgrid, zfgrid = self._tabulate_zform(res=res)

        else:
            print('Tabulating z_form...')
            zgrid, tdgrid, zfgrid = self._tabulate_zform(res=res)

        self.zform_interp = utils.RegularGridInterpolator_JAX((zgrid, tdgrid), zfgrid, bounds_error=False, fill_value=None)
