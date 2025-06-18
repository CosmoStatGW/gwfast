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
from scipy.stats import truncnorm as truncnormscpy

class SpinDistribution(ABC):
    ''' 
    Abstract class to compute spin distributions.

    :param list parameters: List containing the parameters of the spin model.
    :param dict hyperparameters: Dictionary containing the hyperparameters of the spin model as keys and their fiducial value as entry.
    :param dict priorlims_parameters: Dictionary containing the parameters of the spin model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.
    ''' 
    
    def __init__(self, is_Precessing=True):
        self.par_list = []
        self.hyperpar_dict = {}
        self.priorlims_dict = {}
        self.is_Precessing = is_Precessing

    def set_parameters(self, parameters):
        '''
        Setter method for the parameters of the spin model.

        :param list parameters: List containing the parameters of the spin model.
        '''
        self.par_list = parameters

    def set_hyperparameters(self, hyperparameters):
        '''
        Setter method for the hyperparameters of the spin model.

        :param dict new_hyperparameters: Dictionary containing the hyperparameters of the spin model as keys and their new value as entry.
        '''
        self.hyperpar_dict = hyperparameters
    
    def update_hyperparameters(self, new_hyperparameters):
        '''
        Method to update the hyperparameters of the spin model.

        :param dict new_hyperparameters: Dictionary containing the new hyperparameters of the spin model as keys and their new value as entry.
        '''
        for key in new_hyperparameters.keys():
            if key in self.hyperpar_dict.keys():
                self.hyperpar_dict[key] = new_hyperparameters[key]
            else:
                raise ValueError('The hyperparameter '+key+' is not present in the hyperparameter dictionary.')

    def set_priorlimits(self, limits):
        '''
        Setter method for the prior limits on the parameters of the spin model.

        :param dict limits: Dictionary containing the parameters of the spin model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.
        '''
        self.priorlims_dict = limits
    
    def update_priorlimits(self, new_limits):
        '''
        Method to update the prior limits on the parameters of the spin model.
        
        :param dict new_limits: Dictionary containing the new prior limits on the parameters of the spin model as keys and their new value as entry, given as a tuple :math:`(l, h)`.
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
    def spin_function(self,):
        ''' 
        Mass function of the population model.

        :return: Mass function value.
        :rtype: float
        ''' 
        pass

    @abstractmethod
    def sample_population(self, size):
        ''' 
        Function to sample the spin distribution.

        :param int size: number of samples
        ''' 
        pass

    @abstractmethod
    def spin_function_derivative(self,):
        ''' 
        Derivative of the spin function of the population model.

        :return: Derivative of the spin function.
        :rtype: float
        ''' 
        pass

class DefaultPrecessing_SpinDistribution(SpinDistribution):
    ''' 
    Default spin model

    Parameters:
        * alpha_chi: alpha shape parameter of the Beta distribution for the spin magnitudes.
        * beta_chi: beta shape parameter of the Beta distribution for the spin magnitudes.
        * zeta_chi: Mixing fraction of mergers from the truncated Gaussian component for spin orientations.
        * sigma_t_chi: Width of the truncated Gaussian component for spin orientations.
    
    :param dict, optional hyperparameters: Dictionary containing the hyperparameters of the spin model as keys and their fiducial value as entry.
    :param dict, optional priorlims_parameters: Dictionary containing the parameters of the spin model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.

    ''' 

    def __init__(self, hyperparameters=None, priorlims_parameters=None):

        self.expected_hyperpars = ['alpha_chi', 'beta_chi', 'zeta_chi', 'sigma_t_chi']
        super().__init__(is_Precessing=True)

        self.set_parameters(['chi1', 'chi2', 'tilt1', 'tilt2', 'phiJL', 'phi12'])

        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)
        else:
            print('Expected hyperparameters: ', self.expected_hyperpars)
            basevalues = {'alpha_chi':1.6, 'beta_chi':4.12, 'zeta_chi':0.66, 'sigma_t_chi':1.5}
            print('Base values are: ', list(basevalues.items()))
            self.set_hyperparameters(basevalues)

        if priorlims_parameters is not None:
            self.set_priorlimits(priorlims_parameters)
        else:
            self.set_priorlimits({'chi1': (0, 1), 'chi2': (0, 1), 'tilt1': (0, np.pi), 'tilt2': (0, np.pi), 'phi12': (0, 2*np.pi), 'phiJL': (0, 2*np.pi)})
        
        self.derivative_par_nums = {'alpha_chi':0, 'beta_chi':1, 'zeta_chi':2, 'sigma_t_chi':3}

    def _chimagnitude_function(self, chi, alpha_chi=None, beta_chi=None):
        '''
        Spin magnitude function.
        
        :param array chi: Spin magnitude.
        :param float, optional alpha_chi: alpha shape parameter of the Beta distribution for the spin magnitudes.
        :param float, optional beta_chi: beta shape parameter of the Beta distribution for the spin magnitudes.
        
        :return: Spin magnitude function value at the input spins.
        :rtype: array
        '''

        alpha = self.hyperpar_dict['alpha_chi'] if alpha_chi is None else alpha_chi
        beta  = self.hyperpar_dict['beta_chi'] if beta_chi is None else beta_chi

        goodsamples = self._isin_prior_range('chi1', chi)
        prob = utils.beta_distrib(chi, alpha, beta)
        
        return jnp.where(goodsamples, prob, 0.)

    def _coschitilt_function(self, costilt, zeta_chi=None, sigma_t_chi=None):
        '''
        Cosine of the tilt angle function.
        
        :param array costilt: Cosine of the tilt angle.
        :param float, optional zeta_chi: Mixing fraction of mergers from the truncated Gaussian component for spin orientations.
        :param float, optional sigma_t_chi: Width of the truncated Gaussian component for spin orientations.
        
        :return: Cosine of the tilt angle function value at the input angles.
        :rtype: array
        '''

        zeta    = self.hyperpar_dict['zeta_chi'] if zeta_chi is None else zeta_chi
        sigma_t = self.hyperpar_dict['sigma_t_chi'] if sigma_t_chi is None else sigma_t_chi

        goodsamples = self._isin_prior_range('tilt1', jnp.arccos(costilt))
        
        prob = zeta*utils.trunc_gaussian_norm(costilt, mu=1., sigma=sigma_t, lower=-1., upper=1.) + 0.5 * (1.-zeta)

        return jnp.where(goodsamples, prob, 0.)
    
    def spin_function(self, chi1, chi2, tilt1, tilt2, phiJL, phi12, alpha_chi=None, beta_chi=None, zeta_chi=None, sigma_t_chi=None, uselog=False):
        '''
        Spin function.
        
        :param array chi1: Spin magnitude of the first BH.
        :param array chi2: Spin magnitude of the second BH.
        :param array tilt1: Tilt angle of the first BH.
        :param array tilt2: Tilt angle of the second BH.
        :param array phiJL: Azimuthal angle of the orbital angular momentum relative to the total angular momentum.
        :param array phi12: Difference in azimuthal angle between the spins.
        :param float, optional alpha_chi: alpha shape parameter of the Beta distribution for the spin magnitudes.
        :param float, optional beta_chi: beta shape parameter of the Beta distribution for the spin magnitudes.
        :param float, optional zeta_chi: Mixing fraction of mergers from the truncated Gaussian component for spin orientations.
        :param float, optional sigma_t_chi: Width of the truncated Gaussian component for spin orientations.
        :param bool, optional uselog: Boolean specifying whether to return the probability or log-probability, defaults to False.
        
        :return: Spin function value at the input spins.
        :rtype: array
        '''

        alpha   = self.hyperpar_dict['alpha_chi'] if alpha_chi is None else alpha_chi
        beta    = self.hyperpar_dict['beta_chi'] if beta_chi is None else beta_chi
        zeta    = self.hyperpar_dict['zeta_chi'] if zeta_chi is None else zeta_chi
        sigma_t = self.hyperpar_dict['sigma_t_chi'] if sigma_t_chi is None else sigma_t_chi

        goodsamples = self._isin_prior_range('phiJL', phiJL) & self._isin_prior_range('phi12', phi12)

        if not uselog:
            return jnp.where(goodsamples, self._chimagnitude_function(chi1, alpha, beta)*self._chimagnitude_function(chi2, alpha, beta)*self._coschitilt_function(jnp.cos(tilt1), zeta, sigma_t)*self._coschitilt_function(jnp.cos(tilt2), zeta, sigma_t)/(4.*np.pi*np.pi), 0.)
        else:
            return jnp.where(goodsamples, jnp.log(self._chimagnitude_function(chi1, alpha, beta)) + jnp.log(self._chimagnitude_function(chi2, alpha, beta))  + jnp.log(self._coschitilt_function(jnp.cos(tilt1), zeta, sigma_t)) + jnp.log(self._coschitilt_function(jnp.cos(tilt2), zeta, sigma_t)) - jnp.log(4.*np.pi*np.pi), -jnp.inf)
                
    def sample_population(self, size):
        '''
        Function to sample the spin model.
        
        :param int size: Size of the spin sample.

        :return: Sampled spin components for the two objects.
        :rtype: dict(array, array, ...)
        '''
        
        chi1 = np.random.beta(self.hyperpar_dict['alpha_chi'], self.hyperpar_dict['beta_chi'], size=size)
        chi2 = np.random.beta(self.hyperpar_dict['alpha_chi'], self.hyperpar_dict['beta_chi'], size=size)
        
        if (self.hyperpar_dict['alpha_chi']<1.) | (self.hyperpar_dict['beta_chi']<1.):
            tmpeps = 1.0e-4
            chi1 = np.where(chi1<=tmpeps, tmpeps, np.where(chi1>=1.-tmpeps, 1.-tmpeps, chi1))
            chi2 = np.where(chi2<=tmpeps, tmpeps, np.where(chi2>=1.-tmpeps, 1.-tmpeps, chi2))

        costilt1 = utils.inverse_cdf_sampling(self._coschitilt_function, size, (-1., 1.))
        costilt2 = utils.inverse_cdf_sampling(self._coschitilt_function, size, (-1., 1.))

        phiJL = np.random.uniform(0, 2*np.pi, size)
        phi12 = np.random.uniform(0, 2*np.pi, size)

        return {'chi1': chi1, 'chi2': chi2, 'tilt1': np.arccos(costilt1), 'tilt2': np.arccos(costilt2), 'phiJL': phiJL, 'phi12': phi12}
    
    def spin_function_derivative(self, chi1, chi2, tilt1, tilt2, phiJL, phi12, alpha_chi=None, beta_chi=None, zeta_chi=None, sigma_t_chi=None, uselog=False):
        '''
        Derivative with respect to the hyperparameters of the spin function.
        
        :param array chi1: Spin magnitude of the first BH.
        :param array chi2: Spin magnitude of the second BH.
        :param array tilt1: Tilt angle of the first BH.
        :param array tilt2: Tilt angle of the second BH.
        :param array phiJL: Azimuthal angle of the orbital angular momentum relative to the total angular momentum.
        :param array phi12: Difference in azimuthal angle between the spins.
        :param float, optional alpha_chi: alpha shape parameter of the Beta distribution for the spin magnitudes.
        :param float, optional beta_chi: beta shape parameter of the Beta distribution for the spin magnitudes.
        :param float, optional zeta_chi: Mixing fraction of mergers from the truncated Gaussian component for spin orientations.
        :param float, optional sigma_t_chi: Width of the truncated Gaussian component for spin orientations.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Derivative of the spin function with respect to the hyperparameters.
        :rtype: array
        '''

        alpha   = self.hyperpar_dict['alpha_chi'] if alpha_chi is None else alpha_chi
        beta    = self.hyperpar_dict['beta_chi'] if beta_chi is None else beta_chi
        zeta    = self.hyperpar_dict['zeta_chi'] if zeta_chi is None else zeta_chi
        sigma_t = self.hyperpar_dict['sigma_t_chi'] if sigma_t_chi is None else sigma_t_chi
        
        funder = lambda chi1, chi2, tilt1, tilt2, phiJL, phi12, alpha_chi, beta_chi, zeta_chi, sigma_t_chi: self.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, alpha_chi, beta_chi, zeta_chi, sigma_t_chi, uselog=uselog)

        derivs_all = np.squeeze(np.asarray(jacrev(funder, argnums=(6,7,8,9))(chi1, chi2, tilt1, tilt2, phiJL, phi12, jnp.array([alpha]), jnp.array([beta]), jnp.array([zeta]), jnp.array([sigma_t]))))
        
        return derivs_all
    
    def spin_function_hessian(self, chi1, chi2, tilt1, tilt2, phiJL, phi12, alpha_chi=None, beta_chi=None, zeta_chi=None, sigma_t_chi=None, uselog=False):
        '''
        Hessians with respect to the hyperparameters of the spin function.
        
        :param array chi1: Spin magnitude of the first BH.
        :param array chi2: Spin magnitude of the second BH.
        :param array tilt1: Tilt angle of the first BH.
        :param array tilt2: Tilt angle of the second BH.
        :param array phiJL: Azimuthal angle of the orbital angular momentum relative to the total angular momentum.
        :param array phi12: Difference in azimuthal angle between the spins.
        :param float, optional alpha_chi: alpha shape parameter of the Beta distribution for the spin magnitudes.
        :param float, optional beta_chi: beta shape parameter of the Beta distribution for the spin magnitudes.
        :param float, optional zeta_chi: Mixing fraction of mergers from the truncated Gaussian component for spin orientations.
        :param float, optional sigma_t_chi: Width of the truncated Gaussian component for spin orientations.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Hessians of the spin function with respect to the hyperparameters.
        :rtype: array
        '''

        alpha   = self.hyperpar_dict['alpha_chi'] if alpha_chi is None else alpha_chi
        beta    = self.hyperpar_dict['beta_chi'] if beta_chi is None else beta_chi
        zeta    = self.hyperpar_dict['zeta_chi'] if zeta_chi is None else zeta_chi
        sigma_t = self.hyperpar_dict['sigma_t_chi'] if sigma_t_chi is None else sigma_t_chi

        funder = lambda chi1, chi2, tilt1, tilt2, phiJL, phi12, alpha_chi, beta_chi, zeta_chi, sigma_t_chi: self.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, alpha_chi, beta_chi, zeta_chi, sigma_t_chi, uselog=uselog)
        
        derivs_all = np.squeeze(np.asarray(hessian(funder, argnums=(6,7,8,9))(chi1, chi2, tilt1, tilt2, phiJL, phi12, jnp.array([alpha]), jnp.array([beta]), jnp.array([zeta]), jnp.array([sigma_t]))))
        
        return derivs_all
        
class SameFlatNonPrecessing_SpinDistribution(SpinDistribution):
    ''' 
    Independent Flat spin model with fixed absolute value

    Parameters:
        * abs_chi: maximum absolute value of the spin
        
    :param dict, optional hyperparameters: Dictionary containing the hyperparameters of the spin model as keys and their fiducial value as entry.
    :param dict, optional priorlims_parameters: Dictionary containing the parameters of the spin model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.

    ''' 

    def __init__(self, hyperparameters=None, priorlims_parameters=None):

        self.expected_hyperpars = ['abs_chi']
        super().__init__(is_Precessing=False)

        self.set_parameters(['chi1z', 'chi2z'])

        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)
        else:
            print('Expected hyperparameters: ', self.expected_hyperpars)
            basevalues = {'abs_chi':0.05}
            print('Base values are: ', list(basevalues.items()))
            self.set_hyperparameters(basevalues)
        
        if priorlims_parameters is not None:
            self.set_priorlimits(priorlims_parameters)
        else:
            # Define the prior limits for 'm1' and 'q'
            self.set_priorlimits({'chi1z': (-1., 1.), 'chi2z': (-1., 1.)})
        
        self.derivative_par_nums = {'abs_chi':0}
    
    def spin_function(self, chi1z, chi2z, abs_chi=None, uselog=False):
        '''
        Spin function.
        
        :param array chi1z: Spin component of the first object.
        :param array chi2z: Spin component of the second object.
        :param float, optional abs_chi: maximum absolute value of the spin.
        :param bool, optional uselog: Boolean specifying whether to return the probability or log-probability, defaults to False.
        
        :return: Spin function value at the input spins.
        :rtype: array
        '''

        abs_chi = self.hyperpar_dict['abs_chi'] if abs_chi is None else abs_chi
    
        goodsamples = self._isin_prior_range('chi1z', chi1z) & self._isin_prior_range('chi2z', chi2z)

        if not uselog:
            return jnp.where(goodsamples, jnp.where((abs(chi1z)<=abs_chi) & (abs(chi2z)<=abs_chi), 1./((2.*abs_chi)**2), 0.), 0.)
        else:
            return jnp.where(goodsamples, jnp.where((abs(chi1z)<=abs_chi) & (abs(chi2z)<=abs_chi), -jnp.log((2.*abs_chi)**2), -jnp.inf), -jnp.inf)
        
    def sample_population(self, size):
        '''
        Function to sample the spin model.
        
        :param int size: Size of the spin sample.

        :return: Sampled spin components for the two objects.
        :rtype: dict(array, array, ...)
        '''
    
        chi1z = np.random.uniform(low=max(-self.hyperpar_dict['abs_chi'], self.priorlims_dict['chi1z'][0]), high=min(self.hyperpar_dict['abs_chi'], self.priorlims_dict['chi1z'][1]), size=size)
        chi2z = np.random.uniform(low=max(-self.hyperpar_dict['abs_chi'], self.priorlims_dict['chi2z'][0]), high=min(self.hyperpar_dict['abs_chi'], self.priorlims_dict['chi2z'][1]), size=size)

        tilt1 = np.zeros_like(chi1z)
        tilt2 = np.zeros_like(chi1z)
        phiJL = np.zeros_like(chi1z)
        phi12 = np.zeros_like(chi1z)

        return {'chi1': chi1z, 'chi2': chi1z, 'tilt1': tilt1, 'tilt2': tilt2, 'phiJL': phiJL, 'phi12': phi12, 'chi1z':chi1z, 'chi2z':chi2z, 'chi1x':tilt1, 'chi2x':tilt1, 'chi1y':tilt1, 'chi2y':tilt1}
    
    def spin_function_derivative(self, chi1z, chi2z, abs_chi=None, uselog=False):
        '''
        Derivative with respect to the hyperparameters of the spin function.
        
        :param array chi1z: Spin component of the first object.
        :param array chi2z: Spin component of the second object.
        :param float, optional abs_chi: maximum absolute value of the spin.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Derivative of the spin function with respect to the hyperparameters.
        :rtype: array
        '''

        abs_chi = self.hyperpar_dict['abs_chi'] if abs_chi is None else abs_chi
        
        funder = lambda chi1z, chi2z, abs_chi: self.spin_function(chi1z, chi2z, abs_chi, uselog=uselog)
        
        derivs_all = np.squeeze(np.asarray(jacrev(funder, argnums=(2))(chi1z, chi2z, jnp.array([abs_chi]))))
        
        return derivs_all[np.newaxis,:]
    
    def spin_function_hessian(self, chi1z, chi2z, abs_chi=None, uselog=False):
        '''
        Hessians with respect to the hyperparameters of the spin function.
        
        :param array chi1z: Spin component of the first object.
        :param array chi2z: Spin component of the second object.
        :param float, optional abs_chi: maximum absolute value of the spin.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Hessians of the spin function with respect to the hyperparameters.
        :rtype: array
        '''

        abs_chi = self.hyperpar_dict['abs_chi'] if abs_chi is None else abs_chi

        funder = lambda chi1z, chi2z, abs_chi: self.spin_function(chi1z, chi2z, abs_chi, uselog=uselog)

        derivs_all = np.squeeze(np.asarray(hessian(funder, argnums=(2))(chi1z, chi2z, jnp.array([abs_chi]))))
        
        return derivs_all[np.newaxis,np.newaxis,:]

class FlatNonPrecessing_SpinDistribution(SpinDistribution):
    ''' 
    Independent Flat spin model

    Parameters:
        * min_chi: minimum value of the spin
        * max_chi: maximum value of the spin

    :param dict, optional hyperparameters: Dictionary containing the hyperparameters of the spin model as keys and their fiducial value as entry.
    :param dict, optional priorlims_parameters: Dictionary containing the parameters of the spin model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.

    ''' 

    def __init__(self, hyperparameters=None, priorlims_parameters=None):

        self.expected_hyperpars = ['min_chi', 'max_chi']
        super().__init__(is_Precessing=False)

        self.set_parameters(['chi1z', 'chi2z'])

        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)
        else:
            print('Expected hyperparameters: ', self.expected_hyperpars)
            basevalues = {'min_chi':-0.05, 'max_chi':0.05}
            print('Base values are: ', list(basevalues.items()))
            self.set_hyperparameters(basevalues)
        
        if priorlims_parameters is not None:
            self.set_priorlimits(priorlims_parameters)
        else:
            # Define the prior limits for 'm1' and 'q'
            self.set_priorlimits({'chi1z': (-1., 1.), 'chi2z': (-1., 1.)})
        
        self.derivative_par_nums = {'min_chi':0, 'max_chi':1}
    
    def spin_function(self, chi1z, chi2z, min_chi=None, max_chi=None, uselog=False):
        '''
        Spin function.
        
        :param array chi1z: Spin component of the first object.
        :param array chi2z: Spin component of the second object.
        :param float, optional min_chi: minimum value of the spin.
        :param float, optional max_chi: maximum value of the spin.
        :param bool, optional uselog: Boolean specifying whether to return the probability or log-probability, defaults to False.
        
        :return: Spin function value at the input spins.
        :rtype: array
        '''

        min_chi = self.hyperpar_dict['min_chi'] if min_chi is None else min_chi
        max_chi = self.hyperpar_dict['max_chi'] if max_chi is None else max_chi
    
        goodsamples = self._isin_prior_range('chi1z', chi1z) & self._isin_prior_range('chi2z', chi2z)

        if not uselog:
            return jnp.where(goodsamples, jnp.where((chi1z>=min_chi) & (chi1z<=max_chi) & (chi2z>=min_chi) & (chi2z<=max_chi), 1./((max_chi - min_chi)**2), 0.), 0.)
        else:
            return jnp.where(goodsamples, jnp.where((chi1z>=min_chi) & (chi1z<=max_chi) & (chi2z>=min_chi) & (chi2z<=max_chi), -jnp.log((max_chi - min_chi)**2), -jnp.inf), -jnp.inf)
        
    def sample_population(self, size):
        '''
        Function to sample the spin model.
        
        :param int size: Size of the spin sample.

        :return: Sampled spin components for the two objects.
        :rtype: dict(array, array, ...)
        '''
    
        chi1z = np.random.uniform(low=max(self.hyperpar_dict['min_chi'], self.priorlims_dict['chi1z'][0]), high=min(self.hyperpar_dict['max_chi'], self.priorlims_dict['chi1z'][1]), size=size)
        chi2z = np.random.uniform(low=max(self.hyperpar_dict['min_chi'], self.priorlims_dict['chi2z'][0]), high=min(self.hyperpar_dict['max_chi'], self.priorlims_dict['chi2z'][1]), size=size)

        tilt1 = np.zeros_like(chi1z)
        tilt2 = np.zeros_like(chi1z)
        phiJL = np.zeros_like(chi1z)
        phi12 = np.zeros_like(chi1z)

        return {'chi1': chi1z, 'chi2': chi1z, 'tilt1': tilt1, 'tilt2': tilt2, 'phiJL': phiJL, 'phi12': phi12, 'chi1z':chi1z, 'chi2z':chi2z, 'chi1x':tilt1, 'chi2x':tilt1, 'chi1y':tilt1, 'chi2y':tilt1}
    
    def spin_function_derivative(self, chi1z, chi2z, min_chi=None, max_chi=None, uselog=False):
        '''
        Derivative with respect to the hyperparameters of the spin function.
        
        :param array chi1z: Spin component of the first object.
        :param array chi2z: Spin component of the second object.
        :param float, optional min_chi: minimum value of the spin.
        :param float, optional max_chi: maximum value of the spin.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Derivative of the spin function with respect to the hyperparameters.
        :rtype: array
        '''
        
        min_chi = self.hyperpar_dict['min_chi'] if min_chi is None else min_chi
        max_chi = self.hyperpar_dict['max_chi'] if max_chi is None else max_chi
        
        funder = lambda chi1z, chi2z, min_chi, max_chi: self.spin_function(chi1z, chi2z, min_chi, max_chi, uselog=uselog)
        
        derivs_all = np.squeeze(np.asarray(jacrev(funder, argnums=(2,3))(chi1z, chi2z, jnp.array([min_chi]), jnp.array([max_chi]))))
        
        return derivs_all
    
    def spin_function_hessian(self, chi1z, chi2z, min_chi=None, max_chi=None, uselog=False):
        '''
        Hessians with respect to the hyperparameters of the spin function.
        
        :param array chi1z: Spin component of the first object.
        :param array chi2z: Spin component of the second object.
        :param float, optional min_chi: minimum value of the spin.
        :param float, optional max_chi: maximum value of the spin.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Hessians of the spin function with respect to the hyperparameters.
        :rtype: array
        '''

        min_chi = self.hyperpar_dict['min_chi'] if min_chi is None else min_chi
        max_chi = self.hyperpar_dict['max_chi'] if max_chi is None else max_chi

        funder = lambda chi1z, chi2z, min_chi, max_chi: self.spin_function(chi1z, chi2z, min_chi, max_chi, uselog=uselog)
        
        derivs_all = np.squeeze(np.asarray(hessian(funder, argnums=(2,3))(chi1z, chi2z, jnp.array([min_chi]), jnp.array([max_chi]))))
        
        return derivs_all
        
class GaussNonPrecessing_SpinDistribution(SpinDistribution):
    ''' 
    Independent truncated Gaussian spin model.

    Parameters:
        * mu_chi: mean of the truncated Gaussian distribution for the spin.
        * sigma_chi: standard deviation of the truncated Gaussian distribution for the spin.
        
    :param dict, optional hyperparameters: Dictionary containing the hyperparameters of the spin model as keys and their fiducial value as entry.
    :param dict, optional priorlims_parameters: Dictionary containing the parameters of the spin model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.

    ''' 

    def __init__(self, hyperparameters=None, priorlims_parameters=None):

        self.expected_hyperpars = ['mu_chi, sigma_chi']
        super().__init__(is_Precessing=False)

        self.set_parameters(['chi1z', 'chi2z'])

        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)
        else:
            print('Expected hyperparameters: ', self.expected_hyperpars)
            basevalues = {'mu_chi':0., 'sigma_chi':0.15}
            print('Base values are: ', list(basevalues.items()))
            self.set_hyperparameters(basevalues)
        
        if priorlims_parameters is not None:
            self.set_priorlimits(priorlims_parameters)
        else:
            # Define the prior limits for 'm1' and 'q'
            self.set_priorlimits({'chi1z': (-1., 1.), 'chi2z': (-1., 1.)})
        
        self.derivative_par_nums = {'mu_chi':0, 'sigma_chi':1}
    
    def spin_function(self, chi1z, chi2z, mu_chi=None, sigma_chi=None, uselog=False):
        '''
        Spin function.
        
        :param array chi1z: Spin component of the first object.
        :param array chi2z: Spin component of the second object.
        :param float, optional mu_chi: mean of the truncated Gaussian distribution for the spin.
        :param float, optional sigma_chi: standard deviation of the truncated Gaussian distribution for the spin.
        :param bool, optional uselog: Boolean specifying whether to return the probability or log-probability, defaults to False.
        
        :return: Spin function value at the input spins.
        :rtype: array
        '''

        mu_chi = self.hyperpar_dict['mu_chi'] if mu_chi is None else mu_chi
        sigma_chi = self.hyperpar_dict['sigma_chi'] if sigma_chi is None else sigma_chi
    
        goodsamples = self._isin_prior_range('chi1z', chi1z) & self._isin_prior_range('chi2z', chi2z)

        if not uselog:
            return jnp.where(goodsamples, utils.trunc_gaussian_norm(chi1z, mu_chi, sigma_chi, lower=self.priorlims_dict['chi1z'][0], upper=self.priorlims_dict['chi1z'][1])*utils.trunc_gaussian_norm(chi2z, mu_chi, sigma_chi, lower=self.priorlims_dict['chi2z'][0], upper=self.priorlims_dict['chi2z'][1]), 0.)
        else:
            return jnp.where(goodsamples, jnp.log(utils.trunc_gaussian_norm(chi1z, mu_chi, sigma_chi, lower=self.priorlims_dict['chi1z'][0], upper=self.priorlims_dict['chi1z'][1])) + jnp.log(utils.trunc_gaussian_norm(chi2z, mu_chi, sigma_chi, lower=self.priorlims_dict['chi2z'][0], upper=self.priorlims_dict['chi2z'][1])), -jnp.inf)
        
    def sample_population(self, size):
        '''
        Function to sample the spin model.
        
        :param int size: Size of the spin sample.

        :return: Sampled spin components for the two objects.
        :rtype: dict(array, array, ...)
        '''
    
        mu_chi = self.hyperpar_dict['mu_chi']
        sigma_chi = self.hyperpar_dict['sigma_chi']
        
        a1, b1 = (self.priorlims_dict['chi1z'][0] - mu_chi) / sigma_chi, (self.priorlims_dict['chi1z'][1] - mu_chi) / sigma_chi
        a2, b2 = (self.priorlims_dict['chi2z'][0] - mu_chi) / sigma_chi, (self.priorlims_dict['chi2z'][1] - mu_chi) / sigma_chi

        chi1z = truncnormscpy.rvs(a1, b1, loc=mu_chi, scale=sigma_chi, size=size)
        chi2z = truncnormscpy.rvs(a2, b2, loc=mu_chi, scale=sigma_chi, size=size)

        tilt1 = np.zeros_like(chi1z)
        tilt2 = np.zeros_like(chi1z)
        phiJL = np.zeros_like(chi1z)
        phi12 = np.zeros_like(chi1z)

        return {'chi1': chi1z, 'chi2': chi2z, 'tilt1': tilt1, 'tilt2': tilt2, 'phiJL': phiJL, 'phi12': phi12, 'chi1z':chi1z, 'chi2z':chi2z, 'chi1x':tilt1, 'chi2x':tilt1, 'chi1y':tilt1, 'chi2y':tilt1}
    
    def spin_function_derivative(self, chi1z, chi2z, mu_chi=None, sigma_chi=None, uselog=False):
        '''
        Derivative with respect to the hyperparameters of the spin function.
        
        :param array chi1z: Spin component of the first object.
        :param array chi2z: Spin component of the second object.
        :param float, optional mu_chi: mean of the truncated Gaussian distribution for the spin.
        :param float, optional sigma_chi: standard deviation of the truncated Gaussian distribution for the spin.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Derivative of the spin function with respect to the hyperparameters.
        :rtype: array
        '''

        mu_chi = self.hyperpar_dict['mu_chi'] if mu_chi is None else mu_chi
        sigma_chi = self.hyperpar_dict['sigma_chi'] if sigma_chi is None else sigma_chi
        
        funder = lambda chi1z, chi2z, mu_chi, sigma_chi: self.spin_function(chi1z, chi2z, mu_chi, sigma_chi, uselog=uselog)
        
        derivs_all = np.squeeze(np.asarray(jacrev(funder, argnums=(2,3))(chi1z, chi2z, jnp.array([mu_chi]), jnp.array([sigma_chi]))))
        
        return derivs_all
    
    def spin_function_hessian(self, chi1z, chi2z, mu_chi=None, sigma_chi=None, uselog=False):
        '''
        Hessians with respect to the hyperparameters of the spin function.
        
        :param array chi1z: Spin component of the first object.
        :param array chi2z: Spin component of the second object.
        :param float, optional mu_chi: mean of the truncated Gaussian distribution for the spin.
        :param float, optional sigma_chi: standard deviation of the truncated Gaussian distribution for the spin.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Hessians of the spin function with respect to the hyperparameters.
        :rtype: array
        '''

        mu_chi = self.hyperpar_dict['mu_chi'] if mu_chi is None else mu_chi
        sigma_chi = self.hyperpar_dict['sigma_chi'] if sigma_chi is None else sigma_chi

        funder = lambda chi1z, chi2z, mu_chi, sigma_chi: self.spin_function(chi1z, chi2z, mu_chi, sigma_chi, uselog=uselog)

        derivs_all = np.squeeze(np.asarray(hessian(funder, argnums=(2,3))(chi1z, chi2z, jnp.array([mu_chi]), jnp.array([sigma_chi]))))
        
        return derivs_all
