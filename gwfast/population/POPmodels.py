#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,sys,jax

jax.config.update("jax_enable_x64", True)


import jax.numpy as jnp
import numpy as np
import copy
from jax import jacrev, vmap, hessian

from abc import ABC, abstractmethod
from gwfast.population import POPutils as utils
#import POPutils as utils


class PopulationModel(ABC):
    '''
    Class to compute population models, collecting mass, spin, and redshift distributions.

    '''
    
    def __init__(self, verbose=False):
        
        self.par_list = []
        self.hyperpar_dict = {}
        self.priorlims_dict = {}
        self.verbose = verbose
        
        if 'psi' not in self.par_list:
            self.par_list.extend(['theta', 'phi', 'thetaJN', 'psi', 'tcoal', 'Phicoal'])
                
        if 'psi' not in self.priorlims_dict:
        # Shortcut to avoid having to specify the prior limits for the angles in each model
            self.priorlims_dict.update({'theta':(0.,np.pi), 'phi':(0.,2.*np.pi), 'iota':(0.,np.pi), 'thetaJN':(0.,np.pi), 'psi':(0.,np.pi), 'tcoal':(0.,1.), 'Phicoal':(0.,2.*np.pi)})
    
    @abstractmethod
    def update_hyperparameters(self, new_hyperparameters):
        pass

    @abstractmethod
    def update_priorlimits(self, new_limits):
        pass

    @abstractmethod
    def sample_population(self, size):
        pass
    
    @abstractmethod
    def pop_function(self, events, uselog=False):
        pass

    @abstractmethod
    def pop_function_derivative(self, events, uselog=False):
        pass

    @abstractmethod
    def pop_function_hessian(self, events, derivs=None, uselog=False):
        pass

    def _isin_prior_range(self, par,  val):
        '''
        Function to check if a value is in the prior range of a parameter.

        :param str par: Parameter name.
        :param float val: Parameter value.
        :return: Boolean value.
        :rtype: bool
        '''

        return (val >= self.priorlims_dict[par][0]) & (val <= self.priorlims_dict[par][1])
    
    def angle_distribution(self, theta, phi, thetaJN, psi, tcoal, Phicoal, uselog=False, **kwargs):
        '''
        Angle distribution of the population model. The adopted sky position angles are :math:`\\theta = \pi/2-{\\rm dec}` and :math:`\phi = {\\rm ra}`.

        :return: Angle distribution value.
        :rtype: float
        '''

        goodsamples = self._isin_prior_range('theta', theta) & self._isin_prior_range('phi', phi) & self._isin_prior_range('thetaJN', thetaJN) & self._isin_prior_range('psi', psi) & self._isin_prior_range('tcoal', tcoal) & self._isin_prior_range('Phicoal', Phicoal)

        costhetamin, costhetamax = jnp.cos(self.priorlims_dict['theta'][0]), jnp.cos(self.priorlims_dict['theta'][1])
        costhetaJNmin, costhetaJNmax = jnp.cos(self.priorlims_dict['thetaJN'][0]), jnp.cos(self.priorlims_dict['thetaJN'][1])

        if not uselog:
            #return jnp.where(goodsamples, (jnp.sin(theta)*0.5)*(0.5/np.pi)*(jnp.sin(thetaJN)*0.5)*(1./np.pi)*(0.5/np.pi), 0.)
            return jnp.where(goodsamples, (jnp.sin(theta)/(costhetamin - costhetamax))*(1./(self.priorlims_dict['phi'][1] - self.priorlims_dict['phi'][0]))*(jnp.sin(thetaJN)/(costhetaJNmin - costhetaJNmax))*(1./(self.priorlims_dict['psi'][1] - self.priorlims_dict['psi'][0]))*(1./(self.priorlims_dict['tcoal'][1] - self.priorlims_dict['tcoal'][0]))*(1./(self.priorlims_dict['Phicoal'][1] - self.priorlims_dict['Phicoal'][0])), 0.)
        else:
            #return jnp.where(goodsamples, jnp.log(jnp.sin(theta)*0.5) + jnp.log(0.5/np.pi) + jnp.log(jnp.sin(thetaJN)*0.5) + jnp.log(1./np.pi) + jnp.log(0.5/np.pi), -jnp.inf)
            return jnp.where(goodsamples, jnp.log(jnp.sin(theta)/(costhetamin - costhetamax)) + jnp.log(1./(self.priorlims_dict['phi'][1] - self.priorlims_dict['phi'][0])) + jnp.log(jnp.sin(thetaJN)/(costhetaJNmin - costhetaJNmax)) + jnp.log(1./(self.priorlims_dict['psi'][1] - self.priorlims_dict['psi'][0])) + jnp.log(1./(self.priorlims_dict['tcoal'][1] - self.priorlims_dict['tcoal'][0])) + jnp.log(1./(self.priorlims_dict['Phicoal'][1] - self.priorlims_dict['Phicoal'][0])), -jnp.inf)
    
    def _sample_angles(self, size, is_Precessing=False):
        '''
        Function to sample the angles of the population model.

        :return: Sampled angles.
        :rtype: dict
        '''
        costhetamin, costhetamax = jnp.cos(self.priorlims_dict['theta'][0]), jnp.cos(self.priorlims_dict['theta'][1])
        costhetaJNmin, costhetaJNmax = jnp.cos(self.priorlims_dict['thetaJN'][0]), jnp.cos(self.priorlims_dict['thetaJN'][1])

        theta   = np.arccos(np.random.uniform(costhetamax, costhetamin, size=size))
        phi     = np.random.uniform(self.priorlims_dict['phi'][0], self.priorlims_dict['phi'][1], size=size)
        thetaJN = np.arccos(np.random.uniform(costhetaJNmax, costhetaJNmin, size=size))
        psi     = np.random.uniform(self.priorlims_dict['psi'][0], self.priorlims_dict['psi'][1], size=size)
        tcoal   = np.random.uniform(self.priorlims_dict['tcoal'][0], self.priorlims_dict['tcoal'][1], size=size) # in days
        Phicoal = np.random.uniform(self.priorlims_dict['Phicoal'][0], self.priorlims_dict['Phicoal'][1], size=size)
        
        if is_Precessing:
            return {'theta': theta, 'phi': phi, 'thetaJN': thetaJN, 'psi': psi, 'tcoal': tcoal, 'Phicoal': Phicoal}
        else:
            return {'theta': theta, 'phi': phi, 'thetaJN': thetaJN, 'psi': psi, 'tcoal': tcoal, 'Phicoal': Phicoal, 'iota':thetaJN}
            
    @abstractmethod
    def N_per_yr(self,):
        pass
    
    def set_EoS(self, EoS):
        
        self.EoS = EoS
    
    def get_Lambda(self, m1_src, m2_src, EoS='rand'):
        '''
        Method to get the individual quadrupolar tidal deformability from the source masses.

        :param array or float m1_src: Source-frame primary mass(es).
        :param array or float m2_src: Source-frame secondary mass(es).
        :param array or str EoS: Either a path to a file containing the EoS, or 'rand' to randomly sample the tidal deformabilities, or 2-D array. Both the file and the array must have the first column containing the mass (in solar masses) and the second column containing the tidal deformability.

        :return: Dictionary containing the individual tidal deformabilities of the objects.
        :rtype: dict(array, array)
        '''
        
        if self.object_type == 'BBH':
            return {'Lambda1':np.zeros_like(m1_src), 'Lambda2':np.zeros_like(m2_src)}
        
        elif (self.object_type == 'BNS') | (self.object_type == 'NSBH'):
            
            if EoS == 'rand':
                Lam1 = np.random.uniform(0., 5000., size=len(m1_src))
                Lam2 = np.random.uniform(0., 5000., size=len(m1_src))
            else:
                if isinstance(EoS, np.ndarray):
                    if EoS.ndim==2:
                        EoS_use = EoS
                    else:
                        raise ValueError('EoS array must be either a 2D, or a path to a file containing the EoS, or rand.')
                elif isinstance(EoS, str):
                    try:
                        EoS_use = np.loadtxt(EoS)
                    except:
                        raise ValueError('EoS array must be either a 2D, or a path to a file containing the EoS, or rand.')
                else:
                    raise ValueError('EoS array must be either a 2D, or a path to a file containing the EoS, or rand.')
                
                MTOV = np.amax(EoS_use[:, 0])
                
                if (self.object_type == 'BNS') & ((np.any(m1_src > MTOV)) | (np.any(m2_src > MTOV))):
                    if self.verbose:
                        print('Some masses exceed the maximum mass of the EoS! They will be given zero tidal deformability.')
                elif (self.object_type == 'NSBH') & (np.any(m2_src > MTOV)):
                    if self.verbose:
                        print('Some masses exceed the maximum mass of the EoS! They will be given zero tidal deformability.')
                # Cut the EoS in case there is a non-physical turnaround
                maxIdx = np.argmax(EoS_use[:,0])
                EoS_use = EoS_use[:maxIdx,:]

                Lam1 = np.exp(np.interp(m1_src, EoS_use[:,0], np.log(EoS_use[:,1]), right=-jnp.inf))
                Lam2 = np.exp(np.interp(m2_src, EoS_use[:,0], np.log(EoS_use[:,1]), right=-jnp.inf))

            if self.object_type == 'BNS':
                return {'Lambda1':Lam1, 'Lambda2':Lam2}
            else:
                return {'Lambda1':np.zeros_like(m1_src), 'Lambda2':Lam2}

class MassSpinRedshiftIndependent_PopulationModel(PopulationModel):
    '''
    Class to compute population models in which the mass, spin and redshift distributions are independent.

    :param list parameters: List containing the parameters of the population model.
    :param dict hyperparameters: Dictionary containing the hyperparameters of the population model as keys and their fiducial value as entry.
    :param dict priorlims_parameters: Dictionary containing the parameters of the population model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.
    :param dict kwargs: Keyword arguments passed to the constructor of the population model.
    '''
    
    def __init__(self, mass_function, rate_function, spin_function, verbose=False):
        '''
        Constructor method
        '''

        super().__init__(verbose=verbose)

        self.mass_function = mass_function
        self.rate_function = rate_function
        self.spin_function = spin_function

        self.par_list.extend([x for x in self.mass_function.par_list if x not in self.par_list])
        self.par_list.extend([x for x in self.rate_function.par_list if x not in self.par_list])
        self.par_list.extend([x for x in self.spin_function.par_list if x not in self.par_list])
        
        self.hyperpar_dict.update(self.mass_function.hyperpar_dict)
        self.hyperpar_dict.update(self.rate_function.hyperpar_dict)
        self.hyperpar_dict.update(self.spin_function.hyperpar_dict)
        
        self.priorlims_dict.update(self.mass_function.priorlims_dict)
        self.priorlims_dict.update(self.rate_function.priorlims_dict)
        self.priorlims_dict.update(self.spin_function.priorlims_dict)
        
        self.derivative_par_nums = copy.deepcopy(self.mass_function.derivative_par_nums)
        self.derivative_par_nums_mass = copy.deepcopy(self.mass_function.derivative_par_nums)
        tmppnums = copy.deepcopy(self.rate_function.derivative_par_nums)
        for key in tmppnums.keys():
            tmppnums[key] += len(list(self.derivative_par_nums.keys()))
        self.derivative_par_nums.update(tmppnums)
        self.derivative_par_nums_rate = tmppnums
        tmppnums = copy.deepcopy(self.spin_function.derivative_par_nums)
        for key in tmppnums.keys():
            tmppnums[key] += len(list(self.derivative_par_nums.keys()))
        self.derivative_par_nums.update(tmppnums)
        self.derivative_par_nums_spin = tmppnums
        
        self.object_type = self.mass_function.object_type
        
        if (self.object_type=='BNS') | (self.object_type=='NSBH'):
            if self.verbose:
                print('It is possible to set the EoS using the set_EoS method.')
            self.set_EoS('rand')
        
        self.angular_pars = ['theta', 'phi', 'thetaJN', 'psi', 'tcoal', 'Phicoal']
    
    def update_hyperparameters(self, new_hyperparameters):
        '''
        Method to update the hyperparameters of the population model.

        :param dict new_hyperparameters: Dictionary containing the new hyperparameters of the population model as keys and their new value as entry.
        '''
        for key in new_hyperparameters.keys():
            if key in self.mass_function.hyperpar_dict.keys():
                self.mass_function.hyperpar_dict[key] = new_hyperparameters[key]
            elif key in self.rate_function.hyperpar_dict.keys():
                self.rate_function.hyperpar_dict[key] = new_hyperparameters[key]
            elif key in self.spin_function.hyperpar_dict.keys():
                self.spin_function.hyperpar_dict[key] = new_hyperparameters[key]
            else:
                raise ValueError('The hyperparameter '+key+' is not present in the hyperparameter dictionary.')
        
        self.hyperpar_dict.update(self.mass_function.hyperpar_dict)
        self.hyperpar_dict.update(self.rate_function.hyperpar_dict)
        self.hyperpar_dict.update(self.spin_function.hyperpar_dict)
    
    def update_priorlimits(self, new_limits):
        '''
        Method to update the prior limits of the population model.

        :param dict new_limits: Dictionary containing the new prior limits of the population model as keys and their new value as entry.
        '''
        for key in new_limits.keys():
            if key in self.mass_function.priorlims_dict.keys():
                self.mass_function.priorlims_dict[key] = new_limits[key]
            elif key in self.rate_function.priorlims_dict.keys():
                self.rate_function.priorlims_dict[key] = new_limits[key]
            elif key in self.spin_function.priorlims_dict.keys():
                self.spin_function.priorlims_dict[key] = new_limits[key]
            elif key in self.angular_pars:
                self.priorlims_dict[key] = new_limits[key]
            else:
                raise ValueError('The parameter '+key+' is not present in the prior limits dictionary.')
        
        self.priorlims_dict.update(self.mass_function.priorlims_dict)
        self.priorlims_dict.update(self.rate_function.priorlims_dict)
        self.priorlims_dict.update(self.spin_function.priorlims_dict)

    def sample_population(self, size):
        '''
        Function to sample the population model.
        
        :param int size: Size of the population sample.

        :return: Sampled population.
        :rtype: dict(array, array, ...)
        '''

        res = {}

        angles = self._sample_angles(size, is_Precessing=self.spin_function.is_Precessing)
        masses = self.mass_function.sample_population(size)
        spins  = self.spin_function.sample_population(size)
        z      = self.rate_function.sample_population(size)

        res.update(angles)
        res.update(masses)
        res.update(spins)
        res.update(z)
        
        utils.check_masses(res)
                
        if (self.object_type=='BNS') | (self.object_type=='NSBH'):
            res.update(self.get_Lambda(res['m1_src'], res['m2_src'], EoS=self.EoS))
        
        return res
    
    def N_per_yr(self, R0=1.):
        '''
        Compute the number of events per year for the chosen distribution and parameters.
        
        :param float R0: Normalization of the local merger rate, in :math:`{\\rm Gpc}^{-3}\,{\\rm yr}^{-1}`.
        
        :return: Number of events per year.
        :rtype: float
        '''
    
        return self.rate_function.N_per_yr(R0=R0)
    
    def pop_function(self, events, uselog=False):
        '''
        Population function pdf of the model.

        :param dict(array, array, ...) events: Dictionary containing the events parameters.
        :param bool, optional uselog: Boolean specifying whether to return the probability or log-probability, defaults to False.
        
        :return: Population function value.
        :rtype: array or float
        '''

        parsmass = {key:events[key] for key in self.mass_function.par_list}
        parsspin = {key:events[key] for key in self.spin_function.par_list}

        mass_value = self.mass_function.mass_function(uselog=uselog, **parsmass)
        rate_value = self.rate_function.rate_function(events['z'], uselog=uselog)
        spin_value = self.spin_function.spin_function(uselog=uselog, **parsspin)
        ang_value  = self.angle_distribution(uselog=uselog, **events)

        if not uselog:
            return mass_value*rate_value*spin_value*ang_value
        else:
            return mass_value + rate_value + spin_value + ang_value

    
    def pop_function_derivative(self, events, uselog=False):
        '''
        Derivative with respect to the hyperparameters of the population function pdf of the model.

        :param dict(array, array, ...) events: Dictionary containing the events parameters.
        :param bool uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the derivatives of the population function.
        :rtype: array 
        '''
    
        parsmass = {key:events[key] for key in self.mass_function.par_list}
        parsspin = {key:events[key] for key in self.spin_function.par_list}
                
        if not uselog:
            mass_value = self.mass_function.mass_function(uselog=uselog, **parsmass)
            rate_value = self.rate_function.rate_function(events['z'], uselog=uselog)
            spin_value = self.spin_function.spin_function(uselog=uselog, **parsspin)
            ang_value  = self.angle_distribution(uselog=uselog, **events)

            mass_der = self.mass_function.mass_function_derivative(**parsmass)*ang_value
            z_der    = self.rate_function.rate_function_derivative(events['z'])*ang_value
            spin_der = self.spin_function.spin_function_derivative(**parsspin)*ang_value
            
            return np.vstack((mass_der*rate_value*spin_value, z_der*mass_value*spin_value, spin_der*mass_value*rate_value))
        else:
            
            mass_der = self.mass_function.mass_function_derivative(uselog=True, **parsmass) 
            z_der    = self.rate_function.rate_function_derivative(events['z'], uselog=True) 
            spin_der = self.spin_function.spin_function_derivative(uselog=True, **parsspin)
            
            return np.vstack((mass_der, z_der, spin_der))
    
    def pop_function_hessian(self, events, derivs=None, uselog=False):
        '''
        Hessian with respect to the hyperparameters of the population function pdf of the model.

        :param dict(array, array, ...) events: Dictionary containing the events parameters.
        :param array, optional derivs: Array containing the derivatives of the population function.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the Hessians of the population function.
        :rtype: array 
        '''
    
        if (derivs is None) & (not uselog):
            derivs = self.pop_function_derivative(events, uselog=uselog)
            
        parsmass = {key:events[key] for key in self.mass_function.par_list}
        parsspin = {key:events[key] for key in self.spin_function.par_list}
        
        Nhyperpar = len(self.hyperpar_dict.keys())
        outmatr = np.zeros((Nhyperpar, Nhyperpar, len(events['z'])))
        
        mhp = len(list(self.mass_function.hyperpar_dict.keys()))
        rhp = len(list(self.rate_function.hyperpar_dict.keys()))
        shp = len(list(self.spin_function.hyperpar_dict.keys()))
        
        if not uselog:
            mass_value = self.mass_function.mass_function(uselog=uselog, **parsmass)
            rate_value = self.rate_function.rate_function(events['z'], uselog=uselog)
            spin_value = self.spin_function.spin_function(uselog=uselog, **parsspin)
            ang_value  = self.angle_distribution(uselog=uselog, **events)
            pop_value  = mass_value*rate_value*spin_value*ang_value
            
            mass_hess = self.mass_function.mass_function_hessian(**parsmass)*rate_value*spin_value*ang_value
            z_hess    = self.rate_function.rate_function_hessian(events['z'])*mass_value*spin_value*ang_value
            spin_hess = self.spin_function.spin_function_hessian(**parsspin)*mass_value*rate_value*ang_value
            
        else:
            mass_hess = self.mass_function.mass_function_hessian(uselog=True, **parsmass)
            z_hess    = self.rate_function.rate_function_hessian(events['z'], uselog=True)
            spin_hess = self.spin_function.spin_function_hessian(uselog=True, **parsspin)

        if not uselog:
            for i in range(Nhyperpar):
                for j in range(Nhyperpar):
                    if (i in self.derivative_par_nums_mass.values()) & (j in self.derivative_par_nums_mass.values()):
                        outmatr[i,j,:] = mass_hess[i,j,:]
                    elif (i in self.derivative_par_nums_rate.values()) & (j in self.derivative_par_nums_rate.values()):
                        outmatr[i,j,:] = z_hess[i-mhp,j-mhp,:]
                    elif (i in self.derivative_par_nums_spin.values()) & (j in self.derivative_par_nums_spin.values()):
                        outmatr[i,j,:] = spin_hess[i-mhp-rhp,j-mhp-rhp,:]
                    elif ((i in self.derivative_par_nums_mass.values()) & (j in self.derivative_par_nums_rate.values())) | ((i in self.derivative_par_nums_mass.values()) & (j in self.derivative_par_nums_spin.values())) | ((i in self.derivative_par_nums_rate.values()) & (j in self.derivative_par_nums_spin.values())):
                        outmatr[i,j,:] = outmatr[j,i,:] = derivs[i,:]*derivs[j,:]/pop_value
        else:
            outmatr[np.ix_(list(range(mhp)),list(range(mhp)))] = mass_hess
            outmatr[np.ix_(list(range(mhp, mhp+rhp)), list(range(mhp, mhp+rhp)))] = z_hess
            outmatr[np.ix_(list(range(mhp+rhp, mhp+rhp+shp)), list(range(mhp+rhp, mhp+rhp+shp)))] = spin_hess
        
        return outmatr
        
    def pop_function_hessian_termII(self, events, FIMs, ParNums, ParPrior=None):
        '''
        Function to compute the matrices appearing in the second term of the population Fisher.

        :param dict events: Dictionary containing the events parameters.
        :param array FIMs: Array containing the Fisher matrices of the single events.
        :param dict ParNums: Dictionary containing the parameter names as keys and their position in the Fisher matrix as entry.
        :param dict ParPrior: Dictionary containing the names of the parameters on which a prior in the single-event FIM has to be added as keys and the prior value as entry.

        :return: Array containing the matrices appearing in the second term of the population Fisher.
        :rtype: array
        '''

        ParNums = {k: v for k, v in sorted(ParNums.items(), key=lambda item: item[1])}
        
        tmp_par_list = []
        tmp_par_list.extend([x for x in self.mass_function.par_list if x not in tmp_par_list])
        tmp_par_list.extend([x for x in self.rate_function.par_list if x not in tmp_par_list])
        tmp_par_list.extend([x for x in self.spin_function.par_list if x not in tmp_par_list])

        if ParPrior is not None:
            diag = np.array([ParPrior[key] if key in ParPrior.keys() else 0. for key in ParNums.keys()])
            #pp   = np.eye(FIM_use.shape[0])*diag
            pp   = np.eye(FIMs.shape[0])*diag
            if FIMs.ndim==2:
                FIMs = pp + FIMs
            else:
                FIMs = pp[:,:,np.newaxis] + FIMs

        FIM_nums = [ParNums[par] for par in tmp_par_list]
        FIM_use = FIMs[np.ix_(FIM_nums, FIM_nums)]
        
        FIM_use = FIM_use.T

        parsmass = {key:events[key] for key in self.mass_function.par_list}
        parsspin = {key:events[key] for key in self.spin_function.par_list}

        hyperpars_m = [self.mass_function.hyperpar_dict[key] for key in self.mass_function.hyperpar_dict.keys()]
        hyperpars_r = [self.rate_function.hyperpar_dict[key] for key in self.rate_function.hyperpar_dict.keys()]
        hyperpars_s = [self.spin_function.hyperpar_dict[key] for key in self.spin_function.hyperpar_dict.keys()]

        Nhyperpar_m = len(self.mass_function.hyperpar_dict.keys())
        Nhyperpar_r = len(self.rate_function.hyperpar_dict.keys())
        Nhyperpar_s = len(self.spin_function.hyperpar_dict.keys())

        if len(self.spin_function.par_list) == 2:
            if Nhyperpar_m == 3:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1, pars2: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 4:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1, pars2: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 5:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 6:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 9:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 11:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            else:
                raise ValueError('The number of hyperparameters for the mass function is not supported.')
        elif len(self.spin_function.par_list) == 6:
            if Nhyperpar_m == 3:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, pars1, pars2, pars3, pars4: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2, pars3, pars4: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 4:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, pars1, pars2, pars3, pars4: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2, pars3, pars4: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 5:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2, pars3, pars4: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2, pars3, pars4: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 6:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2, pars3, pars4: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2, pars3, pars4: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 9:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2, pars3, pars4: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2, pars3, pars4: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 11:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2, pars3, pars4: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2, pars3, pars4: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            else:
                raise ValueError('The number of hyperparameters for the mass function is not supported.')
            
        return derivs_all
    
    def pop_function_hessian_termIII(self, events, FIMs, ParNums, ParPrior=None):
        '''
        Function to compute the matrices appearing in the third term of the population Fisher.

        :param dict events: Dictionary containing the events parameters.
        :param array FIMs: Array containing the Fisher matrices of the single events.
        :param dict ParNums: Dictionary containing the parameter names as keys and their position in the Fisher matrix as entry.
        :param dict ParPrior: Dictionary containing the names of the parameters on which a prior in the single-event FIM has to be added as keys and the prior value as entry.

        :return: Array containing the matrices appearing in the third term of the population Fisher.
        :rtype: array
        '''

        ParNums = {k: v for k, v in sorted(ParNums.items(), key=lambda item: item[1])}
        
        tmp_par_list = []
        tmp_par_list.extend([x for x in self.mass_function.par_list if x not in tmp_par_list])
        tmp_par_list.extend([x for x in self.rate_function.par_list if x not in tmp_par_list])
        tmp_par_list.extend([x for x in self.spin_function.par_list if x not in tmp_par_list])

        if ParPrior is not None:
            diag = np.array([ParPrior[key] if key in ParPrior.keys() else 0. for key in ParNums.keys()])
            #pp   = np.eye(FIM_use.shape[0])*diag
            pp   = np.eye(FIMs.shape[0])*diag
            if FIMs.ndim==2:
                FIMs = pp + FIMs
            else:
                FIMs = pp[:,:,np.newaxis] + FIMs

        FIM_nums = [ParNums[par] for par in tmp_par_list]
        FIM_use = FIMs[np.ix_(FIM_nums, FIM_nums)]
        
        FIM_use = FIM_use.T

        parsmass = {key:events[key] for key in self.mass_function.par_list}
        parsspin = {key:events[key] for key in self.spin_function.par_list}

        hyperpars_m = [self.mass_function.hyperpar_dict[key] for key in self.mass_function.hyperpar_dict.keys()]
        hyperpars_r = [self.rate_function.hyperpar_dict[key] for key in self.rate_function.hyperpar_dict.keys()]
        hyperpars_s = [self.spin_function.hyperpar_dict[key] for key in self.spin_function.hyperpar_dict.keys()]

        Nhyperpar_m = len(self.mass_function.hyperpar_dict.keys())
        Nhyperpar_r = len(self.rate_function.hyperpar_dict.keys())
        Nhyperpar_s = len(self.spin_function.hyperpar_dict.keys())

        if len(self.spin_function.par_list) == 2:
            if Nhyperpar_m == 3:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_s[0]))), FIM_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1, pars2: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1, pars2)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0]))), FIM_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0]))), FIM_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 4:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_s[0]))), FIM_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1, pars2: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1, pars2)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0]))), FIM_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0]))), FIM_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 5:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_s[0]))), FIM_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0]))), FIM_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0]))), FIM_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 6:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_s[0]))), FIM_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0]))), FIM_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0]))), FIM_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 9:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_s[0]))), FIM_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0]))), FIM_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0]))), FIM_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 11:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_s[0]))), FIM_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0]))), FIM_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0]))), FIM_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            else:
                raise ValueError('The number of hyperparameters for the mass function is not supported.')
        elif len(self.spin_function.par_list) == 6:
            if Nhyperpar_m == 3:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, pars1, pars2, pars3, pars4: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2, pars3, pars4: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 4:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, pars1, pars2, pars3, pars4: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2, pars3, pars4: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 5:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2, pars3, pars4: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2, pars3, pars4: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 6:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2, pars3, pars4: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2, pars3, pars4: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 9:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2, pars3, pars4: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2, pars3, pars4: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 11:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2, pars3, pars4: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2, pars3, pars4: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), FIM_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            else:
                raise ValueError('The number of hyperparameters for the mass function is not supported.')
            
        return derivs_all
    
    def pop_function_hessian_termIV(self, events, FIMs, Pdet_ders, ParNums, ParPrior=None):
        '''
        Function to compute the matrices appearing in the fourth term of the population Fisher.

        :param dict events: Dictionary containing the events parameters.
        :param array FIMs: Array containing the Fisher matrices of the single events.
        :param array Pdet_ders: Array containing the derivatives of the detection probability with respect to the parameters, ordered as the Fisher matrices of the single events.
        :param dict ParNums: Dictionary containing the parameter names as keys and their position in the Fisher matrix as entry.
        :param dict ParPrior: Dictionary containing the names of the parameters on which a prior in the single-event FIM has to be added as keys and the prior value as entry.

        :return: Array containing the matrices appearing in the fourth term of the population Fisher.
        :rtype: array
        '''

        ParNums = {k: v for k, v in sorted(ParNums.items(), key=lambda item: item[1])}
        
        tmp_par_list = []
        tmp_par_list.extend([x for x in self.mass_function.par_list if x not in tmp_par_list])
        tmp_par_list.extend([x for x in self.rate_function.par_list if x not in tmp_par_list])
        tmp_par_list.extend([x for x in self.spin_function.par_list if x not in tmp_par_list])

        if ParPrior is not None:
            diag = np.array([ParPrior[key] if key in ParPrior.keys() else 0. for key in ParNums.keys()])
            #pp   = np.eye(FIM_use.shape[0])*diag
            pp   = np.eye(FIMs.shape[0])*diag
            if FIMs.ndim==2:
                FIMs = pp + FIMs
            else:
                FIMs = pp[:,:,np.newaxis] + FIMs
        FIM_nums = [ParNums[par] for par in tmp_par_list]
        FIM_use = FIMs[np.ix_(FIM_nums, FIM_nums)]
        Pdet_ders_use = Pdet_ders[FIM_nums].T
        FIM_use = FIM_use.T

        parsmass = {key:events[key] for key in self.mass_function.par_list}
        parsspin = {key:events[key] for key in self.spin_function.par_list}

        hyperpars_m = [self.mass_function.hyperpar_dict[key] for key in self.mass_function.hyperpar_dict.keys()]
        hyperpars_r = [self.rate_function.hyperpar_dict[key] for key in self.rate_function.hyperpar_dict.keys()]
        hyperpars_s = [self.spin_function.hyperpar_dict[key] for key in self.spin_function.hyperpar_dict.keys()]

        Nhyperpar_m = len(self.mass_function.hyperpar_dict.keys())
        Nhyperpar_r = len(self.rate_function.hyperpar_dict.keys())
        Nhyperpar_s = len(self.spin_function.hyperpar_dict.keys())

        if len(self.spin_function.par_list) == 2:
            if Nhyperpar_m == 3:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_s[0]))), Pdet_ders_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1, pars2: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1, pars2)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0]))), Pdet_ders_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0]))), Pdet_ders_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 4:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_s[0]))), Pdet_ders_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1, pars2: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1, pars2)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0]))), Pdet_ders_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0]))), Pdet_ders_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 5:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_s[0]))), Pdet_ders_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0]))), Pdet_ders_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0]))), Pdet_ders_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 6:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_s[0]))), Pdet_ders_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0]))), Pdet_ders_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0]))), Pdet_ders_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 9:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_s[0]))), Pdet_ders_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0]))), Pdet_ders_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0]))), Pdet_ders_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 11:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_s[0]))), Pdet_ders_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0]))), Pdet_ders_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0]))), Pdet_ders_use)
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            else:
                raise ValueError('The number of hyperparameters for the mass function is not supported.')
        elif len(self.spin_function.par_list) == 6:
            if Nhyperpar_m == 3:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, pars1, pars2, pars3, pars4: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, pars1, pars2, pars3, pars4)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2, pars3, pars4: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 4:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, pars1, pars2, pars3, pars4: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, pars1, pars2, pars3, pars4)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2, pars3, pars4: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 5:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2, pars3, pars4: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2, pars3, pars4)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2, pars3, pars4: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 6:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2, pars3, pars4: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2, pars3, pars4)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2, pars3, pars4: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 9:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2, pars3, pars4: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2, pars3, pars4)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2, pars3, pars4: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 11:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2, pars3, pars4: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2, pars3, pars4)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2, pars3, pars4: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T))
                        derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3]))), Pdet_ders_use)
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            else:
                raise ValueError('The number of hyperparameters for the mass function is not supported.')
            
        return derivs_all
    
    def pop_function_hessian_termV(self, events, FIMs, ParNums, ParPrior=None):
        '''
        Function to compute the matrices appearing in the fifth term of the population Fisher.

        :param dict events: Dictionary containing the events parameters.
        :param array FIMs: Array containing the Fisher matrices of the single events.
        :param dict ParNums: Dictionary containing the parameter names as keys and their position in the Fisher matrix as entry.
        :param dict ParPrior: Dictionary containing the names of the parameters on which a prior in the single-event FIM has to be added as keys and the prior value as entry.

        :return: Array containing the matrices appearing in the fifth term of the population Fisher.
        :rtype: array
        '''
        
        ParNums = {k: v for k, v in sorted(ParNums.items(), key=lambda item: item[1])}

        tmp_par_list = []
        tmp_par_list.extend([x for x in self.mass_function.par_list if x not in tmp_par_list])
        tmp_par_list.extend([x for x in self.rate_function.par_list if x not in tmp_par_list])
        tmp_par_list.extend([x for x in self.spin_function.par_list if x not in tmp_par_list])

        if ParPrior is not None:
            diag = np.array([ParPrior[key] if key in ParPrior.keys() else 0. for key in ParNums.keys()])
            #pp   = np.eye(FIM_use.shape[0])*diag
            pp   = np.eye(FIMs.shape[0])*diag
            if FIMs.ndim==2:
                FIMs = pp + FIMs
            else:
                FIMs = pp[:,:,np.newaxis] + FIMs

        FIM_nums = [ParNums[par] for par in tmp_par_list]
        FIM_use = FIMs[np.ix_(FIM_nums, FIM_nums)]
        FIM_use = FIM_use.T

        parsmass = {key:events[key] for key in self.mass_function.par_list}
        parsspin = {key:events[key] for key in self.spin_function.par_list}

        hyperpars_m = [self.mass_function.hyperpar_dict[key] for key in self.mass_function.hyperpar_dict.keys()]
        hyperpars_r = [self.rate_function.hyperpar_dict[key] for key in self.rate_function.hyperpar_dict.keys()]
        hyperpars_s = [self.spin_function.hyperpar_dict[key] for key in self.spin_function.hyperpar_dict.keys()]

        Nhyperpar_m = len(self.mass_function.hyperpar_dict.keys())
        Nhyperpar_r = len(self.rate_function.hyperpar_dict.keys())
        Nhyperpar_s = len(self.spin_function.hyperpar_dict.keys())

        if len(self.spin_function.par_list) == 2:
            if Nhyperpar_m == 3:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1, pars2: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1, pars2)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 4:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1, pars2: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1, pars2)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 5:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 6:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 9:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 11:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 1:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0])))
                    elif Nhyperpar_s == 2:
                        funder = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, pars1, pars2, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4)), in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1z'], parsspin['chi2z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            else:
                raise ValueError('The number of hyperparameters for the mass function is not supported.')
        elif len(self.spin_function.par_list) == 6:
            if Nhyperpar_m == 3:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, pars1, pars2, pars3, pars4: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, pars1, pars2, pars3, pars4)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2, pars3, pars4: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 4:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, pars1, pars2, pars3, pars4: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, pars1, pars2, pars3, pars4)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2, pars3, pars4: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 5:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2, pars3, pars4: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2, pars3, pars4)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2, pars3, pars4: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 6:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2, pars3, pars4: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2, pars3, pars4)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2, pars3, pars4: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 9:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2, pars3, pars4: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2, pars3, pars4)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2, pars3, pars4: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            elif Nhyperpar_m == 11:
                if Nhyperpar_r == 1:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2, pars3, pars4: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2, pars3, pars4)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 3:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2, pars3, pars4: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                elif Nhyperpar_r == 5:
                    if Nhyperpar_s == 4:
                        funder = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True) + self.spin_function.spin_function(chi1, chi2, tilt1, tilt2, phiJL, phi12, pars1, pars2, pars3, pars4, uselog=True)
                        parshess = lambda m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2,3,4,5,6,7,8)), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, chi1, chi2, tilt1, tilt2, phiJL, phi12, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5, pars1, pars2, pars3, pars4)).T)
                        derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], parsspin['chi1'], parsspin['chi2'], parsspin['tilt1'], parsspin['tilt2'], parsspin['phiJL'], parsspin['phi12'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4], hyperpars_s[0], hyperpars_s[1], hyperpars_s[2], hyperpars_s[3])))
                    else:
                        raise ValueError('The number of hyperparameters for the spin function is not supported.')
                else:
                    raise ValueError('The number of hyperparameters for the rate function is not supported.')
            else:
                raise ValueError('The number of hyperparameters for the mass function is not supported.')
            
        return derivs_all

class MassRedshiftIndependent_PopulationModel(PopulationModel):
    '''
    Class to compute population models in which the mass and redshift distributions are independent.

    :param list parameters: List containing the parameters of the population model.
    :param dict hyperparameters: Dictionary containing the hyperparameters of the population model as keys and their fiducial value as entry.
    :param dict priorlims_parameters: Dictionary containing the parameters of the population model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.
    :param dict kwargs: Keyword arguments passed to the constructor of the population model.
    '''
    
    def __init__(self, mass_function, rate_function, verbose=False):
        '''
        Constructor method
        '''

        super().__init__(verbose=verbose)

        self.mass_function = mass_function
        self.rate_function = rate_function

        self.par_list.extend([x for x in self.mass_function.par_list if x not in self.par_list])
        self.par_list.extend([x for x in self.rate_function.par_list if x not in self.par_list])
        
        self.hyperpar_dict.update(self.mass_function.hyperpar_dict)
        self.hyperpar_dict.update(self.rate_function.hyperpar_dict)
        
        self.priorlims_dict.update(self.mass_function.priorlims_dict)
        self.priorlims_dict.update(self.rate_function.priorlims_dict)
        
        self.derivative_par_nums = copy.deepcopy(self.mass_function.derivative_par_nums)
        self.derivative_par_nums_mass = copy.deepcopy(self.mass_function.derivative_par_nums)
        tmppnums = copy.deepcopy(self.rate_function.derivative_par_nums)
        for key in tmppnums.keys():
            tmppnums[key] += len(list(self.derivative_par_nums.keys()))
        self.derivative_par_nums.update(tmppnums)
        self.derivative_par_nums_rate = tmppnums
        
        self.object_type = self.mass_function.object_type
        
        if (self.object_type=='BNS') | (self.object_type=='NSBH'):
            if self.verbose:
                print('It is possible to set the EoS using the set_EoS method.')
            self.set_EoS('rand')
        
        self.angular_pars = ['theta', 'phi', 'thetaJN', 'psi', 'tcoal', 'Phicoal']
    
    def update_hyperparameters(self, new_hyperparameters):
        '''
        Method to update the hyperparameters of the population model.

        :param dict new_hyperparameters: Dictionary containing the new hyperparameters of the population model as keys and their new value as entry.
        '''
        for key in new_hyperparameters.keys():
            if key in self.mass_function.hyperpar_dict.keys():
                self.mass_function.hyperpar_dict[key] = new_hyperparameters[key]
            elif key in self.rate_function.hyperpar_dict.keys():
                self.rate_function.hyperpar_dict[key] = new_hyperparameters[key]
            else:
                raise ValueError('The hyperparameter '+key+' is not present in the hyperparameter dictionary.')
        
        self.hyperpar_dict.update(self.mass_function.hyperpar_dict)
        self.hyperpar_dict.update(self.rate_function.hyperpar_dict)
        
    def update_priorlimits(self, new_limits):
        '''
        Method to update the prior limits of the population model.

        :param dict new_limits: Dictionary containing the new prior limits of the population model as keys and their new value as entry.
        '''
        for key in new_limits.keys():
            if key in self.mass_function.priorlims_dict.keys():
                self.mass_function.priorlims_dict[key] = new_limits[key]
            elif key in self.rate_function.priorlims_dict.keys():
                self.rate_function.priorlims_dict[key] = new_limits[key]
            elif key in self.angular_pars:
                self.priorlims_dict[key] = new_limits[key]
            else:
                raise ValueError('The parameter '+key+' is not present in the prior limits dictionary.')
        
        self.priorlims_dict.update(self.mass_function.priorlims_dict)
        self.priorlims_dict.update(self.rate_function.priorlims_dict)
        
    def sample_population(self, size):
        '''
        Function to sample the population model.
        
        :param int size: Size of the population sample.

        :return: Sampled population.
        :rtype: dict(array, array, ...)
        '''

        res = {}

        angles = self._sample_angles(size, is_Precessing=False)
        masses = self.mass_function.sample_population(size)
        spins  = {'chi1z':np.full(size, 1.0e-5), 'chi2z':np.full(size, 1.0e-5)}
        z      = self.rate_function.sample_population(size)

        res.update(angles)
        res.update(masses)
        res.update(spins)
        res.update(z)
        
        utils.check_masses(res)
                
        if (self.object_type=='BNS') | (self.object_type=='NSBH'):
            res.update(self.get_Lambda(res['m1_src'], res['m2_src'], EoS=self.EoS))
        
        return res
    
    def N_per_yr(self, R0=1.):
        '''
        Compute the number of events per year for the chosen distribution and parameters.
        
        :param float R0: Normalization of the local merger rate, in :math:`{\\rm Gpc}^{-3}\,{\\rm yr}^{-1}`.
        
        :return: Number of events per year.
        :rtype: float
        '''
    
        return self.rate_function.N_per_yr(R0=R0)
    
    def pop_function(self, events, uselog=False):
        '''
        Population function pdf of the model.

        :param dict(array, array, ...) events: Dictionary containing the events parameters.
        :param bool, optional uselog: Boolean specifying whether to return the probability or log-probability, defaults to False.
        
        :return: Population function value.
        :rtype: array or float
        '''

        parsmass = {key:events[key] for key in self.mass_function.par_list}
        
        mass_value = self.mass_function.mass_function(uselog=uselog, **parsmass)
        rate_value = self.rate_function.rate_function(events['z'], uselog=uselog)
        ang_value  = self.angle_distribution(uselog=uselog, **events)

        if not uselog:
            return mass_value*rate_value*ang_value
        else:
            return mass_value + rate_value + ang_value

    
    def pop_function_derivative(self, events, uselog=False):
        '''
        Derivative with respect to the hyperparameters of the population function pdf of the model.

        :param dict(array, array, ...) events: Dictionary containing the events parameters.
        :param bool uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the derivatives of the population function.
        :rtype: array 
        '''
    
        parsmass = {key:events[key] for key in self.mass_function.par_list}
                
        if not uselog:
            mass_value = self.mass_function.mass_function(uselog=uselog, **parsmass)
            rate_value = self.rate_function.rate_function(events['z'], uselog=uselog)
            ang_value  = self.angle_distribution(uselog=uselog, **events)

            mass_der = self.mass_function.mass_function_derivative(**parsmass)*ang_value
            z_der    = self.rate_function.rate_function_derivative(events['z'])*ang_value
            
            return np.vstack((mass_der*rate_value, z_der*mass_value))
        else:
            
            mass_der = self.mass_function.mass_function_derivative(uselog=True, **parsmass) 
            z_der    = self.rate_function.rate_function_derivative(events['z'], uselog=True) 
            
            return np.vstack((mass_der, z_der))
    
    def pop_function_hessian(self, events, derivs=None, uselog=False):
        '''
        Hessian with respect to the hyperparameters of the population function pdf of the model.

        :param dict(array, array, ...) events: Dictionary containing the events parameters.
        :param array, optional derivs: Array containing the derivatives of the population function.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the Hessians of the population function.
        :rtype: array 
        '''
    
        if (derivs is None) & (not uselog):
            derivs = self.pop_function_derivative(events, uselog=uselog)
            
        parsmass = {key:events[key] for key in self.mass_function.par_list}
        
        Nhyperpar = len(self.hyperpar_dict.keys())
        outmatr = np.zeros((Nhyperpar, Nhyperpar, len(events['z'])))
        
        mhp = len(list(self.mass_function.hyperpar_dict.keys()))
        rhp = len(list(self.rate_function.hyperpar_dict.keys()))
        
        if not uselog:
            mass_value = self.mass_function.mass_function(uselog=uselog, **parsmass)
            rate_value = self.rate_function.rate_function(events['z'], uselog=uselog)
            ang_value  = self.angle_distribution(uselog=uselog, **events)
            pop_value  = mass_value*rate_value*ang_value
            
            mass_hess = self.mass_function.mass_function_hessian(**parsmass)*rate_value*ang_value
            z_hess    = self.rate_function.rate_function_hessian(events['z'])*mass_value*ang_value
        
        else:
            mass_hess = self.mass_function.mass_function_hessian(uselog=True, **parsmass)
            z_hess    = self.rate_function.rate_function_hessian(events['z'], uselog=True)
            
        if not uselog:
            for i in range(Nhyperpar):
                for j in range(Nhyperpar):
                    if (i in self.derivative_par_nums_mass.values()) & (j in self.derivative_par_nums_mass.values()):
                        outmatr[i,j,:] = mass_hess[i,j,:]
                    elif (i in self.derivative_par_nums_rate.values()) & (j in self.derivative_par_nums_rate.values()):
                        outmatr[i,j,:] = z_hess[i-mhp,j-mhp,:]
                    elif (i in self.derivative_par_nums_mass.values()) & (j in self.derivative_par_nums_rate.values()):
                        outmatr[i,j,:] = outmatr[j,i,:] = derivs[i,:]*derivs[j,:]/pop_value
        else:
            outmatr[np.ix_(list(range(mhp)),list(range(mhp)))] = mass_hess
            outmatr[np.ix_(list(range(mhp, mhp+rhp)), list(range(mhp, mhp+rhp)))] = z_hess
            
        return outmatr
        
    def pop_function_hessian_termII(self, events, FIMs, ParNums, ParPrior=None):
        '''
        Function to compute the matrices appearing in the second term of the population Fisher.

        :param dict events: Dictionary containing the events parameters.
        :param array FIMs: Array containing the Fisher matrices of the single events.
        :param dict ParNums: Dictionary containing the parameter names as keys and their position in the Fisher matrix as entry.
        :param dict ParPrior: Dictionary containing the names of the parameters on which a prior in the single-event FIM has to be added as keys and the prior value as entry.

        :return: Array containing the matrices appearing in the second term of the population Fisher.
        :rtype: array
        '''

        ParNums = {k: v for k, v in sorted(ParNums.items(), key=lambda item: item[1])}
        
        tmp_par_list = []
        tmp_par_list.extend([x for x in self.mass_function.par_list if x not in tmp_par_list])
        tmp_par_list.extend([x for x in self.rate_function.par_list if x not in tmp_par_list])
        
        if ParPrior is not None:
            diag = np.array([ParPrior[key] if key in ParPrior.keys() else 0. for key in ParNums.keys()])
            #pp   = np.eye(FIM_use.shape[0])*diag
            pp   = np.eye(FIMs.shape[0])*diag
            if FIMs.ndim==2:
                FIMs = pp + FIMs
            else:
                FIMs = pp[:,:,np.newaxis] + FIMs

        FIM_nums = [ParNums[par] for par in tmp_par_list]
        FIM_use = FIMs[np.ix_(FIM_nums, FIM_nums)]
        
        FIM_use = FIM_use.T

        parsmass = {key:events[key] for key in self.mass_function.par_list}
        
        hyperpars_m = [self.mass_function.hyperpar_dict[key] for key in self.mass_function.hyperpar_dict.keys()]
        hyperpars_r = [self.rate_function.hyperpar_dict[key] for key in self.rate_function.hyperpar_dict.keys()]
        
        Nhyperpar_m = len(self.mass_function.hyperpar_dict.keys())
        Nhyperpar_r = len(self.rate_function.hyperpar_dict.keys())
        
        if Nhyperpar_m == 3:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parz1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parz1)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0])))
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2])))
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4])))
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
        
        elif Nhyperpar_m == 4:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parz1)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0])))
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2])))
            elif Nhyperpar_r == 5:
                    funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                    parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5)).T)
                    derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4])))
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
            
        elif Nhyperpar_m == 5:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0])))
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2])))
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4])))
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
            
        elif Nhyperpar_m == 6:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0])))
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2])))
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4])))
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
        elif Nhyperpar_m == 9:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0])))
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2])))
                
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4])))
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
        elif Nhyperpar_m == 11:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0])))
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2])))
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4])))
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
        else:
            raise ValueError('The number of hyperparameters for the mass function is not supported.')
            
        return derivs_all
    
    def pop_function_hessian_termIII(self, events, FIMs, ParNums, ParPrior=None):
        '''
        Function to compute the matrices appearing in the third term of the population Fisher.

        :param dict events: Dictionary containing the events parameters.
        :param array FIMs: Array containing the Fisher matrices of the single events.
        :param dict ParNums: Dictionary containing the parameter names as keys and their position in the Fisher matrix as entry.
        :param dict ParPrior: Dictionary containing the names of the parameters on which a prior in the single-event FIM has to be added as keys and the prior value as entry.

        :return: Array containing the matrices appearing in the third term of the population Fisher.
        :rtype: array
        '''

        ParNums = {k: v for k, v in sorted(ParNums.items(), key=lambda item: item[1])}
        
        tmp_par_list = []
        tmp_par_list.extend([x for x in self.mass_function.par_list if x not in tmp_par_list])
        tmp_par_list.extend([x for x in self.rate_function.par_list if x not in tmp_par_list])
        
        if ParPrior is not None:
            diag = np.array([ParPrior[key] if key in ParPrior.keys() else 0. for key in ParNums.keys()])
            #pp   = np.eye(FIM_use.shape[0])*diag
            pp   = np.eye(FIMs.shape[0])*diag
            if FIMs.ndim==2:
                FIMs = pp + FIMs
            else:
                FIMs = pp[:,:,np.newaxis] + FIMs

        FIM_nums = [ParNums[par] for par in tmp_par_list]
        FIM_use = FIMs[np.ix_(FIM_nums, FIM_nums)]
        
        FIM_use = FIM_use.T

        parsmass = {key:events[key] for key in self.mass_function.par_list}
        
        hyperpars_m = [self.mass_function.hyperpar_dict[key] for key in self.mass_function.hyperpar_dict.keys()]
        hyperpars_r = [self.rate_function.hyperpar_dict[key] for key in self.rate_function.hyperpar_dict.keys()]
        
        Nhyperpar_m = len(self.mass_function.hyperpar_dict.keys())
        Nhyperpar_r = len(self.rate_function.hyperpar_dict.keys())
        
        if Nhyperpar_m == 3:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parz1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parz1)).T)
                derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0]))), FIM_use)
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3)).T)
                derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2]))), FIM_use)
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5)).T)
                derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4]))), FIM_use)
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
        elif Nhyperpar_m == 4:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parz1)).T)
                derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0]))), FIM_use)
            elif Nhyperpar_r == 3:            
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3)).T)
                derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2]))), FIM_use)
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5)).T)
                derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4]))), FIM_use)
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
        elif Nhyperpar_m == 5:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1)).T)
                derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0]))), FIM_use)
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3)).T)
                derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2]))), FIM_use)
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5)).T)
                derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4]))), FIM_use)
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
        elif Nhyperpar_m == 6:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1)).T)
                derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0]))), FIM_use)
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3)).T)
                derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2]))), FIM_use)
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5)).T)
                derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4]))), FIM_use)
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
        elif Nhyperpar_m == 9:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1)).T)
                derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0]))), FIM_use)
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3)).T)
                derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2]))), FIM_use)
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5)).T)
                derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4]))), FIM_use)
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
        elif Nhyperpar_m == 11:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1)).T)
                derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0]))), FIM_use)
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3)).T)
                derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2]))), FIM_use)
                
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5)).T)
                derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4]))), FIM_use)
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
        else:
            raise ValueError('The number of hyperparameters for the mass function is not supported.')
            
        return derivs_all
    
    def pop_function_hessian_termIV(self, events, FIMs, Pdet_ders, ParNums, ParPrior=None):
        '''
        Function to compute the matrices appearing in the fourth term of the population Fisher.

        :param dict events: Dictionary containing the events parameters.
        :param array FIMs: Array containing the Fisher matrices of the single events.
        :param array Pdet_ders: Array containing the derivatives of the detection probability with respect to the parameters, ordered as the Fisher matrices of the single events.
        :param dict ParNums: Dictionary containing the parameter names as keys and their position in the Fisher matrix as entry.
        :param dict ParPrior: Dictionary containing the names of the parameters on which a prior in the single-event FIM has to be added as keys and the prior value as entry.

        :return: Array containing the matrices appearing in the fourth term of the population Fisher.
        :rtype: array
        '''

        ParNums = {k: v for k, v in sorted(ParNums.items(), key=lambda item: item[1])}
        
        tmp_par_list = []
        tmp_par_list.extend([x for x in self.mass_function.par_list if x not in tmp_par_list])
        tmp_par_list.extend([x for x in self.rate_function.par_list if x not in tmp_par_list])
        
        if ParPrior is not None:
            diag = np.array([ParPrior[key] if key in ParPrior.keys() else 0. for key in ParNums.keys()])
            #pp   = np.eye(FIM_use.shape[0])*diag
            pp   = np.eye(FIMs.shape[0])*diag
            if FIMs.ndim==2:
                FIMs = pp + FIMs
            else:
                FIMs = pp[:,:,np.newaxis] + FIMs
        FIM_nums = [ParNums[par] for par in tmp_par_list]
        FIM_use = FIMs[np.ix_(FIM_nums, FIM_nums)]
        Pdet_ders_use = Pdet_ders[FIM_nums].T
        FIM_use = FIM_use.T

        parsmass = {key:events[key] for key in self.mass_function.par_list}
        
        hyperpars_m = [self.mass_function.hyperpar_dict[key] for key in self.mass_function.hyperpar_dict.keys()]
        hyperpars_r = [self.rate_function.hyperpar_dict[key] for key in self.rate_function.hyperpar_dict.keys()]
        
        Nhyperpar_m = len(self.mass_function.hyperpar_dict.keys())
        Nhyperpar_r = len(self.rate_function.hyperpar_dict.keys())
        
        if Nhyperpar_m == 3:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parz1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parz1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parz1)).T))
                derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0]))), Pdet_ders_use)
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3)).T))
                derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2]))), Pdet_ders_use)
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5)).T))
                derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4]))), Pdet_ders_use)
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
        elif Nhyperpar_m == 4:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parz1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parz1)).T))
                derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0]))), Pdet_ders_use)
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3)).T))
                derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2]))), Pdet_ders_use)
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5)).T))
                derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4]))), Pdet_ders_use)
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
        elif Nhyperpar_m == 5:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1)).T))
                derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0]))), Pdet_ders_use)
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3)).T))
                derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2]))), Pdet_ders_use)
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5)).T))
                derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4]))), Pdet_ders_use)
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
        elif Nhyperpar_m == 6:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1)).T))
                derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0]))), Pdet_ders_use)
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3)).T))
                derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2]))), Pdet_ders_use)
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5)).T))
                derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4]))), Pdet_ders_use)
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
        elif Nhyperpar_m == 9:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1)).T))
                derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0]))), Pdet_ders_use)
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3)).T))
                derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2]))), Pdet_ders_use)
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5)).T))
                derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4]))), Pdet_ders_use)
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
        elif Nhyperpar_m == 11:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1)).T))
                derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0]))), Pdet_ders_use)
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3)).T))
                derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2]))), Pdet_ders_use)
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5)).T))
                derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4]))), Pdet_ders_use)
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
        else:
            raise ValueError('The number of hyperparameters for the mass function is not supported.')
            
        return derivs_all
    
    def pop_function_hessian_termV(self, events, FIMs, ParNums, ParPrior=None):
        '''
        Function to compute the matrices appearing in the fifth term of the population Fisher.

        :param dict events: Dictionary containing the events parameters.
        :param array FIMs: Array containing the Fisher matrices of the single events.
        :param dict ParNums: Dictionary containing the parameter names as keys and their position in the Fisher matrix as entry.
        :param dict ParPrior: Dictionary containing the names of the parameters on which a prior in the single-event FIM has to be added as keys and the prior value as entry.

        :return: Array containing the matrices appearing in the fifth term of the population Fisher.
        :rtype: array
        '''
        
        ParNums = {k: v for k, v in sorted(ParNums.items(), key=lambda item: item[1])}

        tmp_par_list = []
        tmp_par_list.extend([x for x in self.mass_function.par_list if x not in tmp_par_list])
        tmp_par_list.extend([x for x in self.rate_function.par_list if x not in tmp_par_list])
        
        if ParPrior is not None:
            diag = np.array([ParPrior[key] if key in ParPrior.keys() else 0. for key in ParNums.keys()])
            #pp   = np.eye(FIM_use.shape[0])*diag
            pp   = np.eye(FIMs.shape[0])*diag
            if FIMs.ndim==2:
                FIMs = pp + FIMs
            else:
                FIMs = pp[:,:,np.newaxis] + FIMs

        FIM_nums = [ParNums[par] for par in tmp_par_list]
        FIM_use = FIMs[np.ix_(FIM_nums, FIM_nums)]
        FIM_use = FIM_use.T

        parsmass = {key:events[key] for key in self.mass_function.par_list}
        
        hyperpars_m = [self.mass_function.hyperpar_dict[key] for key in self.mass_function.hyperpar_dict.keys()]
        hyperpars_r = [self.rate_function.hyperpar_dict[key] for key in self.rate_function.hyperpar_dict.keys()]
        
        Nhyperpar_m = len(self.mass_function.hyperpar_dict.keys())
        Nhyperpar_r = len(self.rate_function.hyperpar_dict.keys())
        
        if Nhyperpar_m == 3:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parz1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parz1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parz1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parz1)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0])))
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2])))
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parz1, parz2, parz3, parz4, parz5)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4])))
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
        elif Nhyperpar_m == 4:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parz1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parz1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parz1)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0])))
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2])))
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parz1, parz2, parz3, parz4, parz5)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4])))
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
        elif Nhyperpar_m == 5:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0])))
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2])))
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parz1, parz2, parz3, parz4, parz5)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4])))
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
        elif Nhyperpar_m == 6:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0])))
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2])))
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parz1, parz2, parz3, parz4, parz5)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4])))
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
        elif Nhyperpar_m == 9:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0])))
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2])))
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parz1, parz2, parz3, parz4, parz5)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4])))
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
        elif Nhyperpar_m == 11:
            if Nhyperpar_r == 1:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13,14)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0])))
            elif Nhyperpar_r == 3:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13,14,15,16)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2])))
            elif Nhyperpar_r == 5:
                funder = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True) + self.rate_function.rate_function(z, parz1, parz2, parz3, parz4, parz5, uselog=True)
                parshess = lambda m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1,2)), in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, z, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, parz1, parz2, parz3, parz4, parz5)).T)
                derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18)))(parsmass['m1_src'], parsmass['m2_src'], events['z'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10], hyperpars_r[0], hyperpars_r[1], hyperpars_r[2], hyperpars_r[3], hyperpars_r[4])))
            else:
                raise ValueError('The number of hyperparameters for the rate function is not supported.')
        else:
            raise ValueError('The number of hyperparameters for the mass function is not supported.')
            
        return derivs_all

class MassOnly_PopulationModel(PopulationModel):
    '''
    Class to compute population models in which we only consider the mass distribution for the FIM but keep the and redshift distribution for sampling.

    :param list parameters: List containing the parameters of the population model.
    :param dict hyperparameters: Dictionary containing the hyperparameters of the population model as keys and their fiducial value as entry.
    :param dict priorlims_parameters: Dictionary containing the parameters of the population model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.
    :param dict kwargs: Keyword arguments passed to the constructor of the population model.
    '''
    
    def __init__(self, mass_function, rate_function, verbose=False):
        '''
        Constructor method
        '''

        super().__init__(verbose=verbose)

        self.mass_function = mass_function
        self.rate_function = rate_function

        self.par_list.extend([x for x in self.mass_function.par_list if x not in self.par_list])
        
        self.hyperpar_dict.update(self.mass_function.hyperpar_dict)
        
        self.priorlims_dict.update(self.mass_function.priorlims_dict)
        
        self.derivative_par_nums = copy.deepcopy(self.mass_function.derivative_par_nums)
        self.derivative_par_nums_mass = copy.deepcopy(self.mass_function.derivative_par_nums)
        
        self.object_type = self.mass_function.object_type
        
        if (self.object_type=='BNS') | (self.object_type=='NSBH'):
            if self.verbose:
                print('It is possible to set the EoS using the set_EoS method.')
            self.set_EoS('rand')
        
        self.angular_pars = ['theta', 'phi', 'thetaJN', 'psi', 'tcoal', 'Phicoal']
    
    def update_hyperparameters(self, new_hyperparameters):
        '''
        Method to update the hyperparameters of the population model.

        :param dict new_hyperparameters: Dictionary containing the new hyperparameters of the population model as keys and their new value as entry.
        '''
        for key in new_hyperparameters.keys():
            if key in self.mass_function.hyperpar_dict.keys():
                self.mass_function.hyperpar_dict[key] = new_hyperparameters[key]
            else:
                raise ValueError('The hyperparameter '+key+' is not present in the hyperparameter dictionary.')
        
        self.hyperpar_dict.update(self.mass_function.hyperpar_dict)
        
    def update_priorlimits(self, new_limits):
        '''
        Method to update the prior limits of the population model.

        :param dict new_limits: Dictionary containing the new prior limits of the population model as keys and their new value as entry.
        '''
        for key in new_limits.keys():
            if key in self.mass_function.priorlims_dict.keys():
                self.mass_function.priorlims_dict[key] = new_limits[key]
            elif key in self.angular_pars:
                self.priorlims_dict[key] = new_limits[key]
            else:
                raise ValueError('The parameter '+key+' is not present in the prior limits dictionary.')
        
        self.priorlims_dict.update(self.mass_function.priorlims_dict)
        
    def sample_population(self, size):
        '''
        Function to sample the population model.

        :return: Sampled population.
        :rtype: dict
        '''

        res = {}

        angles = self._sample_angles(size, is_Precessing=False)
        masses = self.mass_function.sample_population(size)
        spins  = {'chi1z':np.full(size, 1.0e-5), 'chi2z':np.full(size, 1.0e-5)}
        z      = self.rate_function.sample_population(size)

        res.update(angles)
        res.update(masses)
        res.update(spins)
        res.update(z)
        
        utils.check_masses(res)
                
        if (self.object_type=='BNS') | (self.object_type=='NSBH'):
            res.update(self.get_Lambda(res['m1_src'], res['m2_src'], EoS=self.EoS))
        
        return res
    
    def N_per_yr(self, R0=1.):
        '''
        Compute the number of events per year for the chosen distribution and parameters.
        
        :param float R0: Normalization of the local merger rate, in :math:`{\\rm Gpc}^{-3}\,{\\rm yr}^{-1}`.
        
        :return: Number of events per year.
        :rtype: float
        '''
    
        return self.rate_function.N_per_yr(R0=R0)
    
    def pop_function(self, events, uselog=False):
        '''
        Population function pdf of the model.

        :param dict(array, array, ...) events: Dictionary containing the events parameters.
        :param bool, optional uselog: Boolean specifying whether to return the probability or log-probability, defaults to False.
        
        :return: Population function value.
        :rtype: array or float
        '''

        parsmass = {key:events[key] for key in self.mass_function.par_list}
        
        mass_value = self.mass_function.mass_function(uselog=uselog, **parsmass)
        ang_value  = self.angle_distribution(uselog=uselog, **events)

        if not uselog:
            return mass_value*ang_value
        else:
            return mass_value + ang_value

    
    def pop_function_derivative(self, events, uselog=False):
        '''
        Derivative with respect to the hyperparameters of the population function pdf of the model.

        :param dict(array, array, ...) events: Dictionary containing the events parameters.
        :param bool uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the derivatives of the population function.
        :rtype: array 
        '''
    
        parsmass = {key:events[key] for key in self.mass_function.par_list}
                
        if not uselog:
            ang_value  = self.angle_distribution(uselog=uselog, **events)

            mass_der = self.mass_function.mass_function_derivative(**parsmass)*ang_value
            
            return mass_der
        else:
            
            mass_der = self.mass_function.mass_function_derivative(uselog=True, **parsmass) 
            
            return mass_der
    
    def pop_function_hessian(self, events, derivs=None, uselog=False):
        '''
        Hessian with respect to the hyperparameters of the population function pdf of the model.

        :param dict(array, array, ...) events: Dictionary containing the events parameters.
        :param array, optional derivs: Array containing the derivatives of the population function.
        :param bool, optional uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the Hessians of the population function.
        :rtype: array 
        '''
    
        if (derivs is None) & (not uselog):
            derivs = self.pop_function_derivative(events, uselog=uselog)
            
        parsmass = {key:events[key] for key in self.mass_function.par_list}
        
        Nhyperpar = len(self.hyperpar_dict.keys())
        outmatr = np.zeros((Nhyperpar, Nhyperpar, len(events['z'])))
        
        mhp = len(list(self.mass_function.hyperpar_dict.keys()))
        
        if not uselog:
            ang_value  = self.angle_distribution(uselog=uselog, **events)
            mass_hess = self.mass_function.mass_function_hessian(**parsmass)*ang_value
            
        else:
            mass_hess = self.mass_function.mass_function_hessian(uselog=True, **parsmass)
            
        if not uselog:
            for i in range(Nhyperpar):
                for j in range(Nhyperpar):
                    outmatr[i,j,:] = mass_hess[i,j,:]
        else:
            outmatr[np.ix_(list(range(mhp)),list(range(mhp)))] = mass_hess
            
        return outmatr
        
    def pop_function_hessian_termII(self, events, FIMs, ParNums, ParPrior=None):
        '''
        Function to compute the matrices appearing in the second term of the population Fisher.

        :param dict events: Dictionary containing the events parameters.
        :param array FIMs: Array containing the Fisher matrices of the single events.
        :param dict ParNums: Dictionary containing the parameter names as keys and their position in the Fisher matrix as entry.
        :param dict ParPrior: Dictionary containing the names of the parameters on which a prior in the single-event FIM has to be added as keys and the prior value as entry.

        :return: Array containing the matrices appearing in the second term of the population Fisher.
        :rtype: array
        '''

        ParNums = {k: v for k, v in sorted(ParNums.items(), key=lambda item: item[1])}
        
        tmp_par_list = []
        tmp_par_list.extend([x for x in self.mass_function.par_list if x not in tmp_par_list])
        
        if ParPrior is not None:
            diag = np.array([ParPrior[key] if key in ParPrior.keys() else 0. for key in ParNums.keys()])
            #pp   = np.eye(FIM_use.shape[0])*diag
            pp   = np.eye(FIMs.shape[0])*diag
            if FIMs.ndim==2:
                FIMs = pp + FIMs
            else:
                FIMs = pp[:,:,np.newaxis] + FIMs

        FIM_nums = [ParNums[par] for par in tmp_par_list]
        FIM_use = FIMs[np.ix_(FIM_nums, FIM_nums)]
        
        FIM_use = FIM_use.T

        parsmass = {key:events[key] for key in self.mass_function.par_list}
        
        hyperpars_m = [self.mass_function.hyperpar_dict[key] for key in self.mass_function.hyperpar_dict.keys()]
        
        Nhyperpar_m = len(self.mass_function.hyperpar_dict.keys())
        
        if Nhyperpar_m == 3:
            funder = lambda m1, m2, parm1, parm2, parm3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None))(m1, m2, parm1, parm2, parm3)).T)
            derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(2,3,4)))(parsmass['m1_src'], parsmass['m2_src'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2])))
        
        elif Nhyperpar_m == 4:
            funder = lambda m1, m2, parm1, parm2, parm3, parm4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3, parm4: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4)).T)
            derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(2,3,4,5)))(parsmass['m1_src'], parsmass['m2_src'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3])))
            
        elif Nhyperpar_m == 5:
            funder = lambda m1, m2, parm1, parm2, parm3, parm4, parm5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3, parm4, parm5: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5)).T)
            derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(2,3,4,5,6)))(parsmass['m1_src'], parsmass['m2_src'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4])))
            
        elif Nhyperpar_m == 6:
            funder = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6)).T)
            derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(2,3,4,5,6,7)))(parsmass['m1_src'], parsmass['m2_src'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5])))
        elif Nhyperpar_m == 9:
            funder = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9)).T)
            derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(2,3,4,5,6,7,8,9,10)))(parsmass['m1_src'], parsmass['m2_src'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8])))
            
        elif Nhyperpar_m == 11:
            funder = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11: utils.logdet_stabilizecond(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11)).T)
            derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(2,3,4,5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10])))
        else:
            raise ValueError('The number of hyperparameters for the mass function is not supported.')
            
        return derivs_all
    
    def pop_function_hessian_termIII(self, events, FIMs, ParNums, ParPrior=None):
        '''
        Function to compute the matrices appearing in the third term of the population Fisher.

        :param dict events: Dictionary containing the events parameters.
        :param array FIMs: Array containing the Fisher matrices of the single events.
        :param dict ParNums: Dictionary containing the parameter names as keys and their position in the Fisher matrix as entry.
        :param dict ParPrior: Dictionary containing the names of the parameters on which a prior in the single-event FIM has to be added as keys and the prior value as entry.

        :return: Array containing the matrices appearing in the third term of the population Fisher.
        :rtype: array
        '''

        ParNums = {k: v for k, v in sorted(ParNums.items(), key=lambda item: item[1])}
        
        tmp_par_list = []
        tmp_par_list.extend([x for x in self.mass_function.par_list if x not in tmp_par_list])
        
        if ParPrior is not None:
            diag = np.array([ParPrior[key] if key in ParPrior.keys() else 0. for key in ParNums.keys()])
            #pp   = np.eye(FIM_use.shape[0])*diag
            pp   = np.eye(FIMs.shape[0])*diag
            if FIMs.ndim==2:
                FIMs = pp + FIMs
            else:
                FIMs = pp[:,:,np.newaxis] + FIMs

        FIM_nums = [ParNums[par] for par in tmp_par_list]
        FIM_use = FIMs[np.ix_(FIM_nums, FIM_nums)]
        
        FIM_use = FIM_use.T

        parsmass = {key:events[key] for key in self.mass_function.par_list}
        
        hyperpars_m = [self.mass_function.hyperpar_dict[key] for key in self.mass_function.hyperpar_dict.keys()]
        
        Nhyperpar_m = len(self.mass_function.hyperpar_dict.keys())
        
        if Nhyperpar_m == 3:
            funder = lambda m1, m2, parm1, parm2, parm3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None))(m1, m2, parm1, parm2, parm3)).T)
            derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(2,3,4)))(parsmass['m1_src'], parsmass['m2_src'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2]))), FIM_use)
        elif Nhyperpar_m == 4:
            funder = lambda m1, m2, parm1, parm2, parm3, parm4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3, parm4: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4)).T)
            derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(2,3,4,5)))(parsmass['m1_src'], parsmass['m2_src'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3]))), FIM_use)
        elif Nhyperpar_m == 5:
            funder = lambda m1, m2, parm1, parm2, parm3, parm4, parm5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3, parm4, parm5: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5)).T)
            derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(2,3,4,5,6)))(parsmass['m1_src'], parsmass['m2_src'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4]))), FIM_use)
        elif Nhyperpar_m == 6:
            funder = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6)).T)
            derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(2,3,4,5,6,7)))(parsmass['m1_src'], parsmass['m2_src'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5]))), FIM_use)
        elif Nhyperpar_m == 9:
            funder = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9)).T)
            derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(2,3,4,5,6,7,8,9,10)))(parsmass['m1_src'], parsmass['m2_src'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8]))), FIM_use)
        elif Nhyperpar_m == 11:
            funder = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11: utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11)).T)
            derivs_all = np.einsum('...ijk,ijk->...i', np.squeeze(np.array((hessian(parshess, argnums=(2,3,4,5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10]))), FIM_use)
        else:
            raise ValueError('The number of hyperparameters for the mass function is not supported.')
            
        return derivs_all
    
    def pop_function_hessian_termIV(self, events, FIMs, Pdet_ders, ParNums, ParPrior=None):
        '''
        Function to compute the matrices appearing in the fourth term of the population Fisher.

        :param dict events: Dictionary containing the events parameters.
        :param array FIMs: Array containing the Fisher matrices of the single events.
        :param array Pdet_ders: Array containing the derivatives of the detection probability with respect to the parameters, ordered as the Fisher matrices of the single events.
        :param dict ParNums: Dictionary containing the parameter names as keys and their position in the Fisher matrix as entry.
        :param dict ParPrior: Dictionary containing the names of the parameters on which a prior in the single-event FIM has to be added as keys and the prior value as entry.

        :return: Array containing the matrices appearing in the fourth term of the population Fisher.
        :rtype: array
        '''

        ParNums = {k: v for k, v in sorted(ParNums.items(), key=lambda item: item[1])}
        
        tmp_par_list = []
        tmp_par_list.extend([x for x in self.mass_function.par_list if x not in tmp_par_list])
        
        if ParPrior is not None:
            diag = np.array([ParPrior[key] if key in ParPrior.keys() else 0. for key in ParNums.keys()])
            #pp   = np.eye(FIM_use.shape[0])*diag
            pp   = np.eye(FIMs.shape[0])*diag
            if FIMs.ndim==2:
                FIMs = pp + FIMs
            else:
                FIMs = pp[:,:,np.newaxis] + FIMs
        FIM_nums = [ParNums[par] for par in tmp_par_list]
        FIM_use = FIMs[np.ix_(FIM_nums, FIM_nums)]
        Pdet_ders_use = Pdet_ders[FIM_nums].T
        FIM_use = FIM_use.T

        parsmass = {key:events[key] for key in self.mass_function.par_list}
        
        hyperpars_m = [self.mass_function.hyperpar_dict[key] for key in self.mass_function.hyperpar_dict.keys()]
        
        Nhyperpar_m = len(self.mass_function.hyperpar_dict.keys())
        
        if Nhyperpar_m == 3:
            funder = lambda m1, m2, parm1, parm2, parm3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1)), in_axes=(0,0,None,None,None))(m1, m2, parm1, parm2, parm3)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None))(m1, m2, parm1, parm2, parm3)).T))
            derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(2,3,4)))(parsmass['m1_src'], parsmass['m2_src'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2]))), Pdet_ders_use)
        elif Nhyperpar_m == 4:
            funder = lambda m1, m2, parm1, parm2, parm3, parm4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3, parm4: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4)).T))
            derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(2,3,4,5)))(parsmass['m1_src'], parsmass['m2_src'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3]))), Pdet_ders_use)
        elif Nhyperpar_m == 5:
            funder = lambda m1, m2, parm1, parm2, parm3, parm4, parm5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3, parm4, parm5: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5)).T))
            derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(2,3,4,5,6)))(parsmass['m1_src'], parsmass['m2_src'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4]))), Pdet_ders_use)
        elif Nhyperpar_m == 6:
            funder = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6)).T))
            derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(2,3,4,5,6,7)))(parsmass['m1_src'], parsmass['m2_src'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5]))), Pdet_ders_use)
        elif Nhyperpar_m == 9:
            funder = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9)).T))
            derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(2,3,4,5,6,7,8,9,10)))(parsmass['m1_src'], parsmass['m2_src'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8]))), Pdet_ders_use)
        elif Nhyperpar_m == 11:
            funder = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11: jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11)).T))
            derivs_all = np.einsum('...ij,ij->...i', np.squeeze(np.array((hessian(parshess, argnums=(2,3,4,5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10]))), Pdet_ders_use)
        else:
            raise ValueError('The number of hyperparameters for the mass function is not supported.')
            
        return derivs_all
    
    def pop_function_hessian_termV(self, events, FIMs, ParNums, ParPrior=None):
        '''
        Function to compute the matrices appearing in the fifth term of the population Fisher.

        :param dict events: Dictionary containing the events parameters.
        :param array FIMs: Array containing the Fisher matrices of the single events.
        :param dict ParNums: Dictionary containing the parameter names as keys and their position in the Fisher matrix as entry.
        :param dict ParPrior: Dictionary containing the names of the parameters on which a prior in the single-event FIM has to be added as keys and the prior value as entry.

        :return: Array containing the matrices appearing in the fifth term of the population Fisher.
        :rtype: array
        '''
        
        ParNums = {k: v for k, v in sorted(ParNums.items(), key=lambda item: item[1])}

        tmp_par_list = []
        tmp_par_list.extend([x for x in self.mass_function.par_list if x not in tmp_par_list])
        
        if ParPrior is not None:
            diag = np.array([ParPrior[key] if key in ParPrior.keys() else 0. for key in ParNums.keys()])
            #pp   = np.eye(FIM_use.shape[0])*diag
            pp   = np.eye(FIMs.shape[0])*diag
            if FIMs.ndim==2:
                FIMs = pp + FIMs
            else:
                FIMs = pp[:,:,np.newaxis] + FIMs

        FIM_nums = [ParNums[par] for par in tmp_par_list]
        FIM_use = FIMs[np.ix_(FIM_nums, FIM_nums)]
        FIM_use = FIM_use.T

        parsmass = {key:events[key] for key in self.mass_function.par_list}
        
        hyperpars_m = [self.mass_function.hyperpar_dict[key] for key in self.mass_function.hyperpar_dict.keys()]
        
        Nhyperpar_m = len(self.mass_function.hyperpar_dict.keys())
        
        if Nhyperpar_m == 3:
            funder = lambda m1, m2, parm1, parm2, parm3: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1)), in_axes=(0,0,None,None,None))(m1, m2, parm1, parm2, parm3)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None))(m1, m2, parm1, parm2, parm3)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1)), in_axes=(0,0,None,None,None))(m1, m2, parm1, parm2, parm3)).T)
            derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(2,3,4)))(parsmass['m1_src'], parsmass['m2_src'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2])))
        elif Nhyperpar_m == 4:
            funder = lambda m1, m2, parm1, parm2, parm3, parm4: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3, parm4: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4)).T)
            derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(2,3,4,5)))(parsmass['m1_src'], parsmass['m2_src'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3])))
        elif Nhyperpar_m == 5:
            funder = lambda m1, m2, parm1, parm2, parm3, parm4, parm5: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3, parm4, parm5: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5)).T)
            derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(2,3,4,5,6)))(parsmass['m1_src'], parsmass['m2_src'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4])))
        elif Nhyperpar_m == 6:
            funder = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6)).T)
            derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(2,3,4,5,6,7)))(parsmass['m1_src'], parsmass['m2_src'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5])))
        elif Nhyperpar_m == 9:
            funder = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9)).T)
            derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(2,3,4,5,6,7,8,9,10)))(parsmass['m1_src'], parsmass['m2_src'],  hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8])))
        elif Nhyperpar_m == 11:
            funder = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11: self.mass_function.mass_function(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11, uselog=True)
            parshess = lambda m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11: jnp.einsum('...ij,ij->...i', jnp.einsum('ij,...ijk->...ik', jnp.asarray(vmap(jacrev(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11)).T, utils.invmatrix(FIM_use - jnp.asarray(vmap(hessian(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11)).T)), jnp.asarray(vmap(jacrev(funder, argnums=(0,1)), in_axes=(0,0,None,None,None,None,None,None,None,None,None,None,None))(m1, m2, parm1, parm2, parm3, parm4, parm5, parm6, parm7, parm8, parm9, parm10, parm11)).T)
            derivs_all = np.squeeze(np.array((hessian(parshess, argnums=(2,3,4,5,6,7,8,9,10,11,12)))(parsmass['m1_src'], parsmass['m2_src'], hyperpars_m[0], hyperpars_m[1], hyperpars_m[2], hyperpars_m[3], hyperpars_m[4], hyperpars_m[5], hyperpars_m[6], hyperpars_m[7], hyperpars_m[8], hyperpars_m[9], hyperpars_m[10])))
        else:
            raise ValueError('The number of hyperparameters for the mass function is not supported.')
            
        return derivs_all