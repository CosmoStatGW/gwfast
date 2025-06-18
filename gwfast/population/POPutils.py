"""
A collection of functions to convert between parameters describing
gravitational-wave sources.
"""

import os,sys,h5py,jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.special import betaln, erf, erfc
from jax import custom_jvp, jit
from jax.lax import associative_scan, cond, fori_loop
from functools import partial
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd())))
GWFAST_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))  
sys.path.append(GWFAST_DIR)
import gwfast
from gwfast.gwfastUtils import  get_events_subset, save_detectors, load_population, save_data
from gwfast.fisherTools import CovMatr

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
        



### Sampling functions ###

def inverse_cdf_sampling(fun, size, prange, res=1e5, **kwargs):
    """
    Sample from a probability distribution using inverse cdf sampling.

    :param function fun: PDF to sample from. It has to take an array of values as input and return an array containing their probabilities.
    :param int size: Size of the sample to generate.
    :param tuple prange: Range of the sample to generate.

    :return: Samples extracted from the desired PDF.
    :rtype: array
    """

    # generate uniform samples
    u = np.random.uniform(size=size)

    # compute inverse cdf
    x = np.linspace(prange[0],prange[1], int(res))
    y = fun(x, **kwargs)
    cdf_y = np.cumsum(y)
    cdf_y = cdf_y/cdf_y.max()
    res = np.interp(u, cdf_y,x)
    return res

def inverse_cdf_sampling_conditional(fun, size, prange, condpar):
    """
    
    Sample from a probability distribution using inverse cdf sampling.

    :param function fun: PDF to sample from. It has to take an array of values as input and return an array containing their probabilities.
    :param int size: Size of the sample to generate.
    :param tuple prange: Range of the sample to generate.


    :return: Samples extracted from the desired PDF.
    :rtype: array
    """

    # generate uniform samples
    u = np.random.uniform(size=size)
    
    # compute inverse cdf
    x = np.array([np.linspace(prange[0],prange[1], int(1e4)) for i in range(size)]).T
    y = fun(x, condpar)
    cdf_y = np.cumsum(y, axis=0)
    cdf_y = cdf_y/np.amax(cdf_y, axis=0)#cdf_y.max()
    res = np.array([np.interp(u[i], cdf_y[:,i], x[:,i]) for i in range(size)])
    return res

def inverse_cdf_sampling_uppercond(pdf, lower, upper):

    nSamples = len(upper)
    res = 100000
    eps=1e-02
    x = np.linspace(lower+eps, upper.max()-eps, res)
    cdf = np.cumsum(pdf(x))
    cdf = cdf / cdf[-1]
    probTilUpper = np.interp(upper, x, cdf)
    return np.interp(probTilUpper*np.random.uniform(size=nSamples), cdf, x)

def inverse_cdf_sampling_uppercond_OLD(fun, size, lower, upper):
        
    # generate uniform samples
    u = np.random.uniform(size=size)

    # compute inverse cdf
    x = np.linspace(lower, upper.max(), int(1e5))
    y = fun(x, upper.max())
    cdf_y = np.cumsum(y, axis=0)
    cdf_y = cdf_y/cdf_y.max()
    # adjust to upper limit
    probTilUpper = np.interp(upper, x, cdf_y)
    return np.interp(probTilUpper*u, cdf_y, x)
    
### Useful generic functions ###

def powerlaw(par, slope, limits):
    """
    Power law function.

    :param array par: Parameter value.
    :param float slope: Slope of the power law.
    :param tuple(float, float) limits: Limits of the power law.

    :return: Power law value.
    :rtype: float
    """

    if slope == -1.:
        norm = jnp.log(limits[1] / limits[0])
    else:
        norm = (limits[1]**(1 + slope) - limits[0]**(1 + slope)) / (1 + slope)

    return (par**(slope))/norm

def inversepowerlaw(par, slope, limits):
    """
    Power law function.

    :param array par: Parameter value.
    :param float slope: Slope of the power law.
    :param tuple(float, float) limits: Limits of the power law.

    :return: Power law value.
    :rtype: float
    """

    if slope == 1.:
        norm = jnp.log(limits[1] / limits[0])
    else:
        norm = (limits[1]**(1 - slope) - limits[0]**(1 - slope)) / (1 - slope)

    return (par**(-slope))/norm

def gaussian_norm(par, mu, sigma):
    """
    Gaussian PDF.

    :param array par: Parameter value.
    :param float mu: Mean of the Gaussian.
    :param float sigma: Standard deviation of the Gaussian.

    :return: Gaussian value.
    :rtype: float
    """

    return jnp.exp(-0.5 * ((par - mu) / sigma)**2) / (sigma * jnp.sqrt(2 * jnp.pi))

def beta_distrib(x, alpha=1., beta=1.):
    """
    Beta distribution PDF.
    
    :param array or float x: Parameter value.
    :param float alpha: Alpha parameter of the beta distribution.
    :param float beta: Beta parameter of the beta distribution.
    
    :return: Beta distribution PDF value.
    :rtype: array or float
    """

    ln_beta = (alpha - 1) * jnp.log(x) + (beta - 1) * jnp.log(1. - x) - betaln(alpha, beta)
    
    return jnp.where((x >= 0.) & (x <= 1.), jnp.nan_to_num(jnp.exp(ln_beta)), 0.)

def trunc_gaussian_norm(x, mu=1., sigma=1., lower=0., upper=100.):
    """
    Truncated Gaussian PDF.
    
    :param array or float x: Parameter value.
    :param float mu: Mean of the Gaussian.
    :param float sigma: Standard deviation of the Gaussian.
    :param float lower: Lower limit of the truncation.
    :param float upper: Upper limit of the truncation.
    
    :return: Truncated Gaussian PDF value.
    :rtype: array or float
    """
        
    phi = jnp.exp(-0.5 * ((x - mu)/sigma)**2) / (sigma * jnp.sqrt(2 * jnp.pi))
    Phib = 0.5 * (1.+ erf((upper - mu)/(sigma * jnp.sqrt(2.))))
    Phia = 0.5 * (1.+ erf((lower - mu)/(sigma * jnp.sqrt(2.))))

    return jnp.where((x >= lower) & (x <= upper), phi/(Phib - Phia), 0.)
        
@custom_jvp
def planck_taper(x, xmin, deltax):
    eps = 1e-10 

    return jnp.where(x >= xmin+deltax, 1., jnp.where(x < xmin, 0., 1./(jnp.exp(deltax*(1./(x-xmin+ eps) + 1./(x - xmin - deltax+ eps))) + 1.)))

@custom_jvp
def planck_taper_der_x(x, xmin, deltax):
    
    tangent_out = (-1.) * deltax * jnp.exp(deltax * (((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1))) * ((1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))))**(-2) * ((-1.) * ((x + (-1.) * xmin))**(-2) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-2))
    
    tangent_out = jnp.nan_to_num(tangent_out)
    return tangent_out

def planck_taper_dder_x(x, xmin, deltax):
    
    tangent_out = deltax * jnp.exp(deltax * (1. * ((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1))) * ((1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))))**(-3) * ((-1.) * (1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))) * (2. * ((x + (-1.) * xmin))**(-3) + (-2.) * ((deltax + ((-1.) * x + xmin)))**(-3)) + (2. * deltax * jnp.exp(deltax * (1. * ((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1))) * ((1. * ((x + (-1.) * xmin))**(-2) + 1. * ((deltax + ((-1.) * x + xmin)))**(-2)))**2 + (-1.) * deltax * (1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))) * ((1. * ((x + (-1.) * xmin))**(-2) + 1. * ((deltax + ((-1.) * x + xmin)))**(-2)))**2))
    
    tangent_out = jnp.nan_to_num(tangent_out)
    return tangent_out

def planck_taper_der_x_xmin(x, xmin, deltax):

    tangent_out = (-2.) * deltax * jnp.exp(deltax * (1. * ((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1))) * ((1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))))**(-3) * ((x + -1 * xmin))**(-4) * ((deltax + (-1 * x + xmin)))**(-4) * (1. * deltax * jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1))) * ((1. * ((x + (-1.) * xmin))**2 + 1. * ((deltax + ((-1.) * x + xmin)))**2))**2 + ((-0.5) * deltax * (1. + jnp.exp(deltax * (1. * ((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1)))) * ((1. * ((x + (-1.) * xmin))**2 + 1. * ((deltax + ((-1.) * x + xmin)))**2))**2 + 1. * (1. + jnp.exp(deltax * (1. * ((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1)))) * (x + -1 * xmin) * (deltax + (-1 * x + xmin)) * (1. * ((x + (-1.) * xmin))**3 + (-1.) * ((deltax + ((-1.) * x + xmin)))**3)))
    
    tangent_out = jnp.nan_to_num(tangent_out)
    return tangent_out

def planck_taper_der_x_deltax(x, xmin, deltax):

    tangent_out = jnp.exp(deltax * (1. * ((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1))) * ((1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))))**(-3) * ((-2.) * deltax * (1. + jnp.exp(deltax * (1. * ((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1)))) * ((deltax + ((-1.) * x + xmin)))**(-3) + ((-2.) * deltax * jnp.exp(deltax * (1. * ((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1))) * ((x + (-1.) * xmin))**(-3) * (((-1.) * deltax + (x + (-1.) * xmin)))**(-4) * ((1. * (deltax)**2 + ((-2.) * deltax * x + (2. * (x)**2 + (2. * deltax * xmin + ((-4.) * x * xmin + 2. * (xmin)**2))))))**2 + (1. * deltax * (1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))) * ((x + (-1.) * xmin))**(-3) * (((-1.) * deltax + (x + (-1.) * xmin)))**(-4) * ((1. * (deltax)**2 + ((-2.) * deltax * x + (2. * (x)**2 + (2. * deltax * xmin + ((-4.) * x * xmin + 2. * (xmin)**2))))))**2 + (-1.) * (1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))) * ((-1.) * ((x + (-1.) * xmin))**(-2) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-2)))))
    
    tangent_out = jnp.nan_to_num(tangent_out)
    return tangent_out

@custom_jvp
def planck_taper_der_xmin(x, xmin, deltax):

    tangent_out = jnp.where(x >= xmin+deltax, 0., jnp.where(x < xmin, 0., -deltax*jnp.exp(deltax*(1./(x-xmin) + 1./(x - xmin - deltax)))*(1./(x-xmin)**2. + 1./(x - xmin - deltax)**2.)/(1. + jnp.exp(deltax*(1./(x-xmin) + 1./(x - xmin - deltax))))**2))
    
    tangent_out = jnp.nan_to_num(tangent_out)
    return tangent_out

def planck_taper_dder_xmin(x, xmin, deltax):

    tangent_out = deltax * jnp.exp(deltax * (1. * ((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1))) * ((1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))))**(-3) * ((-1.) * (1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))) * (2. * ((x + (-1.) * xmin))**(-3) + (-2.) * ((deltax + ((-1.) * x + xmin)))**(-3)) + (2. * deltax * jnp.exp(deltax * (1. * ((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1))) * ((1. * ((x + (-1.) * xmin))**(-2) + 1. * ((deltax + ((-1.) * x + xmin)))**(-2)))**2 + (-1.) * deltax * (1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))) * ((1. * ((x + (-1.) * xmin))**(-2) + 1. * ((deltax + ((-1.) * x + xmin)))**(-2)))**2))
    
    tangent_out = jnp.nan_to_num(tangent_out)
    return tangent_out

def planck_taper_der_xmin_x(x, xmin, deltax):

    tangent_out = (-2.) * deltax * jnp.exp(deltax * (1. * ((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1))) * ((1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))))**(-3) * ((x + -1 * xmin))**(-4) * ((deltax + (-1 * x + xmin)))**(-4) * (1. * deltax * jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1))) * ((1. * ((x + (-1.) * xmin))**2 + 1. * ((deltax + ((-1.) * x + xmin)))**2))**2 + ((-0.5) * deltax * (1. + jnp.exp(deltax * (1. * ((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1)))) * ((1. * ((x + (-1.) * xmin))**2 + 1. * ((deltax + ((-1.) * x + xmin)))**2))**2 + 1. * (1. + jnp.exp(deltax * (1. * ((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1)))) * (x + -1 * xmin) * (deltax + (-1 * x + xmin)) * (1. * ((x + (-1.) * xmin))**3 + (-1.) * ((deltax + ((-1.) * x + xmin)))**3)))
    
    tangent_out = jnp.nan_to_num(tangent_out)
    return tangent_out

def planck_taper_der_xmin_deltax(x, xmin, deltax):

    tangent_out = jnp.exp(deltax * (1. * ((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1))) * ((1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))))**(-3) * (2. * deltax * (1. + jnp.exp(deltax * (1. * ((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1)))) * ((deltax + ((-1.) * x + xmin)))**(-3) + (2. * deltax * jnp.exp(deltax * (1. * ((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1))) * ((x + (-1.) * xmin))**(-3) * (((-1.) * deltax + (x + (-1.) * xmin)))**(-4) * ((1. * (deltax)**2 + ((-2.) * deltax * x + (2. * (x)**2 + (2. * deltax * xmin + ((-4.) * x * xmin + 2. * (xmin)**2))))))**2 + ((-1.) * deltax * (1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))) * ((x + (-1.) * xmin))**(-3) * (((-1.) * deltax + (x + (-1.) * xmin)))**(-4) * ((1. * (deltax)**2 + ((-2.) * deltax * x + (2. * (x)**2 + (2. * deltax * xmin + ((-4.) * x * xmin + 2. * (xmin)**2))))))**2 + (-1.) * (1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))) * (1. * ((x + (-1.) * xmin))**(-2) + 1. * ((deltax + ((-1.) * x + xmin)))**(-2)))))
    
    tangent_out = jnp.nan_to_num(tangent_out)
    return tangent_out

@custom_jvp
def planck_taper_der_deltax(x, xmin, deltax):

    tangent_out = jnp.where(x >= xmin+deltax, 0., jnp.where(x < xmin, 0., (-0.25*deltax*deltax + 0.5*deltax*x - 0.5*x*x -0.5*deltax*xmin + x*xmin - 0.5*xmin*xmin) / (jnp.cosh(0.5*deltax*(1./(x-xmin) + 1./(x - xmin - deltax)))**2) / ((x-xmin) * (deltax -x + xmin)**2)))
    
    tangent_out = jnp.nan_to_num(tangent_out)
    return tangent_out

def planck_taper_dder_deltax(x, xmin, deltax):

    tangent_out = jnp.exp(deltax * (1. * ((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1))) * ((1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))))**(-3) * (2. * (1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))) * (1. * x + (-1.) * xmin) * ((deltax + ((-1.) * x + xmin)))**(-3) + (2. * jnp.exp(deltax * (1. * ((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1))) * ((1. * ((x + (-1.) * xmin))**(-1) + (1. * deltax * ((deltax + ((-1.) * x + xmin)))**(-2) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1))))**2 + (-1.) * (1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))) * ((1. * ((x + (-1.) * xmin))**(-1) + (1. * deltax * ((deltax + ((-1.) * x + xmin)))**(-2) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1))))**2))
    
    tangent_out = jnp.nan_to_num(tangent_out)
    return tangent_out

def planck_taper_der_deltax_x(x, xmin, deltax):

    tangent_out = jnp.exp(deltax * (1. * ((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1))) * ((1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))))**(-3) * ((-2.) * deltax * jnp.exp(deltax * (1. * ((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1))) * ((x + (-1.) * xmin))**(-3) * (((-1.) * deltax + (x + (-1.) * xmin)))**(-4) * ((1. * (deltax)**2 + ((-2.) * deltax * x + (2. * (x)**2 + (2. * deltax * xmin + ((-4.) * x * xmin + 2. * (xmin)**2))))))**2 + (1. * deltax * (1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))) * ((x + (-1.) * xmin))**(-3) * (((-1.) * deltax + (x + (-1.) * xmin)))**(-4) * ((1. * (deltax)**2 + ((-2.) * deltax * x + (2. * (x)**2 + (2. * deltax * xmin + ((-4.) * x * xmin + 2. * (xmin)**2))))))**2 + (-1.) * (1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))) * ((-1.) * ((x + (-1.) * xmin))**(-2) + (2. * deltax * ((deltax + ((-1.) * x + xmin)))**(-3) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-2)))))
    
    tangent_out = jnp.nan_to_num(tangent_out)
    return tangent_out

def planck_taper_der_deltax_xmin(x, xmin, deltax):

    tangent_out = jnp.exp(deltax * (1. * ((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1))) * ((1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))))**(-3) * (2. * deltax * jnp.exp(deltax * (1. * ((x + (-1.) * xmin))**(-1) + (-1.) * ((deltax + ((-1.) * x + xmin)))**(-1))) * ((x + (-1.) * xmin))**(-3) * (((-1.) * deltax + (x + (-1.) * xmin)))**(-4) * ((1. * (deltax)**2 + ((-2.) * deltax * x + (2. * (x)**2 + (2. * deltax * xmin + ((-4.) * x * xmin + 2. * (xmin)**2))))))**2 + ((-1.) * deltax * (1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))) * ((x + (-1.) * xmin))**(-3) * (((-1.) * deltax + (x + (-1.) * xmin)))**(-4) * ((1. * (deltax)**2 + ((-2.) * deltax * x + (2. * (x)**2 + (2. * deltax * xmin + ((-4.) * x * xmin + 2. * (xmin)**2))))))**2 + (-1.) * (1. + jnp.exp((1. * deltax * ((x + (-1.) * xmin))**(-1) + (-1.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-1)))) * (1. * ((x + (-1.) * xmin))**(-2) + ((-2.) * deltax * ((deltax + ((-1.) * x + xmin)))**(-3) + 1. * ((deltax + ((-1.) * x + xmin)))**(-2)))))
    
    tangent_out = jnp.nan_to_num(tangent_out)
    return tangent_out

planck_taper.defjvps(lambda x_dot, primal_out, x, xmin, deltax: planck_taper_der_x(x,xmin,deltax) * x_dot,
                     lambda xmin_dot, primal_out, x, xmin, deltax: planck_taper_der_xmin(x,xmin,deltax) * xmin_dot,
                     lambda deltax_dot, primal_out, x, xmin, deltax: planck_taper_der_deltax(x,xmin,deltax) * deltax_dot)

planck_taper_der_x.defjvps(lambda x_dot, primal_out, x, xmin, deltax: planck_taper_dder_x(x,xmin,deltax) * x_dot,
                     lambda xmin_dot, primal_out, x, xmin, deltax: planck_taper_der_x_xmin(x,xmin,deltax) * xmin_dot,
                     lambda deltax_dot, primal_out, x, xmin, deltax: planck_taper_der_x_deltax(x,xmin,deltax) * deltax_dot)

planck_taper_der_xmin.defjvps(lambda x_dot, primal_out, x, xmin, deltax: planck_taper_der_xmin_x(x,xmin,deltax) * x_dot,
                     lambda xmin_dot, primal_out, x, xmin, deltax: planck_taper_dder_xmin(x,xmin,deltax) * xmin_dot,
                     lambda deltax_dot, primal_out, x, xmin, deltax: planck_taper_der_xmin_deltax(x,xmin,deltax) * deltax_dot)

planck_taper_der_deltax.defjvps(lambda x_dot, primal_out, x, xmin, deltax: planck_taper_der_deltax_x(x,xmin,deltax) * x_dot,
                     lambda xmin_dot, primal_out, x, xmin, deltax: planck_taper_der_deltax_xmin(x,xmin,deltax) * xmin_dot,
                     lambda deltax_dot, primal_out, x, xmin, deltax: planck_taper_dder_deltax(x,xmin,deltax) * deltax_dot)

@custom_jvp
def exp_masfunfit_NSBH(x, a1, a2, a3, b1, b2, b3):
    return  1./(1./(a1*1e11*jnp.exp(-b1*x) + a2*1e2*jnp.exp(-b2*x)) + 1./(a3*1e-3*jnp.exp(b3*x)))

def exp_masfunfit_NSBH_der_x(x, a1, a2, a3, b1, b2, b3):
     return (-100.) * a3 * jnp.exp(b3 * x) * ((1e5 * a2 * jnp.exp(b1 * x) + (1e14 * a1 * jnp.exp(b2 * x) + a3 * jnp.exp((b1 + (b2 + b3)) * x))))**(-2) * ((-1e5) * (a2)**2 * b3 * jnp.exp(2 * b1 * x) + ((-1.e23) * (a1)**2 * b3 * jnp.exp(2 * b2 * x) + ((-0.2e15) * a1 * a2 * b3 * jnp.exp((b1 + b2) * x) + (a2 * a3 * b2 * jnp.exp((2 * b1 + (b2 + b3)) * x) + 1e9 * a1 * a3 * b1 * jnp.exp((b1 + (2. * b2 + b3)) * x)))))
def exp_masfunfit_NSBH_der_a1(x, a1, a2, a3, b1, b2, b3):
    return 1e11 * (a3)**2 * jnp.exp((b1 + 2 * (b2 + b3)) * x) * ((1e5 * a2 * jnp.exp(b1 * x) + (1e14 * a1 * jnp.exp(b2 * x) + a3 * jnp.exp((b1 + (b2 + b3)) * x))))**(-2)
def exp_masfunfit_NSBH_der_a2(x, a1, a2, a3, b1, b2, b3):
    return 1e2 * (a3)**2 * jnp.exp((2 * b1 + (b2 + 2 * b3)) * x) * ((1e5 * a2 * jnp.exp(b1 * x) + (1e14 * a1 * jnp.exp(b2 * x) + a3 * jnp.exp((b1 + (b2 + b3)) * x))))**(-2)
def exp_masfunfit_NSBH_der_a3(x, a1, a2, a3, b1, b2, b3):
    return 1e-3 * jnp.exp(b3 * x) * ((a2 * jnp.exp(b1 * x) + 1e9 * a1 * jnp.exp(b2 * x)))**2 * ((a2 * jnp.exp(b1 * x) + (1e9 * a1 * jnp.exp(b2 * x) + 1e-5 * a3 * jnp.exp((b1 + (b2 + b3)) * x))))**(-2)
def exp_masfunfit_NSBH_der_b1(x, a1, a2, a3, b1, b2, b3):
    return (-1e11) * a1 * (a3)**2 * jnp.exp((b1 + 2 * (b2 + b3)) * x) * ((1e5 * a2 * jnp.exp(b1 * x) + (1e14 * a1 * jnp.exp(b2 * x) + a3 * jnp.exp((b1 + (b2 + b3)) * x))))**(-2) * x
def exp_masfunfit_NSBH_der_b2(x, a1, a2, a3, b1, b2, b3):
    return (-1e-8) * a2 * (a3)**2 * jnp.exp((2 * b1 + (b2 + 2 * b3)) * x) * ((a2 * jnp.exp(b1 * x) + (1e9 * a1 * jnp.exp(b2 * x) + 1e-5 * a3 * jnp.exp((b1 + (b2 + b3)) * x))))**(-2) * x
def exp_masfunfit_NSBH_der_b3(x, a1, a2, a3, b1, b2, b3):
    return 1e7 * a3 * jnp.exp(b3 * x) * ((a2 * jnp.exp(b1 * x) + 1e9 * a1 * jnp.exp(b2 * x)))**2 * ((1e5 * a2 * jnp.exp(b1 * x) + (1e14 * a1 * jnp.exp(b2 * x) + a3 * jnp.exp((b1 + (b2 + b3)) * x))))**(-2) * x    

exp_masfunfit_NSBH.defjvps(lambda x_dot, primal_out, x, a1, a2, a3, b1, b2, b3: exp_masfunfit_NSBH_der_x(x, a1, a2, a3, b1, b2, b3) * x_dot,
                     lambda a1_dot, primal_out, x, a1, a2, a3, b1, b2, b3: exp_masfunfit_NSBH_der_a1(x, a1, a2, a3, b1, b2, b3) * a1_dot,
                     lambda a2_dot, primal_out, x, a1, a2, a3, b1, b2, b3: exp_masfunfit_NSBH_der_a2(x, a1, a2, a3, b1, b2, b3) * a2_dot,
                     lambda a3_dot, primal_out, x, a1, a2, a3, b1, b2, b3: exp_masfunfit_NSBH_der_a3(x, a1, a2, a3, b1, b2, b3) * a3_dot,
                     lambda b1_dot, primal_out, x, a1, a2, a3, b1, b2, b3: exp_masfunfit_NSBH_der_b1(x, a1, a2, a3, b1, b2, b3) * b1_dot,
                     lambda b2_dot, primal_out, x, a1, a2, a3, b1, b2, b3: exp_masfunfit_NSBH_der_b2(x, a1, a2, a3, b1, b2, b3) * b2_dot,
                     lambda b3_dot, primal_out, x, a1, a2, a3, b1, b2, b3: exp_masfunfit_NSBH_der_b3(x, a1, a2, a3, b1, b2, b3) * b3_dot)

def normCDF_filter(x, edge, sig):
    """
    Smoothing function exploiting the standard normal cumulative distribution function.

    :param x: array of values to be smoothed
    :param edge: lower bound of the smoothing function
    :param sig: width of the smoothing function on the lower side
    
    :return: smoothed array
    :return type: array
    """
    return 0.5*(1. + erf((jnp.log(x)-jnp.log(edge))/(np.sqrt(2.)*sig)))

def normCDF_filter_hl(x, xl, sigl, xh, sigh):
    """
    Smoothing function exploiting the standard normal cumulative distribution function. Acts both on the lower and upper side of the function.

    :param x: array of values to be smoothed
    :param xl: lower bound of the smoothing function
    :param sigl: width of the smoothing function on the lower side
    :param xh: upper bound of the smoothing function
    :param sigh: width of the smoothing function on the upper side

    :return: smoothed array
    :return type: array
    """
    return normCDF_filter(x, xl, sigl)*(1. - normCDF_filter(x, xh, sigh))

def polynomial_filter(x, edge, sig):
    """
    Smoothing function exploiting a tuned polynomial function.

    :param x: array of values to be smoothed
    :param edge: lower bound of the smoothing function
    :param sig: width of the smoothing function on the lower side
    
    :return: smoothed array
    :return type: array
    """

    return jnp.where(x<edge, 0., jnp.where(x<=edge+sig, ((x-edge)**3) * (10.*sig**2 - 15.*(x-edge)*sig + 6.*(x-edge)**2) / (sig**5), 1.))

def polynomial_filter_hl(x, xl, sigl, xh, sigh):
    """
    Smoothing function exploiting a tuned polynomial function. Acts both on the lower and upper side of the function.

    :param x: array of values to be smoothed
    :param xl: lower bound of the smoothing function
    :param sigl: width of the smoothing function on the lower side
    :param xh: upper bound of the smoothing function
    :param sigh: width of the smoothing function on the upper side

    :return: smoothed array
    :return type: array
    """
    return polynomial_filter(x, xl-sigl, sigl)*(1. - polynomial_filter(x, xh, sigh))

def polynomial_filter_hl_integral(xl, sigl, xh, sigh):
    """
    Integral of smoothing function exploiting a tuned polynomial function.

    :param xl: lower bound of the smoothing function
    :param sigl: width of the smoothing function on the lower side
    :param xh: upper bound of the smoothing function
    :param sigh: width of the smoothing function on the upper side

    :return: integral of the smoothing function exploiting a tuned polynomial function
    :return type: array
    """
    return 0.5 * (sigh + (sigl + (2 * xh - 2 * xl)))

def polynomial_filter_hl_invpowerlaw_integral(xl, sigl, xh, sigh, alpha):
    """
    Integral of smoothing function exploiting a tuned polynomial function times an inverse power-law.

    :param xl: lower bound of the smoothing function
    :param sigl: width of the smoothing function on the lower side
    :param xh: upper bound of the smoothing function
    :param sigh: width of the smoothing function on the upper side
    :param alpha: power-law exponent

    :return: integral of the smoothing function exploiting a tuned polynomial function times an inverse power-law
    :return type: array
    """
    return 60 * ((-6 + alpha))**(-1) * ((-5 + alpha))**(-1) * ((-4 + alpha))**(-1) * ((-3 + alpha))**(-1) * ((-2 + alpha))**(-1) * ((-1 + alpha))**(-1) * (sigh)**(-5) * (sigl)**(-5) * ((xh)**(2) * (sigh + xh) * xl * (-1 * sigl + xl))**(-1 * alpha) * (6 * (-1 + alpha) * alpha * (sigh)**(4) * (sigl)**(5) * (xh)**((2 + 2 * alpha)) * (xl * (-1 * sigl + xl))**(alpha) + (4 * alpha * (4 + alpha) * (sigh)**(3) * (sigl)**(5) * (xh)**((3 + 2 * alpha)) * (xl * (-1 * sigl + xl))**(alpha) + ((-3 + alpha) * (-2 + alpha) * (sigh)**(6) * (sigl)**(5) * ((xh)**(2) * xl * (-1 * sigl + xl))**(alpha) + (12 * (sigl)**(5) * (xh)**(6) * ((xh)**(2 * alpha) * (xl * (-1 * sigl + xl))**(alpha) + -1 * (xh * (sigh + xh) * xl * (-1 * sigl + xl))**(alpha)) + (6 * sigh * (sigl)**(5) * (xh)**(5) * ((6 + alpha) * (xh)**(2 * alpha) * (xl * (-1 * sigl + xl))**(alpha) + (-6 + alpha) * (xh * (sigh + xh) * xl * (-1 * sigl + xl))**(alpha)) + ((sigh)**(2) * (sigl)**(5) * (xh)**(4) * ((30 + alpha * (19 + alpha)) * (xh)**(2 * alpha) * (xl * (-1 * sigl + xl))**(alpha) + -1 * (-6 + alpha) * (-5 + alpha) * (xh * (sigh + xh) * xl * (-1 * sigl + xl))**(alpha)) + (sigh)**(5) * ((-3 + alpha) * (-2 + alpha) * (sigl)**(6) * ((xh)**(2) * (sigh + xh) * xl)**(alpha) + (6 * (-1 + alpha) * alpha * (sigl)**(4) * (xl)**(2) * ((xh)**(2) * (sigh + xh) * xl)**(alpha) + (-4 * alpha * (4 + alpha) * (sigl)**(3) * (xl)**(3) * ((xh)**(2) * (sigh + xh) * xl)**(alpha) + (12 * (xl)**(6) * (((xh)**(2) * (sigh + xh) * xl)**(alpha) + -1 * (xh)**(2 * alpha) * ((sigh + xh) * (-1 * sigl + xl))**(alpha)) + (6 * sigl * (xl)**(5) * (-1 * (6 + alpha) * ((xh)**(2) * (sigh + xh) * xl)**(alpha) + -1 * (-6 + alpha) * (xh)**(2 * alpha) * ((sigh + xh) * (-1 * sigl + xl))**(alpha)) + ((sigl)**(2) * (xl)**(4) * ((30 + alpha * (19 + alpha)) * ((xh)**(2) * (sigh + xh) * xl)**(alpha) + -1 * (-6 + alpha) * (-5 + alpha) * (xh)**(2 * alpha) * ((sigh + xh) * (-1 * sigl + xl))**(alpha)) + -2 * (-2 + alpha) * (-3 + 2 * alpha) * (sigl)**(5) * (xl * ((xh)**(2) * (sigh + xh) * xl)**(alpha) + -1 * xh * ((xh)**(2) * xl * (-1 * sigl + xl))**(alpha))))))))))))))

def polynomial_filter_hl_invpowerlaw_integral_uptox(xextr, xl, sigl, xh, sigh, alpha):
    """
    Integral of smoothing function exploiting a tuned polynomial function times an inverse power-law up to value in range.

    :param xextr: upper extreme of the integral
    :param xl: lower bound of the smoothing function
    :param sigl: width of the smoothing function on the lower side
    :param xh: upper bound of the smoothing function
    :param sigh: width of the smoothing function on the upper side
    :param alpha: power-law exponent

    :return: integral of the smoothing function exploiting a tuned polynomial function times an inverse power-law
    :return type: array
    """

    def res_xextr_in_low(xextr, xl, sigl, xh, sigh, alpha):
        return (sigl)**(-5) * (6 * ((-1 * sigl + xl))**((6 + -1 * alpha)) * (6 * ((-6 + alpha))**(-1) * ((-5 + alpha))**(-1) * ((-4 + alpha))**(-1) * ((-3 + alpha))**(-1) + ((6 + -1 * alpha))**(-1) * (1 + (3 * (-6 + alpha) * ((-5 + alpha))**(-1) * (xextr)**(-1) * (sigl + -1 * xl) + (3 * (-6 + alpha) * ((-4 + alpha))**(-1) * (xextr)**(-2) * ((sigl + -1 * xl))**(2) + (-6 + alpha) * ((-3 + alpha))**(-1) * (xextr)**(-3) * ((sigl + -1 * xl))**(3)))) * ((xextr)**(-1) * (-1 * sigl + xl))**((-6 + alpha))) + (-3 * sigl * ((-1 * sigl + xl))**((5 + -1 * alpha)) * (6 * ((-5 + alpha))**(-1) * ((-4 + alpha))**(-1) * ((-3 + alpha))**(-1) * ((-2 + alpha))**(-1) + ((5 + -1 * alpha))**(-1) * (1 + (3 * (-5 + alpha) * ((-4 + alpha))**(-1) * (xextr)**(-1) * (sigl + -1 * xl) + (3 * (-5 + alpha) * ((-3 + alpha))**(-1) * (xextr)**(-2) * ((sigl + -1 * xl))**(2) + (-5 + alpha) * ((-2 + alpha))**(-1) * (xextr)**(-3) * ((sigl + -1 * xl))**(3)))) * ((xextr)**(-1) * (-1 * sigl + xl))**((-5 + alpha))) + (-12 * xl * ((-1 * sigl + xl))**((5 + -1 * alpha)) * (6 * ((-5 + alpha))**(-1) * ((-4 + alpha))**(-1) * ((-3 + alpha))**(-1) * ((-2 + alpha))**(-1) + ((5 + -1 * alpha))**(-1) * (1 + (3 * (-5 + alpha) * ((-4 + alpha))**(-1) * (xextr)**(-1) * (sigl + -1 * xl) + (3 * (-5 + alpha) * ((-3 + alpha))**(-1) * (xextr)**(-2) * ((sigl + -1 * xl))**(2) + (-5 + alpha) * ((-2 + alpha))**(-1) * (xextr)**(-3) * ((sigl + -1 * xl))**(3)))) * ((xextr)**(-1) * (-1 * sigl + xl))**((-5 + alpha))) + ((sigl)**(2) * ((-1 * sigl + xl))**((4 + -1 * alpha)) * (6 * ((24 + (-50 * alpha + (35 * (alpha)**(2) + (-10 * (alpha)**(3) + (alpha)**(4))))))**(-1) + ((4 + -1 * alpha))**(-1) * (1 + (3 * (-4 + alpha) * ((-3 + alpha))**(-1) * (xextr)**(-1) * (sigl + -1 * xl) + (3 * (-4 + alpha) * ((-2 + alpha))**(-1) * (xextr)**(-2) * ((sigl + -1 * xl))**(2) + (-4 + alpha) * ((-1 + alpha))**(-1) * (xextr)**(-3) * ((sigl + -1 * xl))**(3)))) * ((xextr)**(-1) * (-1 * sigl + xl))**((-4 + alpha))) + (3 * sigl * xl * ((-1 * sigl + xl))**((4 + -1 * alpha)) * (6 * ((24 + (-50 * alpha + (35 * (alpha)**(2) + (-10 * (alpha)**(3) + (alpha)**(4))))))**(-1) + ((4 + -1 * alpha))**(-1) * (1 + (3 * (-4 + alpha) * ((-3 + alpha))**(-1) * (xextr)**(-1) * (sigl + -1 * xl) + (3 * (-4 + alpha) * ((-2 + alpha))**(-1) * (xextr)**(-2) * ((sigl + -1 * xl))**(2) + (-4 + alpha) * ((-1 + alpha))**(-1) * (xextr)**(-3) * ((sigl + -1 * xl))**(3)))) * ((xextr)**(-1) * (-1 * sigl + xl))**((-4 + alpha))) + 6 * (xl)**(2) * ((-1 * sigl + xl))**((4 + -1 * alpha)) * (6 * ((24 + (-50 * alpha + (35 * (alpha)**(2) + (-10 * (alpha)**(3) + (alpha)**(4))))))**(-1) + ((4 + -1 * alpha))**(-1) * (1 + (3 * (-4 + alpha) * ((-3 + alpha))**(-1) * (xextr)**(-1) * (sigl + -1 * xl) + (3 * (-4 + alpha) * ((-2 + alpha))**(-1) * (xextr)**(-2) * ((sigl + -1 * xl))**(2) + (-4 + alpha) * ((-1 + alpha))**(-1) * (xextr)**(-3) * ((sigl + -1 * xl))**(3)))) * ((xextr)**(-1) * (-1 * sigl + xl))**((-4 + alpha))))))))
    
    def res_xextr_in_btw(xextr, xl, sigl, xh, sigh, alpha):
        return (xextr**(1. - alpha) - xl**(1. - alpha)) / (1. - alpha)#((-1 + alpha))**(-1) * (xextr * (sigl + xl))**(-1 * alpha) * ((xextr)**(alpha) * (sigl + xl) + -1 * xextr * ((sigl + xl))**(alpha))
    
    def res_xextr_in_high(xextr, xl, sigl, xh, sigh, alpha):
        return ((-6 + alpha))**(-1) * ((-5 + alpha))**(-1) * ((-4 + alpha))**(-1) * ((-3 + alpha))**(-1) * ((-2 + alpha))**(-1) * ((-1 + alpha))**(-1) * (sigh)**(-5) * (xextr * xh)**(-1 * alpha) * (-720 * (xextr)**(alpha) * (xh)**(6) + (6 * xextr * (xh)**(alpha) * ((-5 + alpha) * (-4 + alpha) * (-3 + alpha) * (-2 + alpha) * (-1 + alpha) * (xextr)**(5) + (-5 * (-6 + alpha) * (-4 + alpha) * (-3 + alpha) * (-2 + alpha) * (-1 + alpha) * (xextr)**(4) * xh + (10 * (-6 + alpha) * (-5 + alpha) * (-3 + alpha) * (-2 + alpha) * (-1 + alpha) * (xextr)**(3) * (xh)**(2) + (-10 * (-6 + alpha) * (-5 + alpha) * (-4 + alpha) * (-2 + alpha) * (-1 + alpha) * (xextr)**(2) * (xh)**(3) + (5 * (-6 + alpha) * (-5 + alpha) * (-4 + alpha) * (-3 + alpha) * (-1 + alpha) * xextr * (xh)**(4) + -1 * (-6 + alpha) * (-5 + alpha) * (-4 + alpha) * (-3 + alpha) * (-2 + alpha) * (xh)**(5)))))) + (-1 * (-6 + alpha) * (-5 + alpha) * (-4 + alpha) * (-3 + alpha) * (-2 + alpha) * (sigh)**(5) * (-1 * (xextr)**(alpha) * xh + xextr * (xh)**(alpha)) + (10 * (-6 + alpha) * (-5 + alpha) * (sigh)**(2) * (-6 * (xextr)**(alpha) * (xh)**(4) + xextr * (xh)**(alpha) * ((-3 + alpha) * (-2 + alpha) * (-1 + alpha) * (xextr)**(3) + (-3 * (-4 + alpha) * (-2 + alpha) * (-1 + alpha) * (xextr)**(2) * xh + (3 * (-4 + alpha) * (-3 + alpha) * (-1 + alpha) * xextr * (xh)**(2) + -1 * (-4 + alpha) * (-3 + alpha) * (-2 + alpha) * (xh)**(3))))) + -15 * (-6 + alpha) * sigh * (-24 * (xextr)**(alpha) * (xh)**(5) + xextr * (xh)**(alpha) * ((-4 + alpha) * (-3 + alpha) * (-2 + alpha) * (-1 + alpha) * (xextr)**(4) + (-4 * (-5 + alpha) * (-3 + alpha) * (-2 + alpha) * (-1 + alpha) * (xextr)**(3) * xh + (6 * (-5 + alpha) * (-4 + alpha) * (-2 + alpha) * (-1 + alpha) * (xextr)**(2) * (xh)**(2) + (-4 * (-5 + alpha) * (-4 + alpha) * (-3 + alpha) * (-1 + alpha) * xextr * (xh)**(3) + (-5 + alpha) * (-4 + alpha) * (-3 + alpha) * (-2 + alpha) * (xh)**(4))))))))))
    
    def res_integ_upto_xl(xl, sigl, xh, sigh, alpha):
        return ((-6 + alpha))**(-1) * ((-5 + alpha))**(-1) * ((-4 + alpha))**(-1) * ((-3 + alpha))**(-1) * ((-2 + alpha))**(-1) * ((-1 + alpha))**(-1) * (sigl)**(-5) * ((-1 * sigl + xl))**(-1 * alpha) * (60 * ((sigl + -1 * xl))**(4) * ((-3 + alpha) * (-2 + alpha) * (sigl)**(2) + (-6 * (-2 + alpha) * sigl * xl + 12 * (xl)**(2))) + -1 * ((1 + -1 * sigl * (xl)**(-1)))**(alpha) * xl * ((-6 + alpha) * (-5 + alpha) * (-4 + alpha) * (-3 + alpha) * (-2 + alpha) * (sigl)**(5) + (60 * (-6 + alpha) * (-5 + alpha) * (sigl)**(2) * (xl)**(3) + (360 * (-6 + alpha) * sigl * (xl)**(4) + 720 * (xl)**(5)))))
    
    def res_integ_upto_xh(xl, sigl, xh, sigh, alpha):
        return (xh**(1. - alpha) - xl**(1. - alpha)) / (1. - alpha)#((1 + -1 * alpha))**(-1) * ((xh)**((1 + -1 * alpha)) + -1 * ((-1 * sigl + xl))**((1 + -1 * alpha)))
    
    return jnp.where(xextr<=xl-sigl, 0., jnp.where(xextr<xl, res_xextr_in_low(xextr, xl, sigl, xh, sigh, alpha), jnp.where(xextr==xl, res_integ_upto_xl(xl, sigl, xh, sigh, alpha), jnp.where(xextr<xh, res_xextr_in_btw(xextr, xl, sigl, xh, sigh, alpha) + res_integ_upto_xl(xl, sigl, xh, sigh, alpha), jnp.where(xextr==xh, res_integ_upto_xl(xl, sigl, xh, sigh, alpha) + res_integ_upto_xh(xl, sigl, xh, sigh, alpha), jnp.where(xextr<xh+sigh, res_xextr_in_high(xextr, xl, sigl, xh, sigh, alpha) + res_integ_upto_xl(xl, sigl, xh, sigh, alpha) + res_integ_upto_xh(xl, sigl, xh, sigh, alpha), polynomial_filter_hl_invpowerlaw_integral(xl, sigl, xh, sigh, alpha)))))))

def polynomial_filter_hl_invpowerlawnorm_integral(xl, sigl, xh, sigh, alpha):
    """
    Integral of smoothing function exploiting a tuned polynomial function times a normalized inverse power-law.

    :param xl: lower bound of the smoothing function
    :param sigl: width of the smoothing function on the lower side
    :param xh: upper bound of the smoothing function
    :param sigh: width of the smoothing function on the upper side
    :param alpha: power-law exponent

    :return: integral of the smoothing function exploiting a tuned polynomial function times a normalized inverse power-law
    :return type: array
    """

    def res_alpha_not_1(xl, sigl, xh, sigh, alpha):
        return 60 * ((-6 + alpha))**(-1) * ((-5 + alpha))**(-1) * ((-4 + alpha))**(-1) * ((-3 + alpha))**(-1) * ((-2 + alpha))**(-1) * (sigh)**(-5) * (sigl)**(-5) * (xh)**(-1 * alpha) * ((((sigh + xh))**(alpha) * (sigl + -1 * xl) + (sigh + xh) * ((-1 * sigl + xl))**(alpha)))**(-1) * (-6 * (-1 + alpha) * alpha * (sigh)**(4) * (sigl)**(5) * (xh)**((2 + alpha)) * ((-1 * sigl + xl))**(alpha) + (-4 * alpha * (4 + alpha) * (sigh)**(3) * (sigl)**(5) * (xh)**((3 + alpha)) * ((-1 * sigl + xl))**(alpha) + (-1 * (-3 + alpha) * (-2 + alpha) * (sigh)**(6) * (sigl)**(5) * (xh * (-1 * sigl + xl))**(alpha) + (12 * (sigl)**(5) * (xh)**(6) * (-1 * (xh * (-1 * sigl + xl))**(alpha) + ((sigh + xh) * (-1 * sigl + xl))**(alpha)) + (6 * sigh * (sigl)**(5) * (xh)**(5) * (-1 * (6 + alpha) * (xh * (-1 * sigl + xl))**(alpha) + -1 * (-6 + alpha) * ((sigh + xh) * (-1 * sigl + xl))**(alpha)) + ((sigh)**(2) * (sigl)**(5) * (xh)**(4) * (-1 * (30 + alpha * (19 + alpha)) * (xh * (-1 * sigl + xl))**(alpha) + (-6 + alpha) * (-5 + alpha) * ((sigh + xh) * (-1 * sigl + xl))**(alpha)) + (sigh)**(5) * (-1 * (-3 + alpha) * (-2 + alpha) * (sigl)**(6) * (xh * (sigh + xh))**(alpha) + (-6 * (-1 + alpha) * alpha * (sigl)**(4) * (xh * (sigh + xh))**(alpha) * (xl)**(2) + (4 * alpha * (4 + alpha) * (sigl)**(3) * (xh * (sigh + xh))**(alpha) * (xl)**(3) + (-2 * (-2 + alpha) * (-3 + 2 * alpha) * (sigl)**(5) * (-1 * (xh * (sigh + xh))**(alpha) * xl + xh * (xh * (-1 * sigl + xl))**(alpha)) + (-12 * (xl)**(6) * ((xh * (sigh + xh))**(alpha) + -1 * (xh * (sigh + xh) * (xl)**(-1) * (-1 * sigl + xl))**(alpha)) + (6 * sigl * (xl)**(5) * ((6 + alpha) * (xh * (sigh + xh))**(alpha) + (-6 + alpha) * (xh * (sigh + xh) * (xl)**(-1) * (-1 * sigl + xl))**(alpha)) + (sigl)**(2) * (xl)**(4) * (-1 * (30 + alpha * (19 + alpha)) * (xh * (sigh + xh))**(alpha) + (-6 + alpha) * (-5 + alpha) * (xh * (sigh + xh) * (xl)**(-1) * (-1 * sigl + xl))**(alpha))))))))))))))
    
    def res_alpha_equal_1(xl, sigl, xh, sigh):
        return 0.5 * (sigh)**(-5) * (sigl)**(-5) * (jnp.log((sigh + xh) * ((-1 * sigl + xl))**(-1)))**(-1) * (3 * sigh * sigl * (sigl * (sigh + xh) + -1 * sigh * xl) * (sigl * xh + sigh * xl) * (-4 * sigh * (sigl)**(2) * xh + (-4 * (sigl)**(2) * (xh)**(2) + (sigh)**(2) * ((sigl)**(2) + (4 * sigl * xl + -4 * (xl)**(2))))) + (2 * (sigl)**(5) * (xh)**(3) * (10 * (sigh)**(2) + (15 * sigh * xh + 6 * (xh)**(2))) * jnp.log((xh)**(-1) * (sigh + xh)) + (2 * (sigh)**(5) * (xl)**(3) * (10 * (sigl)**(2) + (-15 * sigl * xl + 6 * (xl)**(2))) * jnp.log((1 + -1 * sigl * (xl)**(-1))) + 2 * (sigh)**(5) * (sigl)**(5) * (jnp.log((sigh + xh)) + -1 * jnp.log((-1 * sigl + xl))))))
    
    return jnp.where(alpha==1., res_alpha_equal_1(xl, sigl, xh, sigh), res_alpha_not_1(xl, sigl, xh, sigh, alpha))

def polynomial_filter_hl_gaussian_integral_nonreg(xl, sigl, xh, sigh, mu, sigma):
    """
    Integral of smoothing function exploiting a tuned polynomial function times a Gaussian distribution. Do not use it when mmax-mum is higher that 85 or numerical instabilities will arise. 

    :param xl: lower bound of the smoothing function
    :param sigl: width of the smoothing function on the lower side
    :param xh: upper bound of the smoothing function
    :param sigh: width of the smoothing function on the upper side
    :param mu: mean of the Gaussian distribution
    :param sigma: standard deviation of the Gaussian distribution

    :return: integral of the smoothing function exploiting a tuned polynomial function times a Gaussian distribution
    :return type: array
    """
    return 0.5 * (jnp.pi)**(-0.5) * (sigh)**(-5) * (sigl)**(-5) * (-30 * (sigl)**(5) * (xh)**(2) * ((sigh + xh))**(2) * ((2)**(0.5) * (jnp.exp(-0.5 * ((xh + -1 * mu))**(2) * (sigma)**(-2)) + -1 * jnp.exp(-0.5 * ((sigh + (xh + -1 * mu)))**(2) * (sigma)**(-2))) * sigma + -1 * (jnp.pi)**(0.5) * mu * (erf((2)**(-0.5) * (xh + -1 * mu) * (sigma)**(-1)) + -1 * erf((2)**(-0.5) * (sigh + (xh + -1 * mu)) * (sigma)**(-1)))) + (-1 * (jnp.pi)**(0.5) * (sigl)**(5) * ((sigh + xh))**(3) * ((sigh)**(2) + (-3 * sigh * xh + 6 * (xh)**(2))) * (erf((2)**(-0.5) * (xh + -1 * mu) * (sigma)**(-1)) + -1 * erf((2)**(-0.5) * (sigh + (xh + -1 * mu)) * (sigma)**(-1))) + (-3 * (2)**(0.5) * jnp.exp(-0.5 * ((sigh)**(2) + (2 * sigh * (xh + -1 * mu) + 2 * ((xh + -1 * mu))**(2))) * (sigma)**(-2)) * (sigl)**(5) * (2 * jnp.exp(0.5 * ((sigh + (xh + -1 * mu)))**(2) * (sigma)**(-2)) * sigma * ((xh)**(4) + ((xh)**(3) * mu + ((xh)**(2) * (mu)**(2) + (xh * (mu)**(3) + ((mu)**(4) + ((4 * (xh)**(2) + (7 * xh * mu + 9 * (mu)**(2))) * (sigma)**(2) + 8 * (sigma)**(4))))))) + (-2 * jnp.exp(0.5 * ((xh + -1 * mu))**(2) * (sigma)**(-2)) * sigma * ((sigh)**(4) + (4 * (sigh)**(3) * xh + (6 * (sigh)**(2) * (xh)**(2) + (4 * sigh * (xh)**(3) + ((xh)**(4) + ((sigh)**(3) * mu + (3 * (sigh)**(2) * xh * mu + (3 * sigh * (xh)**(2) * mu + ((xh)**(3) * mu + ((sigh)**(2) * (mu)**(2) + (2 * sigh * xh * (mu)**(2) + ((xh)**(2) * (mu)**(2) + (sigh * (mu)**(3) + (xh * (mu)**(3) + ((mu)**(4) + ((4 * ((sigh + xh))**(2) + (7 * (sigh + xh) * mu + 9 * (mu)**(2))) * (sigma)**(2) + 8 * (sigma)**(4))))))))))))))))) + jnp.exp(0.5 * (((xh + -1 * mu))**(2) + ((sigh + (xh + -1 * mu)))**(2)) * (sigma)**(-2)) * (2 * jnp.pi)**(0.5) * mu * ((mu)**(4) + (10 * (mu)**(2) * (sigma)**(2) + 15 * (sigma)**(4))) * (-1 * erf((2)**(-0.5) * (xh + -1 * mu) * (sigma)**(-1)) + erf((2)**(-0.5) * (sigh + (xh + -1 * mu)) * (sigma)**(-1))))) + (-30 * jnp.exp(-0.5 * (((xh + -1 * mu))**(2) + ((sigh + (xh + -1 * mu)))**(2)) * (sigma)**(-2)) * (sigl)**(5) * xh * ((sigh)**(2) + (3 * sigh * xh + 2 * (xh)**(2))) * ((2)**(0.5) * jnp.exp(0.5 * ((xh + -1 * mu))**(2) * (sigma)**(-2)) * (sigh + (xh + mu)) * sigma + -1 * jnp.exp(0.5 * ((sigh + (xh + -1 * mu)))**(2) * (sigma)**(-2)) * ((2)**(0.5) * (xh + mu) * sigma + jnp.exp(0.5 * ((xh + -1 * mu))**(2) * (sigma)**(-2)) * (jnp.pi)**(0.5) * ((mu)**(2) + (sigma)**(2)) * (-1 * erf((2)**(-0.5) * (xh + -1 * mu) * (sigma)**(-1)) + erf((2)**(-0.5) * (sigh + (xh + -1 * mu)) * (sigma)**(-1))))) + (10 * jnp.exp(-0.5 * (((xh + -1 * mu))**(2) + ((sigh + (xh + -1 * mu)))**(2)) * (sigma)**(-2)) * (sigl)**(5) * ((sigh)**(2) + (6 * sigh * xh + 6 * (xh)**(2))) * ((2)**(0.5) * jnp.exp(0.5 * ((xh + -1 * mu))**(2) * (sigma)**(-2)) * sigma * (((sigh + xh))**(2) + ((sigh + xh) * mu + ((mu)**(2) + 2 * (sigma)**(2)))) + -1 * jnp.exp(0.5 * ((sigh + (xh + -1 * mu)))**(2) * (sigma)**(-2)) * ((2)**(0.5) * sigma * ((xh)**(2) + (xh * mu + ((mu)**(2) + 2 * (sigma)**(2)))) + jnp.exp(0.5 * ((xh + -1 * mu))**(2) * (sigma)**(-2)) * (jnp.pi)**(0.5) * mu * ((mu)**(2) + 3 * (sigma)**(2)) * (-1 * erf((2)**(-0.5) * (xh + -1 * mu) * (sigma)**(-1)) + erf((2)**(-0.5) * (sigh + (xh + -1 * mu)) * (sigma)**(-1))))) + (-15 * jnp.exp(-0.5 * (((xh + -1 * mu))**(2) + ((sigh + (xh + -1 * mu)))**(2)) * (sigma)**(-2)) * (sigl)**(5) * (sigh + 2 * xh) * ((2)**(0.5) * jnp.exp(0.5 * ((xh + -1 * mu))**(2) * (sigma)**(-2)) * sigma * ((sigh)**(3) + ((xh)**(3) + ((xh)**(2) * mu + (xh * (mu)**(2) + ((mu)**(3) + ((sigh)**(2) * (3 * xh + mu) + ((3 * xh + 5 * mu) * (sigma)**(2) + sigh * (3 * (xh)**(2) + (2 * xh * mu + ((mu)**(2) + 3 * (sigma)**(2))))))))))) + -1 * jnp.exp(0.5 * ((sigh + (xh + -1 * mu)))**(2) * (sigma)**(-2)) * ((2)**(0.5) * sigma * ((xh)**(3) + ((xh)**(2) * mu + (xh * (mu)**(2) + ((mu)**(3) + (3 * xh + 5 * mu) * (sigma)**(2))))) + jnp.exp(0.5 * ((xh + -1 * mu))**(2) * (sigma)**(-2)) * (jnp.pi)**(0.5) * ((mu)**(4) + (6 * (mu)**(2) * (sigma)**(2) + 3 * (sigma)**(4))) * (-1 * erf((2)**(-0.5) * (xh + -1 * mu) * (sigma)**(-1)) + erf((2)**(-0.5) * (sigh + (xh + -1 * mu)) * (sigma)**(-1))))) + ((jnp.pi)**(0.5) * (sigh)**(5) * (sigl)**(5) * (erf((2)**(-0.5) * (xh + -1 * mu) * (sigma)**(-1)) + -1 * erf((2)**(-0.5) * (xl + -1 * mu) * (sigma)**(-1))) + ((jnp.pi)**(0.5) * (sigh)**(5) * ((sigl + -1 * xl))**(3) * ((sigl)**(2) + (3 * sigl * xl + 6 * (xl)**(2))) * (erf((2)**(-0.5) * (xl + -1 * mu) * (sigma)**(-1)) + erf((2)**(-0.5) * (sigl + (-1 * xl + mu)) * (sigma)**(-1))) + (30 * (sigh)**(5) * ((sigl + -1 * xl))**(2) * (xl)**(2) * (-1 * (2)**(0.5) * jnp.exp(-0.5 * ((sigl + (-1 * xl + mu)))**(2) * (sigma)**(-2)) * (-1 + jnp.exp(0.5 * sigl * (sigl + (-2 * xl + 2 * mu)) * (sigma)**(-2))) * sigma + (jnp.pi)**(0.5) * mu * (erf((2)**(-0.5) * (xl + -1 * mu) * (sigma)**(-1)) + erf((2)**(-0.5) * (sigl + (-1 * xl + mu)) * (sigma)**(-1)))) + (-3 * (2)**(0.5) * jnp.exp(-0.5 * ((sigl)**(2) + (2 * ((xl + -1 * mu))**(2) + 2 * sigl * (-1 * xl + mu))) * (sigma)**(-2)) * (sigh)**(5) * (2 * jnp.exp(0.5 * ((sigl + (-1 * xl + mu)))**(2) * (sigma)**(-2)) * sigma * ((xl)**(4) + ((xl)**(3) * mu + ((xl)**(2) * (mu)**(2) + (xl * (mu)**(3) + ((mu)**(4) + ((4 * (xl)**(2) + (7 * xl * mu + 9 * (mu)**(2))) * (sigma)**(2) + 8 * (sigma)**(4))))))) + (-2 * jnp.exp(0.5 * ((xl + -1 * mu))**(2) * (sigma)**(-2)) * sigma * ((sigl)**(4) + ((xl)**(4) + ((xl)**(3) * mu + ((xl)**(2) * (mu)**(2) + (xl * (mu)**(3) + ((mu)**(4) + (-1 * (sigl)**(3) * (4 * xl + mu) + ((4 * (xl)**(2) + (7 * xl * mu + 9 * (mu)**(2))) * (sigma)**(2) + (8 * (sigma)**(4) + ((sigl)**(2) * (6 * (xl)**(2) + (3 * xl * mu + ((mu)**(2) + 4 * (sigma)**(2)))) + -1 * sigl * (4 * (xl)**(3) + (3 * (xl)**(2) * mu + (2 * xl * (mu)**(2) + ((mu)**(3) + (8 * xl + 7 * mu) * (sigma)**(2))))))))))))))) + -1 * jnp.exp(0.5 * (((xl + -1 * mu))**(2) + ((sigl + (-1 * xl + mu)))**(2)) * (sigma)**(-2)) * (2 * jnp.pi)**(0.5) * mu * ((mu)**(4) + (10 * (mu)**(2) * (sigma)**(2) + 15 * (sigma)**(4))) * (erf((2)**(-0.5) * (xl + -1 * mu) * (sigma)**(-1)) + erf((2)**(-0.5) * (sigl + (-1 * xl + mu)) * (sigma)**(-1))))) + (-30 * jnp.exp(-0.5 * ((sigl)**(2) + (2 * ((xl + -1 * mu))**(2) + 2 * sigl * (-1 * xl + mu))) * (sigma)**(-2)) * (sigh)**(5) * xl * ((sigl)**(2) + (-3 * sigl * xl + 2 * (xl)**(2))) * ((2)**(0.5) * jnp.exp(0.5 * ((xl + -1 * mu))**(2) * (sigma)**(-2)) * (-1 * sigl + (xl + mu)) * sigma + jnp.exp(0.5 * ((sigl + (-1 * xl + mu)))**(2) * (sigma)**(-2)) * (-1 * (2)**(0.5) * (xl + mu) * sigma + jnp.exp(0.5 * ((xl + -1 * mu))**(2) * (sigma)**(-2)) * (jnp.pi)**(0.5) * ((mu)**(2) + (sigma)**(2)) * (erf((2)**(-0.5) * (xl + -1 * mu) * (sigma)**(-1)) + erf((2)**(-0.5) * (sigl + (-1 * xl + mu)) * (sigma)**(-1))))) + (10 * jnp.exp(-0.5 * (((xl + -1 * mu))**(2) + ((sigl + (-1 * xl + mu)))**(2)) * (sigma)**(-2)) * (sigh)**(5) * ((sigl)**(2) + (-6 * sigl * xl + 6 * (xl)**(2))) * ((2)**(0.5) * jnp.exp(0.5 * ((xl + -1 * mu))**(2) * (sigma)**(-2)) * sigma * (((sigl + -1 * xl))**(2) + ((-1 * sigl + xl) * mu + ((mu)**(2) + 2 * (sigma)**(2)))) + jnp.exp(0.5 * ((sigl + (-1 * xl + mu)))**(2) * (sigma)**(-2)) * (-1 * (2)**(0.5) * sigma * ((xl)**(2) + (xl * mu + ((mu)**(2) + 2 * (sigma)**(2)))) + jnp.exp(0.5 * ((xl + -1 * mu))**(2) * (sigma)**(-2)) * (jnp.pi)**(0.5) * mu * ((mu)**(2) + 3 * (sigma)**(2)) * (erf((2)**(-0.5) * (xl + -1 * mu) * (sigma)**(-1)) + erf((2)**(-0.5) * (sigl + (-1 * xl + mu)) * (sigma)**(-1))))) + 15 * jnp.exp(-0.5 * ((sigl)**(2) + (2 * ((xl + -1 * mu))**(2) + 2 * sigl * (-1 * xl + mu))) * (sigma)**(-2)) * (sigh)**(5) * (sigl + -2 * xl) * ((2)**(0.5) * jnp.exp(0.5 * ((xl + -1 * mu))**(2) * (sigma)**(-2)) * sigma * (-1 * (sigl)**(3) + ((xl)**(3) + ((xl)**(2) * mu + (xl * (mu)**(2) + ((mu)**(3) + ((sigl)**(2) * (3 * xl + mu) + ((3 * xl + 5 * mu) * (sigma)**(2) + -1 * sigl * (3 * (xl)**(2) + (2 * xl * mu + ((mu)**(2) + 3 * (sigma)**(2))))))))))) + jnp.exp(0.5 * ((sigl + (-1 * xl + mu)))**(2) * (sigma)**(-2)) * (-1 * (2)**(0.5) * sigma * ((xl)**(3) + ((xl)**(2) * mu + (xl * (mu)**(2) + ((mu)**(3) + (3 * xl + 5 * mu) * (sigma)**(2))))) + jnp.exp(0.5 * ((xl + -1 * mu))**(2) * (sigma)**(-2)) * (jnp.pi)**(0.5) * ((mu)**(4) + (6 * (mu)**(2) * (sigma)**(2) + 3 * (sigma)**(4))) * (erf((2)**(-0.5) * (xl + -1 * mu) * (sigma)**(-1)) + erf((2)**(-0.5) * (sigl + (-1 * xl + mu)) * (sigma)**(-1)))))))))))))))))

def polynomial_filter_hl_gaussian_integral(xl, sigl, xh, sigh, mu, sigma):
    """
    Integral of smoothing function exploiting a tuned polynomial function times a Gaussian distribution.

    :param xl: lower bound of the smoothing function
    :param sigl: width of the smoothing function on the lower side
    :param xh: upper bound of the smoothing function
    :param sigh: width of the smoothing function on the upper side
    :param mu: mean of the Gaussian distribution
    :param sigma: standard deviation of the Gaussian distribution

    :return: integral of the smoothing function exploiting a tuned polynomial function times a Gaussian distribution
    :return type: array
    """
    F=0.5 * (jnp.pi)**(-0.5) * (sigh)**(-5) * (sigl)**(-5) * (-30 * (sigl)**(5) * (xh)**(2) * ((sigh + xh))**(2) * ((2)**(0.5) * (jnp.exp(-0.5 * ((xh + -1 * mu))**(2) * (sigma)**(-2)) + -1 * jnp.exp(-0.5 * ((sigh + (xh + -1 * mu)))**(2) * (sigma)**(-2))) * sigma + -1 * (jnp.pi)**(0.5) * mu * (erf((2)**(-0.5) * (xh + -1 * mu) * (sigma)**(-1)) + -1 * erf((2)**(-0.5) * (sigh + (xh + -1 * mu)) * (sigma)**(-1)))) + (-1 * (jnp.pi)**(0.5) * (sigl)**(5) * ((sigh + xh))**(3) * ((sigh)**(2) + (-3 * sigh * xh + 6 * (xh)**(2))) * (erf((2)**(-0.5) * (xh + -1 * mu) * (sigma)**(-1)) + -1 * erf((2)**(-0.5) * (sigh + (xh + -1 * mu)) * (sigma)**(-1))) + (-3 * (2)**(0.5) * jnp.exp(-0.5 * ((sigh)**(2) + (2 * sigh * (xh + -1 * mu) + 2 * ((xh + -1 * mu))**(2))) * (sigma)**(-2)) * (sigl)**(5) * (2 * jnp.exp(0.5 * ((sigh + (xh + -1 * mu)))**(2) * (sigma)**(-2)) * sigma * ((xh)**(4) + ((xh)**(3) * mu + ((xh)**(2) * (mu)**(2) + (xh * (mu)**(3) + ((mu)**(4) + ((4 * (xh)**(2) + (7 * xh * mu + 9 * (mu)**(2))) * (sigma)**(2) + 8 * (sigma)**(4))))))) + (-2 * jnp.exp(0.5 * ((xh + -1 * mu))**(2) * (sigma)**(-2)) * sigma * ((sigh)**(4) + (4 * (sigh)**(3) * xh + (6 * (sigh)**(2) * (xh)**(2) + (4 * sigh * (xh)**(3) + ((xh)**(4) + ((sigh)**(3) * mu + (3 * (sigh)**(2) * xh * mu + (3 * sigh * (xh)**(2) * mu + ((xh)**(3) * mu + ((sigh)**(2) * (mu)**(2) + (2 * sigh * xh * (mu)**(2) + ((xh)**(2) * (mu)**(2) + (sigh * (mu)**(3) + (xh * (mu)**(3) + ((mu)**(4) + ((4 * ((sigh + xh))**(2) + (7 * (sigh + xh) * mu + 9 * (mu)**(2))) * (sigma)**(2) + 8 * (sigma)**(4))))))))))))))))) + jnp.exp(0.5 * (((xh + -1 * mu))**(2) + ((sigh + (xh + -1 * mu)))**(2)) * (sigma)**(-2)) * (2 * jnp.pi)**(0.5) * mu * ((mu)**(4) + (10 * (mu)**(2) * (sigma)**(2) + 15 * (sigma)**(4))) * (-1 * erf((2)**(-0.5) * (xh + -1 * mu) * (sigma)**(-1)) + erf((2)**(-0.5) * (sigh + (xh + -1 * mu)) * (sigma)**(-1))))) + (-30 * jnp.exp(-0.5 * (((xh + -1 * mu))**(2) + ((sigh + (xh + -1 * mu)))**(2)) * (sigma)**(-2)) * (sigl)**(5) * xh * ((sigh)**(2) + (3 * sigh * xh + 2 * (xh)**(2))) * ((2)**(0.5) * jnp.exp(0.5 * ((xh + -1 * mu))**(2) * (sigma)**(-2)) * (sigh + (xh + mu)) * sigma + -1 * jnp.exp(0.5 * ((sigh + (xh + -1 * mu)))**(2) * (sigma)**(-2)) * ((2)**(0.5) * (xh + mu) * sigma + jnp.exp(0.5 * ((xh + -1 * mu))**(2) * (sigma)**(-2)) * (jnp.pi)**(0.5) * ((mu)**(2) + (sigma)**(2)) * (-1 * erf((2)**(-0.5) * (xh + -1 * mu) * (sigma)**(-1)) + erf((2)**(-0.5) * (sigh + (xh + -1 * mu)) * (sigma)**(-1))))) + (10 * jnp.exp(-0.5 * (((xh + -1 * mu))**(2) + ((sigh + (xh + -1 * mu)))**(2)) * (sigma)**(-2)) * (sigl)**(5) * ((sigh)**(2) + (6 * sigh * xh + 6 * (xh)**(2))) * ((2)**(0.5) * jnp.exp(0.5 * ((xh + -1 * mu))**(2) * (sigma)**(-2)) * sigma * (((sigh + xh))**(2) + ((sigh + xh) * mu + ((mu)**(2) + 2 * (sigma)**(2)))) + -1 * jnp.exp(0.5 * ((sigh + (xh + -1 * mu)))**(2) * (sigma)**(-2)) * ((2)**(0.5) * sigma * ((xh)**(2) + (xh * mu + ((mu)**(2) + 2 * (sigma)**(2)))) + jnp.exp(0.5 * ((xh + -1 * mu))**(2) * (sigma)**(-2)) * (jnp.pi)**(0.5) * mu * ((mu)**(2) + 3 * (sigma)**(2)) * (-1 * erf((2)**(-0.5) * (xh + -1 * mu) * (sigma)**(-1)) + erf((2)**(-0.5) * (sigh + (xh + -1 * mu)) * (sigma)**(-1))))) + (-15 * jnp.exp(-0.5 * (((xh + -1 * mu))**(2) + ((sigh + (xh + -1 * mu)))**(2)) * (sigma)**(-2)) * (sigl)**(5) * (sigh + 2 * xh) * ((2)**(0.5) * jnp.exp(0.5 * ((xh + -1 * mu))**(2) * (sigma)**(-2)) * sigma * ((sigh)**(3) + ((xh)**(3) + ((xh)**(2) * mu + (xh * (mu)**(2) + ((mu)**(3) + ((sigh)**(2) * (3 * xh + mu) + ((3 * xh + 5 * mu) * (sigma)**(2) + sigh * (3 * (xh)**(2) + (2 * xh * mu + ((mu)**(2) + 3 * (sigma)**(2))))))))))) + -1 * jnp.exp(0.5 * ((sigh + (xh + -1 * mu)))**(2) * (sigma)**(-2)) * ((2)**(0.5) * sigma * ((xh)**(3) + ((xh)**(2) * mu + (xh * (mu)**(2) + ((mu)**(3) + (3 * xh + 5 * mu) * (sigma)**(2))))) + jnp.exp(0.5 * ((xh + -1 * mu))**(2) * (sigma)**(-2)) * (jnp.pi)**(0.5) * ((mu)**(4) + (6 * (mu)**(2) * (sigma)**(2) + 3 * (sigma)**(4))) * (-1 * erf((2)**(-0.5) * (xh + -1 * mu) * (sigma)**(-1)) + erf((2)**(-0.5) * (sigh + (xh + -1 * mu)) * (sigma)**(-1))))) + ((jnp.pi)**(0.5) * (sigh)**(5) * (sigl)**(5) * (erf((2)**(-0.5) * (xh + -1 * mu) * (sigma)**(-1)) + -1 * erf((2)**(-0.5) * (xl + -1 * mu) * (sigma)**(-1))) + ((jnp.pi)**(0.5) * (sigh)**(5) * ((sigl + -1 * xl))**(3) * ((sigl)**(2) + (3 * sigl * xl + 6 * (xl)**(2))) * (erf((2)**(-0.5) * (xl + -1 * mu) * (sigma)**(-1)) + erf((2)**(-0.5) * (sigl + (-1 * xl + mu)) * (sigma)**(-1))) + (30 * (sigh)**(5) * ((sigl + -1 * xl))**(2) * (xl)**(2) * (-1 * (2)**(0.5) * jnp.exp(-0.5 * ((sigl + (-1 * xl + mu)))**(2) * (sigma)**(-2)) * (-1 + jnp.exp(0.5 * sigl * (sigl + (-2 * xl + 2 * mu)) * (sigma)**(-2))) * sigma + (jnp.pi)**(0.5) * mu * (erf((2)**(-0.5) * (xl + -1 * mu) * (sigma)**(-1)) + erf((2)**(-0.5) * (sigl + (-1 * xl + mu)) * (sigma)**(-1)))) + (-3 * (2)**(0.5) * jnp.exp(-0.5 * ((sigl)**(2) + (2 * ((xl + -1 * mu))**(2) + 2 * sigl * (-1 * xl + mu))) * (sigma)**(-2)) * (sigh)**(5) * (2 * jnp.exp(0.5 * ((sigl + (-1 * xl + mu)))**(2) * (sigma)**(-2)) * sigma * ((xl)**(4) + ((xl)**(3) * mu + ((xl)**(2) * (mu)**(2) + (xl * (mu)**(3) + ((mu)**(4) + ((4 * (xl)**(2) + (7 * xl * mu + 9 * (mu)**(2))) * (sigma)**(2) + 8 * (sigma)**(4))))))) + (-2 * jnp.exp(0.5 * ((xl + -1 * mu))**(2) * (sigma)**(-2)) * sigma * ((sigl)**(4) + ((xl)**(4) + ((xl)**(3) * mu + ((xl)**(2) * (mu)**(2) + (xl * (mu)**(3) + ((mu)**(4) + (-1 * (sigl)**(3) * (4 * xl + mu) + ((4 * (xl)**(2) + (7 * xl * mu + 9 * (mu)**(2))) * (sigma)**(2) + (8 * (sigma)**(4) + ((sigl)**(2) * (6 * (xl)**(2) + (3 * xl * mu + ((mu)**(2) + 4 * (sigma)**(2)))) + -1 * sigl * (4 * (xl)**(3) + (3 * (xl)**(2) * mu + (2 * xl * (mu)**(2) + ((mu)**(3) + (8 * xl + 7 * mu) * (sigma)**(2))))))))))))))) + -1 * jnp.exp(0.5 * (((xl + -1 * mu))**(2) + ((sigl + (-1 * xl + mu)))**(2)) * (sigma)**(-2)) * (2 * jnp.pi)**(0.5) * mu * ((mu)**(4) + (10 * (mu)**(2) * (sigma)**(2) + 15 * (sigma)**(4))) * (erf((2)**(-0.5) * (xl + -1 * mu) * (sigma)**(-1)) + erf((2)**(-0.5) * (sigl + (-1 * xl + mu)) * (sigma)**(-1))))) + (-30 * jnp.exp(-0.5 * ((sigl)**(2) + (2 * ((xl + -1 * mu))**(2) + 2 * sigl * (-1 * xl + mu))) * (sigma)**(-2)) * (sigh)**(5) * xl * ((sigl)**(2) + (-3 * sigl * xl + 2 * (xl)**(2))) * ((2)**(0.5) * jnp.exp(0.5 * ((xl + -1 * mu))**(2) * (sigma)**(-2)) * (-1 * sigl + (xl + mu)) * sigma + jnp.exp(0.5 * ((sigl + (-1 * xl + mu)))**(2) * (sigma)**(-2)) * (-1 * (2)**(0.5) * (xl + mu) * sigma + jnp.exp(0.5 * ((xl + -1 * mu))**(2) * (sigma)**(-2)) * (jnp.pi)**(0.5) * ((mu)**(2) + (sigma)**(2)) * (erf((2)**(-0.5) * (xl + -1 * mu) * (sigma)**(-1)) + erf((2)**(-0.5) * (sigl + (-1 * xl + mu)) * (sigma)**(-1))))) + (10 * jnp.exp(-0.5 * (((xl + -1 * mu))**(2) + ((sigl + (-1 * xl + mu)))**(2)) * (sigma)**(-2)) * (sigh)**(5) * ((sigl)**(2) + (-6 * sigl * xl + 6 * (xl)**(2))) * ((2)**(0.5) * jnp.exp(0.5 * ((xl + -1 * mu))**(2) * (sigma)**(-2)) * sigma * (((sigl + -1 * xl))**(2) + ((-1 * sigl + xl) * mu + ((mu)**(2) + 2 * (sigma)**(2)))) + jnp.exp(0.5 * ((sigl + (-1 * xl + mu)))**(2) * (sigma)**(-2)) * (-1 * (2)**(0.5) * sigma * ((xl)**(2) + (xl * mu + ((mu)**(2) + 2 * (sigma)**(2)))) + jnp.exp(0.5 * ((xl + -1 * mu))**(2) * (sigma)**(-2)) * (jnp.pi)**(0.5) * mu * ((mu)**(2) + 3 * (sigma)**(2)) * (erf((2)**(-0.5) * (xl + -1 * mu) * (sigma)**(-1)) + erf((2)**(-0.5) * (sigl + (-1 * xl + mu)) * (sigma)**(-1))))) + 15 * jnp.exp(-0.5 * ((sigl)**(2) + (2 * ((xl + -1 * mu))**(2) + 2 * sigl * (-1 * xl + mu))) * (sigma)**(-2)) * (sigh)**(5) * (sigl + -2 * xl) * ((2)**(0.5) * jnp.exp(0.5 * ((xl + -1 * mu))**(2) * (sigma)**(-2)) * sigma * (-1 * (sigl)**(3) + ((xl)**(3) + ((xl)**(2) * mu + (xl * (mu)**(2) + ((mu)**(3) + ((sigl)**(2) * (3 * xl + mu) + ((3 * xl + 5 * mu) * (sigma)**(2) + -1 * sigl * (3 * (xl)**(2) + (2 * xl * mu + ((mu)**(2) + 3 * (sigma)**(2))))))))))) + jnp.exp(0.5 * ((sigl + (-1 * xl + mu)))**(2) * (sigma)**(-2)) * (-1 * (2)**(0.5) * sigma * ((xl)**(3) + ((xl)**(2) * mu + (xl * (mu)**(2) + ((mu)**(3) + (3 * xl + 5 * mu) * (sigma)**(2))))) + jnp.exp(0.5 * ((xl + -1 * mu))**(2) * (sigma)**(-2)) * (jnp.pi)**(0.5) * ((mu)**(4) + (6 * (mu)**(2) * (sigma)**(2) + 3 * (sigma)**(4))) * (erf((2)**(-0.5) * (xl + -1 * mu) * (sigma)**(-1)) + erf((2)**(-0.5) * (sigl + (-1 * xl + mu)) * (sigma)**(-1)))))))))))))))))
    Freg=jnp.nan_to_num(F, nan=1.)
    return Freg

def check_masses(events):

    par_list = events.keys()
    
    if 'm1_src' in par_list:
        events['m1'] = events['m1_src']*(1.+events['z'])
        if 'q' in par_list:
            events['m2_src'] = events['m1_src']*events['q']
            events['m2']     = events['m1']*events['q']
            events['eta']    = events['q']/((1.+events['q'])**2)
            events['Mc']     = ((events['m1']*events['m2'])**(3./5.))/((events['m1']+events['m2'])**(1./5.))
        elif 'm2_src' in par_list:
            events['m2']  = events['m2_src']*(1.+events['z'])
            events['q']   = events['m2_src']/events['m1_src']
            events['eta'] = events['q']/((1.+events['q'])**2)
            events['Mc']  = ((events['m1']*events['m2'])**(3./5.))/((events['m1']+events['m2'])**(1./5.))
        elif 'eta' in par_list:
            Seta             = np.sqrt(np.where(events['eta']<0.25, 1.0 - 4.0*events['eta'], 0.))
            events['q']      = (1.- Seta)/(1. + Seta)
            events['m2_src'] = events['m1_src']*events['q']
            events['m2']     = events['m1']*events['q']
            events['Mc']     = ((events['m1']*events['m2'])**(3./5.))/((events['m1']+events['m2'])**(1./5.))
    elif 'Mc_src' in par_list:
        events['Mc'] = events['Mc_src']*(1.+events['z'])
        if 'eta' in par_list:
            Seta             = np.sqrt(np.where(events['eta']<0.25, 1.0 - 4.0*events['eta'], 0.))
            events['q']      = (1.- Seta)/(1. + Seta)
            events['m1_src'] = 0.5*(events['Mc_src']/(events['eta']**(3./5.)))*(1. + Seta)
            events['m2_src'] = 0.5*(events['Mc_src']/(events['eta']**(3./5.)))*(1. - Seta)
            events['m1']     = events['m1_src']*(1.+events['z'])
            events['m2']     = events['m2_src']*(1.+events['z'])
        if 'q' in par_list:
            events['eta']    = events['q']/((1.+events['q'])**2)
            Seta             = np.sqrt(np.where(events['eta']<0.25, 1.0 - 4.0*events['eta'], 0.))
            events['m1_src'] = 0.5*(events['Mc_src']/(events['eta']**(3./5.)))*(1. + Seta)
            events['m2_src'] = 0.5*(events['Mc_src']/(events['eta']**(3./5.)))*(1. - Seta)
            events['m1']     = events['m1_src']*(1.+events['z'])
            events['m2']     = events['m2_src']*(1.+events['z'])
    elif 'Mtot_src' in par_list:
        if 'q' in par_list:
            events['eta']    = events['q']/((1.+events['q'])**2)
            Seta             = np.sqrt(np.where(events['eta']<0.25, 1.0 - 4.0*events['eta'], 0.))
            events['m1_src'] = 0.5*events['Mtot_src']*(1. + Seta)
            events['m2_src'] = 0.5*events['Mtot_src']*(1. - Seta)
            events['m1']     = events['m1_src']*(1.+events['z'])
            events['m2']     = events['m2_src']*(1.+events['z'])
            events['Mc']     = ((events['m1']*events['m2'])**(3./5.))/((events['m1']+events['m2'])**(1./5.))
        elif 'eta' in par_list:
            Seta             = np.sqrt(np.where(events['eta']<0.25, 1.0 - 4.0*events['eta'], 0.))
            events['q']      = (1.- Seta)/(1. + Seta)
            events['m1_src'] = 0.5*events['Mtot_src']*(1. + Seta)
            events['m2_src'] = 0.5*events['Mtot_src']*(1. - Seta)
            events['m1']     = events['m1_src']*(1.+events['z'])
            events['m2']     = events['m2_src']*(1.+events['z'])
            events['Mc']     = ((events['m1']*events['m2'])**(3./5.))/((events['m1']+events['m2'])**(1./5.))


def logdiffexp(x, y):
    '''
    computes log( e^x - e^y)
    '''
    return x + np.log1p(-np.exp(y-x))

class RegularGridInterpolator_JAX:
    """
    Implementation of ``SciPy`` 's :py:class:`RegularGridInterpolator` in a ``JAX`` usable way. Essentially ``numpy`` in the original code is changed to ``jax.numpy`` because of assignment issues, arising when using ``vmap`` and ``jacrev``. We also changed the ``+=`` syntax which creates issues in ``JAX``.
    
    NOTE: ``bounds_error=True`` still does not work with ``vmap`` and jacrev``.
    
    """
    """
    Interpolation on a regular grid in arbitrary dimensions
    The data must be defined on a regular grid; the grid spacing however may be
    uneven. Linear and nearest-neighbor interpolation are supported. After
    setting up the interpolator object, the interpolation method (*linear* or
    *nearest*) may be chosen at each evaluation.
    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions.
    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.
    method : str, optional
        The method of interpolation to perform. Supported are "linear" and
        "nearest". This parameter will become the default for the object's
        ``__call__`` method. Default is "linear".
    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.
    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated.
    
    References
    ----------
    .. [1] Python package *regulargrid* by Johannes Buchner, see
           https://pypi.python.org/pypi/regulargrid/
    .. [2] Wikipedia, "Trilinear interpolation",
           https://en.wikipedia.org/wiki/Trilinear_interpolation
    .. [3] Weiser, Alan, and Sergio E. Zarantonello. "A note on piecewise linear
           and multilinear table interpolation in many dimensions." MATH.
           COMPUT. 50.181 (1988): 189-196.
           https://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917826-0/S0025-5718-1988-0917826-0.pdf
    """
    # This class is based on code originally programmed by Johannes Buchner,
    # see https://github.com/JohannesBuchner/regulargrid
    # and the original SciPy code
    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html

    def __init__(self, points, values, method="linear", bounds_error=False,
                 fill_value=jnp.nan):
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)
        self.method = method
        self.bounds_error = bounds_error

        if not hasattr(values, 'ndim'):
            # allow reasonable duck-typed values
            values = jnp.asarray(values)

        if len(points) > values.ndim:
            raise ValueError("There are %d point arrays, but values has %d "
                             "dimensions" % (len(points), values.ndim))

        if hasattr(values, 'dtype') and hasattr(values, 'astype'):
            if not jnp.issubdtype(values.dtype, jnp.inexact):
                values = values.astype(float)

        self.fill_value = fill_value
        if fill_value is not None:
            fill_value_dtype = jnp.asarray(fill_value).dtype
            if (hasattr(values, 'dtype') and not
                    jnp.can_cast(fill_value_dtype, values.dtype,
                                casting='same_kind')):
                raise ValueError("fill_value must be either 'None' or "
                                 "of a type compatible with values")

        for i, p in enumerate(points):
            if not jnp.all(jnp.diff(p) > 0.):
                raise ValueError("The points in dimension %d must be strictly "
                                 "ascending" % i)
            if not jnp.asarray(p).ndim == 1:
                raise ValueError("The points in dimension %d must be "
                                 "1-dimensional" % i)
            if not values.shape[i] == len(p):
                raise ValueError("There are %d points and %d values in "
                                 "dimension %d" % (len(p), values.shape[i], i))
        
        self.grid = tuple([jnp.asarray(p) for p in points])
        self.values = values

    def __call__(self, xi, method=None):
        """
        Interpolation at coordinates
        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at
        method : str
            The method of interpolation to perform. Supported are "linear" and
            "nearest".
        """
        method = self.method if method is None else method
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)

        ndim = len(self.grid)
        #xi = _ndim_coords_from_arrays(xi, ndim=ndim) # Skip this checks and conversions to avoid conflicts
        if xi.shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                             "%d, but this RegularGridInterpolator has "
                             "dimension %d" % (xi.shape[1], ndim))

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if not jnp.logical_and(jnp.all(self.grid[i][0] <= p),
                                      jnp.all(p <= self.grid[i][-1])):
                    raise ValueError("One of the requested xi is out of bounds "
                                     "in dimension %d" % i)

        indices, norm_distances, out_of_bounds = self._find_indices(xi.T)
        if method == "linear":
            result = self._evaluate_linear(indices,
                                           norm_distances,
                                           out_of_bounds)
        elif method == "nearest":
            result = self._evaluate_nearest(indices,
                                            norm_distances,
                                            out_of_bounds)
        if not self.bounds_error and self.fill_value is not None:
            result = jnp.where(out_of_bounds>0, self.fill_value, result)

        return result.reshape(xi_shape[:-1] + self.values.shape[ndim:])

    def _evaluate_linear(self, indices, norm_distances, out_of_bounds):
        # slice for broadcasting over trailing dimensions in self.values
        from itertools import product
        vslice = (slice(None),) + (None,)*(self.values.ndim - len(indices))

        # find relevant values
        # each i and i+1 represents a edge
        edges = product(*[[i, i + 1] for i in indices])
        values = 0.
        for edge_indices in edges:
            weight = 1.
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                weight = weight*jnp.where(ei == i, 1 - yi, yi)
            values = values + jnp.asarray(self.values[edge_indices]) * weight[vslice]
        return values

    def _evaluate_nearest(self, indices, norm_distances, out_of_bounds):
        print('nearest method not checked in this implementation')
        idx_res = [jnp.where(yi <= .5, i, i + 1)
                   for i, yi in zip(indices, norm_distances)]
        return self.values[tuple(idx_res)]

    def _find_indices(self, xi):
        # find relevant edges between which xi are situated
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []
        # check for out of bounds xi
        out_of_bounds = jnp.zeros((xi.shape[1]), dtype=bool)
        # iterate through dimensions
        for x, grid in zip(xi, self.grid):
            i = jnp.searchsorted(grid, x) - 1
            i = jnp.where(i < 0, 0, i)
            i = jnp.where(i > grid.size - 2, grid.size - 2, i)
            indices.append(i)
            norm_distances.append((x - grid[i]) /
                                  (grid[i + 1] - grid[i]))
            if not self.bounds_error:
                out_of_bounds = out_of_bounds + x < grid[0]
                out_of_bounds = out_of_bounds + x > grid[-1]
        return indices, norm_distances, out_of_bounds

def cumtrapz_JAX(y, x=None, dx=1.0, axis=-1, initial=None):
    """
    Implementation of ``SciPy`` 's :py:class:`cumtrapz` in a ``JAX`` usable way.
    
    Cumulatively integrate y(x) using the composite trapezoidal rule.

    Parameters
    ----------
    y : array_like
        Values to integrate.
    x : array_like, optional
        The coordinate to integrate along.  If None (default), use spacing `dx`
        between consecutive elements in `y`.
    dx : int, optional
        Spacing between elements of `y`.  Only used if `x` is None.
    axis : int, optional
        Specifies the axis to cumulate.  Default is -1 (last axis).
    initial : scalar, optional
        If given, uses this value as the first value in the returned result.
        Typically this value should be 0.  Default is None, which means no
        value at ``x[0]`` is returned and `res` has one element less than `y`
        along the axis of integration.

    Returns
    -------
    res : ndarray
        The result of cumulative integration of `y` along `axis`.
        If `initial` is None, the shape is such that the axis of integration
        has one less value than `y`.  If `initial` is given, the shape is equal
        to that of `y`.

    Examples
    --------
    >>> from scipy import integrate
    >>> x = np.linspace(-2, 2, num=20)
    >>> y = x
    >>> y_int = integrate.cumtrapz(y, x, initial=0)
    >>> plt.plot(x, y_int, 'ro', x, y[0] + 0.5 * x**2, 'b-')
    >>> plt.show()

    """
    def tupleset(t, i, value):
        l = list(t)
        l[i] = value
        return tuple(l)

    y = jnp.asarray(y)
    if x is None:
        d = dx
    else:
        d = jnp.diff(x, axis=axis)

    nd = len(y.shape)
    slice1 = tupleset((slice(None),)*nd, axis, slice(1, None))
    slice2 = tupleset((slice(None),)*nd, axis, slice(None, -1))
    res = associative_scan(jnp.add, d * (y[slice1] + y[slice2]) / 2.0, axis=axis) 

    if initial is not None:
        if not jnp.isscalar(initial):
            raise ValueError("`initial` parameter should be a scalar.")

        shape = list(res.shape)
        shape[axis] = 1
        res = jnp.concatenate([jnp.ones(shape, dtype=res.dtype) * initial, res], axis=axis)

    return res

def svd_inv(matr):
    '''
    Invert a matrix using SVD decomposition.

    :param (..., M, M) array_like matr: array of matrices to be inverted

    :return: Inverted matrices
    :rtype: (..., M, M) array_like
    '''
    u, s, v = jnp.linalg.svd(matr, full_matrices=False)
    sdiag = jnp.zeros_like(u)
    sdiag = fori_loop(0, s.shape[0], lambda i, x: x.at[i].set(jnp.diag(1./s[i])), sdiag)
    tmpinv = jnp.matmul(jnp.matmul(v.transpose(0,2,1), sdiag), u.transpose(0,2,1))
    tmpinv = (tmpinv + tmpinv.transpose(0,2,1))/2.
    return tmpinv

def svd_inv_stabilizecond(matr):
    '''
    Invert a matrix using SVD decomposition and stabilize the condition number normalizing the rows and columns to the diagonal.

    :param (..., M, M) array_like matr: array of matrices to be inverted

    :return: Inverted matrices
    :rtype: (..., M, M) array_like
    '''
    ws = 1./jnp.sqrt(abs(jnp.diagonal(matr, offset=0, axis1=-2, axis2=-1)))
    wsdiag = jnp.zeros_like(matr)
    wsdiag = fori_loop(0, ws.shape[0], lambda i, x: x.at[i].set(jnp.diag(ws[i])), wsdiag)
    newmatr = jnp.matmul(jnp.matmul(wsdiag, matr), wsdiag)
    u, s, v = jnp.linalg.svd(newmatr, full_matrices=False)
    sdiag = jnp.zeros_like(u)
    sdiag = fori_loop(0, s.shape[0], lambda i, x: x.at[i].set(jnp.diag(1./s[i])), sdiag)
    tmpinv = jnp.matmul(jnp.matmul(v.transpose(0,2,1), sdiag), u.transpose(0,2,1))
    tmpinv = (tmpinv + tmpinv.transpose(0,2,1))/2.
    return jnp.matmul(jnp.matmul(wsdiag, tmpinv), wsdiag)

def inv_stabilizecond(matr):
    '''
    Invert a matrix computing the multiplicative inverse and stabilize the condition number normalizing the rows and columns to the diagonal.

    :param (..., M, M) array_like matr: array of matrices to be inverted

    :return: Inverted matrices
    :rtype: (..., M, M) array_like
    '''
    ws = 1./jnp.sqrt(abs(jnp.diagonal(matr, offset=0, axis1=-2, axis2=-1)))
    wsdiag = jnp.zeros_like(matr)
    wsdiag = fori_loop(0, ws.shape[0], lambda i, x: x.at[i].set(jnp.diag(ws[i])), wsdiag)
    newmatr = jnp.matmul(jnp.matmul(wsdiag, matr), wsdiag)
    tmpinv = jnp.linalg.inv(newmatr)
    tmpinv = (tmpinv + tmpinv.transpose(0,2,1))/2.
    return jnp.matmul(jnp.matmul(wsdiag, tmpinv), wsdiag)

def pinv_stabilizecond(matr):
    '''
    Invert a matrix computing the pseudo-inverse and stabilize the condition number normalizing the rows and columns to the diagonal.

    :param (..., M, M) array_like matr: array of matrices to be inverted

    :return: Inverted matrices
    :rtype: (..., M, M) array_like
    '''
    ws = 1./jnp.sqrt(abs(jnp.diagonal(matr, offset=0, axis1=-2, axis2=-1)))
    wsdiag = jnp.zeros_like(matr)
    wsdiag = fori_loop(0, ws.shape[0], lambda i, x: x.at[i].set(jnp.diag(ws[i])), wsdiag)
    newmatr = jnp.matmul(jnp.matmul(wsdiag, matr), wsdiag)
    tmpinv = jnp.linalg.pinv(newmatr)
    tmpinv = (tmpinv + tmpinv.transpose(0,2,1))/2.
    return jnp.matmul(jnp.matmul(wsdiag, tmpinv), wsdiag)

def logdet_stabilizecond(matr):
    '''
    Compute the log-determinant of a matrix and stabilize the condition number normalizing the rows and columns to the diagonal.

    :param (..., M, M) array_like matr: array of matrices for which the log-determinant is to be computed

    :return: Log-determinant of the matrices
    :rtype: (..., M) array_like
    '''
    ws = 1./jnp.sqrt(abs(jnp.diagonal(matr, offset=0, axis1=-2, axis2=-1)))
    wsdiag = jnp.zeros_like(matr)
    wsdiag = fori_loop(0, ws.shape[0], lambda i, x: x.at[i].set(jnp.diag(ws[i])), wsdiag)
    newmatr = jnp.matmul(jnp.matmul(wsdiag, matr), wsdiag)
    tmpdet = jnp.linalg.slogdet(newmatr)[1]
    return tmpdet - 2.*jnp.log(ws.prod(axis=-1))

partial(jit, static_argnums=(1,2))
def invmatrix(matr, normalize=True, inv_method='inv'):
    '''
    Wrapper for the matrix inversion function. It allows to choose between the pseudo-inverse, the multiplicative inverse and the SVD decomposition. It allows also to normalize the matrix before inverting it, in order to stabilize the condition number.

    :param (..., M, M) array_like matr: array of matrices to be inverted
    :param bool normalize: if True, normalize the rows and columns of the matrices to the diagonal before inverting it
    :param str inv_method: method to be used for the inversion. Allowed values are 'inv', 'pinv' and 'svd'

    :return: Inverted matrices
    :rtype: (..., M, M) array_like
    '''
    invmatrix = cond(normalize, 
                     lambda x: cond(inv_method=='inv', lambda y: inv_stabilizecond(y), #lambda y: pinv_stabilizecond(y), x), 
                                    lambda y: cond(inv_method=='pinv', lambda z: pinv_stabilizecond(z), lambda z: svd_inv_stabilizecond(z), y), x),
                     lambda x: cond(inv_method=='inv', lambda y: jnp.linalg.inv(y), #lambda y: jnp.linalg.pinv(y), x), 
                                    lambda y: cond(inv_method=='pinv', lambda z: jnp.linalg.pinv(z), lambda z: svd_inv(z), y), x),
                     matr)
    return invmatrix







def open_h5py(path_MC_samples):
    '''
    Open h5py file and return the dictionary of the data.
    :param str path_MC_samples: path to the h5py file.
    :return dict: dictionary of the data.
    '''
    open_MC_samples=h5py.File(path_MC_samples , 'r') 
    with h5py.File(path_MC_samples, 'r') as h5_file:
        MC_samples = {}
        for key in open_MC_samples.keys():
            data = h5_file[key][()] 
            MC_samples[key] = data 
    return MC_samples

def load_SNRders(name, topkey='derivative'):
    # name is the path to the hdf5 file containing the derivatives
    out={}
    with h5py.File(name, 'r') as f:
        #print(f.keys())
        for key in f[topkey].keys():
            out[key] = np.array(f[topkey][key])
            
    return out


def print_diagonal_elements(matrix,population_model_recovery):
    '''
    Print the diagonal elements of the matrix and the corresponding hyperparameters.
    :param array matrix: population Fisher matrix.
    :param class population_model_recovery: population model.
    :return None: print the diagonal elements of the Fishers and the values of corresponding hyperparameters.
    '''
    hyperpar_dict=population_model_recovery.hyperpar_dict
    rows = len(matrix)
    print('number of hyperparameters=',rows)
    cols = len(matrix[0])  
    names = [key for key in hyperpar_dict.keys()]
    hyperpar = np.array(names)
    for i in range(rows):
        print(matrix[i][i],'   ----->  ',hyperpar[i])


def plot_mass_rate_spin_distributions(p_pop,p_draw=None,samples_pdraw=None,samples_pop=None):
    '''
    Plot the mass, rate and spin distributions of the injected and recovered populations.
    :param class p_draw: injected population.
    :param class p_pop: recovered population.
    :return None: plot the mass, rate and spin distributions of the injected and recovered populations.
    '''
    fig, axs = plt.subplots(3, 2, figsize=(10, 10.))
    fontsize=12
    ###############################################################
    # MASSES
    ###############################################################

    if p_draw is not None:
        if isinstance(p_draw.mass_function,PowerLawPlusPeak_modsmooth_MassDistribution) or isinstance(p_draw.mass_function,TruncatedPowerLaw_modsmooth_MassDistribution):
            xhigh=max(p_draw.hyperpar_dict['m_max']+p_draw.hyperpar_dict['sigma_h'],p_pop.hyperpar_dict['m_max']+p_pop.hyperpar_dict['sigma_h'])
        elif isinstance(p_draw.mass_function,PowerLawPlusPeak_MassDistribution) or isinstance(p_draw.mass_function,TruncatedPowerLaw_MassDistribution):
            xhigh=max(p_draw.hyperpar_dict['m_max'],p_pop.hyperpar_dict['m_max'])
    else:
        if isinstance(p_pop.mass_function,PowerLawPlusPeak_modsmooth_MassDistribution) or isinstance(p_pop.mass_function,TruncatedPowerLaw_modsmooth_MassDistribution):
            xhigh=p_pop.hyperpar_dict['m_max']+p_pop.hyperpar_dict['sigma_h']
        elif isinstance(p_pop.mass_function,PowerLawPlusPeak_MassDistribution) or isinstance(p_pop.mass_function,TruncatedPowerLaw_MassDistribution):
            xhigh=p_pop.hyperpar_dict['m_max']
    mgrid=np.linspace(0.01,xhigh,100)
    m2grid=np.linspace(0.01,xhigh,100)
    j,k=0,0
    axs[j,k].plot(mgrid,p_pop.mass_function._mass1_function(mgrid)/np.trapz(p_pop.mass_function._mass1_function(mgrid),mgrid),color='C2',lw=2.,label=r'$p_{\mathrm pop}(\theta|\lambda)$')
    axs[j,k].set_xlabel('$m_1~[M_\odot]$',fontsize=fontsize)
    
    j,k=0,1
    axs[j,k].set_xlabel('$m_2~[M_\odot]$',fontsize=fontsize)
    m1,m2=np.meshgrid(mgrid,m2grid,indexing='ij')
    pdf_joint_rec=np.where(m2<=m1,p_pop.mass_function.mass_function(m1,m2),0)
    
    p_m2_rec=np.trapz(pdf_joint_rec,m1,axis=0)
    axs[j,k].plot(m2grid,p_m2_rec,color='C2',lw=2.,ls='solid',label='rec')
    if p_draw is not None:
        '''plot the injected mass distribution'''
        axs[0,0].plot(mgrid,p_draw.mass_function._mass1_function(mgrid)/np.trapz(p_draw.mass_function._mass1_function(mgrid),mgrid),color='C3',lw=2.,label=r'$p_{\mathrm draw}(\theta)$')
        pdf_joint_inj=np.where(m2<=m1,p_draw.mass_function.mass_function(m1,m2),0)
        p_m2_inj=np.trapz(pdf_joint_inj,m1,axis=0)
        axs[0,1].plot(m2grid,p_m2_inj,color='C3',lw=2.,ls='solid',label='inj')
    if samples_pdraw is not None:
        '''plot the mass samples from the injected population'''
        axs[0,0].hist(samples_pdraw['m1_src'], density=True, histtype='step', color='C3', label='samples $p_\mathrm{draw}$')
        axs[0,1].hist(samples_pdraw['m2_src'],density=True,histtype='step',color='C3')
    if samples_pop is not None:
        '''plot the mass samples from the true population'''
        axs[0,0].hist(samples_pop['m1_src'], density=True, histtype='step', color='C2', label='samples $p_\mathrm{pop}$')
        axs[0,1].hist(samples_pop['m2_src'],density=True,histtype='step',color='C2')

    ###############################################################
    # rate
    ###############################################################
    
    j,k=1,0
    rgrid=np.linspace(0.01,10,100)
    axs[j,k].plot(rgrid,p_pop.rate_function.rate_function(rgrid),color='C2',lw=2.)
    axs[j,k].set_xlabel('$z$',fontsize=fontsize)
    if p_draw is not None:
        axs[j,k].set_xlim(p_draw.priorlims_dict['z'][0],p_draw.priorlims_dict['z'][-1])
    else:
        axs[j,k].set_xlim(p_pop.priorlims_dict['z'][0],p_pop.priorlims_dict['z'][-1])
    if p_draw is not None:
        '''plot the injected redshift distribution'''
        axs[j,k].plot(rgrid,p_draw.rate_function.rate_function(rgrid),color='C3',lw=3.)
    if samples_pdraw is not None:
        '''plot the redshift samples from the injected population'''        
        axs[j,k].hist(samples_pdraw['z'],density=True,histtype='step',color='C3')       
    if samples_pop is not None:
        '''plot the redshift samples from the true population'''
        axs[j,k].hist(samples_pop['z'],density=True,histtype='step',color='C2') 

    ###############################################################
    # spins
    ###############################################################
    
    j,k=2,0
    chi_grid=np.linspace(0.0001,1.,100)
    xx=p_pop.spin_function._chimagnitude_function(chi_grid)
    axs[j,k].plot(chi_grid,xx,color='C2',lw=2.,label='rec')
    axs[j,k].set_xlabel('$\chi$',fontsize=fontsize)
    if p_draw is not None:
        axs[j,k].set_xlim(p_draw.priorlims_dict['chi1'][0],p_draw.priorlims_dict['chi1'][-1])
    else:
        axs[j,k].set_xlim(p_pop.priorlims_dict['chi1'][0],p_pop.priorlims_dict['chi1'][-1])
    if p_draw is not None:
        '''plot the injected spim distribution'''
        x=p_draw.spin_function._chimagnitude_function(chi_grid)
        axs[j,k].plot(chi_grid,x,color='C3',lw=2.)
    if samples_pdraw is not None:
        '''plot the spin samples from the injected population'''
        axs[j,k].hist(samples_pdraw['chi1'],density=True,histtype='step',color='C3')   
    if samples_pop is not None:
        '''plot the spin samples from the true population'''
        axs[j,k].hist(samples_pop['chi1'],density=True,histtype='step',color='C2')


    ###############################################################
    # tilts
    ###############################################################
    
    j,k=2,1
    tilt_grid=np.linspace(-1,1,100)
    xx=p_pop.spin_function._coschitilt_function(tilt_grid)
    axs[j,k].plot(tilt_grid,xx,color='C2',lw=2.)
    axs[j,k].set_xlabel(r'$\theta$',fontsize=fontsize)
    if p_draw is not None:
        axs[j,k].set_xlim(np.cos(p_draw.priorlims_dict['tilt1'][0]),np.cos(p_draw.priorlims_dict['tilt1'][-1]))
    else:
        axs[j,k].set_xlim(np.cos(p_pop.priorlims_dict['tilt1'][0]),np.cos(p_pop.priorlims_dict['tilt1'][-1]))
    if p_draw is not None:
        '''plot the injected tilt distribution'''
        x=p_draw.spin_function._coschitilt_function(tilt_grid)
        axs[j,k].plot(tilt_grid,x,color='C3',lw=2.)
    if samples_pdraw is not None:
        '''plot the tilt samples from the injected population'''
        axs[j,k].hist(np.cos(samples_pdraw['tilt1']),density=True,histtype='step',color='C3')   
    if samples_pop is not None:
        '''plot the tilt samples from the true population'''
        axs[j,k].hist(np.cos(samples_pop['tilt1']),density=True,histtype='step',color='C2')




    for ax_row in axs:
        for ax in ax_row:
            ax.minorticks_on()
            ax.tick_params(axis='both', which='both', direction='in', length=6, width=1, colors='black')
            ax.tick_params(axis='both', which='minor', direction='in', length=3)
            ax.set_ylim(bottom=1e-7, top=1e1)
            ax.grid(True, which='both', linewidth=0.5, alpha=0.5)
            ax.set_yscale('log')
    axs[0,0].legend(loc='upper right')

    return 




def open_catalog(path_pdraw,path_single_FIMs,idx_i,idx_f,popterms=False,path_pop_terms=None):
    #path_pdraw: path to the folder containing the file pdraw.h5
    #path_single_FIMs to folder SingleEventFishers containing the outputs of gwfast: all_derivatives_SNR_{idx_i}_to_{idx_f}.hdf5,fishers_{idx_i}_to_{idx_f}.npy,snrs_{idx_i}_to_{idx_f}.txt
    '''
    Function to load stuff
    '''
    path_MC_samples=path_pdraw+f'pdraw.h5'
    path_der_snr =path_single_FIMs+f'all_derivatives_SNR_{idx_i}_to_{idx_f}.hdf5'
    path_fisher_npy =path_single_FIMs+f'fishers_{idx_i}_to_{idx_f}.npy'
    path_snrs_txt =path_single_FIMs+f'snrs_{idx_i}_to_{idx_f}.txt'
    
            
    theta_samples=open_h5py(path_MC_samples)
    N_samp=len(theta_samples['Mc'])
    print('N draw=',N_samp)
    rho=np.loadtxt(path_snrs_txt)
    der_snr=load_SNRders(path_der_snr, topkey='derivative')['net']
    FIMs= np.load(path_fisher_npy)
    if popterms==False:
        '''
        if you want to load the number of events, the theta samples, snr, derivative of snr and single-event FIMs
        '''
        assert FIMs.shape[-1]==der_snr.shape[-1]==rho.shape[0]==theta_samples['Mc'].shape[0]
        return N_samp,theta_samples,rho,der_snr,FIMs
    elif popterms==True:
        if path_pop_terms is None:
            raise ValueError("path_pop_terms not defined, but popterms=True requires path_pop_terms.")
        '''
        if you want to load also the integrands of GammaII-GammaIII
        '''
        termI_der=np.load(path_pop_terms+f'termI_der_{idx_i}_to_{idx_f}.npy')
        termI_hess=np.load(path_pop_terms+f'termI_hess_{idx_i}_to_{idx_f}.npy')
        termII=np.load(path_pop_terms+f'termII_matr_{idx_i}_to_{idx_f}.npy')
        termIII=np.load(path_pop_terms+f'termIII_matr_{idx_i}_to_{idx_f}.npy')
        termIV=np.load(path_pop_terms+f'termIV_matr_{idx_i}_to_{idx_f}.npy')
        termV=np.load(path_pop_terms+f'termV_matr_{idx_i}_to_{idx_f}.npy')
        assert FIMs.shape[-1]==der_snr.shape[-1]==rho.shape[0]==theta_samples['Mc'].shape[0]
        return N_samp,theta_samples,rho,der_snr,FIMs,termI_der,termI_hess,termII,termIII,termIV,termV







