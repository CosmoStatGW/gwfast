#
#    Copyright (c) 2022 Francesco Iacovelli <francesco.iacovelli@unige.ch>, Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.


#from jax.config import config
#config.update("jax_enable_x64", True)
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import json
import h5py
import ast
from gwfast import gwfastGlobals as glob

##############################################################################
# LOADING AND SAVING CATALOGS
##############################################################################


def get_event(evs, idx):
    """
    Select events from a catalog by index.

    :param dict(array, array, ...) evs: The dictionary conatining the parameters of the events, as in :py:data:`events`.
    :param list(int) or array(int) or int idx: The indexes of the events to select.
    
    :return: The dictionary conatining the subsample of events.
    :rtype: dict(array, array, ...)

    """
    
    res = {k: np.squeeze(np.array([evs[k][idx], ] )) for k in evs.keys()}
    try:
        len(res['Mc'])
    except:
        res = {k: np.array( [res[k], ] )  for k in res.keys()}
    return res
        
def get_events_subset(evs, detected):
    """
    Select events from a catalog given condition.

    :param dict(array, array, ...) evs: The dictionary conatining the parameters of the events, as in :py:data:`events`.
    :param list(bool) or array(bool) detected: Mask with the events to select, with the same shape as the arrays containing the events parameters.
    
    :return: The dictionary conatining the subsample of events.
    :rtype: dict(array, array, ...)

    """
    return get_event(evs, np.argwhere(detected))


def save_detectors(fname, detectors):
    """
    Store a collection of dictionaries containing the detector characteristics in ``json`` file.
    
    :param str fname: The name of the file to store the detector characteristics in. This has to include the path and the ``json`` extension.
    :param dict(dict, dict, ...) detectors: The collection of dictionaries conatining the detector characteristics (``lat``, ``long``, ``xax`` and ``psd_path`` if desired), as in :py:data:`gwfast.gwfastGlobals.detectors`.
    
    """
    with open(fname, 'w') as fp:
        json.dump(detectors, fp)
    

def save_data(fname, data, ):
    """
    Store a dictionary containing the events parameters in ``h5`` file.
    
    :param str fname: The name of the file to store the events in. This has to include the path and the ``h5`` or ``hdf5`` extension.
    :param dict(array, array, ...) data: The dictionary conatining the parameters of the events, as in :py:data:`events`.
    
    """
    print('Saving to %s '%fname)
    with h5py.File(fname, 'w') as out:
            
                    
        def cd(n, d):
            d = np.array(d)
            out.create_dataset(n, data=d, compression='gzip', shuffle=True)
        
        for key in data.keys():
            cd(key, data[key])

def load_population(name, nEventsUse=None, calculate_params=[], keys_skip=[]):
    
    """
    Load a dictionary containing the events parameters in h5 file, compute some useful cobinations and perform checks.
    
    :param str name: The name of the file to load the events from. This has to include the path and the ``h5`` or ``hdf5`` extension.
    :param int or None nEventsUse: Number of the events in the given file to load.
    :type kind: int or None
    :param list(str) calculate_params: Parameters not present in the file to compute. The supported parameters are ``'LambdaTilde'``, ``'deltaLambda'``, ``'Lambda1'``, ``'Lambda2'``, ``'theta'``, ``'phi'``, ``'ra'``, ``'dec'``.
    :param list(str) keys_skip: Parameters present in the file to skip.
    
    :return: Dictionary conatining the loaded events, as in :py:data:`events`.
    :rtype: dict(array, array, ...)
    
    """
    
    events={}
    with h5py.File(name, 'r') as f:
        for key in f.keys(): 
            if key not in keys_skip:
                events[key] = np.array(f[key])
            else:
                print('Skipping %s' %key)
        if nEventsUse is not None:
            for key in f.keys(): 
                events[key]=events[key][:nEventsUse]
    
    plist = list(events.keys())
    #print('Keys in load_population: %s' %str(events.keys()))   
    #computed_L = False
    #computed_L1 = False
    #for p in calculate_params:
    if ('LambdaTilde' in calculate_params) or ('deltaLambda' in calculate_params):
        print('Computing LambdaTilde, deltaLambda from Lambda1, Lambda2...')
        events['LambdaTilde'], events['deltaLambda'] = Lamt_delLam_from_Lam12(events['Lambda1'], events['Lambda2'], events['eta'])
    
    if (('Lambda1' in calculate_params) or ('Lambda2' in calculate_params)) and not ('Lambda1' in plist):
        print('Computing Lambda1, Lambda2 from LambdaTilde, deltaLambda...')
        events['Lambda1'], events['Lambda2'] = Lam12_from_Lamt_delLam(events['LambdaTilde'], events['deltaLambda'], events['eta'])
        #computed_L1 = True
    if (('theta' in calculate_params) or ('phi' in calculate_params)) and not ('theta' in plist):
        print('Computing theta, phi from ra, dec...')
        events['theta'], events['phi'] = th_phi_from_ra_dec_rad(events['ra'], events['dec'])
    if (('ra' in calculate_params) or ('dec' in calculate_params)) and not ('ra' in plist):
        print('Computing ra, dec from theta, phi...')
        events['ra'], events['dec'] = ra_dec_from_th_phi_rad(events['theta'], events['phi'])
        
        #else:
        #    raise NotImplementedError('Only conversion between Lambda1, Lambda2 and LambdaTilde, deltaLambda supported so far')
            
    
    events = check_evparams(events)
    return events


##############################################################################
# ANGLES
##############################################################################

# See http://spiff.rit.edu/classes/phys440/lectures/coords/coords.html
# Check: https://www.vercalendario.info/en/how/convert-ra-degrees-hours.html

def ra_dec_from_th_phi_rad(theta, phi):
    """
    Compute :math:`\\alpha` and :math:`\delta` in :math:`\\rm rad` from :math:`\\theta` and :math:`\phi` in :math:`\\rm rad`.
    
    :param array or float theta: The :math:`\\theta` sky position angle(s) to convert, in :math:`\\rm rad`.
    :param array or float phi: The :math:`\phi` sky position angle(s) to convert, in :math:`\\rm rad`.
    
    :return: :math:`\\alpha` and :math:`\delta` in :math:`\\rm rad`.
    :rtype: tuple(array, array) or tuple(float, float)
    
    """
    ra = phi #np.rad2deg(phi)
    dec = 0.5*np.pi - theta #np.rad2deg(0.5 * np.pi - theta)
    return ra, dec

def th_phi_from_ra_dec_rad(ra, dec):
    """
    Compute :math:`\\theta` and :math:`\phi` in :math:`\\rm rad` from :math:`\\alpha` and :math:`\delta` in :math:`\\rm rad`.
    
    :param array or float ra: The :math:`\\alpha` sky position angle(s) to convert, in :math:`\\rm rad`.
    :param array or float dec: The The :math:`\delta` sky position angle(s) angle(s) to convert, in :math:`\\rm rad`.
    
    :return: :math:`\\theta` and :math:`\phi` in :math:`\\rm rad`.
    :rtype: tuple(array, array) or tuple(float, float)
    
    """
    theta = 0.5 * np.pi - dec
    phi = ra
    return theta, phi


def ra_dec_from_th_phi(theta, phi):
    """
    Compute :math:`\\alpha` and :math:`\delta` in :math:`\\rm deg` from :math:`\\theta` and :math:`\phi` in :math:`\\rm rad`.
    
    :param array or float theta: The :math:`\\theta` sky position angle(s) to convert, in :math:`\\rm rad`.
    :param array or float phi: The :math:`\phi` sky position angle(s) to convert, in :math:`\\rm rad`.
    
    :return: :math:`\\alpha` and :math:`\delta` in :math:`\\rm deg`.
    :rtype: tuple(array, array) or tuple(float, float)
    
    """
    ra = np.rad2deg(phi)
    dec = np.rad2deg(0.5 * np.pi - theta)
    return ra, dec

  
def th_phi_from_ra_dec(ra, dec):
    """
    Compute :math:`\\theta` and :math:`\phi` in :math:`\\rm rad` from :math:`\\alpha` and :math:`\delta` in :math:`\\rm deg`.
    
    :param array or float ra: The :math:`\\alpha` sky position angle(s) to convert, in :math:`\\rm deg`.
    :param array or float dec: The The :math:`\delta` sky position angle(s) angle(s) to convert, in :math:`\\rm deg`.
    
    :return: :math:`\\theta` and :math:`\phi` in :math:`\\rm rad`.
    :rtype: tuple(array, array) or tuple(float, float)
    
    """
    theta = 0.5 * np.pi - np.deg2rad(dec)
    phi = np.deg2rad(ra)
    return theta, phi

def deg_min_sec_to_decimal_deg(d, m, s):
    """
    Convert one or multiple angles in degrees, minutes, seconds to decimal degrees.
    
    :param array or float d: The degrees of the angle(s) to convert.
    :param array or float m: The minutes of the angle(s) to convert.
    :param array or float s: The seconds of the angle(s) to convert.
    
    :return: The angle(s) in decimal degrees.
    :rtype: array or float
    
    """
    return d + m/60 + s/3600

def hr_min_sec_to_decimal_deg(h, m, s):
    """
    Convert one or multiple angles in hours, minutes, seconds to decimal degrees.
    
    :param array or float h: The hours of the angle(s) to convert.
    :param array or float m: The minutes of the angle(s) to convert.
    :param array or float s: The seconds of the angle(s) to convert.
    
    :return: The angle(s) in decimal degrees.
    :rtype: array or float
    
    """
    # decimal degrees=15*h+15*m/60+15*s/3600.
    
    return 15*(h+m/60+s/3600)


def deg_min_sec_to_rad(d, m, s):
    """
    Convert one or multiple angles in degrees, minutes, seconds to :math:`\\rm rad`.
    
    :param array or float d: The degrees of the angle(s) to convert.
    :param array or float m: The minutes of the angle(s) to convert.
    :param array or float s: The seconds of the angle(s) to convert.
    
    :return: The angle(s) in :math:`\\rm rad`.
    :rtype: array or float
    
    """
    return deg_min_sec_to_decimal_deg(d, m, s)*np.pi/180

def hr_min_sec_to_rad(h, m, s):
    """
    Convert one or multiple angles in hours, minutes, seconds to :math:`\\rm rad`.
    
    :param array or float h: The hours of the angle(s) to convert.
    :param array or float m: The minutes of the angle(s) to convert.
    :param array or float s: The seconds of the angle(s) to convert.
    
    :return: The angle(s) in :math:`\\rm rad`.
    :rtype: array or float
    
    """
    return hr_min_sec_to_decimal_deg(h, m, s)*np.pi/180


def rad_to_deg_min_sec(rad):
    """
    Convert one or multiple angles in :math:`\\rm rad` to degrees, minutes, seconds.
    
    Checks have been performed with `<https://www.calculatorsoup.com/calculators/conversions/convert-decimal-degrees-to-degrees-minutes-seconds.php>`_.
    
    :param array or float rad: The angle(s) in :math:`\\rm rad`.
    
    :return: The angle(s)' degrees, minutes, seconds.
    :rtype: tuple(array, array, array) or tuple(float, float, float)
    
    """
    # check: https://www.calculatorsoup.com/calculators/conversions/convert-decimal-degrees-to-degrees-minutes-seconds.php
    
    d = np.floor(rad).astype(int)  
    
    m_exact = (rad-d)*60    
    m = np.floor(m_exact).astype(int)

    s = np.round((m_exact - m)*60, 0).astype(int)
    
    return d, m, s

def rad_to_hr_min_sec(rad):
    """
    Convert one or multiple angles in :math:`\\rm rad` to hours, minutes, seconds.
    
    :param array or float rad: The angle(s) in :math:`\\rm rad`.
    
    :return: The angle(s)' hours, minutes, seconds.
    :rtype: tuple(array, array, array) or tuple(float, float, float)
    
    """
    hh = rad/15
    h = np.floor(hh).astype(int)
    
    m_exact = (hh-h)*60
    m = np.floor(m_exact).astype(int)

    s = np.round((m_exact - m)*60, 0).astype(int)
    
    return h, m, s

def hr_min_sec_string(h,m,s):
    """
    Convert one or multiple angles in hours, minutes, seconds to strings.
    
    :param array or float h: The hours of the angle(s) to convert.
    :param array or float m: The minutes of the angle(s) to convert.
    :param array or float s: The seconds of the angle(s) to convert.
    
    :return: The string(s) containing the angle(s).
    :rtype: list(str) or str
    
    """
    #h,m,s = np.asarray(h), np.asarray(m), np.asarray(s)
    #s = int(np.round(s,0))
    try:
        return [ str((h[i]))+'h'+str((m[i]))+'m'+str(s[i])+'s' for i in range(len(h))]
    except TypeError:
        return str((h))+'h'+str((m))+'m'+str(s)+'s'

def deg_min_sec_string(d,m,s):
    """
    Convert one or multiple angles in degrees, minutes, seconds to strings.
    
    :param array or float d: The degrees of the angle(s) to convert.
    :param array or float m: The minutes of the angle(s) to convert.
    :param array or float s: The seconds of the angle(s) to convert.
    
    :return: The string(s) containing the angle(s).
    :rtype: list(str) or str
    
    """
    #d,m,s = np.asarray(d), np.asarray(m), np.asarray(s)
    #s = int(s)
    
    try:
        return [ str((d[i]))+'°'+str((m[i]))+'m'+str(s[i])+'s' for i in range(len(d))]
    except TypeError:
        return  str((d))+'°'+str((m))+'m'+str(s)+'s'
    
def theta_to_dec_degminsec(theta):
    """
    Compute :math:`\\delta` in degree, minutes, seconds from :math:`\\theta`.
    
    :param array or float theta: The :math:`\\theta` sky position angle(s) to convert.
    
    :return: :math:`\\delta` in degree, minutes, seconds.
    :rtype: list(str) or str
    
    """
    dec = np.rad2deg(0.5 * np.pi - theta)
    return deg_min_sec_string(*rad_to_deg_min_sec(dec))

def phi_to_ra_hrms(phi):
    """
    Compute :math:`\\alpha` in hours, minutes, seconds from :math:`\phi`.
    
    :param array or float phi: The :math:`\phi` sky position angle(s) to convert.
    
    :return: :math:`\\alpha` in hours, minutes, seconds.
    :rtype: list(str) or str
    
    """
    ra = np.rad2deg(phi)
    return hr_min_sec_string(*rad_to_hr_min_sec(ra))

def phi_to_ra_degminsec(phi):
    """
    Compute :math:`\\alpha` in degree, minutes, seconds from :math:`\phi`.
    
    :param array or float phi: The :math:`\phi` sky position angle(s) to convert.
    
    :return: :math:`\\alpha` in degree, minutes, seconds.
    :rtype: list(str) or str
    
    """
    ra = np.rad2deg(phi)
    return deg_min_sec_string(*rad_to_deg_min_sec(ra)) #hr_min_sec_string(*rad_to_hr_min_sec(ra))

##############################################################################
# TIDAL PARAMETERS
##############################################################################

def Lamt_delLam_from_Lam12(Lambda1, Lambda2, eta):
    """
    Compute the dimensionless tidal deformability combinations :math:`\\tilde{\Lambda}` and :math:`\delta\\tilde{\Lambda}`, defined in `arXiv:1402.5156 <https://arxiv.org/abs/1402.5156>`_ eq. (5) and (6), as a function of the dimensionless tidal deformabilities of the two objects and the symmetric mass ratio.
    
    :param array or float Lambda1: Tidal deformability of object 1, :math:`\Lambda_1`.
    :param array or float Lambda2: Tidal deformability of object 2, :math:`\Lambda_2`.
    :param array or float eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    :return: :math:`\\tilde{\Lambda}` and :math:`\delta\\tilde{\Lambda}`.
    :rtype: tuple(array, array) or tuple(float, float)
    
    """
    eta2 = eta*eta
    # This is needed to stabilize JAX derivatives
    Seta = jnp.sqrt(jnp.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        
    Lamt = (8./13.)*((1. + 7.*eta - 31.*eta2)*(Lambda1 + Lambda2) + Seta*(1. + 9.*eta - 11.*eta2)*(Lambda1 - Lambda2))
    
    delLam = 0.5*(Seta*(1. - 13272./1319.*eta + 8944./1319.*eta2)*(Lambda1 + Lambda2) + (1. - 15910./1319.*eta + 32850./1319.*eta2 + 3380./1319.*eta2*eta)*(Lambda1 - Lambda2))
    
    return Lamt, delLam
    
def Lam12_from_Lamt_delLam(Lamt, delLam, eta):
    """
    Compute the dimensionless tidal deformabilities of the two objects as a function of the dimensionless tidal deformability combinations :math:`\\tilde{\Lambda}` and :math:`\delta\\tilde{\Lambda}`, defined in `arXiv:1402.5156 <https://arxiv.org/abs/1402.5156>`_ eq. (5) and (6), and the symmetric mass ratio.
    
    :param array or float Lamt: Tidal deformability combination :math:`\\tilde{\Lambda}`.
    :param array or float delLam: Tidal deformability combination :math:`\delta\\tilde{\Lambda}`.
    :param array or float eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    :return: :math:`\Lambda_1` and :math:`\Lambda_2`.
    :rtype: tuple(array, array) or tuple(float, float)
    
    """
        
    eta2 = eta*eta
    Seta = jnp.sqrt(jnp.where(eta<0.25, 1.0 - 4.0*eta, 0.))
    
    mLp=(8./13.)*(1.+ 7.*eta-31.*eta2)
    mLm=(8./13.)*Seta*(1.+ 9.*eta-11.*eta2)
    mdp=Seta*(1.-(13272./1319.)*eta+(8944./1319.)*eta2)*0.5
    mdm=(1.-(15910./1319.)*eta+(32850./1319.)*eta2+(3380./1319.)*(eta2*eta))*0.5

    det=(306656./1319.)*(eta**5)-(5936./1319.)*(eta**4)

    Lambda1 = ((mdp-mdm)*Lamt+(mLm-mLp)*delLam)/det
    Lambda2 = ((-mdm-mdp)*Lamt+(mLm+mLp)*delLam)/det
    
    return Lambda1, Lambda2

##############################################################################
# MASSES
##############################################################################

def m1m2_from_Mceta(Mc, eta):
    """
    Compute the component masses of a binary given its chirp mass and symmetric mass ratio.
    
    :param array or float Mc: Chirp mass of the binary, :math:`{\cal M}_c`.
    :param array or float eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    :return: :math:`m_1` and :math:`m_2`.
    :rtype: tuple(array, array) or tuple(float, float)
    
    """
    Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
    m1 = 0.5*(Mc/(eta**(3./5.)))*(1. + Seta)
    m2 = 0.5*(Mc/(eta**(3./5.)))*(1. - Seta)

    return m1, m2
    
def Mceta_from_m1m2(m1, m2):
    """
    Compute the chirp mass and symmetric mass ratio of a binary given its component masses.
    
    :param array or float m1: Mass of the primary object, :math:`m_1`.
    :param array or float m2: Mass of the secondary object, :math:`m_2`.
    :return: :math:`{\cal M}_c` and :math:`\eta`.
    :rtype: tuple(array, array) or tuple(float, float)
    
    """
    Mc  = ((m1*m2)**(3./5.))/((m1+m2)**(1./5.))
    eta = (m1*m2)/((m1+m2)*(m1+m2))
    
    return Mc, eta

##############################################################################
# SPINS
##############################################################################

def zrot(angle, vx, vy, vz):
    """
    Perofrm a rotation of the components of a vector around the :math:`z` axis by a given angle.
    
    :param array or float angle: Rotation angle(s).
    :param array or float vx: The :math:`x` component(s) of the vector(s).
    :param array or float vy: The :math:`y` component(s) of the vector(s).
    :param array or float vz: The :math:`z` component(s) of the vector(s).
    
    :return: The components of the rotated vector(s) around :math:`z`.
    :rtype: tuple(array, array, array) or tuple(float, float, float)
    
    """
    # Function to perofrm a rotation of the components of a vector around the z axis by a given angle
    tmp = vx*np.cos(angle) - vy*np.sin(angle)
    yy  = vx*np.sin(angle) + vy*np.cos(angle)
    xx  = tmp
    return xx, yy, vz

def yrot(angle, vx, vy, vz):
    """
    Perofrm a rotation of the components of a vector around the :math:`y` axis by a given angle.
    
    :param array or float angle: Rotation angle(s).
    :param array or float vx: The :math:`x` component(s) of the vector(s).
    :param array or float vy: The :math:`y` component(s) of the vector(s).
    :param array or float vz: The :math:`z` component(s) of the vector(s).
    
    :return: The components of the rotated vector(s) around :math:`y`.
    :rtype: tuple(array, array, array) or tuple(float, float, float)
    
    """
    # Function to perofrm a rotation of the components of a vector around the y axis by a given angle
    tmp = vx*np.cos(angle) + vz*np.sin(angle)
    zz  = - vx*np.sin(angle) + vz*np.cos(angle)
    xx  = tmp
    return xx, vy, zz

def TransformPrecessing_angles2comp(thetaJN, phiJL, theta1, theta2, phi12, chi1, chi2, Mc, eta, fRef, phiRef):
    """
    Compute the components of the spin in cartesian frame given the angular variables.
    Adapted from :py:class:`LALSimInspiral.c`, function :py:class:`XLALSimInspiralTransformPrecessingNewInitialConditions`, line 5885.
    For a scheme of the conventions, see `<https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group__lalsimulation__inference.html>`_.
    
    :param array or float thetaJN: Inclination between total angular momentum (:math:`J`) and the direction of propagation, :math:`\\theta_{JN}` (so that :math:`\\theta_{JN} \\to \iota` for :math:`\\chi_1 + \\chi_2 \\to 0`).
    :param array or float phiJL: Azimuthal angle of the Newtonian orbital angular momentum :math:`L_N` on its cone about the total angular momentum :math:`J`, :math:`\phi_{JL}`.
    :param array or float theta1: Inclination (tilt angle) of object 1 measured from the Newtonian orbital angular momentum (:math:`L_N`), :math:`\\theta_{s,1}`.
    :param array or float theta2: Inclination (tilt angle) of object 2 measured from the Newtonian orbital angular momentum (:math:`L_N`), :math:`\\theta_{s,2}`.
    :param array or float phi12: Difference in azimuthal angles between the two spins, :math:`\phi_{1,2}`.
    :param array or float chi1: Dimensionless spin magnitude of object 1, :math:`\chi_1`.
    :param array or float chi2: Dimensionless spin magnitude of object 2, :math:`\chi_2`.
    :param array or float Mc: Chirp mass of the binary, :math:`{\cal M}_c`, in units of :math:`\\rm M_{\odot}`.
    :param array or float eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    :param array or float fRef: Reference frequency, in :math:`\\rm Hz`.
    :param array or float phiRef: Reference phase, in :math:`\\rm rad`.
    
    :return: :math:`\iota`, :math:`\chi_{1,x}`, :math:`\chi_{1,y}`, :math:`\chi_{1,z}`, :math:`\chi_{2,x}`, :math:`\chi_{2,y}`, :math:`\chi_{2,z}`.
    :rtype: tuple(array, array, array, array, array, array, array) or tuple(float, float, float, float, float, float, float)
    
    """
    
    LNhx = 0.
    LNhy = 0.
    LNhz = 1.

    s1hatx = np.sin(theta1) * np.cos(phiRef)
    s1haty = np.sin(theta1) * np.sin(phiRef)
    s1hatz = np.cos(theta1)
    s2hatx = np.sin(theta2) * np.cos(phi12+phiRef)
    s2haty = np.sin(theta2) * np.sin(phi12+phiRef)
    s2hatz = np.cos(theta2)

    m1, m2 = m1m2_from_Mceta(Mc, eta)
    M = m1+m2
    v0 = (M * glob.GMsun_over_c3 * np.pi * fRef)**(1./3.)

    # Define S1, S2, J with proper magnitudes
    Lmag = (M*M*eta/v0)*(1. + v0*v0*(1.5 + eta/6.))
    
    s1x = m1 * m1 * chi1 * s1hatx
    s1y = m1 * m1 * chi1 * s1haty
    s1z = m1 * m1 * chi1 * s1hatz
    s2x = m2 * m2 * chi2 * s2hatx
    s2y = m2 * m2 * chi2 * s2haty
    s2z = m2 * m2 * chi2 * s2hatz
    Jx = s1x + s2x
    Jy = s1y + s2y
    Jz = Lmag + s1z + s2z

    # Normalize J to Jhat, find its angles in starting frame

    Jnorm = np.sqrt(Jx*Jx + Jy*Jy + Jz*Jz)
    Jhatx = Jx / Jnorm
    Jhaty = Jy / Jnorm
    Jhatz = Jz / Jnorm
    theta0 = np.arccos(Jhatz)
    phi0 = np.arctan2(np.real(Jhaty), np.real(Jhatx))
    
    # Rotation 1: Rotate about z-axis by -phi0 to put Jhat in x-z plane
    s1hatx, s1haty, s1hatz = zrot(-phi0, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = zrot(-phi0, s2hatx, s2haty, s2hatz)

    # Rotation 2: Rotate about new y-axis by -theta0 to put Jhat along z-axis
    LNhx, LNhy, LNhz       = yrot(-theta0, LNhx, LNhy, LNhz)
    s1hatx, s1haty, s1hatz = yrot(-theta0, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = yrot(-theta0, s2hatx, s2haty, s2hatz)

    # Rotation 3: Rotate about new z-axis by phiJL to put L at desired azimuth about J.
    # Note that is currently in x-z plane towards -x (i.e. azimuth=pi). Hence we rotate about z by phiJL - pi
    LNhx, LNhy, LNhz       = zrot(phiJL - np.pi, LNhx, LNhy, LNhz)
    s1hatx, s1haty, s1hatz = zrot(phiJL - np.pi, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = zrot(phiJL - np.pi, s2hatx, s2haty, s2hatz)
    
    # The cosine of the angle between L and N is the scalar product of the two vectors, no further rotation needed
    
    Nx=0.
    Ny=np.sin(thetaJN)
    Nz=np.cos(thetaJN)
    iota=np.arccos(Nx*LNhx+Ny*LNhy+Nz*LNhz)

    # Rotation 4-5: Now J is along z and N in y-z plane, inclined from J by thetaJN and with >ve component along y.
    # Now we bring L into the z axis to get spin components.
    thetaLJ = np.arccos(LNhz)
    phiL    = np.arctan2(np.real(LNhy), np.real(LNhx))
    
    s1hatx, s1haty, s1hatz = zrot(-phiL, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = zrot(-phiL, s2hatx, s2haty, s2hatz)
    Nx, Ny, Nz             = zrot(-phiL, Nx, Ny, Nz)
    
    s1hatx, s1haty, s1hatz = yrot(-thetaLJ, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = yrot(-thetaLJ, s2hatx, s2haty, s2hatz)
    Nx, Ny, Nz             = yrot(-thetaLJ, Nx, Ny, Nz)
    
    # Rotation 6: Now L is along z and we have to bring N in the y-z plane with >ve y components.
    
    phiN = np.arctan2(np.real(Ny), np.real(Nx))
    
    s1hatx, s1haty, s1hatz = zrot(np.pi/2.-phiN-phiRef, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = zrot(np.pi/2.-phiN-phiRef, s2hatx, s2haty, s2hatz)
    
    S1x = s1hatx*chi1
    S1y = s1haty*chi1
    S1z = s1hatz*chi1
    S2x = s2hatx*chi2
    S2y = s2haty*chi2
    S2z = s2hatz*chi2
    
    return iota, S1x, S1y, S1z, S2x, S2y, S2z

def TransformPrecessing_comp2angles(iota, S1x, S1y, S1z, S2x, S2y, S2z, Mc, eta, fRef, phiRef):
    """
    Compute the angular variables of the spins given the components in cartesian frame
    Adapted from :py:class:`LALSimInspiral.c`, function :py:class:`XLALSimInspiralTransformPrecessingWvf2PE`, line 6105.
    For a scheme of the conventions, see `<https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group__lalsimulation__inference.html>`_.
    
    :param array or float iota: Inclination between the orbital angular momentum and the direction of propagation.
    :param array or float S1x: spin of object 1 along the axis :math:`x`, :math:`\chi_{1,x}`.
    :param array or float S1y: spin of object 1 along the axis :math:`y`, :math:`\chi_{1,y}`.
    :param array or float S1z: spin of object 1 along the axis :math:`z`, :math:`\chi_{1,z}`.
    :param array or float S2x: spin of object 2 along the axis :math:`x`, :math:`\chi_{2,x}`.
    :param array or float S2y: spin of object 2 along the axis :math:`y`, :math:`\chi_{2,y}`.
    :param array or float S2z: spin of object 2 along the axis :math:`z`, :math:`\chi_{2,z}`.
    :param array or float Mc: Chirp mass of the binary, :math:`{\cal M}_c`, in units of :math:`\\rm M_{\odot}`.
    :param array or float eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    :param array or float fRef: Reference frequency, in :math:`\\rm Hz`.
    :param array or float phiRef: Reference phase, in :math:`\\rm rad`.
    
    :return: :math:`\\theta_{JN}`, :math:`\phi_{JL}`, :math:`\\theta_{s,1}`, :math:`\\theta_{s,2}`, :math:`\phi_{1,2}`, :math:`\chi_1`, :math:`\chi_2`.
    :rtype: tuple(array, array, array, array, array, array, array) or tuple(float, float, float, float, float, float, float)
    
    """
    
    LNhx = 0.
    LNhy = 0.
    LNhz = 1.
    chi1 = np.sqrt(S1x*S1x + S1y*S1y + S1z*S1z)
    chi2 = np.sqrt(S2x*S2x + S2y*S2y + S2z*S2z)
    
    s1hatx = np.where(chi1>0., S1x/(chi1), 0.)
    s1haty = np.where(chi1>0., S1y/(chi1), 0.)
    s1hatz = np.where(chi1>0., S1z/(chi1), 0.)
    s2hatx = np.where(chi2>0., S2x/(chi2), 0.)
    s2haty = np.where(chi2>0., S2y/(chi2), 0.)
    s2hatz = np.where(chi2>0., S2z/(chi2), 0.)
    
    phi1 = np.arctan2(np.real(s1haty), np.real(s1hatx))
    phi2 = np.arctan2(np.real(s2haty), np.real(s2hatx))
    
    phi12 = np.where(phi2 - phi1 < 0., 2.*np.pi + (phi2 - phi1), phi2 - phi1)
    
    theta1 = np.arccos(s1hatz)
    theta2 = np.arccos(s2hatz)
    
    m1, m2 = m1m2_from_Mceta(Mc, eta)
    M = m1+m2
    v0 = (M * glob.GMsun_over_c3 * np.pi * fRef)**(1./3.)#np.cbrt(M * glob.GMsun_over_c3 * np.pi * fRef)
    # Define S1, S2, J with proper magnitudes
    Lmag = (M*M*eta/v0)*(1. + v0*v0*(1.5 + eta/6.))
    
    s1x = m1 * m1 * S1x
    s1y = m1 * m1 * S1y
    s1z = m1 * m1 * S1z
    s2x = m2 * m2 * S2x
    s2y = m2 * m2 * S2y
    s2z = m2 * m2 * S2z
    Jx = s1x + s2x
    Jy = s1y + s2y
    Jz = Lmag*LNhz + s1z + s2z
    
    # Normalize J to Jhat, find its angles in starting frame
    
    Jnorm = np.sqrt(Jx*Jx + Jy*Jy + Jz*Jz)
    Jhatx = Jx / Jnorm
    Jhaty = Jy / Jnorm
    Jhatz = Jz / Jnorm
    thetaJL = np.arccos(Jhatz)
    phiJ    = np.arctan2(np.real(Jhaty), np.real(Jhatx))
    
    phiO = np.pi/2. - phiRef
    Nx = np.sin(iota)*np.cos(phiO);
    Ny = np.sin(iota)*np.sin(phiO);
    Nz = np.cos(iota)
    
    thetaJN = np.arccos(Jhatx*Nx + Jhaty*Ny + Jhatz*Nz)
    
    # The easiest way to define the phiJL is to rotate to the frame where J is along z and N is in the y-z plane
    Nx, Ny, Nz = zrot(-phiJ, Nx, Ny, Nz)
    Nx, Ny, Nz = yrot(-thetaJL, Nx, Ny, Nz)
    
    LNhx, LNhy, LNhz = zrot(-phiJ, LNhx, LNhy, LNhz)
    LNhx, LNhy, LNhz = yrot(-thetaJL, LNhx, LNhy, LNhz)
    
    phiN = np.arctan2(np.real(Ny), np.real(Nx))
    
    # After rotation defined below N should be in y-z plane inclined by thetaJN to J=z
    LNhx, LNhy, LNhz = zrot(np.pi/2. - phiN, LNhx, LNhy, LNhz)
    
    phiJL = np.arctan2(np.real(LNhy), np.real(LNhx))
    phiJL = np.where(phiJL<0., phiJL+2.*np.pi, phiJL)
    
    return thetaJN, phiJL, theta1, theta2, phi12, chi1, chi2
    
##############################################################################
# TIMES
##############################################################################

def GPSt_to_J200t(t_GPS):
    # According to https://www.andrews.edu/~tzs/timeconv/timedisplay.php the GPS time of J2000 is 630763148 s
    return t_GPS - 630763148.0
        
def GPSt_to_LMST(t_GPS, lat, long):
    """
    Compute the Local Mean Sidereal Time (LMST) in units of fraction of day, from GPS time and location (given as latitude and longitude in degrees)
    
    :param array or float t_GPS: GPS time(s) to convert, in seconds.
    :param float lat: Latitude of the chosen location, in :math:`\\rm deg`.
    :param float long: Longitude of the chosen location, in :math:`\\rm deg`.
    
    :return: Local Mean Sidereal Time(s).
    :rtype: array or float
    
    """
    from astropy.coordinates import EarthLocation
    import astropy.time as aspyt
    import astropy.units as u
    # Uncomment the next two lines in case of troubles with IERS
    #import astropy
    #astropy.utils.iers.conf.iers_degraded_accuracy='ignore'
    loc = EarthLocation(lat=lat*u.deg, lon=long*u.deg)
    t = aspyt.Time(t_GPS, format='gps', location=(loc))
    LMST = t.sidereal_time('mean').value
    return jnp.array(LMST/24.)

def GPSt_to_GMST_alt(t_GPS):
    """
    Compute the Greenwich Mean Sidereal Time (GMST) in units of fraction of day, from GPS time. This function does not rely on external libraries but is **approximate**.
    The implementation is taken from `GWFish <https://github.com/janosch314/GWFish/tree/main>`_.
    
    :param array or float t_GPS: GPS time(s) to convert, in seconds.
    
    :return: Greenwich Mean Sidereal Time(s).
    :rtype: array or float
    
    """
    
    return jnp.mod(9.533088395981618 + (t_GPS - 1126260000.) / 3600. * 24. / glob.siderealDay, 24.) / 24.

##############################################################################
# SPHERICAL HARMONICS
##############################################################################

def Add_Higher_Modes(Ampl, Phi, iota, phi=0.):
    """
    Compute the total signal from a collection of different modes.
    
    :param dict(array, array, ...) Ampl: Dictionary containing the amplitudes for each mode computed on a grid of frequencies. The keys are expected to be stings made up of :math:`l` and :math:`m`, e.g. for :math:`(2,2)` --> key= ``'22'``.
    :param dict(array, array, ...) Phi: Dictionary containing the phases for each mode computed on a grid of frequencies.
    :param array or float iota: The inclination angle(s) of the system(s) with respect to orbital angular momentum, :math:`\iota`, in :math:`\\rm rad`.
    :param array or float phi: The second angular direction of the spherical coordinate system.
    :return: Plus and cross polarisations of the GW for the chosen events evaluated on the frequency grid.
    :rtype: tuple(array, array)
    
    """
    # Function to compute the total signal from a collection of different modes
    # Ampl and Phi have to be dictionaries containing the amplitudes and phases, computed on a grid of frequencies, for
    # each mode. The keys are expected to be stings made up of l and m, e.g. for (2,2) -> key='22'
    
    def SpinWeighted_SphericalHarmonic(theta, phi, l, m, s=-2):
        # Taken from arXiv:0709.0093v3 eq. (II.7), (II.8) and LALSimulation for the s=-2 case and up to l=4
        
        if s != -2:
            raise ValueError('The only spin-weight implemented for the moment is s = -2.')
            
        if (2 == l):
            if (-2 == m):
                res = jnp.sqrt( 5.0 / ( 64.0 * jnp.pi ) ) * ( 1.0 - jnp.cos( theta ))*( 1.0 - jnp.cos( theta ))
            elif (-1 == m):
                res = jnp.sqrt( 5.0 / ( 16.0 * jnp.pi ) ) * jnp.sin( theta )*( 1.0 - jnp.cos( theta ))
            elif (0 == m):
                res = jnp.sqrt( 15.0 / ( 32.0 * jnp.pi ) ) * jnp.sin( theta )*jnp.sin( theta )
            elif (1 == m):
                res = jnp.sqrt( 5.0 / ( 16.0 * jnp.pi ) ) * jnp.sin( theta )*( 1.0 + jnp.cos( theta ))
            elif (2 == m):
                res = jnp.sqrt( 5.0 / ( 64.0 * jnp.pi ) ) * ( 1.0 + jnp.cos( theta ))*( 1.0 + jnp.cos( theta ))
            else:
                raise ValueError('Invalid m for l = 2.')
                
        elif (3 == l):
            if (-3 == m):
                res = jnp.sqrt(21.0/(2.0*jnp.pi))*jnp.cos(theta*0.5)*((jnp.sin(theta*0.5))**(5.))
            elif (-2 == m):
                res = jnp.sqrt(7.0/(4.0*jnp.pi))*(2.0 + 3.0*jnp.cos(theta))*((jnp.sin(theta*0.5))**(4.0))
            elif (-1 == m):
                res = jnp.sqrt(35.0/(2.0*jnp.pi))*(jnp.sin(theta) + 4.0*jnp.sin(2.0*theta) - 3.0*jnp.sin(3.0*theta))/32.0
            elif (0 == m):
                res = (jnp.sqrt(105.0/(2.0*jnp.pi))*jnp.cos(theta)*(jnp.sin(theta)*jnp.sin(theta)))*0.25
            elif (1 == m):
                res = -jnp.sqrt(35.0/(2.0*jnp.pi))*(jnp.sin(theta) - 4.0*jnp.sin(2.0*theta) - 3.0*jnp.sin(3.0*theta))/32.0
            elif (2 == m):
                res = jnp.sqrt(7.0/jnp.pi)*((jnp.cos(theta*0.5))**(4.0))*(-2.0 + 3.0*jnp.cos(theta))*0.5
            elif (3 == m):
                res = -jnp.sqrt(21.0/(2.0*jnp.pi))*((jnp.cos(theta/2.0))**(5.0))*jnp.sin(theta*0.5)
            else:
                raise ValueError('Invalid m for l = 3.')
                
        elif (4 == l):
            if (-4 == m):
                res = 3.0*jnp.sqrt(7.0/jnp.pi)*(jnp.cos(theta*0.5)*jnp.cos(theta*0.5))*((jnp.sin(theta*0.5))**6.0)
            elif (-3 == m):
                res = 3.0*jnp.sqrt(7.0/(2.0*jnp.pi))*jnp.cos(theta*0.5)*(1.0 + 2.0*jnp.cos(theta))*((jnp.sin(theta*0.5))**5.0)
            elif (-2 == m):
                res = (3.0*(9.0 + 14.0*jnp.cos(theta) + 7.0*jnp.cos(2.0*theta))*((jnp.sin(theta/2.0))**4.0))/(4.0*jnp.sqrt(jnp.pi))
            elif (-1 == m):
                res = (3.0*(3.0*jnp.sin(theta) + 2.0*jnp.sin(2.0*theta) + 7.0*jnp.sin(3.0*theta) - 7.0*jnp.sin(4.0*theta)))/(32.0*jnp.sqrt(2.0*jnp.pi))
            elif (0 == m):
                res = (3.0*jnp.sqrt(5.0/(2.0*jnp.pi))*(5.0 + 7.0*jnp.cos(2.0*theta))*(jnp.sin(theta)*jnp.sin(theta)))/16.
            elif (1 == m):
                res = (3.0*(3.0*jnp.sin(theta) - 2.0*jnp.sin(2.0*theta) + 7.0*jnp.sin(3.0*theta) + 7.0*jnp.sin(4.0*theta)))/(32.0*jnp.sqrt(2.0*jnp.pi))
            elif (2 == m):
                res = (3.0*((jnp.cos(theta*0.5))**4.0)*(9.0 - 14.0*jnp.cos(theta) + 7.0*jnp.cos(2.0*theta)))/(4.0*jnp.sqrt(jnp.pi))
            elif (3 == m):
                res = -3.0*jnp.sqrt(7.0/(2.0*jnp.pi))*((jnp.cos(theta*0.5))**5.0)*(-1.0 + 2.0*jnp.cos(theta))*jnp.sin(theta*0.5)
            elif (4 == m):
                res = 3.0*jnp.sqrt(7.0/jnp.pi)*((jnp.cos(theta*0.5))**6.0)*(jnp.sin(theta*0.5)*jnp.sin(theta*0.5))
            else:
                raise ValueError('Invalid m for l = 4.')
                
        else:
            raise ValueError('Multipoles with l > 4 not implemented yet.')
        
        return res*jnp.exp(1j*m*phi)
    
    hp = jnp.zeros(Ampl[list(Ampl)[0]].shape)
    hc = jnp.zeros(Ampl[list(Ampl)[0]].shape)
    
    for key in Ampl.keys():
        if key in Phi.keys():
            l, m = int(key[:2//2]), int(key[2//2:])
            Y = SpinWeighted_SphericalHarmonic(iota, phi, l, m)
            if m:
                Ymstar = jnp.conj(SpinWeighted_SphericalHarmonic(iota, phi, l, -m))
            else:
                Ymstar = 0.
            
            hp = hp + Ampl[key]*jnp.exp(-1j*Phi[key])*(0.5*(Y + ((-1)**l)*Ymstar))
            hc = hc + Ampl[key]*jnp.exp(-1j*Phi[key])*(-1j* 0.5 * (Y - ((-1)**l)* Ymstar))
    
    return hp, hc

##############################################################################
# DETECTOR RELATIVE ORIENTATION AND DISTANCE
##############################################################################

def ang_btw_dets_GC(det1, det2):
    """
    Compute the angle between two detectors with respect to the great circle that joins them, see `<https://en.wikipedia.org/wiki/Great-circle_navigation>`_.
    
    :param dict(float, float, float) det1: Dictionary containing the latitude, ``'lat'``, longitude, ``'long'``, and orientation, ``'xax'``, of the first detector (all in degrees), as in :py:data:`gwfast.gwfastGlobals.detectors`.
    :param dict(float, float, float) det2: Dictionary containing the latitude, ``'lat'``, longitude, ``'long'``, and orientation, ``'xax'``, of the second detector (all in degrees), as in :py:data:`gwfast.gwfastGlobals.detectors`.
    
    :return: Angle between the two detectors, in :math:`\\rm deg`.
    :rtype: float
    
    """
    lat1, lat2   = np.deg2rad(det1['lat']), np.deg2rad(det2['lat'])
    long1, long2 = np.deg2rad(det1['long']), np.deg2rad(det2['long'])
    
    def initial_course(lat1, lat2, long1, long2):
        # Compute the course at the initial point given two points
        # See http://www.edwilliams.org/avform147.htm#Crs or https://en.wikipedia.org/wiki/Great-circle_navigation
        a = np.sin(long2-long1)*np.cos(lat2)
        b = np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(long2-long1)

        # If the initial point is a pole we need a "fix"
        return np.rad2deg(np.where(np.isclose(np.cos(lat1), 0.), np.where(lat1 > 0., np.pi, 2.*np.pi), np.arctan2(a,b)))
    
    def final_course(lat1, lat2, long1, long2):
        # Compute the course at the final point given two points
        # See http://www.edwilliams.org/avform147.htm#Crs or https://en.wikipedia.org/wiki/Great-circle_navigation
        a = np.sin(long2-long1)*np.cos(lat1)
        b = -np.cos(lat2)*np.sin(lat1)+np.sin(lat2)*np.cos(lat1)*np.cos(long2-long1)

        # If the final point is a pole we need a "fix"
        return np.rad2deg(np.where(np.isclose(np.cos(lat2), 0.), np.where(lat2 > 0., np.pi, 2.*np.pi), np.arctan2(a,b)))
    
    # Compute the course at the first detector
    ang1 = initial_course(lat1, lat2, long1, long2)
    # Compute the course at the second detector
    ang2 = final_course(lat1, lat2, long1, long2)

    angdiff = 360.-(ang2-ang1)

    return (det1['xax'] - det2['xax']) + np.where(angdiff<180.,angdiff, angdiff-360.)

def dist_btw_dets_GC(det1, det2):
    """
    Compute the great circle distance between two detectors using the Vincenty formula in spherical case, see `<https://en.wikipedia.org/wiki/Great-circle_distance>`_.
    
    :param dict(float, float, float) det1: Dictionary containing the latitude, ``'lat'``, longitude, ``'long'``, and orientation, ``'xax'``, of the first detector (all in degrees), as in :py:data:`gwfast.gwfastGlobals.detectors`.
    :param dict(float, float, float) det2: Dictionary containing the latitude, ``'lat'``, longitude, ``'long'``, and orientation, ``'xax'``, of the second detector (all in degrees), as in :py:data:`gwfast.gwfastGlobals.detectors`.
    
    :return: Great circle distance between the detectors, in :math:`\\rm km`.
    :rtype: float
    
    """

    lat1, lat2   = np.deg2rad(det1['lat']), np.deg2rad(det2['lat'])
    long1, long2 = np.deg2rad(det1['long']), np.deg2rad(det2['long'])
    dlong = long2 - long1

    num = np.sqrt((np.cos(lat2)*np.sin(dlong))**2 + (np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlong))**2)
    den = np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(dlong)

    return glob.REarth*np.arctan2(num, den)

def dist_btw_dets_Chord(det1, det2):
    """
    Compute the great circle chord length between two detectors, see `<https://en.wikipedia.org/wiki/Great-circle_distance>`_.
    
    :param dict(float, float, float) det1: Dictionary containing the latitude, ``'lat'``, longitude, ``'long'``, and orientation, ``'xax'``, of the first detector (all in degrees), as in :py:data:`gwfast.gwfastGlobals.detectors`.
    :param dict(float, float, float) det2: Dictionary containing the latitude, ``'lat'``, longitude, ``'long'``, and orientation, ``'xax'``, of the second detector (all in degrees), as in :py:data:`gwfast.gwfastGlobals.detectors`.
    
    :return: Great circle chord length between the detectors, in :math:`\\rm km`.
    :rtype: float
    
    """
    
    lat1, lat2   = np.deg2rad(det1['lat']), np.deg2rad(det2['lat'])
    long1, long2 = np.deg2rad(det1['long']), np.deg2rad(det2['long'])

    dx = np.cos(lat2)*np.cos(long2) - np.cos(lat1)*np.cos(long1)
    dy = np.cos(lat2)*np.sin(long2) - np.cos(lat1)*np.sin(long1)
    dz = np.sin(lat2) - np.sin(lat1)

    return glob.REarth*np.sqrt(dx*dx + dy*dy + dz*dz)
    
##############################################################################
# OTHERS
##############################################################################

def check_evparams(evParams):
    """
    Check the format of the events parameters and make the needed conversions.
    
    :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
    
    """
    # Function to check the format of the events' parameters and make the needed conversions
    try:
        _ = evParams['tcoal']
    except KeyError:
        try:
            print('Adding tcoal from tGPS')
            # In the code we use Greenwich Mean Sidereal Time (LMST computed at long = 0. deg) as convention, so convert t_GPS
            evParams['tcoal'] = GPSt_to_LMST(evParams['tGPS'], lat=0., long=0.)
        except KeyError:
            raise ValueError('One among tGPS and tcoal has to be provided.')
    
    try:
        _ = evParams['iota']
    except KeyError:
        try:
            # In the precessing spin case, iota is different from thetaJN, and is computed later. This is just a fix.
            evParams['iota'] = evParams['thetaJN']
        except KeyError:
            raise ValueError('One among iota and thetaJN has to be provided.')
    
    try:
        _ = evParams['Mc']
    except KeyError:
        try:
            print('Adding Mc and eta from the individual detector-frame masses')
            evParams['Mc'], evParams['eta'] = Mceta_from_m1m2(evParams['m1'], evParams['m2'])
        except KeyError:
            raise ValueError('Two among (Mc, eta) and (m1, m2) have to be provided.')
    #try:
    #    _ =evParams['chi1z']
    #except KeyError:
    #    try:
    #        print('Adding chi1z, chi2z from chiS, chiA')
    #        evParams['chi1z'] = evParams['chiS'] + evParams['chiA']
    #        evParams['chi2z'] = evParams['chiS'] - evParams['chiA']
    #    except KeyError:
    #        raise ValueError('Two among chi1z, chi2z and chiS, chiA have to be provided.')
            
    try:
        _ = evParams['theta']
    except KeyError:
        try:
            print('Adding (theta, phi) from (ra, dec)')
            evParams['theta'] = np.pi/2-evParams['dec']
            evParams['phi']=evParams['ra']
        except KeyError:
            raise ValueError('Two among (theta, phi) and (ra, dec) have to be provided.')
    return evParams
                
                
             

class RegularGridInterpolator_JAX:
    """
    Implementation of ``SciPy`` 's :py:class:`RegularGridInterpolator` in a ``JAX`` usable way. Essentially ``numpy`` in the original code is changed to ``jax.numpy`` because of assignement issues, arising when using ``vmap`` and ``jacrev``. We also changed the ``+=`` syntax which creates issues in ``JAX``.
    
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

import os
import sys

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited.
    
    Full credit goes to https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functionsorator

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


class config_conversion():

    '''
    Dummy class used to convert what the configparser read
    into what the argparser create.
    '''

    def __init__(self, config_parser):

        self.__config_parser = config_parser
        for key, value in zip(config_parser.keys(), config_parser.values()):
            try:
                setattr(self, key, ast.literal_eval(value))
            except (ValueError, SyntaxError):
                if "[" and "]" in value:
                    setattr(self, key, value.strip("][").split(", "))
                else:
                    setattr(self, key, value)
