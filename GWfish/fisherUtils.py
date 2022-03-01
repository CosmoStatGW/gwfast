#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

##############################################################################
# ANGLES
##############################################################################

# See http://spiff.rit.edu/classes/phys440/lectures/coords/coords.html
# Check: https://www.vercalendario.info/en/how/convert-ra-degrees-hours.html

def ra_dec_from_th_phi_rad(theta, phi):
    ra = phi #np.rad2deg(phi)
    dec = 0.5*np.pi - theta #np.rad2deg(0.5 * np.pi - theta)
    return ra, dec

def ra_dec_from_th_phi(theta, phi):
        ra = np.rad2deg(phi)
        dec = np.rad2deg(0.5 * np.pi - theta)
        return ra, dec

  
def th_phi_from_ra_dec(ra, dec):
    theta = 0.5 * np.pi - np.deg2rad(dec)
    phi = np.deg2rad(ra)
    return theta, phi

def deg_min_sec_to_decimal_deg(d, m, s):
    return d + m/60 + s/3600

def hr_min_sec_to_decimal_deg(h, m, s):
    # decimal degrees=15*h+15*m/60+15*s/3600.
    
    return 15*(h+m/60+s/3600)


def deg_min_sec_to_rad(d, m, s):
    return deg_min_sec_to_decimal_deg(d, m, s)*np.pi/180

def hr_min_sec_to_rad(h, m, s):
    return hr_min_sec_to_decimal_deg(h, m, s)*np.pi/180


def rad_to_deg_min_sec(rad):
    # check: https://www.calculatorsoup.com/calculators/conversions/convert-decimal-degrees-to-degrees-minutes-seconds.php
    
    d = np.floor(rad).astype(int)  
    
    m_exact = (rad-d)*60    
    m = np.floor(m_exact).astype(int)

    s = np.round((m_exact - m)*60, 0).astype(int)
    
    return d, m, s

def rad_to_hr_min_sec(rad):
    
    hh = rad/15
    h = np.floor(hh).astype(int)
    
    m_exact = (hh-h)*60
    m = np.floor(m_exact).astype(int)

    s = np.round((m_exact - m)*60, 0).astype(int)
    
    return h, m, s

def hr_min_sec_string(h,m,s):
    #h,m,s = np.asarray(h), np.asarray(m), np.asarray(s)
    #s = int(np.round(s,0))
    try:
        return [ str((h[i]))+'h'+str((m[i]))+'m'+str(s[i])+'s' for i in range(len(h))]
    except TypeError:
        return str((h))+'h'+str((m))+'m'+str(s)+'s'

def deg_min_sec_string(d,m,s):
    
    #d,m,s = np.asarray(d), np.asarray(m), np.asarray(s)
    #s = int(s)
    
    try:
        return [ str((d[i]))+'°'+str((m[i]))+'m'+str(s[i])+'s' for i in range(len(d))]
    except TypeError:
        return  str((d))+'°'+str((m))+'m'+str(s)+'s'
    
def theta_to_dec_degminsec(theta):
    dec = np.rad2deg(0.5 * np.pi - theta)
    return deg_min_sec_string(*rad_to_deg_min_sec(dec))

def phi_to_ra_hrms(phi):
    ra = np.rad2deg(phi)
    return hr_min_sec_string(*rad_to_hr_min_sec(ra))

def phi_to_ra_degminsec(phi):
    ra = np.rad2deg(phi)
    return deg_min_sec_string(*rad_to_deg_min_sec(ra)) #hr_min_sec_string(*rad_to_hr_min_sec(ra))

def Lamt_delLam_from_Lam12(Lambda1, Lambda2, eta):
    # Returns the dimensionless tidal deformability parameters Lambda_tilde and delta_Lambda as defined in PhysRevD.89.103012 eq. (5) and (6), as a function of the dimensionless tidal deformabilities of the two objects and the symmetric mass ratio
    eta2 = eta*eta
    # This is needed to stabilize JAX derivatives
    Seta = jnp.sqrt(jnp.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        
    Lamt = (8./13.)*((1. + 7.*eta - 31.*eta2)*(Lambda1 + Lambda2) + Seta*(1. + 9.*eta - 11.*eta2)*(Lambda1 - Lambda2))
    
    delLam = 0.5*(Seta*(1. - 13272./1319.*eta + 8944./1319.*eta2)*(Lambda1 + Lambda2) + (1. - 15910./1319.*eta + 32850./1319.*eta2 + 3380./1319.*eta2*eta)*(Lambda1 - Lambda2))
    
    return Lamt, delLam
    
def Lam12_from_Lamt_delLam(Lamt, delLam, eta):
        # inversion of Wade et al, PhysRevD.89.103012, eq. (5) and (6)
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
# TIMES
##############################################################################

def GPSt_to_J200t(t_GPS):
    # According to https://www.andrews.edu/~tzs/timeconv/timedisplay.php the GPS time of J2000 is 630763148 s
    return t_GPS - 630763148.0



def check_evparams(evParams):
        try:
            evParams['logdL']
        except KeyError:
            try:
                evParams['logdL'] = np.log(evParams['dL'])
            except KeyError:
                raise ValueError('One among dL and logdL has to be provided.')
        try:
            evParams['tcoal']
        except KeyError:
            try:
                # In the code we use Greenwich Mean Sideral Time (LMST computed at long = 0. deg) as convention, so convert t_GPS
                evParams['tcoal'] = GPSt_to_LMST(evParams['tGPS'], lat=0., long=0.)
            except KeyError:
                raise ValueError('One among tGPS and tcoal has to be provided.')
        try:
            evParams['chi1z']
        except KeyError:
            try:
                evParams['chi1z'] = evParams['chiS'] + evParams['chiA']
                evParams['chi2z'] = evParams['chiS'] - evParams['chiA']
            except KeyError:
                raise ValueError('Two among chi1z, chi2z and chiS and chiA have to be provided.')
        
        #try:
        #    evParams['cosiota']
        #except KeyError:
        #    try:
        #        evParams['cosiota'] = np.cos(evParams['iota'])
        #    except KeyError:
        #        raise ValueError('One among dL and logdL has to be provided.')
        
        
def GPSt_to_LMST(t_GPS, lat, long):
  # Returns the Local Mean Sideral Time in units of fraction of day, from GPS time and location (given as latitude and longitude in degrees)
  from astropy.coordinates import EarthLocation
  import astropy.time as aspyt
  import astropy.units as u
  loc = EarthLocation(lat=lat*u.deg, lon=long*u.deg)
  t = aspyt.Time(t_GPS, format='gps', location=(loc))
  LMST = t.sidereal_time('mean').value
  return jnp.array(LMST/24.)

