#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:58:25 2022

@author: Michi
"""

import numpy as np




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
        #try:
        #    evParams['cosiota']
        #except KeyError:
        #    try:
        #        evParams['cosiota'] = np.cos(evParams['iota'])
        #    except KeyError:
        #        raise ValueError('One among dL and logdL has to be provided.')
        
        
def GPSt_to_LMST(t_GPS, lat, long):
  #Returns the Local Mean Sideral Time in units of fraction of day, from GPS time and location (given as latitude and longitude in degrees)
  from astropy.coordinates import EarthLocation
  import astropy.time as aspyt
  import astropy.units as u
  loc = EarthLocation(lat=lat*u.deg, lon=long*u.deg)
  t = aspyt.Time(t_GPS, format='gps', location=(loc))
  LMST = t.sidereal_time('mean').value
  return np.array(LMST/24.)

