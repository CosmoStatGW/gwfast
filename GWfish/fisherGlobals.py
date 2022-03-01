#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 17:12:07 2022

@author: Michi
"""

##############################################################################
# DIRECTORIES
##############################################################################


import os
import numpy as np

dirName  = os.path.dirname( os.path.dirname(os.path.abspath(__file__)))

detPath=os.path.join(dirName, 'psds')

WFfilesPath = os.path.join(dirName, 'WFfiles')

##############################################################################
# PHYSICAL CONSTANTS
##############################################################################
# See http://asa.hmnao.com/static/files/2021/Astronomical_Constants_2021.pdf

GMsun_over_c3 = 4.925491025543575903411922162094833998e-6 # seconds
GMsun_over_c2 = 1.476625061404649406193430731479084713e3 # meters
uGpc = 3.085677581491367278913937957796471611e25 # meters
GMsun_over_c2_Gpc = GMsun_over_c2/uGpc # Gpc

REarth = 6371.00 #km
        
clight = 2.99792458*(10**5) #km/s
clightGpc = clight/3.0856778570831e+22


f_isco=1/(np.sqrt(6)*6*2*np.pi*GMsun_over_c3)

##############################################################################
# POSITIONS OF EXIXTING DETECTORS
##############################################################################


detectors = { 'L1': { 'lat':30.563,
                     'long':-90.774,
                     'xax':242.71636956358617,
                     'shape':'L',
                     
                    },
             
             'H1': { 'lat':46.455,
                     'long':-119.408,
                     'xax':170.99924234706103,
                     'shape':'L',
                    },
             
             'Virgo': { 'lat':43.631,
                     'long':10.504,
                     'xax':115.56756342034298,
                     'shape':'L',
                    }

    
            }


###############################################################################
# Values FOR O1, O2, O3

# Note: xax is the angle between local east and the x arm bisector couterclockwise from east
# i.e. the gamma of P. Jaranowski, A. Krolak, B. F. Schutz, PRD 58, 063001, eq. (10)--(13)
# It is computed from https://www.ligo.org/scientists/GW100916/GW100916-geometry.html

# LIGO H: latLH=46.455 -- longLH=-119.408 -- xLH=170.99924234706103
# LIGO L: latLL=30.563 -- longLL=-90.774 -- xLL=242.71636956358617
# Virgo: latVi=43.631 -- longVi= 10.504 -- xVi=115.56756342034298


# O3 H1 psd : O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt
# O3 L1 psd : O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt
# O3 V1 psd : O3-V1_sensitivity_strain_asd.txt

# O2 H1 psd : 2017-06-10_DCH_C02_H1_O2_Sensitivity_strain_asd.txt
# O2 L1 psd : 2017-08-06_DCH_C02_L1_O2_Sensitivity_strain_asd.txt
# O2 V1 psd : Hrec_hoft_V1O2Repro2A_16384Hz.txt

# O2 duty cycles: H1 0.617, L1 0.606 , V1  0.85
# O3 duty cycles: H1 0.712, L1 0.758 , V1  0.763
# O3b duty cycles: H1 0.788 L1 0.786 V1 0.756


###############################################################################
# Sources

# O3a reprentative strain : https://dcc.ligo.org/LIGO-P2000251/public
# O2 reprentative strain :   all https://dcc.ligo.org/P1800374/public/
#                            H1 https://dcc.ligo.org/LIGO-G1801950/public
#                            L1 https://dcc.ligo.org/LIGO-G1801952/public 
#                            V1 https://dcc.ligo.org/P1800374/public/
# O3b strain: https://zenodo.org/record/5571767#.YYkzey2ZPOQ


# O3a duty cycles: https://www.gw-openscience.org/detector_status/O3a/
# O2: https://www.gw-openscience.org/summary_pages/detector_status/O2/
# O2 Virgo : https://www.virgo-gw.eu/O2.html




##########################################
# Locations used in gwbench, for comparison

def det_angles(loc):
    
    import numpy as np
    
    PI = np.pi
    # return alpha, beta, gamma in radians
    # alpha ... longitude
    # beta  ... pi/2 - latitude
    # gamma ... angle from 'Due East' to y-arm
    if loc == 'H':
        return -2.08406, PI/2.-0.810795, PI-5.65488
    elif loc == 'L':
        return -1.58431, PI/2.-0.533423, PI-4.40318
    elif loc in ('V','ET1','ET2','ET3'):
        return 0.183338, PI/2.-0.761512, PI-0.33916
    elif loc == 'K':
        return 2.3942, PI/2.-0.632682, PI-1.054113
    elif loc == 'I':
        return 1.334013, PI/2.-0.248418, PI-1.570796

    elif loc == 'C':
        return -1.969174, PI/2.-0.764918, 0.
    elif loc == 'N':
        return -1.8584265, PI/2.-0.578751, -PI/3.
    elif loc == 'S':
        return 2.530727, PI/2.+0.593412, PI/4.


def get_filename(tec):
    if tec == 'A+':
        filename = 'a_plus.txt'
        asd = 1
    elif tec == 'V+':
        filename = 'advirgo_plus.txt'
        asd = 1
    elif tec == 'K+':
        filename = 'kagra_plus.txt'
        asd = 1
    elif tec == 'Voyager-CBO':
        filename = 'voyager_cb.txt'
        asd = 1
    elif tec == 'Voyager-PMO':
        filename = 'voyager_pm.txt'
        asd = 1
    elif tec == 'ET':
        filename = 'et.txt'
        asd = 1
    elif tec == 'CE1-10-CBO':
        filename = 'ce1_10km_cb.txt'
        asd = 1
    elif tec == 'CE1-20-CBO':
        filename = 'ce1_20km_cb.txt'
        asd = 1
    elif tec == 'CE1-30-CBO':
        filename = 'ce1_30km_cb.txt'
        asd = 1
    elif tec == 'CE1-40-CBO':
        filename = 'ce1_40km_cb.txt'
        asd = 1
    elif tec == 'CE2-10-CBO':
        filename = 'ce2_10km_cb.txt'
        asd = 1
    elif tec == 'CE2-20-CBO':
        filename = 'ce2_20km_cb.txt'
        asd = 1
    elif tec == 'CE2-30-CBO':
        filename = 'ce2_30km_cb.txt'
        asd = 1
    elif tec == 'CE2-40-CBO':
        filename = 'ce2_40km_cb.txt'
        asd = 1
    elif tec == 'CE1-10-PMO':
        filename = 'ce1_10km_pm.txt'
        asd = 1
    elif tec == 'CE1-20-PMO':
        filename = 'ce1_20km_pm.txt'
        asd = 1
    elif tec == 'CE1-30-PMO':
        filename = 'ce1_30km_pm.txt'
        asd = 1
    elif tec == 'CE1-40-PMO':
        filename = 'ce1_40km_pm.txt'
        asd = 1
    elif tec == 'CE2-10-PMO':
        filename = 'ce2_10km_pm.txt'
        asd = 1
    elif tec == 'CE2-20-PMO':
        filename = 'ce2_20km_pm.txt'
        asd = 1
    elif tec == 'CE2-30-PMO':
        filename = 'ce2_30km_pm.txt'
        asd = 1
    elif tec == 'CE2-40-PMO':
        filename = 'ce2_40km_pm.txt'
        asd = 1
    else: raise ValueError#(f'Specified PSD "{tec}" not known, choose from {tecs}.')

    return filename, asd

