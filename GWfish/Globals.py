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

dirName  = os.path.dirname( os.path.dirname(os.path.abspath(__file__)))

detPath=os.path.join(dirName, 'psds')



##############################################################################
# PHYSICAL CONSTANTS
##############################################################################



GMsun_over_c3 = 4.925491025543575903411922162094833998e-6 # seconds
GMsun_over_c2 = 1.476625061404649406193430731479084713e3 # meters
uGpc = 3.085677581491367278913937957796471611e25 # meters
GMsun_over_c2_Gpc = GMsun_over_c2/uGpc # Gpc

REarth = 6371.00 #km
        
clight = 2.99792458*(10**5) #km/s
clightGpc = clight/3.0856778570831e+22




##############################################################################
# POSITIONS OF EXIXTING DETECTORS
##############################################################################


detectors = { 'L1': { 'lat':30.563,
                     'long':-90.774,
                     'xax':198.,
                     'shape':'L',
                     
                    },
             
             'H1': { 'lat':46.455,
                     'long':-119.408,
                     'xax':126.,
                     'shape':'L',
                    },
             
             'Virgo': { 'lat':43.631,
                     'long':10.504,
                     'xax':71.,
                     'shape':'L',
                    }

    
            }


###############################################################################
# Values FOR O1, O2, O3

# LIGO H: latLH=46.455 -- longLH=-119.408 -- xLH=126.
# LIGO L: latLL=30.563 -- longLL=-90.774 -- xLL=198.
# Virgo: latVi=43.631 -- longVi= 10.504 -- xVi=71.


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