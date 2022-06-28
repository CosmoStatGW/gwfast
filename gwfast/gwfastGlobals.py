#
#    Copyright (c) 2022 Francesco Iacovelli <francesco.iacovelli@unige.ch>, Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

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
uMsun = 1.988409902147041637325262574352366540e30 # kg
GMsun_over_c2_Gpc = GMsun_over_c2/uGpc # Gpc

REarth = 6371.00 #km
        
clight = 2.99792458*(10**5) #km/s
clightGpc = clight/3.0856778570831e+22


f_isco=1./(np.sqrt(6.)*6.*2.*np.pi*GMsun_over_c3)

##############################################################################
# POSITIONS OF DETECTORS
# See https://iopscience.iop.org/article/10.3847/1538-4357/ac4164
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
             
             'Virgo': {'lat':43.631,
                     'long':10.504,
                     'xax':115.56756342034298,
                     'shape':'L',
                    },
             
             'KAGRA': {'lat':36.412,
                     'long':137.306,
                     'xax':15.396,
                     'shape':'L',
                    },
            'LIGOI': {'lat':19.613,
                     'long':77.031,
                     'xax':287.384,
                     'shape':'L',
                    },

            'ETS': { 'lat': 40.+31./60.,
                    'long': 9.+25./60.,
                    'xax':0. ,
                    'shape':'T',
                   },
            'ETMR': {'lat': 50.+43./60.+23./3600.,
                    'long': 5.+55./60.+14./3600.,
                    'xax':0. ,
                    'shape':'T',
                   },
                
            'CE1Id':{'lat':43.827,
                     'long':-112.825,
                     'xax':-45.,
                     'shape':'L',
                  },
            'CE2NM':{'lat':33.160,
                     'long':-106.480,
                     'xax':-105.,
                     'shape':'L',
                  },
            'CE2NSW':{'lat':-34.,
                     'long':145.,
                     'xax':0.,
                     'shape':'L',
                  },
    
            }

