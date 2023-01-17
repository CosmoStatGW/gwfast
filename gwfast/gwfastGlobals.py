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
"""
Path to the ``gwfast`` directory.

:type: str
"""
detPath=os.path.join(dirName, 'psds')
"""
Path to the ``./psds`` directory, containing the provided detector PSDs.

:type: str
"""
WFfilesPath = os.path.join(dirName, 'WFfiles')
"""
Path to the ``./WFfiles`` directory, containing files needed for the waveform evaluation.

:type: str
"""
##############################################################################
# PHYSICAL CONSTANTS
##############################################################################
# See http://asa.hmnao.com/static/files/2021/Astronomical_Constants_2021.pdf

GMsun_over_c3 = 4.925491025543575903411922162094833998e-6 # seconds
"""
Geometrized solar mass :math:`G \, {\\rm M}_{\odot} / c^3`, in seconds (:math:`\\rm s`).

:type: float
"""
GMsun_over_c2 = 1.476625061404649406193430731479084713e3 # meters
"""
Geometrized solar mass :math:`G \, {\\rm M}_{\odot} / c^2`, in meters (:math:`\\rm m`).

:type: float
"""
uGpc = 3.085677581491367278913937957796471611e25 # meters
"""
Gigaparsec (:math:`\\rm Gpc`) in meters (:math:`\\rm m`).

:type: float
"""
uMsun = 1.988409902147041637325262574352366540e30 # kg
"""
Solar mass (:math:`{\\rm M}_{\odot}`) in kilograms (:math:`\\rm kg`).

:type: float
"""
uAU = 149597870.7 # km
"""
Astronomical unit (:math:`\\rm A.U.`) in kilometers (:math:`\\rm km`).

:type: float
"""
GMsun_over_c2_Gpc = GMsun_over_c2/uGpc # Gpc
"""
Geometrized solar mass :math:`G \, {\\rm M}_{\odot} / c^2`, in gigaparsec (:math:`\\rm Gpc`).

:type: float
"""

REarth = 6371.00 # km
"""
Average Earth radius, in kilometers (:math:`\\rm km`).

:type: float
"""
clight = 2.99792458e5 # km/s
"""
Speed of light in vacuum (:math:`c`), in kilometers per second (:math:`\\rm km / s`).

:type: float
"""
clightGpc = clight/3.0856778570831e+22
"""
Speed of light in vacuum (:math:`c`), in gigaparsecs per second (:math:`\\rm Gpc / s`).

:type: float
"""
gravConst = 6.67430e-11
"""
Gravitational constant (:math:`G`), in cubic meters per kilogram per square second (:math:`\\rm m^3\, kg^{-1}\, s^{-2}`).

:type: float
"""
# ISCO frequency coefficient for a Schwarzschild BH
f_isco=1./(np.sqrt(6.)*6.*2.*np.pi*GMsun_over_c3)
"""
ISCO frequency coefficient for a Schwarzschild BH, in :math:`\\rm Hz`.

:type: float
"""
# limit of the quasi-Keplerian approximation, as in arXiv:2108.05861 (see also arXiv:1605.00304), more conservative than the Schwarzschild ISCO
f_qK = 2585. # Hz
"""
Coefficient for the limit of the quasi-Keplerian approximation, in :math:`\\rm Hz`, as in `arXiv:2108.05861 <https://arxiv.org/abs/2108.05861>`_ (see also `arXiv:1605.00304 <https://arxiv.org/abs/1605.00304>`_). This is more conservative than two times the Schwarzschild ISCO.

:type: float
"""
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
"""
Pre-defined detector locations and orientations.

:type: dict(dict, dict, ...)
"""
