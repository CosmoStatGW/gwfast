##############################################################################
# DIRECTORIES
##############################################################################
import os

dirName  = os.path.dirname( os.path.dirname(os.path.abspath(__file__)))
"""
Path to the ``popfisher`` directory.

:type: str
"""
AuxFilesPath = os.path.join(dirName, 'population', 'AuxFiles')
"""
Path to the ``./AuxFiles`` directory, containing files needed for some evaluations.

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
