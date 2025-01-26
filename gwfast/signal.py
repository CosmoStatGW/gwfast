#
#    Copyright (c) 2022 Francesco Iacovelli <francesco.iacovelli@unige.ch>, Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import os
import jax

#Enable 64bit on JAX, fundamental
#from jax.config import config
#config.update("jax_enable_x64", True)
jax.config.update("jax_enable_x64", True)
#config.update("TF_CPP_MIN_LOG_LEVEL", 0)

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'


# We use both the original numpy, denoted as onp, and the JAX implementation of numpy, denoted as np
import numpy as onp
import jax.numpy as np
from jax import pmap, vmap, jacrev, jit
import time
import h5py
import numdifftools as ndt
import copy
from numdifftools.step_generators import MaxStepGenerator

from gwfast import gwfastUtils as utils
from gwfast import gwfastGlobals as glob


class GWSignal(object):
    """
    Class to compute the GW signal emitted by a coalescing binary system as seen by a detector on Earth.

    The functions defined within this class allow to get e.g. the amplitude of the signal, its phase, SNR and Fisher matrix elements.

    :param WaveFormModel wf_model: Object containing the waveform model.
    :param str psd_path: Full path to the file containing the detector's *Power Spectral Density*, PSD, or *Amplitude Spectral Density*, ASD, including the file extension. The file is assumed to have two columns, the first containing the frequencies (in :math:`\\rm Hz`) and the second containing the detector's PSD/ASD at each frequency.
    :param str detector_shape: The shape of the detector, to be chosen among ``'L'`` for an L-shaped detector (90°-arms) and ``'T'`` for a triangular detector (3 nested detectors with 60°-arms).
    :param float det_lat: Latitude of the detector, in degrees.
    :param float det_long: Longitude of the detector, in degrees.
    :param float det_xax: Angle between the bisector of the detector's arms (the first detector in the case of a triangle) and local East, in degrees.
    :param bool, optional verbose: Boolean specifying if the code has to print additional details during execution.
    :param bool, optional is_ASD: Boolean specifying if the provided file is a PSD or an ASD.
    :param bool, optional useEarthMotion: Boolean specifying if the effect of the Earth rotation has to be included in the analysis.
    :param bool, optional noMotion: Boolean specifying if the Earth should be considered fixed at ``tcoal=0``. In the case ``useEarthMotion=False`` the system is rotated depending on ``tcoal`` and then left fixed. This was needed for checks and is not to be used.
    :param float fmin: Minimum frequency to use for the grid in the analysis, in :math:`\\rm Hz`.
    :param float fmax: Maximum frequency to use for the grid in the analysis, in :math:`\\rm Hz`. The cut frequency of the waveform (which depends on the events parameters) will be used as maximum frequency if ``fmax=None`` or if it is smaller than ``fmax``.
    :param str IntTablePath: Deprecated, not used.
    :param float DutyFactor: Duty factor of the detector, between 0 and 1, representing the percentage of time the detector (each detector independently in the case of a triangular detector) is supposed to be operational.
    :param bool, optional compute2arms: Boolean specifying if, in the case of a triangular detector, the computation can be performed only in two of the instruments, using the null-stream to get the signal in the third instrument, speeding up the computation by 1/3.
    :param bool, optional jitCompileDerivs: Boolean specifying if the derivatives function has to be jit compiled.

    """
    '''
    Inputs are an object containing the waveform model, the coordinates of the detector (latitude and longitude in deg),
    its shape (L or T), the angle with respect to East of the bisector of the arms (deg)
    and its ASD or PSD (given in a .txt file containing two columns: one with the frequencies and one with the ASD or PSD values,
    remember ASD=sqrt(PSD))

    '''
    def __init__(self, wf_model,
                psd_path=None,
                detector_shape = 'T',
                det_lat=40.44,
                det_long=9.45,
                det_xax=0.,
                verbose=True,
                is_ASD=True,
                useEarthMotion = False,
                noMotion = False, # use only for checks
                fmin=2., fmax=None,
                IntTablePath=None,
                DutyFactor=None,
                compute2arms=True,
                jitCompileDerivs=False):
        """
        Constructor method
        """
        if (detector_shape!='L') and (detector_shape!='T'):
            raise ValueError('Enter valid detector configuration')

        if psd_path is None:
            raise ValueError('Enter a valid PSD or ASD path')

        if verbose:
            if not is_ASD:
                print('Using PSD from file %s ' %psd_path)
            else:
                print('Using ASD from file %s ' %psd_path)

        if (useEarthMotion) and (wf_model.objType == 'BBH') and (verbose):
            print('WARNING: the motion of Earth gives a negligible contribution for BBH signals, consider switching it off to make the code run faster')
        if (not useEarthMotion) and (wf_model.objType == 'BNS') and (verbose):
            print('WARNING: the motion of Earth gives a relevant contribution for BNS signals, consider switching it on')
        if (not useEarthMotion) and (wf_model.objType == 'NSBH') and (verbose):
            print('WARNING: the motion of Earth gives a relevant contribution for NSBH signals, consider switching it on')

        self.wf_model = wf_model

        self.psd_base_path = ('/').join(psd_path.split('/')[:-1])
        self.psd_file_name = psd_path.split('/')[-1]

        self.verbose = verbose
        self.detector_shape = detector_shape

        self.det_lat_rad  = det_lat*np.pi/180.
        self.det_long_rad = det_long*np.pi/180.

        self.det_xax_rad  = det_xax*np.pi/180.

        self.IntTablePath = IntTablePath
        #This is the percentage of time each arm of the detector (or the whole detector for an L) is supposed to be operational, between 0 and 1, default is None, resulting in a detector always online
        self.DutyFactor = DutyFactor

        noise = onp.loadtxt(psd_path, usecols=(0,1))
        f = noise[:,0]
        if is_ASD:
            S = (noise[:,1])**2
        else:
            S = noise[:,1]

        self.strainFreq = f
        self.noiseCurve = S

        import scipy.integrate as igt
        mask = self.strainFreq >= fmin
        self.strainInteg = igt.cumtrapz(self.strainFreq[mask]**(-7./3.)/S[mask], self.strainFreq[mask], initial = 0)

        self.useEarthMotion = useEarthMotion
        self.noMotion = noMotion
        if self.noMotion and self.useEarthMotion:
            print('noMotion and useEarthMotion are True. switching off useEarthMotion ')
            self.useEarthMotion = False
        self.fmin = fmin #Hz
        self.fmax = fmax #Hz or None

        if detector_shape == 'L':
            self.angbtwArms = 0.5*np.pi
        elif detector_shape == 'T':
            self.angbtwArms = np.pi/3.

        self.IntegInterpArr = None
        self.compute2arms = compute2arms

        onp.random.seed(None)
        self.seedUse = onp.random.randint(2**32 - 1, size=1)
        self.jitCompileDerivs = jitCompileDerivs

        if not self.wf_model.is_LAL:
            self._init_jax()
        else:
            self._SignalDerivatives_use = self._SignalDerivatives


    def _init_jax(self):
        """
        JAX initialisation method
        """
        if self.verbose:
            print('Initializing jax...')
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
        os.environ["TF_CPP_MIN_LOG_LEVEL"]='0'
        #os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count=8'
        if self.verbose:
            print('Jax local device count: %s' %str(jax.local_device_count()))
            print('Jax  device count: %s' %str(jax.device_count()))

        if self.jitCompileDerivs:
            self._SignalDerivatives_use = jit(self._SignalDerivatives, static_argnames=['use_chi1chi2', 'use_m1m2', 'computeAnalyticalDeriv', 'use_prec_ang', 'computeDerivFinDiff', 'stepNDT', 'methodNDT'])
        else:
            self._SignalDerivatives_use = self._SignalDerivatives

        inj_params_init = {'Mc': np.array([77.23905294]),
                           'Phicoal': np.array([3.28297867]),
                           'chi1z': np.array([0.2018924]),
                           'chi2z': np.array([-0.68859213]),
                           'chis': np.array([0.2018924]),
                           'chia': np.array([-0.68859213]),
                           'dL': np.array([22.68426174]),
                           'eta': np.array([0.20586622]),
                           'iota': np.array([4.48411048]),
                           'phi': np.array([0.90252645]),
                           'psi': np.array([3.11843169]),
                           'Lambda1':np.array([300.]),
                           'Lambda2':np.array([300.]),
                           #'snr': np.array([21.20295982]),
                           #'tGPS': np.array([1.78168705e+09]),
                           'tcoal': np.array([0.]),
                           'theta': np.array([3.00702251]),
                           'chi1x': np.array([0.1]),
                           'chi2x': np.array([0.05]),
                           'chi1y': np.array([0.1]),
                           'chi2y': np.array([-0.01]),
                           'ecc': np.array([0.]),
                          }
        verboseOr = self.verbose
        self.verbose = False
        detector_shapeOr = self.detector_shape
        self.detector_shape = 'L' # Get a faster Initialization with an L
        _ = self.SNRInteg(inj_params_init, res=10)
        _ = self.FisherMatr(inj_params_init, res=10)

        if self.verbose:
            print('Done.')
        self.verbose = verboseOr
        self.detector_shape = detector_shapeOr

    def _clear_cache(self):
        if self.jitCompileDerivs:
            print('Clearing cache...')
            self._SignalDerivatives_use = jax.jit(self._SignalDerivatives, static_argnums=(15,16,17,18,19))

    def _update_seed(self, seed=None):
        """
        Update the seed for the duty cycle with a random value or a user input value.

        :param int, optional seed: User input value for the seed.

        """
        onp.random.seed(None)
        if seed is None:
            self.seedUse = onp.random.randint(2**32 - 1, size=1)
        else:
            self.seedUse = seed

    def _tabulateIntegrals(self, res=200, store=True, Mcmin=.9, Mcmax=9., etamin=.1):
        """
        Compute the table of integrals to use :py:class:`GWSignal.SNRFastInsp`.

        .. deprecated:: 1.0.0

        """
        def IntegrandC(f, Mc, tcoal, n):
            t = tcoal - 2.18567 * ((1.21/Mc)**(5./3.)) * ((100/f[:,onp.newaxis])**(8./3.))/(3600.*24)
            return (f[:,onp.newaxis]**(-7./3.))*np.cos(n*2.*np.pi*t)
        def IntegrandS(f, Mc, tcoal, n):
            t = tcoal - 2.18567 * ((1.21/Mc)**(5./3.)) * ((100/f[:,onp.newaxis])**(8./3.))/(3600.*24)
            return (f[:,onp.newaxis]**(-7./3.))*np.sin(n*2.*np.pi*t)

        Mcgrid = onp.linspace(Mcmin, Mcmax, res)
        etagrid = onp.linspace(etamin, 0.25, res)
        tcgrid = onp.linspace(0.,2.*np.pi,res)

        Igrid = onp.zeros((res,res,res,9))

        if self.verbose:
            print('Computing table of integrals...\n')

        in_time=time.time()

        for i,Mc in enumerate(Mcgrid):
            for j,eta in enumerate(etagrid):
                tmpev = {'Mc':np.array([Mc, ]), 'eta':np.array([eta])}
                fcut = self.wf_model.fcut(**tmpev)
                mask = (self.strainFreq >= self.fmin) & (self.strainFreq <= fcut)
                #for k,tc in enumerate(tcgrid):
                fgrids = np.ones((res, len(self.strainFreq[mask])))*self.strainFreq[mask]
                noisegrids = np.ones((res, len(self.noiseCurve[mask])))*self.noiseCurve[mask]
                for m in range(4):
                    tmpIntegrandC = IntegrandC(self.strainFreq[mask], Mc, tcgrid, m+1.)
                    tmpIntegrandS = IntegrandS(self.strainFreq[mask], Mc, tcgrid, m+1.)
                    Igrid[i,j,:,m] = onp.trapz(tmpIntegrandC/noisegrids.T, fgrids.T, axis=0)
                    Igrid[i,j,:,m+4] = onp.trapz(tmpIntegrandS/noisegrids.T, fgrids.T, axis=0)

                tmpIntegrand = IntegrandC(self.strainFreq[mask], Mc, tcgrid, 0.)
                Igrid[i,j,:,8] = onp.trapz(tmpIntegrand/noisegrids.T, fgrids.T, axis=0)

        if self.verbose:
            print('Done in %.2fs \n' %(time.time() - in_time))

        if store:
            print('Saving result...')
            if not os.path.isdir(os.path.join(self.psd_base_path, 'Integral_Tables')):
                os.mkdir(os.path.join(self.psd_base_path, 'Integral_Tables'))

            with h5py.File(os.path.join(self.psd_base_path, 'Integral_Tables', type(self.wf_model).__name__+str(res)+'.h5'), 'w') as out:
                out.create_dataset('Mc', data=Mcgrid, compression='gzip', shuffle=True)
                out.create_dataset('eta', data=etagrid, compression='gzip', shuffle=True)
                out.create_dataset('tc', data=tcgrid, compression='gzip', shuffle=True)
                out.create_dataset('Integs', data=Igrid, compression='gzip', shuffle=True)
                out.attrs['npoints'] = res
                out.attrs['etamin'] = etamin
                out.attrs['Mcmin'] = Mcmin
                out.attrs['Mcmax'] = Mcmax

        return Igrid, Mcgrid, etagrid, tcgrid

    def _make_SNRig_interpolator(self, ):
        """
        Make interpolator of the table of integrals to use :py:class:`GWSignal.SNRFastInsp`.

        .. deprecated:: 1.0.0

        """
        from scipy.interpolate import RegularGridInterpolator
        if self.IntTablePath is not None:
            if os.path.exists(self.IntTablePath):
                if self.verbose:
                    print('Pre-computed optimal integrals grid is present for this waveform. Loading...')
                with h5py.File(self.IntTablePath, 'r') as inp:
                    Mcs = np.array(inp['Mc'])
                    etas = np.array(inp['eta'])
                    tcs = np.array(inp['tc'])
                    Igrid = np.array(inp['Integs'])
                    #res = inp.attrs['npoints']
                    #etamin =  inp.attrs['etamin']
                    #Mcmin = inp.attrs['Mcmin']
                    #Mcmax = inp.attrs['Mcmax']
                    if self.verbose:
                        print('Attributes of pre-computed integrals: ')
                        print([(k, inp.attrs[k]) for k in inp.attrs.keys()])

            else:
                print('Tabulating integrals...')
                Igrid, Mcs, etas, tcs = self._tabulateIntegrals()

        else:
            print('Tabulating integrals...')
            Igrid, Mcs, etas, tcs = self._tabulateIntegrals()

        self.IntegInterpArr = onp.array([])
        for i in range(9):
            # The interpolator contains in the elements from i=0 to 3 the integrals of cos((i+1) Om t)f^{-7/3}
            # in the elements from 4 to 7 the integrals of the sine (from lower to higher i), and in element 8 the
            # integral of f^-7/3 alone

            self.IntegInterpArr =  onp.append(self.IntegInterpArr,RegularGridInterpolator((Mcs, etas, tcs), Igrid[:,:,:,i]))


    def _ra_dec_from_th_phi(self, theta, phi):
        return utils.ra_dec_from_th_phi_rad(theta, phi)


    def _PatternFunction(self, theta, phi, t, psi, rot=0.):
        """
        Compute the value of the so-called pattern functions of the detector for a set of sky coordinates, GW polarisation(s) and time(s).

        For the definition of the pattern functions see `arXiv:gr-qc/9804014 <https://arxiv.org/abs/gr-qc/9804014>`_ eq. (10)--(13).

        :param array or float theta: The :math:`\\theta` sky position angle(s), in :math:`\\rm rad`.
        :param array or float phi: The :math:`\phi` sky position angle(s), in :math:`\\rm rad`.
        :param array or float t: The time(s) given as GMST.
        :param array or float psi: The GW polarisation angle(s) :math:`\psi`, in :math:`\\rm rad`.
        :param float rot: Further rotation of the interferometer with respect to the :py:data:`self.xax` orientation, in degrees, needed for the triangular geometry. In this case, the three arms will have orientations 1 --> :py:data:`self.xax`, 2 --> :py:data:`self.xax` + 60°, 3 --> :py:data:`self.xax` + 120°.
        :return: Plus and cross pattern functions of the detector evaluated at the given parameters.
        :rtype: tuple(array, array) or tuple(float, float)

        """
        # See P. Jaranowski, A. Krolak, B. F. Schutz, PRD 58, 063001, eq. (10)--(13)


        def afun(ra, dec, t, rot):
            phir = self.det_long_rad
            a1 = 0.0625*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad))*(3.-np.cos(2.*dec))*np.cos(2.*(ra - phir - 2.*np.pi*t))
            a2 = 0.25*np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*(3.-np.cos(2.*dec))*np.sin(2.*(ra - phir - 2.*np.pi*t))
            a3 = 0.25*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*np.sin(2.*dec)*np.cos(ra - phir - 2.*np.pi*t)
            a4 = 0.5*np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*np.sin(2.*dec)*np.sin(ra - phir - 2.*np.pi*t)
            a5 = 3.*0.25*np.sin(2*(self.det_xax_rad+rot))*(np.cos(self.det_lat_rad)*np.cos(dec))**2.
            return a1 - a2 + a3 - a4 + a5

        def bfun(ra, dec, t, rot):
            phir = self.det_long_rad
            b1 = np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*np.sin(dec)*np.cos(2.*(ra - phir - 2.*np.pi*t))
            b2 = 0.25*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad))*np.sin(dec)*np.sin(2.*(ra - phir - 2.*np.pi*t))
            b3 = np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*np.cos(dec)*np.cos(ra - phir - 2.*np.pi*t)
            b4 = 0.5*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*np.cos(dec)*np.sin(ra - phir - 2.*np.pi*t)

            return b1 + b2 + b3 + b4

        rot_rad = rot*np.pi/180.

        ras, decs = self._ra_dec_from_th_phi(theta, phi)
        afac = afun(ras, decs, t, rot_rad)
        bfac = bfun(ras, decs, t, rot_rad)

        Fp = np.sin(self.angbtwArms)*(afac*np.cos(2.*psi) + bfac*np.sin(2*psi))
        Fc = np.sin(self.angbtwArms)*(bfac*np.cos(2.*psi) - afac*np.sin(2*psi))

        return Fp, Fc

    def _phiPhase(self, theta, phi, t, iota, psi, Fp=None,Fc=None):
        #The polarization phase contribution (the change in F+ and Fx with time influences also the phase)

        if (Fp is None) or (Fc is None):
            Fp, Fc = self._PatternFunction(theta, phi, t, psi)

        phiP = -np.arctan2(np.cos(iota)*Fc,0.5*(1.+((np.cos(iota))**2))*Fp)

        #The contriution to the amplitude is negligible, so we do not compute it

        return phiP

    def _DeltLoc(self, theta, phi, t):
        """
        Compute the time needed to go from Earth center to detector location for a set of sky coordinates and time(s). The result is given in seconds.

        :param array or float theta: The :math:`\\theta` sky position angle(s), in :math:`\\rm rad`.
        :param array or float phi: The :math:`\phi` sky position angle(s), in :math:`\\rm rad`.
        :param array or float t: The time(s) given as GMST.

        :return: Time shift(s) to go from Earth center to detector location.
        :rtype: array or float

        """
        # Time needed to go from Earth center to detector location

        ras, decs = self._ra_dec_from_th_phi(theta, phi)

        comp1 = np.cos(decs)*np.cos(ras)*np.cos(self.det_lat_rad)*np.cos(self.det_long_rad + 2.*np.pi*t)
        comp2 = np.cos(decs)*np.sin(ras)*np.cos(self.det_lat_rad)*np.sin(self.det_long_rad + 2.*np.pi*t)
        comp3 = np.sin(decs)*np.sin(self.det_lat_rad)
        # The minus sign arises from the definition of the unit vector pointing to the source
        Delt = - glob.REarth*(comp1+comp2+comp3)/glob.clight

        return Delt # in seconds

    def GWAmplitudes(self, evParams, f, rot=0.):
        """
        Compute the amplitude of the signal(s) as seen by the detector, as a function of the parameters, at given frequencies.

        :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
        :param array or float f: The frequency(ies) at which to perform the calculation, in :math:`\\rm Hz`.
        :param float rot: Further rotation of the interferometer with respect to the :py:data:`self.xax` orientation, in degrees, needed for the triangular geometry.
        :return: Plus and cross amplitudes at the detector, evaluated at the given parameters and frequency(ies).
        :rtype: tuple(array, array) or tuple(float, float)

        """
        # evParams are all the parameters characterizing the event(s) under exam. It has to be a dictionary containing the entries:
        # Mc -> chirp mass (Msun), dL -> luminosity distance (Gpc), theta & phi -> sky position (rad), iota -> inclination angle of orbital angular momentum to l.o.s toward the detector,
        # psi -> polarisation angle, tcoal -> time of coalescence as GMST (fraction of days), eta -> symmetric mass ratio, Phicoal -> GW frequency at coalescence.
        # chi1z, chi2z -> dimensionless spin components aligned to orbital angular momentum [-1;1], Lambda1,2 -> tidal parameters of the objects,
        # f is the frequency (Hz)

        theta, phi, iota, psi, tcoal = evParams['theta'], evParams['phi'], evParams['iota'], evParams['psi'], evParams['tcoal']

        if self.noMotion:
            t = 0.
            t = t + self._DeltLoc(theta, phi, t)/(3600.*24.)
        else:
            if self.useEarthMotion:
                t = tcoal - self.wf_model.tau_star(f, **evParams)/(3600.*24)
                t = t + self._DeltLoc(theta, phi, t)/(3600.*24.)
            else:
                t = tcoal #- self.wf_model.tau_star(self.fmin, **evParams)/(3600.*24)
                t = t + self._DeltLoc(theta, phi, t)/(3600.*24.)
        # wfAmpl = self.wf_model.Ampl(f, **evParams)
        Fp, Fc = self._PatternFunction(theta, phi, t, psi, rot=rot)

        if (self.wf_model.is_HigherModes) or (self.wf_model.is_Precessing):
        # If the waveform includes higher modes or precessing spins, it is not possible to compute amplitude and phase separately, make all together
            hp, hc = self.wf_model.hphc(f, **evParams)
            Ap, Ac = abs(hp)*Fp, abs(hc)*Fc
        else:
            wfAmpl = self.wf_model.Ampl(f, **evParams)
            Ap = wfAmpl*Fp*0.5*(1.+(np.cos(iota))**2)
            Ac = wfAmpl*Fc*np.cos(iota)

        return Ap, Ac

    def GWPhase(self, evParams, f):
        """
        Compute the complete phase of the signal(s), as a function of the parameters, at given frequencies.

        :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
        :param array or float f: The frequency(ies) at which to perform the calculation, in :math:`\\rm Hz`.

        :return: Complete signal phase, evaluated at the given parameters and frequency(ies).
        :rtype: array or float

        """
        # Phase of the GW signal
        tcoal, Phicoal =  evParams['tcoal'], evParams['Phicoal']
        PhiGw = self.wf_model.Phi(f, **evParams)

        return 2.*np.pi*f*(tcoal*3600.*24.) - Phicoal - PhiGw

    def GWstrain(self, f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc, rot=0., is_m1m2=False, is_chi1chi2=False, is_prec_ang=False, is_Lam1Lam2=False, return_single_comp=None):
        """
        Compute the full GW strain (complex) as a function of the parameters, at given frequencies.

        :param array or float f: The frequency(ies) at which to perform the calculation, in :math:`\\rm Hz`.
        :param array or float Mc: The chirp mass(es), :math:`{\cal M}_c`, in units of :math:`\\rm M_{\odot}`. If ``is_m1m2=True`` this is interpreted as the primary mass, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
        :param array or float eta:  The symmetric mass ratio(s), :math:`\eta`. If ``is_m1m2=True`` this is interpreted as the secondary mass, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.
        :param array or float dL: The luminosity distance(s), :math:`d_L`, in :math:`\\rm Gpc`.
        :param array or float theta: The :math:`\\theta` sky position angle(s), in :math:`\\rm rad`.
        :param array or float phi: The :math:`\phi` sky position angle(s), in :math:`\\rm rad`.
        :param array or float iota: The inclination angle(s), with respect to orbital angular momentum, :math:`\iota`, in :math:`\\rm rad`. If ``is_prec_ang=True`` this is interpreted as the inclination angle(s) with respect to total angular momentum, :math:`\\theta_{JN}`, in :math:`\\rm rad`.
        :param array or float psi: The polarisation angle(s), :math:`\psi`, in :math:`\\rm rad`.
        :param array or float tcoal: The time(s) of coalescence, :math:`t_{\\rm coal}`, as a GMST.
        :param array or float Phicoal: The phase(s) at coalescence, :math:`\Phi_{\\rm coal}`, in :math:`\\rm rad`.
        :param array or float chiS: The symmetric spin component(s), :math:`\chi_s`. If :py:class:`self.wf_model` is precessing or ``is_chi1chi2=True`` this is interpreted as the spin component(s) of the primary object(s) along the axis :math:`z`, :math:`\chi_{1,z}`. If ``is_prec_ang=True`` this is interpreted as the spin magnitude(s) of the primary object(s), :math:`\chi_1`.
        :param array or float chiA: The antisymmetric spin component(s) :math:`\chi_a`. If :py:class:`self.wf_model` is precessing or ``is_chi1chi2=True`` this is interpreted as the spin component(s) of the secondary object(s) along the axis :math:`z`, :math:`\chi_{2,z}`. If ``is_prec_ang=True`` this is interpreted as the spin magnitude(s) of the secondary object(s), :math:`\chi_2`.
        :param array or float chi1x: The spin component(s) of the primary object(s) along the axis :math:`x`, :math:`\chi_{1,x}`. If ``is_prec_ang=True`` this is interpreted as the spin tilt angle(s) of the primary object(s), :math:`\\theta_{s,1}`, in :math:`\\rm rad`.
        :param array or float chi2x: The spin component(s) of the secondary object(s) along the axis :math:`x`, :math:`\chi_{2,x}`. If ``is_prec_ang=True`` this is interpreted as the spin tilt angle(s) of the secondary object(s), :math:`\\theta_{s,2}`, in :math:`\\rm rad`.
        :param array or float chi1y: spin component(s) of the primary object(s) along the axis :math:`y`, :math:`\chi_{1,y}`. If ``is_prec_ang=True`` this is interpreted as the azimuthal angle(s) of orbital angular momentum relative to total angular momentum, :math:`\phi_{JL}`, in :math:`\\rm rad`.
        :param array or float chi2y: spin component(s) of the secondary object(s) along the axis :math:`y`, :math:`\chi_{2,y}`. If ``is_prec_ang=True`` this is interpreted as the difference(s) in azimuthal angle between spin vectors, :math:`\phi_{1,2}`, in :math:`\\rm rad`.
        :param array or float LambdaTilde: The adimensional tidal deformability(ies) of combination :math:`\\tilde{\Lambda}`.
        :param array or float deltaLambda: The adimensional tidal deformability(ies) of combination :math:`\delta\\tilde{\Lambda}`.
        :param array or float ecc: The orbital eccentricity(ies), :math:`e_0`.
        :param float rot: Further rotation of the interferometer with respect to the :py:data:`self.xax` orientation, in degrees, needed for the triangular geometry.
        :param bool, optional is_m1m2: Boolean specifying if the ``Mc`` and ``eta`` inputs should be interpreted as the primary and secondary mass(es).
        :param bool, optional is_chi1chi2: Boolean specifying if the ``chiS`` and ``chiA`` inputs should be interpreted as the primary and secondary spin components along the axis :math:`z`.
        :param bool, optional is_prec_ang: Boolean specifying if the ``iota`` input should be interpreted as the inclination angle with respect to total angular momentum, ``chiS`` and ``chiA`` as the primary and secondary spin magnitudes, ``chi1x`` and ``chi2x`` as the primary and secondary spin tilts, ``chi1y`` as the azimuthal angle of orbital angular momentum relative to total angular momentum and ``chi2y`` as the difference in azimuthal angle between spin vectors.
        :param bool, optional is_Lam1Lam2: Boolean specifying if the ``LambdaTilde`` and ``deltaLambda`` inputs should be interpreted as the individual tidal deformabilities.
        :param str return_single_comp: String specifying if a single component of the signal should be returned, to be chosen among ``Ap`` and ``Ac``, to return the plus and cross amplitude, :math:`A_+` and :math:`A_{\\times}`, respectively, and ``Psip`` and ``Psic``, to return the plus and cross phase, :math:`\Phi_+` and :math:`\Phi_{\\times}`, respectively.
        :return: Complete signal strain (complex), evaluated at the given parameters and frequency(ies).
        :rtype: array or float

        """
        # Full GW strain expression (complex)
        # Here we have the decompressed parameters and we put them back in a dictionary just to have an easier
        # implementation of the JAX module for derivatives

        if is_m1m2:
            # Interpret Mc as m1 and eta as m2
            McUse, etaUse = utils.Mceta_from_m1m2(Mc, eta)
        else:
            McUse  = Mc
            etaUse = eta

        if not self.wf_model.is_Precessing:
            if is_chi1chi2:
                # Interpret chiS as chi1z and chiA as chi2z
                chi1z = chiS
                chi2z = chiA
            else:
                chi1z = chiS + chiA
                chi2z = chiS - chiA
            chi1xUse, chi2xUse, chi1yUse, chi2yUse = McUse*0., McUse*0., McUse*0., McUse*0.
        else:
            if not is_prec_ang:
                chi1z = chiS
                chi2z = chiA
                chi1xUse = chi1x
                chi2xUse = chi2x
                chi1yUse = chi1y
                chi2yUse = chi2y
            else:
            # convert angles and iota
                iota, chi1xUse, chi1yUse, chi1z, chi2xUse, chi2yUse, chi2z = utils.TransformPrecessing_angles2comp(thetaJN=iota, phiJL=chi1y, theta1=chi1x, theta2=chi2x, phi12=chi2y, chi1=chiS, chi2=chiA, Mc=McUse, eta=etaUse, fRef=self.fmin, phiRef=0.)

        evParams = {'Mc':McUse, 'dL':dL, 'theta':theta, 'phi':phi, 'iota':iota, 'psi':psi, 'tcoal':tcoal, 'eta':etaUse, 'Phicoal':Phicoal, 'chi1z':chi1z, 'chi2z':chi2z, 'chi1x':chi1xUse, 'chi2x':chi2xUse, 'chi1y':chi1yUse, 'chi2y':chi2yUse}

        if self.wf_model.is_tidal:
            if not is_Lam1Lam2:
                Lambda1, Lambda2 = utils.Lam12_from_Lamt_delLam(LambdaTilde, deltaLambda, etaUse)
            else:
                Lambda1, Lambda2 = LambdaTilde, deltaLambda

            evParams['Lambda1'] = Lambda1
            evParams['Lambda2'] = Lambda2

        if self.wf_model.is_eccentric:
            evParams['ecc'] = ecc

        if self.useEarthMotion:
            # Compute Doppler contribution
            t = tcoal - self.wf_model.tau_star(f, **evParams)/(3600.*24.)
            tmpDeltLoc = self._DeltLoc(theta, phi, t) # in seconds
            t = t + tmpDeltLoc/(3600.*24.)
            phiD = Mc*0.
            #phiP is necessary if we write the signal as A*exp(i Psi) with A = sqrt(Ap^2 + Ac^2), uncomment if needed
            #phiP = self._phiPhase(theta, phi, t, iota, psi)
        else:
            phiD = Mc*0.
            #phiP = Mc*0.
            if self.noMotion:
                t = 0.
            else:
                t = tcoal
            tmpDeltLoc = self._DeltLoc(theta, phi, t) # in seconds
            t = t + tmpDeltLoc/(3600.*24.)

        phiL = (2.*np.pi*f)*tmpDeltLoc

        needs_HM = (self.wf_model.is_HigherModes) or (self.wf_model.is_Precessing)
        is_LAL = self.wf_model.is_LAL

        if not (needs_HM or is_LAL):
            # Return the simplest waveform construction
            Ap, Ac = self.GWAmplitudes(evParams, f, rot=rot)
            Psi = self.GWPhase(evParams, f)
            Psi = Psi + phiD + phiL

            if return_single_comp is not None:
                if (return_single_comp == 'Ap'):
                    return Ap
                elif (return_single_comp == 'Ac'):
                    return Ac
                elif (return_single_comp == 'Psip'):
                    return Psi  # np.unwrap(Psi)
                elif (return_single_comp == 'Psic'):
                    return Psi + np.pi*0.5  # np.unwrap(Psi + np.pi*0.5)
                elif (return_single_comp == 'At'):
                    return np.abs(Ap + 1j*Ac)
                elif (return_single_comp == 'Psit'):
                    return Psi + np.arctan2(np.real(Ac), np.real(Ap))
                else:
                    raise ValueError('Single component to return has to be among Ap, Ac, Psip, Psic')
            else:
                return (Ap + 1j*Ac)*np.exp(Psi*1j)
            #return np.sqrt(Ap*Ap + Ac*Ac)*np.exp((Psi+phiP)*1j)

        # If the waveform includes higher modes or precessing spins, it is not possible to compute amplitude and phase separately, make all together
        Fp, Fc = self._PatternFunction(theta, phi, t, psi, rot=rot)
        hp, hc = self.wf_model.hphc(f, **evParams)
        hp *= Fp*np.exp(1j*(phiD + phiL + 2.*np.pi*f*(tcoal*3600.*24.) - Phicoal))
        hc *= Fc*np.exp(1j*(phiD + phiL + 2.*np.pi*f*(tcoal*3600.*24.) - Phicoal))

        if is_LAL:
            hp *= 0.5*(1.+(np.cos(iota))**2)
            hc *= np.cos(iota)

        if return_single_comp is not None:
            if (return_single_comp == 'Ap'):
                return np.abs(hp)
            elif (return_single_comp == 'Ac'):
                return np.abs(hc)
            elif (return_single_comp == 'Psip'):
                return np.unwrap(np.angle(hp))
            elif (return_single_comp == 'Psic'):
                return np.unwrap(np.angle(hc))
            elif (return_single_comp == 'At'):
                return np.abs(hp + hc)
            elif (return_single_comp == 'Psit'):
                return np.unwrap(np.angle(hp + hc))
            else:
                raise ValueError('Single component to return has to be among Ap, Ac, Psip, Psic')
        else:
            return hp + hc

    def SNRInteg(self, evParams, res=1000, return_all=False):
        """
        Compute the *signal-to-noise-ratio*, SNR, as a function of the parameters of the event(s).

        :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
        :param int res: The resolution of the frequency grid to use.
        :param bool, optional return_all: Boolean specifying if, in the case of a triangular detector, the SNRs of the individual instruments have to be returned separately. In this case the return type is *list(array, array, array)*.

        :return: SNR(s) as a function of the parameters of the event(s). The shape is :math:`(N_{\\rm events})`.
        :rtype: 1-D array

        """
        # SNR calculation performing the frequency integral for each signal
        # This is computationally more expensive, but needed for complex waveform models
        if self.DutyFactor is not None:
            onp.random.seed(self.seedUse)

        utils.check_evparams(evParams)

        #if not np.isscalar(evParams['Mc']):
        #    SNR = np.zeros(len(np.asarray(evParams['Mc'])))
        #else:
        #    SNR = 0.

        allSNRsq=[]

        if self.wf_model.is_Precessing:
            try:
                _ =evParams['chi1x']
            except KeyError:
                try:
                    if self.verbose:
                        print('Adding cartesian components of the spins from angular variables')
                    evParams['iota'], evParams['chi1x'], evParams['chi1y'], evParams['chi1z'], evParams['chi2x'], evParams['chi2y'], evParams['chi2z'] = utils.TransformPrecessing_angles2comp(thetaJN=evParams['thetaJN'], phiJL=evParams['phiJL'], theta1=evParams['tilt1'], theta2=evParams['tilt2'], phi12=evParams['phi12'], chi1=evParams['chi1'], chi2=evParams['chi2'], Mc=evParams['Mc'], eta=evParams['eta'], fRef=self.fmin, phiRef=0.)
                except KeyError:
                    raise ValueError('Either the cartesian components of the precessing spins (iota, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z) or their modulus and orientations (thetaJN, chi1, chi2, tilt1, tilt2, phiJL, phi12) have to be provided.')
        else:
            try:
                _ =evParams['chi1z']
            except KeyError:
                try:
                    if self.verbose:
                        print('Adding chi1z, chi2z from chiS, chiA')
                    evParams['chi1z'] = evParams['chiS'] + evParams['chiA']
                    evParams['chi2z'] = evParams['chiS'] - evParams['chiA']
                except KeyError:
                    raise ValueError('Two among chi1z, chi2z and chiS, chiA have to be provided.')

        if self.wf_model.is_tidal:
            try:
                _=evParams['Lambda1']
            except KeyError:
                try:
                    evParams['Lambda1'], evParams['Lambda2'] = utils.Lam12_from_Lamt_delLam(evParams['LambdaTilde'], evParams['deltaLambda'], evParams['eta'])
                except KeyError:
                    raise ValueError('Two among Lambda1, Lambda2 and LambdaTilde and deltaLambda have to be provided.')

        fcut = self.wf_model.fcut(**evParams)

        if self.fmax is not None:
            fcut = np.where(fcut > self.fmax, self.fmax, fcut)

        fminarr = np.full(fcut.shape, self.fmin)
        fgrids = np.geomspace(fminarr,fcut,num=int(res))
        # Out of the provided PSD range, we use a constant value of 1, which results in completely negligible contributions
        strainGrids = np.interp(fgrids, self.strainFreq, self.noiseCurve, left=1., right=1.)

        if self.detector_shape=='L':
            Aps, Acs = self.GWAmplitudes(evParams, fgrids)
            Atot = Aps*Aps + Acs*Acs
            SNRsq = np.trapezoid(Atot/strainGrids, fgrids, axis=0)
            if self.DutyFactor is not None:
                excl = onp.random.choice([0,1],len(evParams['Mc']), p=[1.-self.DutyFactor,self.DutyFactor])
                SNRsq = SNRsq*excl
            allSNRsq.append(SNRsq)
        elif self.detector_shape=='T':
            if not self.compute2arms:
                for i in range(3):
                    Aps, Acs = self.GWAmplitudes(evParams, fgrids, rot=i*60.)
                    Atot = Aps*Aps + Acs*Acs
                    tmpSNRsq = np.trapezoid(Atot/strainGrids, fgrids, axis=0)
                    if self.DutyFactor is not None:
                        excl = onp.random.choice([0,1],len(evParams['Mc']), p=[1.-self.DutyFactor,self.DutyFactor])
                        tmpSNRsq = tmpSNRsq*excl
                    allSNRsq.append(tmpSNRsq)
                    #SNR = SNR + tmpSNRsq
                #SNR = np.sqrt(SNR)
            else:
            # The signal in 3 arms sums to zero for geometrical reasons, so we can use this to skip some calculations
                Aps1, Acs1 = self.GWAmplitudes(evParams, fgrids, rot=0.)
                Atot1 = Aps1*Aps1 + Acs1*Acs1
                Aps2, Acs2 = self.GWAmplitudes(evParams, fgrids, rot=60.)
                Atot2 = Aps2*Aps2 + Acs2*Acs2
                Aps3, Acs3 = - (Aps1 + Aps2), - (Acs1 + Acs2)
                Atot3 = Aps3*Aps3 + Acs3*Acs3
                tmpSNRsq1 = np.trapezoid(Atot1/strainGrids, fgrids, axis=0)
                tmpSNRsq2 = np.trapezoid(Atot2/strainGrids, fgrids, axis=0)
                tmpSNRsq3 = np.trapezoid(Atot3/strainGrids, fgrids, axis=0)
                if self.DutyFactor is not None:
                    excl = onp.random.choice([0,1],len(evParams['Mc']), p=[1.-self.DutyFactor,self.DutyFactor])
                    tmpSNRsq1 = tmpSNRsq1 * excl
                    excl = onp.random.choice([0,1],len(evParams['Mc']), p=[1.-self.DutyFactor,self.DutyFactor])
                    tmpSNRsq2 = tmpSNRsq2 * excl
                    excl = onp.random.choice([0,1],len(evParams['Mc']), p=[1.-self.DutyFactor,self.DutyFactor])
                    tmpSNRsq3 = tmpSNRsq3 * excl
                allSNRsq.append(tmpSNRsq1)
                allSNRsq.append(tmpSNRsq2)
                allSNRsq.append(tmpSNRsq3)
                #SNR = np.sqrt(tmpSNRsq1 + tmpSNRsq2 + tmpSNRsq3)
        allSNRsq = np.array(allSNRsq)

        if return_all:
            if self.detector_shape=='T':
                return 2*np.sqrt(allSNRsq)
            else:
                return np.squeeze(2*np.sqrt(allSNRsq), axis=0)
        elif self.detector_shape=='T':
            return 2*np.sqrt(allSNRsq.sum(axis=0))
        else:
            return np.squeeze(2*np.sqrt(allSNRsq), axis=0)

        # The factor of two arises by cutting the integral from 0 to infinity

    def FisherMatr(self, evParams, res=1000, df=None, spacing='geom',
                   use_m1m2=False, use_chi1chi2=True, use_prec_ang=True,
                   computeDerivFinDiff=False, computeAnalyticalDeriv=True,
                   return_all=False,
                   **kwargs):
        """
        Compute the *Fisher information matrix*, FIM, as a function of the parameters of the event(s).

        :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
        :param int res: The resolution of the frequency grid to use.
        :param float df: The spacing of the frequency grid to use, in :math:`\\rm Hz`. Alternative to ``res``.
        :param str spacing: The kind of spacing of the frequency grid to use. If ``'geom'`` the grid will be spaced evenly on a log scale (geometric progression), if ``'lin'`` it will be spaced evenly on a linear scale.
        :param bool, optional use_m1m2: Boolean specifying if the FIM has to be computed with respect to the individual masses ``m1`` and ``m2`` rather than ``Mc`` and ``eta``.
        :param bool, optional use_chi1chi2: Boolean specifying if, in the non-precessing case, the FIM has to be computed with respect to the individual spins ``chi1z`` and ``chi2z`` rather than ``chiS`` and ``chiA``.
        :param bool, optional use_prec_ang: Boolean specifying if, in the precessing case, the FIM has to be computed with respect to the spin angular variables rather than the spin cartesian components.
        :param bool, optional computeDerivFinDiff: Boolean specifying if the derivatives have to be computed using numerical differentiation (finite differences) through the `numdifftools <https://github.com/pbrod/numdifftools>`_ package.
        :param bool, optional computeAnalyticalDeriv: Boolean specifying if the derivatives with respect to ``dL``, ``theta``, ``phi``, ``psi``, ``tcoal``, ``Phicoal`` and ``iota`` (the latter only for the fundamental mode in the non-precessing case) have to be computed analytically. This considerably speeds up the calculation and provides better accuracy.
        :param bool, optional return_all: Boolean specifying if, in the case of a triangular detector, the FIMs of the individual instruments have to be returned separately. In this case the return type is *list(array, array, array)*.
        :param kwargs: Optional arguments to be passed to :py:class:`gwfast.signal.GWSignal._SignalDerivatives`, such as ``methodNDT``.
        :return: FIM(s) as a function of the parameters of the event(s). The shape is :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
        :rtype: 3-D array

        """
        # If use_m1m2=True the Fisher is computed w.r.t. m1 and m2, not Mc and eta
        # If use_chi1chi2=True the Fisher is computed w.r.t. chi1z and chi2z, not chiS and chiA
        if self.DutyFactor is not None:
            onp.random.seed(self.seedUse)

        utils.check_evparams(evParams)


        McOr, dL, theta, phi = evParams['Mc'].astype('complex128'), evParams['dL'].astype('complex128'), evParams['theta'].astype('complex128'), evParams['phi'].astype('complex128')
        iota, psi, tcoal, etaOr, Phicoal = evParams['iota'].astype('complex128'), evParams['psi'].astype('complex128'), evParams['tcoal'].astype('complex128'), evParams['eta'].astype('complex128'), evParams['Phicoal'].astype('complex128')

        if use_m1m2:
            # In this case Mc represents m1 and eta represents m2
            Mc, eta  = utils.m1m2_from_Mceta(McOr, etaOr)
        else:
            Mc, eta  = McOr, etaOr

        if not self.wf_model.is_Precessing:
            try:
                _ =evParams['chi1z']
            except KeyError:
                try:
                    if self.verbose:
                        print('Adding chi1z, chi2z from chiS, chiA')
                    evParams['chi1z'] = evParams['chiS'] + evParams['chiA']
                    evParams['chi2z'] = evParams['chiS'] - evParams['chiA']
                except KeyError:
                    raise ValueError('Two among chi1z, chi2z and chiS, chiA have to be provided.')

            chi1z, chi2z = evParams['chi1z'].astype('complex128'), evParams['chi2z'].astype('complex128')

            if use_chi1chi2:
            # In this case chiS represents chi1z and chiA represents chi2z
                chiS, chiA = chi1z, chi2z
            else:
                chiS, chiA = 0.5*(chi1z + chi2z), 0.5*(chi1z - chi2z)

            chi1x, chi2x, chi1y, chi2y = np.zeros(Mc.shape), np.zeros(Mc.shape), np.zeros(Mc.shape), np.zeros(Mc.shape)

        else:
            try:
               _=evParams['chi1x']
            except KeyError:
                try:
                    if self.verbose:
                        print('Adding cartesian components of the spins from angular variables')
                    evParams['iota'], evParams['chi1x'], evParams['chi1y'], evParams['chi1z'], evParams['chi2x'], evParams['chi2y'], evParams['chi2z'] = utils.TransformPrecessing_angles2comp(thetaJN=evParams['thetaJN'], phiJL=evParams['phiJL'], theta1=evParams['tilt1'], theta2=evParams['tilt2'], phi12=evParams['phi12'], chi1=evParams['chi1'], chi2=evParams['chi2'], Mc=evParams['Mc'], eta=evParams['eta'], fRef=self.fmin, phiRef=0.)
                except KeyError:
                    raise ValueError('Either the cartesian components of the precessing spins (iota, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z) or their modulus and orientations (thetaJN, chi1, chi2, tilt1, tilt2, phiJL, phi12) have to be provided.')
            if not use_prec_ang:
                chiS, chiA = evParams['chi1z'].astype('complex128'), evParams['chi2z'].astype('complex128')
                chi1x, chi2x, chi1y, chi2y = evParams['chi1x'].astype('complex128'), evParams['chi2x'].astype('complex128'), evParams['chi1y'].astype('complex128'), evParams['chi2y'].astype('complex128')
            else:
                # In this case iota=thetaJN, chi1y=phiJL, chi1x=tilt1, chi2x=tilt2, chi2y=phi12, chiS=chi1, chiA=chi2
                iota, chi1y, chi1x, chi2x, chi2y, chiS, chiA = utils.TransformPrecessing_comp2angles(evParams['iota'].astype('complex128'), evParams['chi1x'].astype('complex128'), evParams['chi1y'].astype('complex128'), evParams['chi1z'].astype('complex128'), evParams['chi2x'].astype('complex128'), evParams['chi2y'].astype('complex128'), evParams['chi2z'].astype('complex128'), McOr, etaOr, fRef=self.fmin, phiRef=0.)

        if self.wf_model.is_tidal:
            try:
                Lambda1, Lambda2 = evParams['Lambda1'].astype('complex128'), evParams['Lambda2'].astype('complex128')
            except KeyError:
                try:
                    Lambda1, Lambda2  = utils.Lam12_from_Lamt_delLam(evParams['LambdaTilde'].astype('complex128'), evParams['deltaLambda'].astype('complex128'), etaOr)
                except KeyError:
                    raise ValueError('Two among Lambda1, Lambda2 and LambdaTilde and deltaLambda have to be provided.')
            LambdaTilde, deltaLambda = utils.Lamt_delLam_from_Lam12(Lambda1, Lambda2, etaOr)

        else:
            Lambda1, Lambda2, LambdaTilde, deltaLambda = np.zeros(Mc.shape), np.zeros(Mc.shape), np.zeros(Mc.shape), np.zeros(Mc.shape)

        if self.wf_model.is_eccentric:
            try:
                ecc = evParams['ecc'].astype('complex128')
            except KeyError:
                raise ValueError('Eccentricity has to be provided.')
        else:
            ecc = np.zeros(Mc.shape)

        fcut = self.wf_model.fcut(**evParams)

        if self.fmax is not None:
            fcut = np.where(fcut > self.fmax, self.fmax, fcut)

        fminarr = np.full(fcut.shape, self.fmin)
        if res is None and df is not None:
            res = np.floor( np.real((1+(fcut-fminarr)/df)))
            res = np.amax(res)
        elif res is None and df is None:
            raise ValueError('Provide either resolution in frequency or step size.')
        if spacing=='lin':
            fgrids = np.linspace(fminarr, fcut, num=int(res))
        elif spacing=='geom':
            fgrids = np.geomspace(fminarr, fcut, num=int(res))

        # Out of the provided PSD range, we use a constant value of 1, which results in completely negligible conntributions
        strainGrids = np.interp(fgrids, self.strainFreq, self.noiseCurve, left=1., right=1.)

        nParams = self.wf_model.nParams
        tcelem = self.wf_model.ParNums['tcoal']

        if (self.wf_model.is_LAL) and (not computeDerivFinDiff):
            computeDerivFinDiff=True
            if self.verbose:
                print('Using LAL or TEOBResumS waveforms it is not possible to compute the derivatives using JAX automatic differentiation routines, being the functions written in C. Proceeding using numdifftools for numerical differentiation (finite differences)')

        allFishers=[]

        if self.detector_shape=='L':
            # Compute derivatives
            FisherDerivs = self._SignalDerivatives_use(fgrids, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc, rot=0., use_m1m2=use_m1m2, use_chi1chi2=use_chi1chi2, use_prec_ang=use_prec_ang, computeAnalyticalDeriv=computeAnalyticalDeriv, computeDerivFinDiff=computeDerivFinDiff, **kwargs)
            # Change the units of the tcoal derivative from days to seconds (this improves conditioning)
            FisherDerivs = onp.array(FisherDerivs)
            FisherDerivs[tcelem,:,:] /= (3600.*24.)

            FisherIntegrands = (onp.conjugate(FisherDerivs[:,:,onp.newaxis,:])*FisherDerivs.transpose(1,0,2))

            Fisher = onp.zeros((nParams,nParams,len(Mc)))
            # This for is unavoidable
            for alpha in range(nParams):
                for beta in range(alpha,nParams):
                    tmpElem = FisherIntegrands[alpha,:,beta,:].T
                    Fisher[alpha,beta, :] = onp.trapz(tmpElem.real/strainGrids.real, fgrids.real, axis=0)*4.

                    Fisher[beta,alpha, :] = Fisher[alpha,beta, :]
            if self.DutyFactor is not None:
                excl = onp.random.choice([0,1],len(evParams['Mc']), p=[1.-self.DutyFactor,self.DutyFactor])
                Fisher = Fisher*excl
            allFishers.append(Fisher)
        else:
            #Fisher = onp.zeros((nParams,nParams,len(Mc)))
            if not self.compute2arms:
                for i in range(3):
                    # Change rot and compute derivatives
                    FisherDerivs = self._SignalDerivatives_use(fgrids, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc, rot=i*60., use_m1m2=use_m1m2, use_chi1chi2=use_chi1chi2, use_prec_ang=use_prec_ang, computeAnalyticalDeriv=computeAnalyticalDeriv, computeDerivFinDiff=computeDerivFinDiff, **kwargs)
                    # Change the units of the tcoal derivative from days to seconds (this improves conditioning)
                    FisherDerivs = onp.array(FisherDerivs)
                    FisherDerivs[tcelem,:,:] /= (3600.*24.)
                    FisherIntegrands = (onp.conjugate(FisherDerivs[:,:,onp.newaxis,:])*FisherDerivs.transpose(1,0,2))

                    tmpFisher = onp.zeros((nParams,nParams,len(Mc)))
                    # This for is unavoidable
                    if self.verbose:
                        print('Filling matrix for arm %s...'%(i+1))

                    for alpha in range(nParams):
                        for beta in range(alpha,nParams):
                            tmpElem = FisherIntegrands[alpha,:,beta,:].T
                            tmpFisher[alpha,beta, :] = onp.trapz(tmpElem.real/strainGrids.real, fgrids.real, axis=0)*4.

                            tmpFisher[beta,alpha, :] = tmpFisher[alpha,beta, :]
                    if self.DutyFactor is not None:
                        excl = onp.random.choice([0,1],len(evParams['Mc']), p=[1.-self.DutyFactor,self.DutyFactor])
                        tmpFisher = tmpFisher*excl
                    allFishers.append(tmpFisher)
                    #Fisher += tmpFisher
            else:
            # The signal in 3 arms sums to zero for geometrical reasons, so we can use this to skip some calculations

                # Compute derivatives
                FisherDerivs1 = self._SignalDerivatives_use(fgrids, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc, rot=0., use_m1m2=use_m1m2, use_chi1chi2=use_chi1chi2, use_prec_ang=use_prec_ang, computeAnalyticalDeriv=computeAnalyticalDeriv, computeDerivFinDiff=computeDerivFinDiff, **kwargs)
                # Change the units of the tcoal derivative from days to seconds (this improves conditioning)
                FisherDerivs1 = onp.array(FisherDerivs1)
                FisherDerivs1[tcelem,:,:] /= (3600.*24.)

                FisherIntegrands = (onp.conjugate(FisherDerivs1[:,:,onp.newaxis,:])*FisherDerivs1.transpose(1,0,2))

                tmpFisher = onp.zeros((nParams,nParams,len(Mc)))
                if self.verbose:
                    print('Filling matrix for arm 1...')
                # This for is unavoidable
                for alpha in range(nParams):
                    for beta in range(alpha,nParams):
                        tmpElem = FisherIntegrands[alpha,:,beta,:].T
                        tmpFisher[alpha,beta, :] = onp.trapz(tmpElem.real/strainGrids.real, fgrids.real, axis=0)*4.

                        tmpFisher[beta,alpha, :] = tmpFisher[alpha,beta, :]
                if self.DutyFactor is not None:
                    excl = onp.random.choice([0,1],len(evParams['Mc']), p=[1.-self.DutyFactor,self.DutyFactor])
                    tmpFisher = tmpFisher*excl
                #Fisher += tmpFisher
                allFishers.append(tmpFisher)


                FisherDerivs2 = self._SignalDerivatives_use(fgrids, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc, rot=60., use_m1m2=use_m1m2, use_chi1chi2=use_chi1chi2, use_prec_ang=use_prec_ang, computeAnalyticalDeriv=computeAnalyticalDeriv, computeDerivFinDiff=computeDerivFinDiff, **kwargs)
                FisherDerivs2 = onp.array(FisherDerivs2)
                FisherDerivs2[tcelem,:,:] /= (3600.*24.)
                FisherIntegrands = (onp.conjugate(FisherDerivs2[:,:,onp.newaxis,:])*FisherDerivs2.transpose(1,0,2))

                tmpFisher = onp.zeros((nParams,nParams,len(Mc)))
                # This for is unavoidable
                if self.verbose:
                    print('Filling matrix for arm 2...')
                for alpha in range(nParams):
                    for beta in range(alpha,nParams):
                        tmpElem = FisherIntegrands[alpha,:,beta,:].T
                        tmpFisher[alpha,beta, :] = onp.trapz(tmpElem.real/strainGrids.real, fgrids.real, axis=0)*4.

                        tmpFisher[beta,alpha, :] = tmpFisher[alpha,beta, :]
                if self.DutyFactor is not None:
                    excl = onp.random.choice([0,1],len(evParams['Mc']), p=[1.-self.DutyFactor,self.DutyFactor])
                    tmpFisher = tmpFisher*excl
                #Fisher += tmpFisher
                allFishers.append(tmpFisher)

                FisherDerivs3 = - (FisherDerivs1 + FisherDerivs2)

                FisherIntegrands = (onp.conjugate(FisherDerivs3[:,:,onp.newaxis,:])*FisherDerivs3.transpose(1,0,2))

                tmpFisher = onp.zeros((nParams,nParams,len(Mc)))
                # This for is unavoidable
                if self.verbose:
                    print('Filling matrix for arm 3...')
                for alpha in range(nParams):
                    for beta in range(alpha,nParams):
                        tmpElem = FisherIntegrands[alpha,:,beta,:].T
                        tmpFisher[alpha,beta, :] = onp.trapz(tmpElem.real/strainGrids.real, fgrids.real, axis=0)*4.

                        tmpFisher[beta,alpha, :] = tmpFisher[alpha,beta, :]
                if self.DutyFactor is not None:
                    excl = onp.random.choice([0,1],len(evParams['Mc']), p=[1.-self.DutyFactor,self.DutyFactor])
                    tmpFisher = tmpFisher*excl
                #Fisher += tmpFisher
                allFishers.append(tmpFisher)

        if return_all:
            return allFishers
        elif self.detector_shape=='T':
            return onp.array(allFishers).sum(axis=0)
        else:
            return allFishers[0]

    def _SignalDerivatives(self, fgrids, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc, rot=0., use_m1m2=False, use_chi1chi2=True, use_prec_ang=True, computeDerivFinDiff=False, computeAnalyticalDeriv=True, stepNDT=MaxStepGenerator(base_step=1e-5), methodNDT='central', **kwargs):
        """
        Compute the derivatives of the GW strain with respect to the parameters of the event(s) at given frequencies (in :math:`\\rm Hz`).

        :param array or float fgrids: The frequency(ies) at which to perform the calculation, in :math:`\\rm Hz`.
        :param array or float Mc: The chirp mass(es), :math:`{\cal M}_c`, in units of :math:`\\rm M_{\odot}`. If ``use_m1m2=True`` this is interpreted as the primary mass, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
        :param array or float eta:  The symmetric mass ratio(s), :math:`\eta`. If ``use_m1m2=True`` this is interpreted as the secondary mass, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.
        :param array or float dL: The luminosity distance(s), :math:`d_L`, in :math:`\\rm Gpc`.
        :param array or float theta: The :math:`\\theta` sky position angle(s), in :math:`\\rm rad`.
        :param array or float phi: The :math:`\phi` sky position angle(s), in :math:`\\rm rad`.
        :param array or float iota: The inclination angle(s), with respect to orbital angular momentum, :math:`\iota`, in :math:`\\rm rad`. If ``is_prec_ang=True`` this is interpreted as the inclination angle(s) with respect to total angular momentum, :math:`\\theta_{JN}`, in :math:`\\rm rad`.
        :param array or float psi: The polarisation angle(s), :math:`\psi`, in :math:`\\rm rad`.
        :param array or float tcoal: The time(s) of coalescence, :math:`t_{\\rm coal}`, as a GMST.
        :param array or float Phicoal: The phase(s) at coalescence, :math:`\Phi_{\\rm coal}`, in :math:`\\rm rad`.
        :param array or float chiS: The symmetric spin component(s), :math:`\chi_s`. If :py:class:`self.wf_model` is precessing or ``use_chi1chi2=True`` this is interpreted as the spin component(s) of the primary object(s) along the axis :math:`z`, :math:`\chi_{1,z}`. If ``use_prec_ang=True`` this is interpreted as the spin magnitude(s) of the primary object(s), :math:`\chi_1`.
        :param array or float chiA: The antisymmetric spin component(s) :math:`\chi_a`. If :py:class:`self.wf_model` is precessing or ``use_chi1chi2=True`` this is interpreted as the spin component(s) of the secondary object(s) along the axis :math:`z`, :math:`\chi_{2,z}`. If ``use_prec_ang=True`` this is interpreted as the spin magnitude(s) of the secondary object(s), :math:`\chi_2`.
        :param array or float chi1x: The spin component(s) of the primary object(s) along the axis :math:`x`, :math:`\chi_{1,x}`. If ``use_prec_ang=True`` this is interpreted as the spin tilt angle(s) of the primary object(s), :math:`\\theta_{s,1}`, in :math:`\\rm rad`.
        :param array or float chi2x: The spin component(s) of the secondary object(s) along the axis :math:`x`, :math:`\chi_{2,x}`. If ``use_prec_ang=True`` this is interpreted as the spin tilt angle(s) of the secondary object(s), :math:`\\theta_{s,2}`, in :math:`\\rm rad`.
        :param array or float chi1y: spin component(s) of the primary object(s) along the axis :math:`y`, :math:`\chi_{1,y}`. If ``use_prec_ang=True`` this is interpreted as the azimuthal angle(s) of orbital angular momentum relative to total angular momentum, :math:`\phi_{JL}`, in :math:`\\rm rad`.
        :param array or float chi2y: spin component(s) of the secondary object(s) along the axis :math:`y`, :math:`\chi_{2,y}`. If ``use_prec_ang=True`` this is interpreted as the difference(s) in azimuthal angle between spin vectors, :math:`\phi_{1,2}`, in :math:`\\rm rad`.
        :param array or float LambdaTilde: The adimensional tidal deformability(ies) of combination :math:`\\tilde{\Lambda}`.
        :param array or float deltaLambda: The adimensional tidal deformability(ies) of combination :math:`\delta\\tilde{\Lambda}`.
        :param array or float ecc: The orbital eccentricity(ies), :math:`e_0`.
        :param float rot: Further rotation of the interferometer with respect to the :py:data:`self.xax` orientation, in degrees, needed for the triangular geometry.
        :param bool, optional use_m1m2: Boolean specifying if the ``Mc`` and ``eta`` inputs should be interpreted as the primary and secondary mass(es). In this case the derivatives are then taken with respect to ``m1`` and ``m2``.
        :param bool, optional use_chi1chi2: Boolean specifying if the ``chiS`` and ``chiA`` inputs should be interpreted as the primary and secondary spin components along the axis :math:`z`. In this case the derivatives are then taken with respect to ``chi1z`` and ``chi2z``.
        :param bool, optional use_prec_ang: Boolean specifying if the ``iota`` input should be interpreted as the inclination angle with respect to total angular momentum, ``chiS`` and ``chiA`` as the primary and secondary spin magnitudes, ``chi1x`` and ``chi2x`` as the primary and secondary spin tilts, ``chi1y`` as the azimuthal angle of orbital angular momentum relative to total angular momentum and ``chi2y`` as the difference in azimuthal angle between spin vectors. In this case the derivatives are then taken with respect to ``thetaJN``, ``chi1``, ``chi2``, ``tilt1``, ``tilt2``, ``phiJL`` and ``phi12``.
        :param bool, optional computeDerivFinDiff: Boolean specifying if the derivatives have to be computed using numerical differentiation (finite differences) through the `numdifftools <https://github.com/pbrod/numdifftools>`_ package.
        :param bool, optional computeAnalyticalDeriv: Boolean specifying if the derivatives with respect to ``dL``, ``theta``, ``phi``, ``psi``, ``tcoal``, ``Phicoal`` and ``iota`` (the latter only for the fundamental mode in the non-precessing case) have to be computed analytically. This considerably speeds up the calculation and provides better accuracy.
        :param stepNDT: The step size to use in the computation with numerical differentiation (finite differences).
        :type stepNDT: float or numdifftools.step_generators.MaxStepGenerator
        :param str methodNDT: The method to use in the computation with numerical differentiation (finite differences). This can be ``'central'``, ``'complex'``, ``'multicomplex'``, ``'forward'`` or ``'backward'``.
        :return: Complete signal strain derivatives (complex), evaluated at the given parameters and frequency(ies).
        :rtype: array

        """
        # `numdifftools.step_generators <https://numdifftools.readthedocs.io/en/latest/reference/numdifftools.html#module-numdifftools.step_generators>`_
        if self.verbose:
            print('Computing derivatives...')
        # Function to compute the derivatives of a GW signal, both with JAX (automatic differentiation) and NumDiffTools (finite differences). It offers the possibility to compute directly the derivative of the complex signal. It is also possible to compute analytically the derivatives w.r.t. dL, theta, phi, psi, tcoal and Phicoal, and also iota in absence of HM or precessing spins.

        if (self.wf_model.is_newtonian):
            if self.verbose:
                print('WARNING: In the Newtonian inspiral case the mass ratio and spins do not enter the waveform, and the corresponding Fisher matrix elements vanish, we then discard them.\n')

            if computeAnalyticalDeriv:
                derivargs = (1)
                inputNumdL, inputNumiota = 1, 2
            else:
                derivargs = (1,3,4,5,6,7,8,9)
        else:
            if computeAnalyticalDeriv:
                if (not self.wf_model.is_HigherModes) and (not self.wf_model.is_Precessing):
                    derivargs = (1,2,10,11,16,17)
                elif self.wf_model.is_Precessing:
                    derivargs = (1,2,6,10,11,12,13,14,15,16,17)
                elif (not self.wf_model.is_Precessing) and self.wf_model.is_HigherModes:
                    derivargs = (1,2,6,10,11,16,17)
                inputNumdL, inputNumiota = 2, 3
            else:
                if not self.wf_model.is_Precessing:
                    derivargs = (1,2,3,4,5,6,7,8,9,10,11,16,17)
                else:
                    derivargs = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17)
        if not self.wf_model.is_tidal:
            derivargs = derivargs[:-2]

        if self.wf_model.is_eccentric:
            derivargs = derivargs + (18,)

        nParams = self.wf_model.nParams

        if not computeDerivFinDiff:
            if self.wf_model.is_holomorphic:
                GWstrainUse = lambda f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc: self.GWstrain(f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, is_prec_ang=use_prec_ang)

                FisherDerivs = np.asarray(vmap(jacrev(GWstrainUse, argnums=derivargs, holomorphic=True))(fgrids.T, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc))
            else:
                # In the non holomorphic case, to improve the accuracy, we compute separately the derivatives of the real and imaginary part of the strain as real functions
                fgrids, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc = np.real(fgrids), np.real(Mc), np.real(eta), np.real(dL), np.real(theta), np.real(phi), np.real(iota), np.real(psi), np.real(tcoal), np.real(Phicoal), np.real(chiS), np.real(chiA), np.real(chi1x), np.real(chi2x), np.real(chi1y), np.real(chi2y), np.real(LambdaTilde), np.real(deltaLambda), np.real(ecc)

                GWstrainUse_real = lambda f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc: np.real(self.GWstrain(f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, is_prec_ang=use_prec_ang))
                GWstrainUse_imag = lambda f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc: np.imag(self.GWstrain(f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, is_prec_ang=use_prec_ang))

                realDerivs = np.asarray(vmap(jacrev(GWstrainUse_real, argnums=derivargs))(fgrids.T, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc))
                imagDerivs = np.asarray(vmap(jacrev(GWstrainUse_imag, argnums=derivargs))(fgrids.T, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc))

                FisherDerivs = realDerivs + 1j*imagDerivs
        else:
            if self.wf_model.is_newtonian:
                if computeAnalyticalDeriv:
                    GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                    evpars = [Mc]
                else:
                    GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], eta, pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                    evpars = [Mc, dL, theta, phi, iota, psi, tcoal, Phicoal]
            elif self.wf_model.is_tidal:
                if self.wf_model.is_Precessing:
                    if not self.wf_model.is_eccentric:
                        if not computeAnalyticalDeriv:
                            GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], pars[10], pars[11], pars[12], pars[13], pars[14], pars[15], pars[16], ecc, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, is_prec_ang=use_prec_ang)
                            evpars = [Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda]
                        else:
                            GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, pars[2], psi, tcoal, Phicoal, pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], pars[10], ecc, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, is_prec_ang=use_prec_ang)
                            evpars = [Mc, eta, iota, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda]
                    else:
                        if not computeAnalyticalDeriv:
                            GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], pars[10], pars[11], pars[12], pars[13], pars[14], pars[15], pars[16], pars[17], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, is_prec_ang=use_prec_ang)
                            evpars = [Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc]
                        else:
                            GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, pars[2], psi, tcoal, Phicoal, pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], pars[10], pars[11], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, is_prec_ang=use_prec_ang)
                            evpars = [Mc, eta, iota, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc]
                else:
                    if not self.wf_model.is_eccentric:
                        if not computeAnalyticalDeriv:
                            GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], pars[10], chi1x, chi2x, chi1y, chi2y, pars[11], pars[12], ecc, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                            evpars = [Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda]
                        else:
                            if not self.wf_model.is_HigherModes:
                                GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, iota, psi, tcoal, Phicoal, pars[2], pars[3], chi1x, chi2x, chi1y, chi2y, pars[4], pars[5], ecc, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                                evpars = [Mc, eta, chiS, chiA, LambdaTilde, deltaLambda]
                            else:
                                GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, pars[2], psi, tcoal, Phicoal, pars[3], pars[4], chi1x, chi2x, chi1y, chi2y, pars[5], pars[6], ecc, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                                evpars = [Mc, eta, iota, chiS, chiA, LambdaTilde, deltaLambda]
                    else:
                        if not computeAnalyticalDeriv:
                            GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], pars[10], chi1x, chi2x, chi1y, chi2y, pars[11], pars[12], pars[13], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                            evpars = [Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda, ecc]
                        else:
                            if not self.wf_model.is_HigherModes:
                                GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, iota, psi, tcoal, Phicoal, pars[2], pars[3], chi1x, chi2x, chi1y, chi2y, pars[4], pars[5], pars[6], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                                evpars = [Mc, eta, chiS, chiA, LambdaTilde, deltaLambda, ecc]
                            else:
                                GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, pars[2], psi, tcoal, Phicoal, pars[3], pars[4], chi1x, chi2x, chi1y, chi2y, pars[5], pars[6], pars[7], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                                evpars = [Mc, eta, iota, chiS, chiA, LambdaTilde, deltaLambda, ecc]
            else:
                if self.wf_model.is_Precessing:
                    if not self.wf_model.is_eccentric:
                        if not computeAnalyticalDeriv:
                            GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], pars[10], pars[11], pars[12], pars[13], pars[14], LambdaTilde, deltaLambda, ecc, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, is_prec_ang=use_prec_ang)
                            evpars = [Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y]
                        else:
                            GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, pars[2], psi, tcoal, Phicoal, pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], LambdaTilde, deltaLambda, ecc, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, is_prec_ang=use_prec_ang)
                            evpars = [Mc, eta, iota, chiS, chiA, chi1x, chi2x, chi1y, chi2y]
                    else:
                        if not computeAnalyticalDeriv:
                            GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], pars[10], pars[11], pars[12], pars[13], pars[14], LambdaTilde, deltaLambda, pars[15], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, is_prec_ang=use_prec_ang)
                            evpars = [Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y, ecc]
                        else:
                            GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, pars[2], psi, tcoal, Phicoal, pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], LambdaTilde, deltaLambda, pars[9], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, is_prec_ang=use_prec_ang)
                            evpars = [Mc, eta, iota, chiS, chiA, chi1x, chi2x, chi1y, chi2y, ecc]
                else:
                    if not self.wf_model.is_eccentric:
                        if not computeAnalyticalDeriv:
                            GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], pars[10], chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                            evpars = [Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA]
                        else:
                            if not self.wf_model.is_HigherModes:
                                GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, iota, psi, tcoal, Phicoal, pars[2], pars[3], chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                                evpars = [Mc, eta, chiS, chiA]
                            else:
                                GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, pars[2], psi, tcoal, Phicoal, pars[3], pars[4], chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                                evpars = [Mc, eta, iota, chiS, chiA]
                    else:
                        if not computeAnalyticalDeriv:
                            GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], pars[10], chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, pars[11], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                            evpars = [Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, ecc]
                        else:
                            if not self.wf_model.is_HigherModes:
                                GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, iota, psi, tcoal, Phicoal, pars[2], pars[3], chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, pars[4], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                                evpars = [Mc, eta, chiS, chiA, ecc]
                            else:
                                GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, pars[2], psi, tcoal, Phicoal, pars[3], pars[4], chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, pars[5], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                                evpars = [Mc, eta, iota, chiS, chiA, ecc]

            dh = ndt.Jacobian(GWstrainUse, step=stepNDT, method=methodNDT, order=2, n=1)
            FisherDerivs = np.asarray(dh(evpars))
            if len(FisherDerivs.shape) == 2: #len(Mc) == 1:
                FisherDerivs = FisherDerivs[:,:,np.newaxis]
            FisherDerivs = FisherDerivs.transpose(1,2,0)

        if computeAnalyticalDeriv:
            # We compute the derivative w.r.t. dL, theta, phi, iota, psi, tcoal and Phicoal analytically, so have to split the matrix and insert them
            if (not self.wf_model.is_HigherModes) and (not self.wf_model.is_Precessing):
                NAnalyticalDerivs = 7
            else:
                NAnalyticalDerivs = 6

            dL_deriv, theta_deriv, phi_deriv, iota_deriv, psi_deriv, tc_deriv, Phicoal_deriv = self._AnalyticalDerivatives(fgrids, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc, rot=rot, use_m1m2=use_m1m2, use_chi1chi2=use_chi1chi2, use_prec_ang=use_prec_ang)
            if (not self.wf_model.is_HigherModes) and (not self.wf_model.is_Precessing):
                if not self.wf_model.is_newtonian:
                    tmpsplit1, tmpsplit2, _ = onp.vsplit(FisherDerivs, onp.array([inputNumdL, nParams-NAnalyticalDerivs]))
                    FisherDerivs = np.vstack((tmpsplit1, np.asarray(dL_deriv).T[np.newaxis,:], np.asarray(theta_deriv).T[np.newaxis,:], np.asarray(phi_deriv).T[np.newaxis,:], np.asarray(iota_deriv).T[np.newaxis,:], np.asarray(psi_deriv).T[np.newaxis,:], np.asarray(tc_deriv).T[np.newaxis,:], np.asarray(Phicoal_deriv).T[np.newaxis,:], tmpsplit2))
                else:
                    FisherDerivs = np.vstack((FisherDerivs[np.newaxis,:], np.asarray(dL_deriv).T[np.newaxis,:], np.asarray(theta_deriv).T[np.newaxis,:], np.asarray(phi_deriv).T[np.newaxis,:], np.asarray(iota_deriv).T[np.newaxis,:], np.asarray(psi_deriv).T[np.newaxis,:], np.asarray(tc_deriv).T[np.newaxis,:], np.asarray(Phicoal_deriv).T[np.newaxis,:]))
            else:
                tmpsplit1, tmpsplit2, tmpsplit3, _ = onp.vsplit(FisherDerivs, onp.array([inputNumdL, inputNumiota, nParams-NAnalyticalDerivs]))
                FisherDerivs = np.vstack((tmpsplit1, np.asarray(dL_deriv).T[np.newaxis,:], np.asarray(theta_deriv).T[np.newaxis,:], np.asarray(phi_deriv).T[np.newaxis,:], tmpsplit2, np.asarray(psi_deriv).T[np.newaxis,:], np.asarray(tc_deriv).T[np.newaxis,:], np.asarray(Phicoal_deriv).T[np.newaxis,:], tmpsplit3))

        return FisherDerivs

    def _AnalyticalDerivatives(self, f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc, rot=0., use_m1m2=False, use_chi1chi2=False, use_prec_ang=False):
        """
        Compute analytical derivatives with respect to ``dL``, ``theta``, ``phi``, ``psi``, ``tcoal``, ``Phicoal`` and ``iota`` (the latter only for the fundamental mode in the non-precessing case).

        :param array or float f: The frequency(ies) at which to perform the calculation, in :math:`\\rm Hz`.
        :param array or float Mc: The chirp mass(es), :math:`{\cal M}_c`, in units of :math:`\\rm M_{\odot}`. If ``use_m1m2=True`` this is interpreted as the primary mass, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
        :param array or float eta:  The symmetric mass ratio(s), :math:`\eta`. If ``use_m1m2=True`` this is interpreted as the secondary mass, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.
        :param array or float dL: The luminosity distance(s), :math:`d_L`, in :math:`\\rm Gpc`.
        :param array or float theta: The :math:`\\theta` sky position angle(s), in :math:`\\rm rad`.
        :param array or float phi: The :math:`\phi` sky position angle(s), in :math:`\\rm rad`.
        :param array or float iota: The inclination angle(s), with respect to orbital angular momentum, :math:`\iota`, in :math:`\\rm rad`. If ``is_prec_ang=True`` this is interpreted as the inclination angle(s) with respect to total angular momentum, :math:`\\theta_{JN}`, in :math:`\\rm rad`.
        :param array or float psi: The polarisation angle(s), :math:`\psi`, in :math:`\\rm rad`.
        :param array or float tcoal: The time(s) of coalescence, :math:`t_{\\rm coal}`, as a GMST.
        :param array or float Phicoal: The phase(s) at coalescence, :math:`\Phi_{\\rm coal}`, in :math:`\\rm rad`.
        :param array or float chiS: The symmetric spin component(s), :math:`\chi_s`. If :py:class:`self.wf_model` is precessing or ``use_chi1chi2=True`` this is interpreted as the spin component(s) of the primary object(s) along the axis :math:`z`, :math:`\chi_{1,z}`. If ``use_prec_ang=True`` this is interpreted as the spin magnitude(s) of the primary object(s), :math:`\chi_1`.
        :param array or float chiA: The antisymmetric spin component(s) :math:`\chi_a`. If :py:class:`self.wf_model` is precessing or ``use_chi1chi2=True`` this is interpreted as the spin component(s) of the secondary object(s) along the axis :math:`z`, :math:`\chi_{2,z}`. If ``use_prec_ang=True`` this is interpreted as the spin magnitude(s) of the secondary object(s), :math:`\chi_2`.
        :param array or float chi1x: The spin component(s) of the primary object(s) along the axis :math:`x`, :math:`\chi_{1,x}`. If ``use_prec_ang=True`` this is interpreted as the spin tilt angle(s) of the primary object(s), :math:`\\theta_{s,1}`, in :math:`\\rm rad`.
        :param array or float chi2x: The spin component(s) of the secondary object(s) along the axis :math:`x`, :math:`\chi_{2,x}`. If ``use_prec_ang=True`` this is interpreted as the spin tilt angle(s) of the secondary object(s), :math:`\\theta_{s,2}`, in :math:`\\rm rad`.
        :param array or float chi1y: spin component(s) of the primary object(s) along the axis :math:`y`, :math:`\chi_{1,y}`. If ``use_prec_ang=True`` this is interpreted as the azimuthal angle(s) of orbital angular momentum relative to total angular momentum, :math:`\phi_{JL}`, in :math:`\\rm rad`.
        :param array or float chi2y: spin component(s) of the secondary object(s) along the axis :math:`y`, :math:`\chi_{2,y}`. If ``use_prec_ang=True`` this is interpreted as the difference(s) in azimuthal angle between spin vectors, :math:`\phi_{1,2}`, in :math:`\\rm rad`.
        :param array or float LambdaTilde: The adimensional tidal deformability(ies) of combination :math:`\\tilde{\Lambda}`.
        :param array or float deltaLambda: The adimensional tidal deformability(ies) of combination :math:`\delta\\tilde{\Lambda}`.
        :param array or float ecc: The orbital eccentricity(ies), :math:`e_0`.
        :param float rot: Further rotation of the interferometer with respect to the :py:data:`self.xax` orientation, in degrees, needed for the triangular geometry.
        :param bool, optional use_m1m2: Boolean specifying if the ``Mc`` and ``eta`` inputs should be interpreted as the primary and secondary mass(es).
        :param bool, optional use_chi1chi2: Boolean specifying if the ``chiS`` and ``chiA`` inputs should be interpreted as the primary and secondary spin components along the axis :math:`z`.
        :param bool, optional use_prec_ang: Boolean specifying if the ``iota`` input should be interpreted as the inclination angle with respect to total angular momentum, ``chiS`` and ``chiA`` as the primary and secondary spin magnitudes, ``chi1x`` and ``chi2x`` as the primary and secondary spin tilts, ``chi1y`` as the azimuthal angle of orbital angular momentum relative to total angular momentum and ``chi2y`` as the difference in azimuthal angle between spin vectors.
        :return: Analytical derivatives with respect to ``dL``, ``theta``, ``phi``, ``iota``, ``psi``, ``tcoal`` and ``Phicoal``. If the :py:class:`self.wf_model` is precessing or includes higher order modes the derivative with respect to ``iota`` will be ``None``
        :rtype: tuple(array, array, array, array, array, array, array)

        """
        # Module to compute analytically the derivatives w.r.t. dL, theta, phi, psi, tcoal, Phicoal and also iota in absence of HM or precessing spins. Each derivative is inserted into its own function with representative name, for ease of check.
        if use_m1m2:
            # Interpret Mc as m1 and eta as m2
            McUse, etaUse = utils.Mceta_from_m1m2(Mc, eta)
        else:
            McUse  = Mc
            etaUse = eta

        if not self.wf_model.is_Precessing:
            if use_chi1chi2:
                # Interpret chiS as chi1z and chiA as chi2z
                chi1z = chiS
                chi2z = chiA
            else:
                chi1z = chiS + chiA
                chi2z = chiS - chiA
            chi1xUse, chi2xUse, chi1yUse, chi2yUse = McUse*0., McUse*0., McUse*0., McUse*0.
        else:
            if not use_prec_ang:
                chi1z = chiS
                chi2z = chiA
                chi1xUse = chi1x
                chi2xUse = chi2x
                chi1yUse = chi1y
                chi2yUse = chi2y
            else:
            # convert angles and iota
                iota, chi1xUse, chi1yUse, chi1z, chi2xUse, chi2yUse, chi2z = utils.TransformPrecessing_angles2comp(thetaJN=iota, phiJL=chi1y, theta1=chi1x, theta2=chi2x, phi12=chi2y, chi1=chiS, chi2=chiA, Mc=McUse, eta=etaUse, fRef=self.fmin, phiRef=0.)

        evParams = {'Mc':McUse, 'dL':dL, 'theta':theta, 'phi':phi, 'iota':iota, 'psi':psi, 'tcoal':tcoal, 'eta':etaUse, 'Phicoal':Phicoal, 'chi1z':chi1z, 'chi2z':chi2z, 'chi1x':chi1xUse, 'chi2x':chi2xUse, 'chi1y':chi1yUse, 'chi2y':chi2yUse}

        if self.wf_model.is_tidal:
            Lambda1, Lambda2 = utils.Lam12_from_Lamt_delLam(LambdaTilde, deltaLambda, etaUse)

            evParams['Lambda1'] = Lambda1
            evParams['Lambda2'] = Lambda2

        if self.wf_model.is_eccentric:
            evParams['ecc'] = ecc

        if (not self.wf_model.is_HigherModes) and (not self.wf_model.is_Precessing):
            wfPhiGw = self.wf_model.Phi(f, **evParams)
            wfAmpl  = self.wf_model.Ampl(f, **evParams)
            wfhp, wfhc = wfAmpl*np.exp(-1j*wfPhiGw)*0.5*(1.+(np.cos(iota))**2), 1j*wfAmpl*np.exp(-1j*wfPhiGw)*np.cos(iota)
        else:
            # If the waveform includes higher modes, it is not possible to compute amplitude and phase separately, make all together
            wfhp, wfhc = self.wf_model.hphc(f, **evParams)

        if self.useEarthMotion:
            # Compute Doppler contribution
            tnoloc = tcoal - self.wf_model.tau_star(f, **evParams)/(3600.*24.)
            tmpDeltLoc = self._DeltLoc(theta, phi, tnoloc) # in seconds
            t = tnoloc + tmpDeltLoc/(3600.*24.)
            phiD = Mc*0.
            #phiP is necessary if we write the signal as A*exp(i Psi) with A = sqrt(Ap^2 + Ac^2), uncomment if necessary
            #phiP = self._phiPhase(theta, phi, t, iota, psi)
        else:
            phiD = Mc*0.
            #phiP = Mc*0.
            if self.noMotion:
                tnoloc=0
            else:
                tnoloc = tcoal
            tmpDeltLoc = self._DeltLoc(theta, phi, tnoloc) # in seconds
            t = tnoloc + tmpDeltLoc/(3600.*24.)

        phiL = (2.*np.pi*f)*tmpDeltLoc

        rot_rad = rot*np.pi/180.

        def afun(ra, dec, t, rot):
            phir = self.det_long_rad
            a1 = 0.0625*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad))*(3.-np.cos(2.*dec))*np.cos(2.*(ra - phir - 2.*np.pi*t))
            a2 = 0.25*np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*(3.-np.cos(2.*dec))*np.sin(2.*(ra - phir - 2.*np.pi*t))
            a3 = 0.25*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*np.sin(2.*dec)*np.cos(ra - phir - 2.*np.pi*t)
            a4 = 0.5*np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*np.sin(2.*dec)*np.sin(ra - phir - 2.*np.pi*t)
            a5 = 3.*0.25*np.sin(2*(self.det_xax_rad+rot))*(np.cos(self.det_lat_rad)*np.cos(dec))**2.
            return a1 - a2 + a3 - a4 + a5

        def bfun(ra, dec, t, rot):
            phir = self.det_long_rad
            b1 = np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*np.sin(dec)*np.cos(2.*(ra - phir - 2.*np.pi*t))
            b2 = 0.25*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad))*np.sin(dec)*np.sin(2.*(ra - phir - 2.*np.pi*t))
            b3 = np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*np.cos(dec)*np.cos(ra - phir - 2.*np.pi*t)
            b4 = 0.5*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*np.cos(dec)*np.sin(ra - phir - 2.*np.pi*t)

            return b1 + b2 + b3 + b4

        ras, decs = self._ra_dec_from_th_phi(theta, phi)

        afac = afun(ras, decs, t, rot_rad)
        bfac = bfun(ras, decs, t, rot_rad)

        Fp = np.sin(self.angbtwArms)*(afac*np.cos(2.*psi) + bfac*np.sin(2*psi))
        Fc = np.sin(self.angbtwArms)*(bfac*np.cos(2.*psi) - afac*np.sin(2*psi))

        hp, hc = wfhp*Fp*np.exp(1j*(2.*np.pi*f*(tcoal*3600.*24.) - Phicoal + phiD + phiL)), wfhc*Fc*np.exp(1j*(2.*np.pi*f*(tcoal*3600.*24.) - Phicoal + phiD + phiL))

        def psi_par_deriv():

            Fp_psider = 2*np.sin(self.angbtwArms)*(-afac*np.sin(2.*psi) + bfac*np.cos(2*psi))
            Fc_psider = 2*np.sin(self.angbtwArms)*(-bfac*np.sin(2.*psi) - afac*np.cos(2*psi))

            return wfhp*Fp_psider*np.exp(1j*(2.*np.pi*f*(tcoal*3600.*24.) - Phicoal + phiD + phiL)) + wfhc*Fc_psider*np.exp(1j*(2.*np.pi*f*(tcoal*3600.*24.) - Phicoal + phiD + phiL))

        def phi_par_deriv():

            def Delt_loc_phider(ra, dec, t):

                comp1 = -np.cos(dec)*np.sin(ra)*np.cos(self.det_lat_rad)*np.cos(self.det_long_rad + 2.*np.pi*t)
                comp2 = np.cos(dec)*np.cos(ra)*np.cos(self.det_lat_rad)*np.sin(self.det_long_rad + 2.*np.pi*t)

                Delt_phider = - glob.REarth*(comp1+comp2)/glob.clight

                return Delt_phider/(3600.*24.) # in days

            def afun_phider(ra, dec, t, rot):
                phir = self.det_long_rad
                a1 = 0.0625*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad))*(3.-np.cos(2.*dec))*(-2.)*np.sin(2.*(ra - phir - 2.*np.pi*t))
                a2 = 0.25*np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*(3.-np.cos(2.*dec))*2.*np.cos(2.*(ra - phir - 2.*np.pi*t))
                a3 = -0.25*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*np.sin(2.*dec)*np.sin(ra - phir - 2.*np.pi*t)
                a4 = 0.5*np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*np.sin(2.*dec)*np.cos(ra - phir - 2.*np.pi*t)

                return a1 - a2 + a3 - a4

            def bfun_phider(ra, dec, t, rot):
                phir = self.det_long_rad

                b1 = np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*np.sin(dec)*(-2.)*np.sin(2.*(ra - phir - 2.*np.pi*t))
                b2 = 0.25*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad))*np.sin(dec)*2.*np.cos(2.*(ra - phir - 2.*np.pi*t))
                b3 = -np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*np.cos(dec)*np.sin(ra - phir - 2.*np.pi*t)
                b4 = 0.5*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*np.cos(dec)*np.cos(ra - phir - 2.*np.pi*t)

                return b1 + b2 + b3 + b4
            locDt_phider = Delt_loc_phider(ras, decs, tnoloc)
            afac_phider = afun_phider(ras, decs, t, rot_rad)*(1.-2.*np.pi*locDt_phider)
            bfac_phider = bfun_phider(ras, decs, t, rot_rad)*(1.-2.*np.pi*locDt_phider)

            Fp_phider = np.sin(self.angbtwArms)*(afac_phider*np.cos(2.*psi) + bfac_phider*np.sin(2*psi))
            Fc_phider = np.sin(self.angbtwArms)*(bfac_phider*np.cos(2.*psi) - afac_phider*np.sin(2*psi))

            ampP_phider = wfhp*Fp_phider*np.exp(1j*(2.*np.pi*f*(tcoal*3600.*24.) - Phicoal + phiD + phiL))
            ampC_phider = wfhc*Fc_phider*np.exp(1j*(2.*np.pi*f*(tcoal*3600.*24.) - Phicoal + phiD + phiL))
            phiD_phideriv = 0.
            phiL_phideriv = 2.*np.pi*f*locDt_phider*(3600.*24.)

            return ampP_phider + 1j*(phiD_phideriv + phiL_phideriv)*hp + ampC_phider + 1j*(phiD_phideriv + phiL_phideriv)*hc

        def theta_par_deriv():
            def Delt_loc_thder(ra, dec, t):

                comp1 = np.sin(dec)*np.cos(ra)*np.cos(self.det_lat_rad)*np.cos(self.det_long_rad + 2.*np.pi*t)
                comp2 = np.sin(dec)*np.sin(ra)*np.cos(self.det_lat_rad)*np.sin(self.det_long_rad + 2.*np.pi*t)
                comp3 = -np.cos(dec)*np.sin(self.det_lat_rad)

                Delt_thder = - glob.REarth*(comp1+comp2+comp3)/glob.clight

                return Delt_thder/(3600.*24.) # in days

            def afun_thder(ra, dec, t, rot, loc_thder):
                phir = self.det_long_rad
                a1 = 0.0625*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad))*(-2.*np.sin(2.*dec))*np.cos(2.*(ra - phir - 2.*np.pi*t)) + 0.0625*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad)) *(3.-np.cos(2.*dec))*np.sin(2.*(ra - phir - 2.*np.pi*t))*(4.*np.pi*loc_thder)
                a2 = 0.25*np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*(-2.*np.sin(2.*dec))*np.sin(2.*(ra - phir - 2.*np.pi*t)) - 0.25*np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*(3.-np.cos(2.*dec))*np.cos(2.*(ra - phir - 2.*np.pi*t))*(4.*np.pi*loc_thder)
                a3 = -0.25*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*2.*np.cos(2.*dec)*np.cos(ra - phir - 2.*np.pi*t) + 0.25*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*np.sin(2.*dec)*np.sin(ra - phir - 2.*np.pi*t)*(2.*np.pi*loc_thder)
                a4 = -0.5*np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*2.*np.cos(2.*dec)*np.sin(ra - phir - 2.*np.pi*t) - 0.5*np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*np.sin(2.*dec)*np.cos(ra - phir - 2.*np.pi*t)*(2.*np.pi*loc_thder)
                a5 = 2.*3.*0.25*np.sin(2*(self.det_xax_rad+rot))*((np.cos(self.det_lat_rad))**2)*np.cos(dec)*np.sin(dec)
                return a1 - a2 + a3 - a4 + a5

            def bfun_thder(ra, dec, t, rot, loc_thder):
                phir = self.det_long_rad
                b1 = -np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*np.cos(dec)*np.cos(2.*(ra - phir - 2.*np.pi*t)) + np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*np.sin(dec)*np.sin(2.*(ra - phir - 2.*np.pi*t))*(4.*np.pi*loc_thder)
                b2 = -0.25*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad))*np.cos(dec)*np.sin(2.*(ra - phir - 2.*np.pi*t)) - 0.25*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad))*np.sin(dec)*np.cos(2.*(ra - phir - 2.*np.pi*t))*(4.*np.pi*loc_thder)
                b3 = np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*np.sin(dec)*np.cos(ra - phir - 2.*np.pi*t) + np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*np.cos(dec)*np.sin(ra - phir - 2.*np.pi*t)*(2.*np.pi*loc_thder)
                b4 = 0.5*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*np.sin(dec)*np.sin(ra - phir - 2.*np.pi*t) - 0.5*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*np.cos(dec)*np.cos(ra - phir - 2.*np.pi*t)*(2.*np.pi*loc_thder)

                return b1 + b2 + b3 + b4

            locDt_thder = Delt_loc_thder(ras, decs, tnoloc)
            afac_thder = afun_thder(ras, decs, t, rot_rad, locDt_thder)
            bfac_thder = bfun_thder(ras, decs, t, rot_rad, locDt_thder)

            Fp_thder = np.sin(self.angbtwArms)*(afac_thder*np.cos(2.*psi) + bfac_thder*np.sin(2*psi))
            Fc_thder = np.sin(self.angbtwArms)*(bfac_thder*np.cos(2.*psi) - afac_thder*np.sin(2*psi))

            ampP_thder = wfhp*Fp_thder*np.exp(1j*(2.*np.pi*f*(tcoal*3600.*24.) - Phicoal + phiD + phiL))
            ampC_thder = wfhc*Fc_thder*np.exp(1j*(2.*np.pi*f*(tcoal*3600.*24.) - Phicoal + phiD + phiL))
            phiD_thderiv = 0.
            phiL_thderiv = 2.*np.pi*f*locDt_thder*(3600.*24.)

            return ampP_thder + 1j*(phiD_thderiv + phiL_thderiv)*hp + ampC_thder + 1j*(phiD_thderiv + phiL_thderiv)*hc

        def tcoal_par_deriv():

            def Delt_loc_tcder(ra, dec, t):

                comp1 = -np.cos(decs)*np.cos(ras)*np.cos(self.det_lat_rad)*np.sin(self.det_long_rad + 2.*np.pi*t)
                comp2 = np.cos(decs)*np.sin(ras)*np.cos(self.det_lat_rad)*np.cos(self.det_long_rad + 2.*np.pi*t)

                Delt_tcder = - 2.*np.pi*glob.REarth*(comp1+comp2)/glob.clight

                return Delt_tcder/(3600.*24.) # in days

            def afun_tcder(ra, dec, t, rot):
                phir = self.det_long_rad
                a1 = 0.0625*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad))*(3.-np.cos(2.*dec))*(-2.)*np.sin(2.*(ra - phir - 2.*np.pi*t))
                a2 = 0.25*np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*(3.-np.cos(2.*dec))*2.*np.cos(2.*(ra - phir - 2.*np.pi*t))
                a3 = -0.25*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*np.sin(2.*dec)*np.sin(ra - phir - 2.*np.pi*t)
                a4 = 0.5*np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*np.sin(2.*dec)*np.cos(ra - phir - 2.*np.pi*t)

                return a1 - a2 + a3 - a4

            def bfun_tcder(ra, dec, t, rot):
                phir = self.det_long_rad
                b1 = np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*np.sin(dec)*(-2.)*np.sin(2.*(ra - phir - 2.*np.pi*t))
                b2 = 0.25*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad))*np.sin(dec)*2.*np.cos(2.*(ra - phir - 2.*np.pi*t))
                b3 = -np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*np.cos(dec)*np.sin(ra - phir - 2.*np.pi*t)
                b4 = 0.5*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*np.cos(dec)*np.cos(ra - phir - 2.*np.pi*t)

                return b1 + b2 + b3 + b4
            locDt_tcder = Delt_loc_tcder(ras, decs, tnoloc)
            afac_tcder = -2.*np.pi*afun_tcder(ras, decs, t, rot_rad)*(1.+locDt_tcder)
            bfac_tcder = -2.*np.pi*bfun_tcder(ras, decs, t, rot_rad)*(1.+locDt_tcder)

            Fp_tcder = np.sin(self.angbtwArms)*(afac_tcder*np.cos(2.*psi) + bfac_tcder*np.sin(2*psi))
            Fc_tcder = np.sin(self.angbtwArms)*(bfac_tcder*np.cos(2.*psi) - afac_tcder*np.sin(2*psi))

            ampP_tcder = wfhp*Fp_tcder*np.exp(1j*(2.*np.pi*f*(tcoal*3600.*24.) - Phicoal + phiD + phiL))
            ampC_tcder = wfhc*Fc_tcder*np.exp(1j*(2.*np.pi*f*(tcoal*3600.*24.) - Phicoal + phiD + phiL))
            phiD_tcderiv = 0.
            phiL_tcderiv = 2.*np.pi*f*locDt_tcder*(3600.*24.)

            return ampP_tcder + 1j*(phiD_tcderiv + phiL_tcderiv + 2.*np.pi*f*3600.*24.)*hp + ampC_tcder + 1j*(phiD_tcderiv + phiL_tcderiv + 2.*np.pi*f*3600.*24.)*hc

        def iota_par_deriv():

            if (not self.wf_model.is_HigherModes) and (not self.wf_model.is_Precessing):
                wfhp_iotader, wfhc_iotader = -wfAmpl*np.exp(-1j*wfPhiGw)*(np.cos(iota)*np.sin(iota)), -1j*wfAmpl*np.exp(-1j*wfPhiGw)*np.sin(iota)
                return wfhp_iotader*Fp*np.exp(1j*(2.*np.pi*f*(tcoal*3600.*24.) - Phicoal + phiD + phiL)) + wfhc_iotader*Fc*np.exp(1j*(2.*np.pi*f*(tcoal*3600.*24.) - Phicoal + phiD + phiL))
            else:
                # This derivative is computed numerically if the waveform contains higher modes
                return None

        dL_deriv = -(hp+hc)/dL
        Phicoal_deriv = -1j*(hp+hc)
        psi_deriv = psi_par_deriv()
        phi_deriv = phi_par_deriv()
        theta_deriv = theta_par_deriv()
        tc_deriv = tcoal_par_deriv()
        iota_deriv = iota_par_deriv()

        return dL_deriv, theta_deriv, phi_deriv, iota_deriv, psi_deriv, tc_deriv, Phicoal_deriv

    def optimal_location(self, tcoal, is_tGPS=False):
        """
        Compute the optimal sky position for a signal to be seen by the detector at a given time.

        The computation assumes :math:`\psi = 0`.

        :param float tcoal: The time at which to compute the optimal location, as GMST in days.
        :param bool, optional is_tGPS: Boolean specifying if the provided time is a GPS time (in seconds) rather than a GMST.

        :return: Optimal :math:`\\theta` and :math:`\phi` sky coordinates, in :math:`\\rm rad`.
        :rtype: array(float, float)

        """
        # Function to compute the optimal theta and phi for a signal to be seen by the detector at a given GMST. The boolean is_tGPS can be used to specify whether the provided time is a GPS time rather than a GMST, so that it will be converted.
        # For a triangle the best location is the same of an L in the same place, as can be shown by explicit geometrical computation.
        # Even if considering Earth rotation, the highest SNR will still be obtained if the source is in the optimal location close to the merger.
        from scipy.optimize import minimize

        if is_tGPS:
            tc = utils.GPSt_to_LMST(tcoal, lat=0., long=0.)
        else:
            tc = tcoal

        def pattern_fixedtpsi(pars, tc=tc):
            theta, phi = pars
            Fp, Fc = self._PatternFunction(theta, phi, t=tc, psi=0)
            return -np.sqrt(Fp**2 + Fc**2)
        # we actually minimize the pattern function times -1, which is the same as maximizing it
        return minimize(pattern_fixedtpsi, [1.,1.], bounds=((0.,onp.pi), (0.,2.*onp.pi))).x

    def SNRFastInsp(self, evParams, checkInterp=False):
        """
        Compute the inspiral SNR taking into account Earth rotation, without the need of performing an integral for each event

        .. deprecated:: 1.0.0
            Use the standard function :py:class:`GWSignal.SNRInteg`.
       """
        # This module allows to compute the inspiral SNR taking into account Earth rotation, without the need
        # of performing an integral for each event

        Mc, dL, theta, phi, iota, psi, tcoal, eta = evParams['Mc'], evParams['dL'], evParams['theta'], evParams['phi'], evParams['iota'], evParams['psi'], evParams['tcoal'], evParams['eta']

        ras, decs = self._ra_dec_from_th_phi(theta, phi)


        if not np.isscalar(Mc):
            SNR = np.zeros(Mc.shape)
        else:
            SNR = 0

        # Factor in front of the integral in the inspiral only case
        fac = np.sqrt(5./6.)/np.pi**(2./3.)*(glob.GMsun_over_c3*Mc)**(5./6.)*glob.clightGpc/dL#*np.exp(-logdL)

        fcut = self.wf_model.fcut(**evParams)
        if self.fmax is not None:
            fcut = np.where(fcut > self.fmax, self.fmax, fcut)
        mask = self.strainFreq >= self.fmin

        def CoeffsRot(ra, dec, psi, rot=0.):
            rot = rot*np.pi/180.
            rasDet = ra - self.det_long_rad
            # Referring to overleaf, I now call VC2 the last vector appearing in the C2 expression, VS2 the one in the S2 expression and so on
            # e1 is the first element and e2 the second

            VC2e1 = 0.0675*np.cos(2*rasDet)*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2*dec))*(3.-np.cos(2.*self.det_lat_rad)) - 0.25*np.sin(2*rasDet)*np.cos(2*(self.det_xax_rad+rot))*(3.-np.cos(2*dec))*np.sin(self.det_lat_rad)
            VC2e2 = 0.25*np.sin(2*rasDet)*np.sin(2*(self.det_xax_rad+rot))*np.sin(dec)*(3.-np.cos(2.*self.det_lat_rad)) + np.cos(2*rasDet)*np.cos(2*(self.det_xax_rad+rot))*np.sin(dec)*np.sin(self.det_lat_rad)
            C2p = (np.cos(2.*psi)*VC2e1 + np.sin(2*psi)*VC2e2)*np.sin(self.angbtwArms)
            C2c = (-np.sin(2*psi)*VC2e1 + np.cos(2.*psi)*VC2e2)*np.sin(self.angbtwArms)

            VS2e1 = 0.0675*np.sin(2*rasDet)*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2*dec))*(3.-np.cos(2.*self.det_lat_rad)) + 0.25*np.cos(2*rasDet)*np.cos(2*(self.det_xax_rad+rot))*(3.-np.cos(2*dec))*np.sin(self.det_lat_rad)
            VS2e2 = -0.25*np.cos(2*rasDet)*np.sin(2*(self.det_xax_rad+rot))*np.sin(dec)*(3.-np.cos(2.*self.det_lat_rad)) + np.sin(2*rasDet)*np.cos(2*(self.det_xax_rad+rot))*np.sin(dec)*np.sin(self.det_lat_rad)
            S2p = (np.cos(2.*psi)*VS2e1 + np.sin(2*psi)*VS2e2)*np.sin(self.angbtwArms)
            S2c = (-np.sin(2*psi)*VS2e1 + np.cos(2.*psi)*VS2e2)*np.sin(self.angbtwArms)

            VC1e1 = 0.25*np.cos(rasDet)*np.sin(2.*(self.det_xax_rad+rot))*np.sin(2*dec)*np.sin(2.*self.det_lat_rad) - 0.5*np.sin(rasDet)*np.cos(2.*(self.det_xax_rad+rot))*np.sin(2*dec)*np.cos(self.det_lat_rad)
            VC1e2 = np.cos(rasDet)*np.cos(2*(self.det_xax_rad+rot))*np.cos(dec)*np.cos(self.det_lat_rad) + 0.5*np.sin(rasDet)*np.sin(2.*(self.det_xax_rad+rot))*np.cos(dec)*np.sin(2.*self.det_lat_rad)
            C1p = (np.cos(2.*psi)*VC1e1 + np.sin(2*psi)*VC1e2)*np.sin(self.angbtwArms)
            C1c = (-np.sin(2*psi)*VC1e1 + np.cos(2.*psi)*VC1e2)*np.sin(self.angbtwArms)

            VS1e1 = 0.25*np.sin(rasDet)*np.sin(2.*(self.det_xax_rad+rot))*np.sin(2*dec)*np.sin(2.*self.det_lat_rad) + 0.5*np.cos(rasDet)*np.cos(2.*(self.det_xax_rad+rot))*np.sin(2*dec)*np.cos(self.det_lat_rad)
            VS1e2 = np.sin(rasDet)*np.cos(2*(self.det_xax_rad+rot))*np.cos(dec)*np.cos(self.det_lat_rad) - 0.5*np.cos(rasDet)*np.sin(2.*(self.det_xax_rad+rot))*np.cos(dec)*np.sin(2.*self.det_lat_rad)
            S1p = (np.cos(2.*psi)*VS1e1 + np.sin(2*psi)*VS1e2)*np.sin(self.angbtwArms)
            S1c = (-np.sin(2*psi)*VS1e1 + np.cos(2.*psi)*VS1e2)*np.sin(self.angbtwArms)

            C0p = 0.75*np.cos(2.*psi)*np.sin(2.*(self.det_xax_rad+rot))*((np.cos(dec)*np.cos(self.det_lat_rad))**2)*np.sin(self.angbtwArms)
            C0c = -0.75*np.sin(2.*psi)*np.sin(2.*(self.det_xax_rad+rot))*((np.cos(dec)*np.cos(self.det_lat_rad))**2)*np.sin(self.angbtwArms)

            return np.array([C2p, C2c]), np.array([S2p, S2c]), np.array([C1p, C1c]), np.array([S1p, S1c]), np.array([C0p, C0c])

        def FpFcsqInt(C2s, S2s, C1s, S1s, C0s, Igs, iota):
            Fp4 = 0.5*(C2s[0]**2 - S2s[0]**2)*Igs[3] +  C2s[0]*S2s[0]*Igs[7]

            Fp3 = (C2s[0]*C1s[0] - S2s[0]*S1s[0])*Igs[2] + (C2s[0]*S1s[0] + S2s[0]*C1s[0])*Igs[6]

            Fp2 = (0.5*(C1s[0]**2 - S1s[0]**2) + 2.*C2s[0]*C0s[0])*Igs[1] + (2.*C0s[0]*S2s[0] + C1s[0]*S1s[0])*Igs[5]

            Fp1 = (2.*C0s[0]*C1s[0] + C1s[0]*C2s[0] + S2s[0]*S1s[0])*Igs[0] + (2.*C0s[0]*S1s[0] + C1s[0]*S2s[0] - S1s[0]*C2s[0])*Igs[4]

            Fp0 = (C0s[0]**2 + 0.5*(C1s[0]**2 + C2s[0]**2 + S1s[0]**2 + S2s[0]**2))*Igs[8]

            FpsqInt = Fp4 + Fp3 + Fp2 + Fp1 + Fp0

            Fc4 = 0.5*(C2s[1]**2 - S2s[1]**2)*Igs[3] +  C2s[1]*S2s[1]*Igs[7]

            Fc3 = (C2s[1]*C1s[1] - S2s[1]*S1s[1])*Igs[2] + (C2s[1]*S1s[1] + S2s[1]*C1s[1])*Igs[6]

            Fc2 = (0.5*(C1s[1]**2 - S1s[1]**2) + 2.*C2s[1]*C0s[1])*Igs[1] + (2.*C0s[1]*S2s[1] + C1s[1]*S1s[1])*Igs[5]

            Fc1 = (2.*C0s[1]*C1s[1] + C1s[1]*C2s[1] + S2s[1]*S1s[1])*Igs[0] + (2.*C0s[1]*S1s[1] + C1s[1]*S2s[1] - S1s[1]*C2s[1])*Igs[4]

            Fc0 = (C0s[1]**2 + 0.5*(C1s[1]**2 + C2s[1]**2 + S1s[1]**2 + S2s[1]**2))*Igs[8]

            FcsqInt = Fc4 + Fc3 + Fc2 + Fc1 + Fc0

            return FpsqInt*(0.5*(1.+(np.cos(iota))**2))**2, FcsqInt*(np.cos(iota))**2

        if not self.useEarthMotion:
            t = tcoal - self.wf_model.tau_star(self.fmin, **evParams)/(3600.*24)
            if self.detector_shape=='L':
                Fp, Fc = self._PatternFunction(theta, phi, t, psi, rot=0.)
                Qsq = (Fp*0.5*(1.+(np.cos(iota))**2))**2 + (Fc*np.cos(iota))**2
                SNR = fac * np.sqrt(Qsq*onp.interp(fcut, self.strainFreq[mask], self.strainInteg, left=1., right=1.))
            elif self.detector_shape=='T':
                for i in range(3):
                    Fp, Fc = self._PatternFunction(theta, phi, t, psi, rot=60.*i)
                    Qsq = (Fp*0.5*(1.+(np.cos(iota))**2))**2 + (Fc*np.cos(iota))**2
                    tmpSNR = fac * np.sqrt(Qsq*onp.interp(fcut, self.strainFreq[mask], self.strainInteg, left=1., right=1.))
                    SNR = SNR + tmpSNR*tmpSNR
                SNR = np.sqrt(SNR)
        else:
            if self.IntegInterpArr is None:
                self._make_SNRig_interpolator()
            Igs = onp.zeros((9,len(Mc)))
            if not checkInterp:
                for i in range(9):
                    Igs[i,:] = self.IntegInterpArr[i](onp.array([Mc, eta, tcoal]).T)
            else:
                def IntegrandC(f, Mc, tcoal, n):
                    t = tcoal - 2.18567 * ((1.21/Mc)**(5./3.)) * ((100/f)**(8./3.))/(3600.*24)
                    return (f**(-7./3.))*np.cos(n*2.*np.pi*t)
                def IntegrandS(f, Mc, tcoal, n):
                    t = tcoal - 2.18567 * ((1.21/Mc)**(5./3.)) * ((100/f)**(8./3.))/(3600.*24)
                    return (f**(-7./3.))*np.sin(n*2.*np.pi*t)

                fminarr = np.full(fcut.shape, self.fmin)
                fgrids = np.geomspace(fminarr,fcut,num=int(5000))
                strainGrids = np.interp(fgrids, self.strainFreq, self.noiseCurve, left=1., right=1.)

                for m in range(4):
                    tmpIntegrandC = IntegrandC(fgrids, Mc, tcoal, m+1.)
                    tmpIntegrandS = IntegrandS(fgrids, Mc, tcoal, m+1.)
                    Igs[m,:] = onp.trapz(tmpIntegrandC/strainGrids, fgrids, axis=0)
                    Igs[m+4,:] = onp.trapz(tmpIntegrandS/strainGrids, fgrids, axis=0)
                tmpIntegrand = IntegrandC(fgrids, Mc, tcoal, 0.)
                Igs[8,:] = onp.trapz(tmpIntegrand/strainGrids, fgrids, axis=0)


            if self.detector_shape=='L':
                C2s, S2s, C1s, S1s, C0s = CoeffsRot(ras, decs, psi, rot=0.)
                FpsqInt, FcsqInt = FpFcsqInt(C2s, S2s, C1s, S1s, C0s, Igs, iota)
                QsqInt = FpsqInt + FcsqInt
                SNR = fac * np.sqrt(QsqInt)
            elif self.detector_shape=='T':
                for i in range(3):
                    C2s, S2s, C1s, S1s, C0s = CoeffsRot(ras, decs, psi, rot=i*60.)
                    FpsqInt, FcsqInt = FpFcsqInt(C2s, S2s, C1s, S1s, C0s, Igs, iota)
                    QsqInt = FpsqInt + FcsqInt
                    tmpSNR = fac * np.sqrt(QsqInt)
                    SNR = SNR + tmpSNR*tmpSNR
                SNR = np.sqrt(SNR)

        return SNR

    def WFOverlap(self, WF1, WF2, evParams1, evParams2, res=1000, return_separate=False, **kwargs):
        """
        Compute the *overlap* of two waveforms in a single detector on two sets of parameters, for one or multiple events.

        :param WaveFormModel WF1: Object containing the first waveform model to analyse.
        :param WaveFormModel WF2: Object containing the second waveform model to analyse.
        :param dict(array, array, ...) evParams1: Dictionary containing the parameters of the event(s) for the first waveform model, as in :py:data:`events`.
        :param dict(array, array, ...) evParams2: Dictionary containing the parameters of the event(s) for the second waveform model, as in :py:data:`events`.
        :param int res: The resolution of the frequency grid to use.
        :param bool, optional return_all: Boolean specifying if, instead of returning the overlap, the function has to return separately product at the numerator of the definition, :math:`(h_1|h_2)`, and the SNRs at the denominator. This is needed to compute the overlap for a detector network. In this case the return type is *tuple(array, array, array)*.
        :param unused kwargs: Optional arguments.

        :return: Overlap(s) of the two waveforms. The shape is :math:`(N_{\\rm events})`.
        :rtype: 1-D array

        """
        utils.check_evparams(evParams1)
        utils.check_evparams(evParams2)

        # Checks on imput parameters for waveform 1

        if WF1.is_Precessing:
                try:
                    _ =evParams1['chi1x']
                except KeyError:
                    try:
                        print('Adding cartesian components of the spins from angular variables')
                        evParams1['iota'], evParams1['chi1x'], evParams1['chi1y'], evParams1['chi1z'], evParams1['chi2x'], evParams1['chi2y'], evParams1['chi2z'] = utils.TransformPrecessing_angles2comp(thetaJN=evParams1['thetaJN'], phiJL=evParams1['phiJL'], theta1=evParams1['tilt1'], theta2=evParams1['tilt2'], phi12=evParams1['phi12'], chi1=evParams1['chi1'], chi2=evParams1['chi2'], Mc=evParams1['Mc'], eta=evParams1['eta'], fRef=self.fmin, phiRef=0.)
                    except KeyError:
                        raise ValueError('Either the cartesian components of the precessing spins (iota, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z) or their modulus and orientations (thetaJN, chi1, chi2, tilt1, tilt2, phiJL, phi12) have to be provided.')
        else:
            try:
                _ =evParams1['chi1z']
                evParams1['chi1x'], evParams1['chi1y'], evParams1['chi2x'], evParams1['chi2y'] = np.zeros_like(evParams1['Mc']), np.zeros_like(evParams1['Mc']) ,np.zeros_like(evParams1['Mc']), np.zeros_like(evParams1['Mc'])
            except KeyError:
                try:
                    print('Adding chi1z, chi2z from chiS, chiA')
                    evParams1['chi1z'] = evParams1['chiS'] + evParams1['chiA']
                    evParams1['chi2z'] = evParams1['chiS'] - evParams1['chiA']
                except KeyError:
                    raise ValueError('Two among chi1z, chi2z and chiS, chiA have to be provided.')

        if WF1.is_tidal:
            try:
                _=evParams1['LambdaTilde']
            except KeyError:
                try:
                    evParams1['LambdaTilde'], evParams1['deltaLambda'] = utils.Lamt_delLam_from_Lam12(evParams1['Lambda1'], evParams1['Lambda2'], evParams1['eta'])
                except KeyError:
                    raise ValueError('Two among Lambda1, Lambda2 and LambdaTilde and deltaLambda have to be provided.')
        else:
            evParams1['LambdaTilde'], evParams1['deltaLambda'] = np.zeros_like(evParams1['Mc']), np.zeros_like(evParams1['Mc'])

        if not WF1.is_eccentric:
            evParams1['ecc'] = np.zeros_like(evParams1['Mc'])

        # Checks on imput parameters for waveform 2

        if WF2.is_Precessing:
                try:
                    _ =evParams2['chi1x']
                except KeyError:
                    try:
                        print('Adding cartesian components of the spins from angular variables')
                        evParams2['iota'], evParams2['chi1x'], evParams2['chi1y'], evParams2['chi1z'], evParams2['chi2x'], evParams2['chi2y'], evParams2['chi2z'] = utils.TransformPrecessing_angles2comp(thetaJN=evParams2['thetaJN'], phiJL=evParams2['phiJL'], theta1=evParams2['tilt1'], theta2=evParams2['tilt2'], phi12=evParams2['phi12'], chi1=evParams2['chi1'], chi2=evParams2['chi2'], Mc=evParams2['Mc'], eta=evParams2['eta'], fRef=self.fmin, phiRef=0.)
                    except KeyError:
                        raise ValueError('Either the cartesian components of the precessing spins (iota, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z) or their modulus and orientations (thetaJN, chi1, chi2, tilt1, tilt2, phiJL, phi12) have to be provided.')
        else:
            try:
                _ =evParams2['chi1z']
                evParams2['chi1x'], evParams2['chi1y'], evParams2['chi2x'], evParams2['chi2y'] = np.zeros_like(evParams2['Mc']), np.zeros_like(evParams2['Mc']) ,np.zeros_like(evParams2['Mc']), np.zeros_like(evParams2['Mc'])
            except KeyError:
                try:
                    print('Adding chi1z, chi2z from chiS, chiA')
                    evParams2['chi1z'] = evParams2['chiS'] + evParams2['chiA']
                    evParams2['chi2z'] = evParams2['chiS'] - evParams2['chiA']
                except KeyError:
                    raise ValueError('Two among chi1z, chi2z and chiS, chiA have to be provided.')

        if WF2.is_tidal:
            try:
                _=evParams2['LambdaTilde']
            except KeyError:
                try:
                    evParams2['LambdaTilde'], evParams2['deltaLambda'] = utils.Lamt_delLam_from_Lam12(evParams2['Lambda1'], evParams2['Lambda2'], evParams2['eta'])
                except KeyError:
                    raise ValueError('Two among Lambda1, Lambda2 and LambdaTilde and deltaLambda have to be provided.')
        else:
            evParams2['LambdaTilde'], evParams2['deltaLambda'] = np.zeros_like(evParams2['Mc']), np.zeros_like(evParams2['Mc'])

        if not WF2.is_eccentric:
            evParams2['ecc'] = np.zeros_like(evParams2['Mc'])

        # The frequency cut is chosen to be the highest among the two
        fcut1 = WF1.fcut(**evParams1)
        fcut2 = WF2.fcut(**evParams2)

        fcutUse = np.where(fcut1>fcut2, fcut1, fcut2)

        if self.fmax is not None:
            fcutUse = np.where(fcutUse > self.fmax, self.fmax, fcut1)
        fminarr = np.full(fcutUse.shape, self.fmin)

        fgrids = np.geomspace(fminarr,fcutUse,num=int(res))
        # Out of the provided PSD range, we use a constant value of 1, which results in completely negligible conntributions
        strainGrids = np.interp(fgrids, self.strainFreq, self.noiseCurve, left=1., right=1.)

        # This is a horrible way of changing the waveform, but the fastest to implement
        WFor = copy.deepcopy(self.wf_model)

        if self.detector_shape=='L':
            self.wf_model = WF1
            h1 = self.GWstrain(fgrids, evParams1['Mc'], evParams1['eta'], evParams1['dL'], evParams1['theta'], evParams1['phi'], evParams1['iota'], evParams1['psi'], evParams1['tcoal'], evParams1['Phicoal'], evParams1['chi1z'], evParams1['chi2z'], evParams1['chi1x'], evParams1['chi2x'], evParams1['chi1y'], evParams1['chi2y'], evParams1['LambdaTilde'], evParams1['deltaLambda'], evParams1['ecc'], is_chi1chi2=True)
            h1sq = np.conjugate(h1)*h1
            SNRh1 = np.sqrt(4.*np.trapezoid(h1sq.real/strainGrids, fgrids, axis=0))
            self.wf_model = WF2
            h2 = self.GWstrain(fgrids, evParams2['Mc'], evParams2['eta'], evParams2['dL'], evParams2['theta'], evParams2['phi'], evParams2['iota'], evParams2['psi'], evParams2['tcoal'], evParams2['Phicoal'], evParams2['chi1z'], evParams2['chi2z'], evParams2['chi1x'], evParams2['chi2x'], evParams2['chi1y'], evParams2['chi2y'], evParams2['LambdaTilde'], evParams2['deltaLambda'], evParams2['ecc'], is_chi1chi2=True)
            h2sq = np.conjugate(h2)*h2
            SNRh2 = np.sqrt(4.*np.trapezoid(h2sq.real/strainGrids, fgrids, axis=0))

            overlap_h1h2 = h1*np.conjugate(h2)
            overlap_int= 4.*np.trapezoid(overlap_h1h2.real/strainGrids, fgrids, axis=0)

        elif self.detector_shape=='T':
            self.wf_model = WF1
            h1_1 = self.GWstrain(fgrids, evParams1['Mc'], evParams1['eta'], evParams1['dL'], evParams1['theta'], evParams1['phi'], evParams1['iota'], evParams1['psi'], evParams1['tcoal'], evParams1['Phicoal'], evParams1['chi1z'], evParams1['chi2z'], evParams1['chi1x'], evParams1['chi2x'], evParams1['chi1y'], evParams1['chi2y'], evParams1['LambdaTilde'], evParams1['deltaLambda'], evParams1['ecc'], is_chi1chi2=True, rot=0.)
            h1_1sq = np.conjugate(h1_1)*h1_1
            SNRh1_1sq = 4.*np.trapezoid(h1_1sq.real/strainGrids, fgrids, axis=0)
            h1_2 = self.GWstrain(fgrids, evParams1['Mc'], evParams1['eta'], evParams1['dL'], evParams1['theta'], evParams1['phi'], evParams1['iota'], evParams1['psi'], evParams1['tcoal'], evParams1['Phicoal'], evParams1['chi1z'], evParams1['chi2z'], evParams1['chi1x'], evParams1['chi2x'], evParams1['chi1y'], evParams1['chi2y'], evParams1['LambdaTilde'], evParams1['deltaLambda'], evParams1['ecc'], is_chi1chi2=True, rot=60.)
            h1_2sq = np.conjugate(h1_2)*h1_2
            SNRh1_2sq = 4.*np.trapezoid(h1_2sq.real/strainGrids, fgrids, axis=0)
            h1_3 = - (h1_1 + h1_2)
            h1_3sq = np.conjugate(h1_3)*h1_3
            SNRh1_3sq = 4.*np.trapezoid(h1_3sq.real/strainGrids, fgrids, axis=0)
            SNRh1 = np.sqrt(SNRh1_1sq + SNRh1_2sq + SNRh1_3sq)

            self.wf_model = WF2
            h2_1 = self.GWstrain(fgrids, evParams2['Mc'], evParams2['eta'], evParams2['dL'], evParams2['theta'], evParams2['phi'], evParams2['iota'], evParams2['psi'], evParams2['tcoal'], evParams2['Phicoal'], evParams2['chi1z'], evParams2['chi2z'], evParams2['chi1x'], evParams2['chi2x'], evParams2['chi1y'], evParams2['chi2y'], evParams2['LambdaTilde'], evParams2['deltaLambda'], evParams2['ecc'], is_chi1chi2=True, rot=0.)
            h2_1sq = np.conjugate(h2_1)*h2_1
            SNRh2_1sq = 4.*np.trapezoid(h2_1sq.real/strainGrids, fgrids, axis=0)
            h2_2 = self.GWstrain(fgrids, evParams2['Mc'], evParams2['eta'], evParams2['dL'], evParams2['theta'], evParams2['phi'], evParams2['iota'], evParams2['psi'], evParams2['tcoal'], evParams2['Phicoal'], evParams2['chi1z'], evParams2['chi2z'], evParams2['chi1x'], evParams2['chi2x'], evParams2['chi1y'], evParams2['chi2y'], evParams2['LambdaTilde'], evParams2['deltaLambda'], evParams2['ecc'], is_chi1chi2=True, rot=60.)
            h2_2sq = np.conjugate(h2_2)*h2_2
            SNRh2_2sq = 4.*np.trapezoid(h2_2sq.real/strainGrids, fgrids, axis=0)
            h2_3 = - (h2_1 + h2_2)
            h2_3sq = np.conjugate(h2_3)*h2_3
            SNRh2_3sq = 4.*np.trapezoid(h2_3sq.real/strainGrids, fgrids, axis=0)
            SNRh2 = np.sqrt(SNRh2_1sq + SNRh2_2sq + SNRh2_3sq)

            overlap_h1h2_1 = h1_1*np.conjugate(h2_1)
            overlap_int_1  = 4.*np.trapezoid(overlap_h1h2_1.real/strainGrids, fgrids, axis=0)
            overlap_h1h2_2 = h1_2*np.conjugate(h2_2)
            overlap_int_2  = 4.*np.trapezoid(overlap_h1h2_2.real/strainGrids, fgrids, axis=0)
            overlap_h1h2_3 = h1_3*np.conjugate(h2_3)
            overlap_int_3  = 4.*np.trapezoid(overlap_h1h2_3.real/strainGrids, fgrids, axis=0)

            overlap_int = overlap_int_1 + overlap_int_2 + overlap_int_3
        # Restore the waveform
        self.wf_model = WFor

        if return_separate:
            return overlap_int, SNRh1, SNRh2
        else:
            return overlap_int/(SNRh1*SNRh2)
