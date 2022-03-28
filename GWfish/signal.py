#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
#os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
import jax
#jax.devices('cpu')

#Enable 64bit on JAX, fundamental
from jax.config import config
config.update("jax_enable_x64", True)
#config.update("TF_CPP_MIN_LOG_LEVEL", 0)

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'       
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='0'

# We use both the original numpy, denoted as onp, and the JAX implementation of numpy, denoted as np
import numpy as onp
#import numpy as np
import jax.numpy as np
from jax.interpreters import xla
from jax import pmap, vmap, jacrev, jit #jacfwd
import time
#import os
import h5py
#import copy

import fisherUtils as utils
import fisherGlobals as glob
#import fisherTools



class GWSignal(object):
    '''
    Class to compute the GW signal emitted by a coalescing binary system as seen by a detector on Earth.
    
    The functions defined within this class allow to get the amplitude of the signal, its phase, and SNR.
    
    Inputs are an object containing the waveform model, the coordinates of the detector (latitude and longitude in deg),
    its shape (L or T), the angle w.r.t. East of the bisector of the arms (deg) 
    and its ASD (given in a .txt file containing two columns: one with the frequencies and one with the ASD values, 
    remember ASD=sqrt(PSD))
    
    '''
    def __init__(self, wf_model, 
                psd_path=None,
                detector_shape = 'T',
                det_lat=40.44,
                det_long=9.45,
                det_xax=0., 
                verbose=True,
                useEarthMotion = False,
                noMotion = False, # use only for checks
                fmin=5, fmax=None,
                IntTablePath=None,
                DutyFactor=None,
                compute2arms=True):
        
        if (detector_shape!='L') and (detector_shape!='T'):
            raise ValueError('Enter valid detector configuration')
        
        if psd_path is None:
            raise ValueError('Enter a valid PSD path')
        
        if verbose:
            print('Using PSD from file %s ' %psd_path)
        
        if (useEarthMotion) and (wf_model.objType == 'BBH'):
            print('WARNING: the motion of Earth gives a negligible contribution for BBH signals, consider switching it off to make the code run faster')
        if (not useEarthMotion) and (wf_model.objType == 'BNS'):
            print('WARNING: the motion of Earth gives a relevant contribution for BNS signals, consider switching it on')
        if (not useEarthMotion) and (wf_model.objType == 'NSBH'):
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
        #This is the percentage of time each arm of the detector (or the whole detector for an L) is supposed to be operational
        self.DutyFactor = DutyFactor
        
        noise = onp.loadtxt(psd_path, usecols=(0,1))
        f = noise[:,0]
        S = (noise[:,1])**2
        
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

        self._init_jax()
        
        
    def _init_jax(self):
        print('Initializing jax...')
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
        os.environ["TF_CPP_MIN_LOG_LEVEL"]='0'
        #os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count=8'
        print('Jax local device count: %s' %str(jax.local_device_count()))
        print('Jax  device count: %s' %str(jax.device_count()))
        
        #self._SignalDerivatives_jit = jax.jit(self._SignalDerivatives, static_argnums=(15,16,17,18,19))
        self._SignalDerivatives_jit = self._SignalDerivatives
        
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
                           'theta': np.array([3.00702251])}
        verboseOr = self.verbose
        self.verbose = False
        detector_shapeOr = self.detector_shape
        self.detector_shape = 'L' # Get a faster Initialization with an L
        _ = self.SNRInteg(inj_params_init, res=10)
        _ = self.FisherMatr(inj_params_init, res=10)
        #McOr, dL, theta, phi = inj_params_init['Mc'].astype('complex128'), inj_params_init['dL'].astype('complex128'), inj_params_init['theta'].astype('complex128'), inj_params_init['phi'].astype('complex128')
        #iota, psi, tcoal, etaOr, Phicoal = inj_params_init['iota'].astype('complex128'), inj_params_init['psi'].astype('complex128'), inj_params_init['tcoal'].astype('complex128'), inj_params_init['eta'].astype('complex128'), inj_params_init['Phicoal'].astype('complex128')
        #chiS, chiA = inj_params_init['chis'].astype('complex128'), inj_params_init['chia'].astype('complex128')
        #_ = self._SignalDerivatives_jit(np.array([[50., 70]]).astype('complex128').T, McOr, etaOr, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, np.array([0.]).astype('complex128'), np.array([0.]).astype('complex128'), )
        print('Done.')
        self.verbose = verboseOr
        self.detector_shape = detector_shapeOr
     
    def _clear_cache(self):
        print('Clearing cache...')
        self._SignalDerivatives_jit = jax.jit(self._SignalDerivatives, static_argnums=(14,15,16,17))
     
    def _update_seed(self,):
        onp.random.seed(None)
        self.seedUse = onp.random.randint(2**32 - 1, size=1)
        
    def _tabulateIntegrals(self, res=200, store=True, Mcmin=.9, Mcmax=9., etamin=.1):
        #import scipy.integrate as igt
        
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
        # See P. Jaranowski, A. Krolak, B. F. Schutz, PRD 58, 063001, eq. (10)--(13)
        # rot (deg) is an additional parameter, needed for the triangle configuration, allowing to specify a further rotation
        # of the interferometer w.r.t. xax. In this case, the three arms will have orientations 1 -> xax, 2 -> xax+60°, 3 -> xax+120° 
        
    
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
    
    def _phiDoppler(self, theta, phi, t, f):
        
        
        phiD = 2.*np.pi*f*(glob.REarth/glob.clight)*np.sin(theta)*np.cos(2.*np.pi*t - phi)
        
        #ddot_phiD = ((2.*np.pi)**3)*f*(glob.REarth/glob.clight)*np.sin(theta)*np.cos(2.*np.pi*t - phi)/((3600.*24)**2.) # This contribution to the amplitude is negligible
        
        return phiD#, ddot_phiD
    
    def _phiPhase(self, theta, phi, t, iota, psi, Fp=None,Fc=None):
        #The polarization phase contribution (the change in F+ and Fx with time influences also the phase)
        
        if (Fp is None) or (Fc is None):
            Fp, Fc = self._PatternFunction(theta, phi, t, psi)
        
        phiP = -np.arctan2(np.cos(iota)*Fc,0.5*(1.+((np.cos(iota))**2))*Fp)
        
        #The contriution to the amplitude is negligible, so we do not compute it
        
        return phiP
    
    def _DeltLoc(self, theta, phi, t, f):
        # Time needed to go from Earth center to detector location
        
        ras, decs = self._ra_dec_from_th_phi(theta, phi)
        
        comp1 = np.cos(decs)*np.cos(ras)*np.cos(self.det_lat_rad)*np.cos(self.det_long_rad + 2.*np.pi*t)
        comp2 = np.cos(decs)*np.sin(ras)*np.cos(self.det_lat_rad)*np.sin(self.det_long_rad + 2.*np.pi*t)
        comp3 = np.sin(decs)*np.sin(self.det_lat_rad)
        # The minus sign arises from the definition of the unit vector pointing to the source
        Delt = - glob.REarth*(comp1+comp2+comp3)/glob.clight
        
        return Delt # in seconds
    
    def GWAmplitudes(self, evParams, f, rot=0.):
        # evParams are all the parameters characterizing the event(s) under exam. It has to be a dictionary containing the entries: 
        # Mc -> chirp mass (Msun), dL -> luminosity distance (Gpc), theta & phi -> sky position (rad), iota -> inclination angle of orbital angular momentum to l.o.s toward the detector,
        # psi -> polarisation angle, tcoal -> time of coalescence as GMST (fraction of days), eta -> symmetric mass ratio, Phicoal -> GW frequency at coalescence.
        # chi1z, chi2z -> dimensionless spin components aligned to orbital angular momentum [-1;1], Lambda1,2 -> tidal parameters of the objects,
        # f is the frequency (Hz)
        
        #self._check_evparams(evParams)
        
        theta, phi, iota, psi, tcoal = evParams['theta'], evParams['phi'], evParams['iota'], evParams['psi'], evParams['tcoal']
        

        if self.noMotion:
            t = 0
            t = t + self._DeltLoc(theta, phi, t, f)/(3600.*24.)
        else:
            if self.useEarthMotion:
                t = tcoal - self.wf_model.tau_star(f, **evParams)/(3600.*24)
                t = t + self._DeltLoc(theta, phi, t, f)/(3600.*24.)
            else:
                t = tcoal #- self.wf_model.tau_star(self.fmin, **evParams)/(3600.*24)
                t = t + self._DeltLoc(theta, phi, t, f)/(3600.*24.)
        # wfAmpl = self.wf_model.Ampl(f, **evParams)
        Fp, Fc = self._PatternFunction(theta, phi, t, psi, rot=rot)
        
        if not self.wf_model.is_HigherModes:
            wfAmpl = self.wf_model.Ampl(f, **evParams)
            Ap = wfAmpl*Fp*0.5*(1.+(np.cos(iota))**2)
            Ac = wfAmpl*Fc*np.cos(iota)
        else:
            # If the waveform includes higher modes, it is not possible to compute amplitude and phase separately, make all together
            #Phis = self.wf_model.Phi(f, **evParams)
            # Now make up the waveform adding the spherical harmonics
            #hp, hc = utils.Add_Higher_Modes(wfAmpl, Phis, iota)
            hp, hc = self.wf_model.hphc(f, **evParams)
            Ap, Ac = abs(hp)*Fp, abs(hc)*Fc
        
        return Ap, Ac
    
    def GWPhase(self, evParams, f):
        # Phase of the GW signal
        tcoal, Phicoal =  evParams['tcoal'], evParams['Phicoal']
        PhiGw = self.wf_model.Phi(f, **evParams)

        return 2.*np.pi*f*(tcoal*3600.*24.) - Phicoal + PhiGw

    def GWstrain(self, f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda, rot=0., is_m1m2=False, is_chi1chi2=False, return_single_comp=None):

        # Full GW strain expression (complex)
        # Here we have the decompressed parameters and we put them back in a dictionary just to have an easier
        # implementation of the JAX module for derivatives
        
        if is_chi1chi2:
            # Interpret chiS as chi1z and chiA as chi2z
            chi1z = chiS
            chi2z = chiA
        else:
            chi1z = chiS + chiA
            chi2z = chiS - chiA
        if is_m1m2:
            # Interpret Mc as m1 and eta as m2
            McUse  = ((Mc*eta)**(3./5.))/((Mc+eta)**(1./5.))
            etaUse = (Mc*eta)/((Mc+eta)**(2.))
        else:
            McUse  = Mc
            etaUse = eta
        
        evParams = {'Mc':McUse, 'dL':dL, 'theta':theta, 'phi':phi, 'iota':iota, 'psi':psi, 'tcoal':tcoal, 'eta':etaUse, 'Phicoal':Phicoal, 'chi1z':chi1z, 'chi2z':chi2z}
        
        if self.wf_model.is_tidal:
            Lambda1, Lambda2 = utils.Lam12_from_Lamt_delLam(LambdaTilde, deltaLambda, etaUse)
            
            evParams['Lambda1'] = Lambda1
            evParams['Lambda2'] = Lambda2
         
        
        if self.useEarthMotion:
            # Compute Doppler contribution
            t = tcoal - self.wf_model.tau_star(f, **evParams)/(3600.*24.)
            tmpDeltLoc = self._DeltLoc(theta, phi, t, f) # in seconds
            t = t + tmpDeltLoc/(3600.*24.)
            phiD = self._phiDoppler(theta, phi, t, f)
            #phiP is necessary if we write the signal as A*exp(i Psi) with A = sqrt(Ap^2 + Ac^2), uncomment if necessary
            #phiP = self._phiPhase(theta, phi, t, iota, psi)
        else:
            phiD = Mc*0.
            #phiP = Mc*0.
            if self.noMotion:
                t=0
            else:
                t = tcoal
            tmpDeltLoc = self._DeltLoc(theta, phi, t, f) # in seconds
            t = t + tmpDeltLoc/(3600.*24.)
        
        phiL = (2.*np.pi*f)*tmpDeltLoc

        
        if not self.wf_model.is_HigherModes:
            Ap, Ac = self.GWAmplitudes(evParams, f, rot=rot)
            Psi = self.GWPhase(evParams, f)
            Psi = Psi + phiD + phiL
        
            if return_single_comp is not None:
                if (return_single_comp == 'Ap'):
                    return Ap
                elif (return_single_comp == 'Ac'):
                    return Ac
                elif (return_single_comp == 'Psip'):
                    return Psi #np.unwrap(Psi)
                elif (return_single_comp == 'Psic'):
                    return Psi + np.pi*0.5 #np.unwrap(Psi + np.pi*0.5)
                else:
                    raise ValueError('Single component to return has to be among Ap, Ac, Psip, Psic')
            else:
                return (Ap + 1j*Ac)*np.exp(Psi*1j)
            #return np.sqrt(Ap*Ap + Ac*Ac)*np.exp((Psi+phiP)*1j)
        else:
            # If the waveform includes higher modes, it is not possible to compute amplitude and phase separately, make all together
            #wfAmpl = self.wf_model.Ampl(f, **evParams)
            Fp, Fc = self._PatternFunction(theta, phi, t, psi, rot=rot)
            #Phis = self.wf_model.Phi(f, **evParams)
            # Now make up the waveform adding the spherical harmonics
            #hp, hc = utils.Add_Higher_Modes(wfAmpl, Phis, iota)
            hp, hc = self.wf_model.hphc(f, **evParams)
            hp = hp*Fp*np.exp(1j*(phiD + phiL + 2.*np.pi*f*(tcoal*3600.*24.) - Phicoal))
            hc = hc*Fc*np.exp(1j*(phiD + phiL + 2.*np.pi*f*(tcoal*3600.*24.) - Phicoal))
            
            if return_single_comp is not None:
                if (return_single_comp == 'Ap'):
                    return np.abs(hp)
                elif (return_single_comp == 'Ac'):
                    return np.abs(hc)
                elif (return_single_comp == 'Psip'):
                    return np.unwrap(np.angle(hp))
                elif (return_single_comp == 'Psic'):
                    return np.unwrap(np.angle(hc))
                else:
                    raise ValueError('Single component to return has to be among Ap, Ac, Psip, Psic')
            else:
                return hp + hc
    
    def SNRInteg(self, evParams, res=1000):
        # SNR calculation performing the frequency integral for each signal
        # This is computationally more expensive, but needed for complex waveform models
        if self.DutyFactor is not None:
            onp.random.seed(self.seedUse)
        
        utils.check_evparams(evParams)
        if not np.isscalar(evParams['Mc']):
            SNR = np.zeros(len(np.asarray(evParams['Mc'])))
        else:
            SNR = 0
        if self.wf_model.is_tidal:
            try:
                evParams['Lambda1']
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
        strainGrids = np.interp(fgrids, self.strainFreq, self.noiseCurve)
        
        if self.detector_shape=='L':    
            Aps, Acs = self.GWAmplitudes(evParams, fgrids)
            Atot = Aps*Aps + Acs*Acs
            SNR = np.sqrt(np.trapz(Atot/strainGrids, fgrids, axis=0))
            if self.DutyFactor is not None:
                excl = onp.random.choice([0,1],len(evParams['Mc']), p=[1.-self.DutyFactor,self.DutyFactor])
                SNR = SNR*excl
        elif self.detector_shape=='T':
            if not self.compute2arms:
                for i in range(3):
                    Aps, Acs = self.GWAmplitudes(evParams, fgrids, rot=i*60.)
                    Atot = Aps*Aps + Acs*Acs
                    tmpSNRsq = np.trapz(Atot/strainGrids, fgrids, axis=0)
                    if self.DutyFactor is not None:
                        excl = onp.random.choice([0,1],len(evParams['Mc']), p=[1.-self.DutyFactor,self.DutyFactor])
                        tmpSNRsq = tmpSNRsq*excl
                    SNR = SNR + tmpSNRsq
                SNR = np.sqrt(SNR)
            else:
            # The signal in 3 arms sums to zero for geometrical reasons, so we can use this to skip some calculations
                Aps1, Acs1 = self.GWAmplitudes(evParams, fgrids, rot=0.)
                Atot1 = Aps1*Aps1 + Acs1*Acs1
                Aps2, Acs2 = self.GWAmplitudes(evParams, fgrids, rot=60.)
                Atot2 = Aps2*Aps2 + Acs2*Acs2
                Aps3, Acs3 = - (Aps1 + Aps2), - (Acs1 + Acs2)
                Atot3 = Aps3*Aps3 + Acs3*Acs3
                tmpSNRsq1 = np.trapz(Atot1/strainGrids, fgrids, axis=0)
                tmpSNRsq2 = np.trapz(Atot2/strainGrids, fgrids, axis=0)
                tmpSNRsq3 = np.trapz(Atot3/strainGrids, fgrids, axis=0)
                if self.DutyFactor is not None:
                    excl = onp.random.choice([0,1],len(evParams['Mc']), p=[1.-self.DutyFactor,self.DutyFactor])
                    tmpSNRsq1 = tmpSNRsq1 * excl
                    excl = onp.random.choice([0,1],len(evParams['Mc']), p=[1.-self.DutyFactor,self.DutyFactor])
                    tmpSNRsq2 = tmpSNRsq2 * excl
                    excl = onp.random.choice([0,1],len(evParams['Mc']), p=[1.-self.DutyFactor,self.DutyFactor])
                    tmpSNRsq3 = tmpSNRsq3 * excl
                
                SNR = np.sqrt(tmpSNRsq1 + tmpSNRsq2 + tmpSNRsq3)
        
        return 2.*SNR # The factor of two arises by cutting the integral from 0 to infinity
    
    
    def FisherMatr(self, evParams, res=None, df=2**-4, spacing='geom', 
                   use_m1m2=False, use_chi1chi2=False, 
                   computeRealDeriv=False, computeDerivFinDiff=False, computeAnalyticalDeriv=False,
                   **kwargs):
                   #stepNDT=1e-9,
                   #parallel=False):
        # If use_m1m2=True the Fisher is computed w.r.t. m1 and m2, not Mc and eta
        # If use_chi1chi2=True the Fisher is computed w.r.t. chi1z and chi2z, not chiS and chiA
        if self.DutyFactor is not None:
            onp.random.seed(self.seedUse)
        
        utils.check_evparams(evParams)
        
        if not computeRealDeriv:
            McOr, dL, theta, phi = evParams['Mc'].astype('complex128'), evParams['dL'].astype('complex128'), evParams['theta'].astype('complex128'), evParams['phi'].astype('complex128')
            iota, psi, tcoal, etaOr, Phicoal = evParams['iota'].astype('complex128'), evParams['psi'].astype('complex128'), evParams['tcoal'].astype('complex128'), evParams['eta'].astype('complex128'), evParams['Phicoal'].astype('complex128')
            chi1z, chi2z = evParams['chi1z'].astype('complex128'), evParams['chi2z'].astype('complex128')
        else:
            McOr, dL, theta, phi = evParams['Mc'].astype('float64'), evParams['dL'].astype('float64'), evParams['theta'].astype('float64'), evParams['phi'].astype('float64')
            iota, psi, tcoal, etaOr, Phicoal = evParams['iota'].astype('float64'), evParams['psi'].astype('float64'), evParams['tcoal'].astype('float64'), evParams['eta'].astype('float64'), evParams['Phicoal'].astype('float64')
            chi1z, chi2z = evParams['chi1z'].astype('float64'), evParams['chi2z'].astype('float64')
            
        if use_m1m2:
            # In this case Mc represents m1 and eta represents m2
            Mc  = 0.5*McOr*(etaOr**(-3./5.))*(1. + np.sqrt(1. - 4.*etaOr))
            eta = 0.5*McOr*(etaOr**(-3./5.))*(1. - np.sqrt(1. - 4.*etaOr))
        else:
            Mc, eta  = McOr, etaOr

        if use_chi1chi2:
            # In this case chiS represents chi1z and chiA represents chi2z
            chiS, chiA = chi1z, chi2z
        else:
            chiS, chiA = 0.5*(chi1z + chi2z), 0.5*(chi1z - chi2z)
        
        if self.wf_model.is_tidal:
            try:
                Lambda1, Lambda2 = evParams['Lambda1'].astype('complex128'), evParams['Lambda2'].astype('complex128')
            except KeyError:
                try:
                    Lambda1, Lambda2  = utils.Lam12_from_Lamt_delLam(evParams['LambdaTilde'].astype('complex128'), evParams['deltaLambda'].astype('complex128'), etaOr)
                except KeyError:
                    raise ValueError('Two among Lambda1, Lambda2 and LambdaTilde and deltaLambda have to be provided.')
            LambdaTilde, deltaLambda = utils.Lamt_delLam_from_Lam12(Lambda1, Lambda2, etaOr)
            if computeRealDeriv:
                Lambda1, Lambda2, LambdaTilde, deltaLambda  = np.real(Lambda1), np.real(Lambda2), np.real(LambdaTilde), np.real(deltaLambda)
        else:
            Lambda1, Lambda2, LambdaTilde, deltaLambda = np.zeros(Mc.shape), np.zeros(Mc.shape), np.zeros(Mc.shape), np.zeros(Mc.shape)
        
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
            
        #fgrids = np.arange(fminarr, np.real(fcut), df, ).astype('complex128') #dtype=fcut.dtype)
        #print('frequency grid: fmin=%s, fmax= %s, step=%s '%(fminarr,fcut, df))
        #
        #print(fgrids)
        strainGrids = np.interp(fgrids, self.strainFreq, self.noiseCurve)

        nParams = self.wf_model.nParams
        tcelem = self.wf_model.ParNums['tcoal']
        
        if self.detector_shape=='L': 
            # Compute derivatives
            FisherDerivs = self._SignalDerivatives_jit(fgrids, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda, rot=0., use_m1m2=use_m1m2, use_chi1chi2=use_chi1chi2, computeRealDeriv=computeRealDeriv, computeAnalyticalDeriv=computeAnalyticalDeriv, **kwargs)
            # Change the units of the tcoal derivative from days to seconds (this improves conditioning)
            FisherDerivs = onp.array(FisherDerivs)
            FisherDerivs[tcelem,:,:] /= (3600.*24.)
            
            FisherIntegrands = (onp.conjugate(FisherDerivs[:,:,onp.newaxis,:])*FisherDerivs.transpose(1,0,2))
    
            Fisher = onp.zeros((nParams,nParams,len(Mc)))
            # This for is unavoidable i think
            for alpha in range(nParams):
                for beta in range(alpha,nParams):
                    tmpElem = FisherIntegrands[alpha,:,beta,:].T
                    Fisher[alpha,beta, :] = onp.trapz(tmpElem.real/strainGrids.real, fgrids.real, axis=0)*4.

                    Fisher[beta,alpha, :] = Fisher[alpha,beta, :]
            if self.DutyFactor is not None:
                excl = onp.random.choice([0,1],len(evParams['Mc']), p=[1.-self.DutyFactor,self.DutyFactor])
                Fisher = Fisher*excl
        else:
            Fisher = onp.zeros((nParams,nParams,len(Mc)))
            if not self.compute2arms:
                for i in range(3):
                    # Change rot and compute derivatives
                    FisherDerivs = self._SignalDerivatives_jit(fgrids, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda, rot=i*60., use_m1m2=use_m1m2, use_chi1chi2=use_chi1chi2, computeRealDeriv=computeRealDeriv, computeAnalyticalDeriv=computeAnalyticalDeriv, **kwargs)
                    # Change the units of the tcoal derivative from days to seconds (this improves conditioning)
                    FisherDerivs = onp.array(FisherDerivs)
                    FisherDerivs[tcelem,:,:] /= (3600.*24.)
                    FisherIntegrands = (onp.conjugate(FisherDerivs[:,:,onp.newaxis,:])*FisherDerivs.transpose(1,0,2))
                    
                    tmpFisher = onp.zeros((nParams,nParams,len(Mc)))
                    # This for is unavoidable i think
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
                    Fisher += tmpFisher
            else:
            # The signal in 3 arms sums to zero for geometrical reasons, so we can use this to skip some calculations
            
                # Compute derivatives
                FisherDerivs1 = self._SignalDerivatives_jit(fgrids, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda, rot=0., use_m1m2=use_m1m2, use_chi1chi2=use_chi1chi2, computeRealDeriv=computeRealDeriv, computeAnalyticalDeriv=computeAnalyticalDeriv, **kwargs)
                # Change the units of the tcoal derivative from days to seconds (this improves conditioning)
                FisherDerivs1 = onp.array(FisherDerivs1)
                FisherDerivs1[tcelem,:,:] /= (3600.*24.)
                #print('Clearing cache...')
                #self._SignalDerivatives_jit = jax.jit(self._SignalDerivatives, static_argnums=(14,15,16,17))

                FisherIntegrands = (onp.conjugate(FisherDerivs1[:,:,onp.newaxis,:])*FisherDerivs1.transpose(1,0,2))
                    
                tmpFisher = onp.zeros((nParams,nParams,len(Mc)))
                if self.verbose:
                    print('Filling matrix for arm 1...')
                # This for is unavoidable i think
                for alpha in range(nParams):
                    for beta in range(alpha,nParams):
                        tmpElem = FisherIntegrands[alpha,:,beta,:].T
                        tmpFisher[alpha,beta, :] = onp.trapz(tmpElem.real/strainGrids.real, fgrids.real, axis=0)*4.
                            
                        tmpFisher[beta,alpha, :] = tmpFisher[alpha,beta, :]
                if self.DutyFactor is not None:
                    excl = onp.random.choice([0,1],len(evParams['Mc']), p=[1.-self.DutyFactor,self.DutyFactor])
                    tmpFisher = tmpFisher*excl
                Fisher += tmpFisher
                
                
                FisherDerivs2 = self._SignalDerivatives_jit(fgrids, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda, rot=60., use_m1m2=use_m1m2, use_chi1chi2=use_chi1chi2, computeRealDeriv=computeRealDeriv, computeAnalyticalDeriv=computeAnalyticalDeriv, **kwargs)
                FisherDerivs2 = onp.array(FisherDerivs2)
                FisherDerivs2[tcelem,:,:] /= (3600.*24.)
                FisherIntegrands = (onp.conjugate(FisherDerivs2[:,:,onp.newaxis,:])*FisherDerivs2.transpose(1,0,2))
                    
                tmpFisher = onp.zeros((nParams,nParams,len(Mc)))
                # This for is unavoidable i think
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
                Fisher += tmpFisher
                
                FisherDerivs3 = - (FisherDerivs1 + FisherDerivs2)
                    
                FisherIntegrands = (onp.conjugate(FisherDerivs3[:,:,onp.newaxis,:])*FisherDerivs3.transpose(1,0,2))
                    
                tmpFisher = onp.zeros((nParams,nParams,len(Mc)))
                # This for is unavoidable i think
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
                Fisher += tmpFisher
            
        return Fisher
    
    
    
    def _SignalDerivatives(self, fgrids, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda, rot=0., use_m1m2=False, use_chi1chi2=False, computeRealDeriv=False, computeDerivFinDiff=False, computeAnalyticalDeriv=False, stepNDT=1e-9):
        if self.verbose:
            print('Computing derivatives...')
        # Function to compute the derivatives of a GW signal, both with JAX (automatic differentiation) and NumDiffTools (finite differences). It offers the possibility to compute directly the derivative of the complex signal (faster) and to compute the derivative of the real functions composing it. It is also possible to compute analytically the derivatives w.r.t. dL, theta, phi, psi, tcoal and Phicoal, and also iota in absence of HM
        
        if self.wf_model.is_newtonian:
            print('WARNING: In the Newtonian inspiral case the mass ratio and spins do not enter the waveform, and the corresponding Fisher matrix elements vanish, we then discard them.\n')
            #derivargs = (1,4,5,6,7,8)
            if computeAnalyticalDeriv:
                derivargs = (1)
                inputNumdL, inputNumiota = 1, 2
            else:
                derivargs = (1,3,4,5,6,7,8,9)
        elif self.wf_model.is_tidal:
            if computeAnalyticalDeriv:
                if not self.wf_model.is_HigherModes:
                    derivargs = (1,2,10,11,12,13)
                else:
                    derivargs = (1,2,6,10,11,12,13)
                inputNumdL, inputNumiota = 2, 3
            else:
                derivargs = (1,2,3,4,5,6,7,8,9,10,11,12,13)
        else:
            if computeAnalyticalDeriv:
                if not self.wf_model.is_HigherModes:
                    derivargs = (1,2,10,11)
                else:
                    derivargs = (1,2,6,10,11)
                inputNumdL, inputNumiota = 2, 3
            else:
                derivargs = (1,2,3,4,5,6,7,8,9,10,11)
                
        nParams = self.wf_model.nParams
        
        if not computeRealDeriv:
            if not computeDerivFinDiff:
                GWstrainUse = lambda f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda: self.GWstrain(f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                
                #dh =  vmap(jacrev(GWstrainUse, argnums=derivargs, holomorphic=True))
                
                FisherDerivs = np.asarray(vmap(jacrev(GWstrainUse, argnums=derivargs, holomorphic=True))(fgrids.T, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda))
            else:
                import numdifftools as nd
                if self.wf_model.is_newtonian:
                    if computeAnalyticalDeriv:
                        GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                        evpars = [Mc]
                    else:
                        GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], eta, pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], chiS, chiA, LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                        evpars = [Mc, dL, theta, phi, iota, psi, tcoal, Phicoal]
                elif self.wf_model.is_tidal:
                    if not computeAnalyticalDeriv:
                        GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], pars[10], pars[11], pars[12], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                        evpars = [Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda]
                    else:
                        if not self.wf_model.is_HigherModes:
                            GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, iota, psi, tcoal, Phicoal, pars[2], pars[3], pars[4], pars[5], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                            evpars = [Mc, eta, chiS, chiA, LambdaTilde, deltaLambda]
                        else:
                            GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, pars[2], psi, tcoal, Phicoal, pars[3], pars[4], pars[5], pars[6], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                            evpars = [Mc, eta, iota, chiS, chiA, LambdaTilde, deltaLambda]
                else:
                    if not computeAnalyticalDeriv:
                        GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], pars[10], LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                        evpars = [Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA]
                    else:
                        if not self.wf_model.is_HigherModes:
                            GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, iota, psi, tcoal, Phicoal, pars[2], pars[3], LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                            evpars = [Mc, eta, chiS, chiA]
                        else:
                            GWstrainUse = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, pars[2], psi, tcoal, Phicoal, pars[3], pars[4], LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2)
                            evpars = [Mc, eta, iota, chiS, chiA]
                dh = nd.Jacobian(GWstrainUse, step=stepNDT, method='central', order=2, n=1)
                FisherDerivs = np.asarray(dh(evpars))
                if len(Mc) == 1:
                    FisherDerivs = FisherDerivs[:,:,np.newaxis]
                FisherDerivs = FisherDerivs.transpose(1,2,0)
        else:
            if not computeDerivFinDiff:
                Apfun = lambda f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda: self.GWstrain(f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Ap')
                Acfun = lambda f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda: self.GWstrain(f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Ac')
                Psipfun = lambda f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda: self.GWstrain(f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Psip')
                Psicfun = lambda f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda: self.GWstrain(f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Psic')
                
                Ap = Apfun(fgrids, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda).T
                Ac = Acfun(fgrids, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda).T
                Psip = Psipfun(fgrids, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda).T
                Psic = Psicfun(fgrids, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda).T
                
                
                dAp = vmap(jacrev(Apfun, argnums=derivargs))
                dAc = vmap(jacrev(Acfun, argnums=derivargs))
                dPsip = vmap(jacrev(Psipfun, argnums=derivargs))
                dPsic = vmap(jacrev(Psicfun, argnums=derivargs))
               
                
                ApDeriv = np.asarray(dAp(fgrids.T, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda))
                AcDeriv = np.asarray(dAc(fgrids.T, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda))
                PsipDeriv = np.asarray(dPsip(fgrids.T, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda))
                PsicDeriv = np.asarray(dPsic(fgrids.T, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda))
                
                FisherDerivs = ApDeriv*np.exp(1j*Psip) + Ap*np.exp(1j*Psip)*1j*PsipDeriv + AcDeriv*np.exp(1j*Psic) + Ac*np.exp(1j*Psic)*1j*PsicDeriv
            else:
                import numdifftools as nd
                
                if self.wf_model.is_newtonian:
                    if not computeAnalyticalDeriv:
                        Apfun = lambda pars: self.GWstrain(fgrids, pars[0], eta, pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], chiS, chiA, LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Ap')
                        Acfun = lambda pars: self.GWstrain(fgrids, pars[0], eta, pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], chiS, chiA, LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Ac')
                        Psipfun = lambda pars: self.GWstrain(fgrids, pars[0], eta, pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], chiS, chiA, LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Psip')
                        Psicfun = lambda pars: self.GWstrain(fgrids, pars[0], eta, pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], chiS, chiA, LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Psic')
                        evpars = [Mc, dL, theta, phi, iota, psi, tcoal, Phicoal]
                    else:
                        Apfun = lambda pars: self.GWstrain(fgrids, pars[0], eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Ap')
                        Acfun = lambda pars: self.GWstrain(fgrids, pars[0], eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Ac')
                        Psipfun = lambda pars: self.GWstrain(fgrids, pars[0], eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Psip')
                        Psicfun = lambda pars: self.GWstrain(fgrids, pars[0], eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Psic')
                        evpars = [Mc]
                elif self.wf_model.is_tidal:
                    if not computeAnalyticalDeriv:
                        Apfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], pars[10], pars[11], pars[12], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Ap')
                        Acfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], pars[10], pars[11], pars[12], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Ac')
                        Psipfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], pars[10], pars[11], pars[12], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Psip')
                        Psicfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], pars[10], pars[11], pars[12], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Psic')
                        evpars = [Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda]
                    else:
                        if not self.wf_model.is_HigherModes:
                            Apfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, iota, psi, tcoal, Phicoal, pars[2], pars[3], pars[4], pars[5], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Ap')
                            Acfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, iota, psi, tcoal, Phicoal, pars[2], pars[3], pars[4], pars[5], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Ac')
                            Psipfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, iota, psi, tcoal, Phicoal, pars[2], pars[3], pars[4], pars[5], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Psip')
                            Psicfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, iota, psi, tcoal, Phicoal, pars[2], pars[3], pars[4], pars[5], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Psic')
                            evpars = [Mc, eta, chiS, chiA, LambdaTilde, deltaLambda]
                        else:
                            Apfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, pars[2], psi, tcoal, Phicoal, pars[3], pars[4], pars[5], pars[6], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Ap')
                            Acfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, pars[2], psi, tcoal, Phicoal, pars[3], pars[4], pars[5], pars[6], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Ac')
                            Psipfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, pars[2], psi, tcoal, Phicoal, pars[3], pars[4], pars[5], pars[6], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Psip')
                            Psicfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, pars[2], psi, tcoal, Phicoal, pars[3], pars[4], pars[5], pars[6], rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Psic')
                            evpars = [Mc, eta, iota, chiS, chiA, LambdaTilde, deltaLambda]
                else:
                    if not computeAnalyticalDeriv:
                        Apfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], pars[10], LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Ap')
                        Acfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], pars[10], LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Ac')
                        Psipfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], pars[10], LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Psip')
                        Psicfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], pars[10], LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Psic')
                        evpars = [Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA]
                    else:
                        if not self.wf_model.is_HigherModes:
                            Apfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, iota, psi, tcoal, Phicoal, pars[2], pars[3], LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Ap')
                            Acfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, iota, psi, tcoal, Phicoal, pars[2], pars[3], LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Ac')
                            Psipfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, iota, psi, tcoal, Phicoal, pars[2], pars[3], LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Psip')
                            Psicfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, iota, psi, tcoal, Phicoal, pars[2], pars[3], LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Psic')
                            evpars = [Mc, eta, chiS, chiA]
                        else:
                            Apfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, pars[2], psi, tcoal, Phicoal, pars[3], pars[4], LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Ap')
                            Acfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, pars[2], psi, tcoal, Phicoal, pars[3], pars[4], LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Ac')
                            Psipfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, pars[2], psi, tcoal, Phicoal, pars[3], pars[4], LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Psip')
                            Psicfun = lambda pars: self.GWstrain(fgrids, pars[0], pars[1], dL, theta, phi, pars[2], psi, tcoal, Phicoal, pars[3], pars[4], LambdaTilde, deltaLambda, rot=rot, is_m1m2=use_m1m2, is_chi1chi2=use_chi1chi2, return_single_comp='Psic')
                            evpars = [Mc, eta, iota, chiS, chiA]
                
                Ap = Apfun(evpars).T
                Ac = Acfun(evpars).T
                Psip = Psipfun(evpars).T
                Psic = Psicfun(evpars).T
                
                dAp = nd.Jacobian(Apfun, step=stepNDT, method='central', order=2, n=1)
                dAc = nd.Jacobian(Acfun, step=stepNDT, method='central', order=2, n=1)
                dPsip = nd.Jacobian(Psipfun, step=stepNDT, method='central', order=2, n=1)
                dPsic = nd.Jacobian(Psicfun, step=stepNDT, method='central', order=2, n=1)
                
                ApDeriv   = np.asarray(dAp(evpars))
                AcDeriv   = np.asarray(dAc(evpars))
                PsipDeriv = np.asarray(dPsip(evpars))
                PsicDeriv = np.asarray(dPsic(evpars))
                
                if len(Mc) == 1:
                    ApDeriv   = ApDeriv[:,:,np.newaxis]
                    AcDeriv   = AcDeriv[:,:,np.newaxis]
                    PsipDeriv = PsipDeriv[:,:,np.newaxis]
                    PsicDeriv = PsicDeriv[:,:,np.newaxis]
                    
                ApDeriv   = ApDeriv.transpose(1,2,0)
                AcDeriv   = AcDeriv.transpose(1,2,0)
                PsipDeriv = PsipDeriv.transpose(1,2,0)
                PsicDeriv = PsicDeriv.transpose(1,2,0)
                
                FisherDerivs = ApDeriv*np.exp(1j*Psip) + Ap*np.exp(1j*Psip)*1j*PsipDeriv + AcDeriv*np.exp(1j*Psic) + Ac*np.exp(1j*Psic)*1j*PsicDeriv
        if computeAnalyticalDeriv:
            # We compute the derivative w.r.t. dL, theta, phi, iota, psi, tcoal and Phicoal analytically, so have to split the matrix and insert them
            if not self.wf_model.is_HigherModes:
                NAnalyticalDerivs = 7
            else:
                NAnalyticalDerivs = 6
                
            dL_deriv, theta_deriv, phi_deriv, iota_deriv, psi_deriv, tc_deriv, Phicoal_deriv = self._AnalyticalDerivatives(fgrids, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda, rot=rot, use_m1m2=use_m1m2, use_chi1chi2=use_chi1chi2)
            if not self.wf_model.is_HigherModes:
                if not self.wf_model.is_newtonian:
                    tmpsplit1, tmpsplit2, _ = onp.vsplit(FisherDerivs, onp.array([inputNumdL, nParams-NAnalyticalDerivs]))
                    FisherDerivs = np.vstack((tmpsplit1, np.asarray(dL_deriv).T[np.newaxis,:], np.asarray(theta_deriv).T[np.newaxis,:], np.asarray(phi_deriv).T[np.newaxis,:], np.asarray(iota_deriv).T[np.newaxis,:], np.asarray(psi_deriv).T[np.newaxis,:], np.asarray(tc_deriv).T[np.newaxis,:], np.asarray(Phicoal_deriv).T[np.newaxis,:], tmpsplit2))
                else:
                    FisherDerivs = np.vstack((FisherDerivs[np.newaxis,:], np.asarray(dL_deriv).T[np.newaxis,:], np.asarray(theta_deriv).T[np.newaxis,:], np.asarray(phi_deriv).T[np.newaxis,:], np.asarray(iota_deriv).T[np.newaxis,:], np.asarray(psi_deriv).T[np.newaxis,:], np.asarray(tc_deriv).T[np.newaxis,:], np.asarray(Phicoal_deriv).T[np.newaxis,:]))
            else:
                tmpsplit1, tmpsplit2, tmpsplit3, _ = onp.vsplit(FisherDerivs, onp.array([inputNumdL, inputNumiota, nParams-NAnalyticalDerivs]))
                FisherDerivs = np.vstack((tmpsplit1, np.asarray(dL_deriv).T[np.newaxis,:], np.asarray(theta_deriv).T[np.newaxis,:], np.asarray(phi_deriv).T[np.newaxis,:], tmpsplit2, np.asarray(psi_deriv).T[np.newaxis,:], np.asarray(tc_deriv).T[np.newaxis,:], np.asarray(Phicoal_deriv).T[np.newaxis,:], tmpsplit3))
            
        # Change the units of the tcoal derivative from days to seconds (this improves conditioning)
        #tcelem = self.wf_model.ParNums['tcoal']
        #FisherDerivs.at[tcelem,:,:].set(FisherDerivs[tcelem,:,:]/(3600.*24.))
        
        #print('Shape of FisherDerivs: %s' %str(FisherDerivs.shape))
        
        return FisherDerivs
        
    def _AnalyticalDerivatives(self, f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA, LambdaTilde, deltaLambda, rot=0., use_m1m2=False, use_chi1chi2=False):
        
        if use_chi1chi2:
            # Interpret chiS as chi1z and chiA as chi2z
            chi1z = chiS
            chi2z = chiA
        else:
            chi1z = chiS + chiA
            chi2z = chiS - chiA
        if use_m1m2:
            # Interpret Mc as m1 and eta as m2
            McUse  = ((Mc*eta)**(3./5.))/((Mc+eta)**(1./5.))
            etaUse = (Mc*eta)/((Mc+eta)**(2.))
        else:
            McUse  = Mc
            etaUse = eta
        
        evParams = {'Mc':McUse, 'dL':dL, 'theta':theta, 'phi':phi, 'iota':iota, 'psi':psi, 'tcoal':tcoal, 'eta':etaUse, 'Phicoal':Phicoal, 'chi1z':chi1z, 'chi2z':chi2z}
        
        if self.wf_model.is_tidal:
            Lambda1, Lambda2 = utils.Lam12_from_Lamt_delLam(LambdaTilde, deltaLambda, etaUse)
            
            evParams['Lambda1'] = Lambda1
            evParams['Lambda2'] = Lambda2
        
        if not self.wf_model.is_HigherModes:
            wfPhiGw = self.wf_model.Phi(f, **evParams)
            wfAmpl  = self.wf_model.Ampl(f, **evParams)
            wfhp, wfhc = wfAmpl*np.exp(1j*wfPhiGw)*0.5*(1.+(np.cos(iota))**2), 1j*wfAmpl*np.exp(1j*wfPhiGw)*np.cos(iota)
        else:
            # If the waveform includes higher modes, it is not possible to compute amplitude and phase separately, make all together
            wfhp, wfhc = self.wf_model.hphc(f, **evParams)
        
        if self.useEarthMotion:
            # Compute Doppler contribution
            tnoloc = tcoal - self.wf_model.tau_star(f, **evParams)/(3600.*24.)
            tmpDeltLoc = self._DeltLoc(theta, phi, tnoloc, f) # in seconds
            t = tnoloc + tmpDeltLoc/(3600.*24.)
            phiD = self._phiDoppler(theta, phi, t, f)
            #phiP is necessary if we write the signal as A*exp(i Psi) with A = sqrt(Ap^2 + Ac^2), uncomment if necessary
            #phiP = self._phiPhase(theta, phi, t, iota, psi)
        else:
            phiD = Mc*0.
            #phiP = Mc*0.
            if self.noMotion:
                tnoloc=0
            else:
                tnoloc = tcoal
            tmpDeltLoc = self._DeltLoc(theta, phi, tnoloc, f) # in seconds
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
        
            phiD_phideriv = 2.*np.pi*f*(glob.REarth/glob.clight)*np.sin(theta)*np.sin(2.*np.pi*t - phi)*(1.-2.*np.pi*locDt_phider)
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
            phiD_thderiv = 2.*np.pi*f*(glob.REarth/glob.clight)*np.cos(theta)*np.cos(2.*np.pi*t - phi) - 2.*np.pi*f*(glob.REarth/glob.clight)*np.sin(theta)*np.sin(2.*np.pi*t - phi)*2.*np.pi*locDt_thder
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
            phiD_tcderiv = -4.*np.pi*np.pi*f*(glob.REarth/glob.clight)*np.sin(theta)*np.sin(2.*np.pi*t - phi)*(1.+locDt_tcder)
            phiL_tcderiv = 2.*np.pi*f*locDt_tcder*(3600.*24.)

            return ampP_tcder + 1j*(phiD_tcderiv + phiL_tcderiv + 2.*np.pi*f*3600.*24.)*hp + ampC_tcder + 1j*(phiD_tcderiv + phiL_tcderiv + 2.*np.pi*f*3600.*24.)*hc
        
        def iota_par_deriv():
            
            if not self.wf_model.is_HigherModes:
                wfhp_iotader, wfhc_iotader = -wfAmpl*np.exp(1j*wfPhiGw)*(np.cos(iota)*np.sin(iota)), -1j*wfAmpl*np.exp(1j*wfPhiGw)*np.sin(iota)
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
        
    def SNRFastInsp(self, evParams, checkInterp=False):
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
                SNR = fac * np.sqrt(Qsq*onp.interp(fcut, self.strainFreq[mask], self.strainInteg))
            elif self.detector_shape=='T':
                for i in range(3):
                    Fp, Fc = self._PatternFunction(theta, phi, t, psi, rot=60.*i)
                    Qsq = (Fp*0.5*(1.+(np.cos(iota))**2))**2 + (Fc*np.cos(iota))**2
                    tmpSNR = fac * np.sqrt(Qsq*onp.interp(fcut, self.strainFreq[mask], self.strainInteg))
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
                strainGrids = np.interp(fgrids, self.strainFreq, self.noiseCurve)
                
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
            
            
