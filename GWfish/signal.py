#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Enable 64bit on JAX, fundamental
from jax.config import config
config.update("jax_enable_x64", True)

# We use both the original numpy, denoted as onp, and the JAX implementation of numpy, denoted as np
import numpy as onp
#import numpy as np
import jax.numpy as np
from jax import vmap, jacrev #jacfwd
import time
import os
import h5py
import copy

import fisherUtils as utils
import fisherGlobals as glob
import fisherTools



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
                verbose=False,
                useEarthMotion = False,
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
        inj_params_init = {'Mc': np.array([77.23905294]),
                           'Phicoal': np.array([3.28297867]),
                           'chi1z': np.array([0.2018924]),
                           'chi2z': np.array([-0.68859213]),
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
        
        _ = self.SNRInteg(inj_params_init, res=10)
        #_ = self.FisherMatr(inj_params_init, res=10)
        print('Done.')
        
    def _update_seed(self,):
        onp.random.seed(None)
        self.seedUse = onp.random.randint(2**32 - 1, size=1)
        
    def _tabulateIntegrals(self, res=200, store=True, Mcmin=.9, Mcmax=9., etamin=.1):
        import scipy.integrate as igt
        
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
    
    def _phiLoc(self, theta, phi, t, f):
        
        
        ras, decs = self._ra_dec_from_th_phi(theta, phi)
        
        comp1 = np.cos(decs)*np.cos(ras)*np.cos(self.det_lat_rad)*np.cos(self.det_long_rad + 2.*np.pi*t)
        comp2 = np.cos(decs)*np.sin(ras)*np.cos(self.det_lat_rad)*np.sin(self.det_long_rad + 2.*np.pi*t)
        comp3 = np.sin(decs)*np.sin(self.det_lat_rad)
        
        phiL = -(2.*np.pi*f/glob.clight)*glob.REarth*(comp1+comp2+comp3)
        
        return phiL
    
    def GWAmplitudes(self, evParams, f, rot=0.):
        # evParams are all the parameters characterizing the event(s) under exam. It has to be a dictionary containing the entries: 
        # Mc -> chirp mass (Msun), dL -> luminosity distance (Gpc), theta & phi -> sky position (rad), iota -> inclination angle of orbital angular momentum to l.o.s toward the detector,
        # psi -> polarisation angle, tcoal -> time of coalescence as GMST (fraction of days), eta -> symmetric mass ratio, Phicoal -> GW frequency at coalescence.
        # chi1z, chi2z -> dimensionless spin components aligned to orbital angular momentum [-1;1], Lambda1,2 -> tidal parameters of the objects,
        # f is the frequency (Hz)
        
        #self._check_evparams(evParams)
        
        Mc, dL, theta, phi, iota, psi, tcoal, eta = evParams['Mc'], evParams['dL'], evParams['theta'], evParams['phi'], evParams['iota'], evParams['psi'], evParams['tcoal'], evParams['eta']
        
        if self.useEarthMotion:
            t = tcoal - self.wf_model.tau_star(f, **evParams)/(3600.*24)
        
        else:
            t = tcoal #- self.wf_model.tau_star(self.fmin, **evParams)/(3600.*24)
        
        wfAmpl = self.wf_model.Ampl(f, **evParams)
        Fp, Fc = self._PatternFunction(theta, phi, t, psi, rot=rot)
        
        if not self.wf_model.is_HigherModes:
            Ap = wfAmpl*Fp*0.5*(1.+(np.cos(iota))**2)
            Ac = wfAmpl*Fc*np.cos(iota)
        else:
            # If the waveform includes higher modes, it is not possible to compute amplitude and phase separately, make all together
            Phis = self.wf_model.Phi(f, **evParams)
            # Now make up the waveform adding the spherical harmonics
            hp, hc = utils.Add_Higher_Modes(wfAmpl, Phis, iota)
            
            Ap, Ac = abs(hp)*Fp, abs(hc)*Fc
        
        return Ap, Ac
    
    def GWPhase(self, evParams, f):
        # Phase of the GW signal
        Mc, eta, tcoal, Phicoal = evParams['Mc'], evParams['eta'], evParams['tcoal'], evParams['Phicoal']
        PhiGw = self.wf_model.Phi(f, **evParams)
        return 2.*np.pi*f*tcoal - Phicoal + PhiGw
        

    def GWstrain(self, f, Mc, dL, theta, phi, iota, psi, tcoal, eta, Phicoal, chiS, chiA, LambdaTilde, deltaLambda, rot=0.):

        # Full GW strain expression (complex)
        # Here we have the decompressed parameters and we put them back in a dictionary just to have an easier
        # implementation of the JAX module for derivatives
        chi1z = chiS + chiA
        chi2z = chiS - chiA
        evParams = {'Mc':Mc, 'dL':dL, 'theta':theta, 'phi':phi, 'iota':iota, 'psi':psi, 'tcoal':tcoal, 'eta':eta, 'Phicoal':Phicoal, 'chi1z':chi1z, 'chi2z':chi2z}
        
        if self.wf_model.is_tidal:
            Lambda1, Lambda2 = utils.Lam12_from_Lamt_delLam(LambdaTilde, deltaLambda, eta)
            
            evParams['Lambda1'] = Lambda1
            evParams['Lambda2'] = Lambda2
            
        if self.useEarthMotion:
            # Compute Doppler contribution
            t = tcoal - self.wf_model.tau_star(f, **evParams)/(3600.*24.)
            phiD = self._phiDoppler(theta, phi, t, f)
            #phiP is necessary if we write the signal as A*exp(i Psi) with A = sqrt(Ap^2 + Ac^2)
            phiP = self._phiPhase(theta, phi, t, iota, psi)
        else:
            phiD, phiP = Mc*0., Mc*0.
            t = tcoal
            
        phiL = self._phiLoc(theta, phi, t, f)
        
        if not self.wf_model.is_HigherModes:
            Ap, Ac = self.GWAmplitudes(evParams, f, rot=rot)
            Psi = self.GWPhase(evParams, f)
            Psi = Psi + phiD + phiL
        
            return (Ap + 1j*Ac)*np.exp(Psi*1j)
            #return np.sqrt(Ap*Ap + Ac*Ac)*np.exp(Psi*1j)
        else:
            # If the waveform includes higher modes, it is not possible to compute amplitude and phase separately, make all together
            wfAmpl = self.wf_model.Ampl(f, **evParams)
            Fp, Fc = self._PatternFunction(theta, phi, t, psi, rot=rot)
            Phis = self.wf_model.Phi(f, **evParams)
            # Now make up the waveform adding the spherical harmonics
            hp, hc = utils.Add_Higher_Modes(wfAmpl, Phis, iota)
            hp = hp*Fp*np.exp(1j*(phiD + phiL + 2.*np.pi*f*tcoal - Phicoal))
            hc = hc*Fc*np.exp(1j*(phiD + phiL + 2.*np.pi*f*tcoal - Phicoal))
            
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
    
    
    def FisherMatr(self, evParams, res=None, df=2**-4, spacing='geom'):
        
        if self.DutyFactor is not None:
            onp.random.seed(self.seedUse)
        
        utils.check_evparams(evParams)
  
        Mc, dL, theta, phi = evParams['Mc'].astype('complex128'), evParams['dL'].astype('complex128'), evParams['theta'].astype('complex128'), evParams['phi'].astype('complex128')
        iota, psi, tcoal, eta, Phicoal = evParams['iota'].astype('complex128'), evParams['psi'].astype('complex128'), evParams['tcoal'].astype('complex128'), evParams['eta'].astype('complex128'), evParams['Phicoal'].astype('complex128')
        chi1z, chi2z = evParams['chi1z'].astype('complex128'), evParams['chi2z'].astype('complex128')
        
        chiS, chiA = 0.5*(chi1z + chi2z), 0.5*(chi1z - chi2z)
        
        if self.wf_model.is_tidal:
            try:
                Lambda1, Lambda2 = evParams['Lambda1'].astype('complex128'), evParams['Lambda2'].astype('complex128')
            except KeyError:
                try:
                    Lambda1, Lambda2  = utils.Lam12_from_Lamt_delLam(evParams['LambdaTilde'].astype('complex128'), evParams['deltaLambda'].astype('complex128'), eta)
                except KeyError:
                    raise ValueError('Two among Lambda1, Lambda2 and LambdaTilde and deltaLambda have to be provided.')
            LambdaTilde, deltaLambda = utils.Lamt_delLam_from_Lam12(Lambda1, Lambda2, eta)
        else:
            Lambda1, Lambda2, LambdaTilde, deltaLambda = np.zeros(Mc.shape), np.zeros(Mc.shape), np.zeros(Mc.shape), np.zeros(Mc.shape)
            
        fcut = self.wf_model.fcut(**evParams)
        
        if self.fmax is not None:
            fcut = np.where(fcut > self.fmax, self.fmax, fcut)
        
        fminarr = np.full(fcut.shape, self.fmin)
        if res is None and df is not None:
            res = np.floor( np.real((1+(fcut-fminarr)/df)))
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

        if self.wf_model.is_newtonian:
            print('WARNING: In the Newtonian inspiral case the mass ratio and spins do not enter the waveform, and the corresponding Fisher matrix elements vanish, we then discard them.\n')

            #derivargs = (1,2,3,4,5,6,7,9)
            derivargs = (1,3,4,5,6,7)
            #nParams = 8
        elif self.wf_model.is_tidal:
            derivargs = (1,3,4,5,6,7,8,10,11,12,13)
            #nParams = 13
        else:
            derivargs = (1,3,4,5,6,7,8,10,11)
            #nParams = 11
        nParams = self.wf_model.nParams
        
        if self.detector_shape=='L': 
            #Build gradient
            dh = vmap(jacrev(self.GWstrain, argnums=derivargs, holomorphic=True))
            
            FisherDerivs = np.asarray(dh(fgrids.T, Mc, dL, theta, phi, iota, psi, tcoal, eta, Phicoal, chiS, chiA, LambdaTilde, deltaLambda))

            # We compute the derivative w.r.t. logdL and Phicoal analytically, so split the matrix and insert them
            tmpsplit1, tmpsplit2, _ = np.vsplit(FisherDerivs, np.array([1, nParams-2]))
            logdLderiv = -onp.asarray(self.GWstrain(fgrids, Mc, dL, theta, phi, iota, psi, tcoal, eta, Phicoal, chiS, chiA, LambdaTilde, deltaLambda)).T
            Phicoalderiv = logdLderiv*1j
            FisherDerivs = np.vstack((tmpsplit1, logdLderiv[onp.newaxis,:], tmpsplit2, Phicoalderiv[onp.newaxis,:]))
            
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
                    # Change rot

                    GWstrainRot = lambda f, Mc, dL, theta, phi, iota, psi, tcoal, eta, Phicoal, chiS, chiA, LambdaTilde, deltaLambda: self.GWstrain(f, Mc, dL, theta, phi, iota, psi, tcoal, eta, Phicoal, chiS, chiA, LambdaTilde, deltaLambda, rot=i*60.)
                    
                    # Build gradient
                    dh = vmap(jacrev(GWstrainRot, argnums=derivargs, holomorphic=True))
                
                    FisherDerivs = onp.asarray(dh(fgrids.T, Mc, dL, theta, phi, iota, psi, tcoal, eta, Phicoal, chiS, chiA, LambdaTilde, deltaLambda))

                    # We compute the derivative w.r.t. logdL and Phicoal analytically, so split the matrix and insert them
                    tmpsplit1, tmpsplit2, _ = onp.vsplit(FisherDerivs, np.array([1, nParams-2]))
                    logdLderiv = -onp.asarray(GWstrainRot(fgrids, Mc, dL, theta, phi, iota, psi, tcoal, eta, Phicoal, chiS, chiA, LambdaTilde, deltaLambda)).T
                    Phicoalderiv = logdLderiv*1j
                    FisherDerivs = onp.vstack((tmpsplit1, logdLderiv[onp.newaxis,:], tmpsplit2, Phicoalderiv[onp.newaxis,:]))
                    
                    FisherIntegrands = (onp.conjugate(FisherDerivs[:,:,onp.newaxis,:])*FisherDerivs.transpose(1,0,2))
                    
                    tmpFisher = onp.zeros((nParams,nParams,len(Mc)))
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
            else:
            # The signal in 3 arms sums to zero for geometrical reasons, so we can use this to skip some calculations
            
                # Build gradient
                dh = vmap(jacrev(self.GWstrain, argnums=derivargs, holomorphic=True))
                
                FisherDerivs = onp.asarray(dh(fgrids.T, Mc, dL, theta, phi, iota, psi, tcoal, eta, Phicoal, chiS, chiA, LambdaTilde, deltaLambda))

                # We compute the derivative w.r.t. logdL and Phicoal analytically, so split the matrix and insert them
                tmpsplit1, tmpsplit2, _ = onp.vsplit(FisherDerivs, np.array([1, nParams-2]))
                logdLderiv = -onp.asarray(self.GWstrain(fgrids, Mc, dL, theta, phi, iota, psi, tcoal, eta, Phicoal, chiS, chiA, LambdaTilde, deltaLambda)).T
                Phicoalderiv = logdLderiv*1j
                
                FisherDerivs1 = onp.vstack((tmpsplit1, logdLderiv[onp.newaxis,:], tmpsplit2, Phicoalderiv[onp.newaxis,:]))
                    
                FisherIntegrands = (onp.conjugate(FisherDerivs1[:,:,onp.newaxis,:])*FisherDerivs1.transpose(1,0,2))
                    
                tmpFisher = onp.zeros((nParams,nParams,len(Mc)))
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
                
                GWstrainRot = lambda f, Mc, dL, theta, phi, iota, psi, tcoal, eta, Phicoal, chiS, chiA, LambdaTilde, deltaLambda: self.GWstrain(f, Mc, dL, theta, phi, iota, psi, tcoal, eta, Phicoal, chiS, chiA, LambdaTilde, deltaLambda, rot=60.)
                    
                # Build gradient
                dh = vmap(jacrev(GWstrainRot, argnums=derivargs, holomorphic=True))
                
                FisherDerivs = onp.asarray(dh(fgrids.T, Mc, dL, theta, phi, iota, psi, tcoal, eta, Phicoal, chiS, chiA, LambdaTilde, deltaLambda))

                # We compute the derivative w.r.t. logdL and Phicoal analytically, so split the matrix and insert them
                tmpsplit1, tmpsplit2, _ = onp.vsplit(FisherDerivs, np.array([1, nParams-2]))
                logdLderiv = -onp.asarray(GWstrainRot(fgrids, Mc, dL, theta, phi, iota, psi, tcoal, eta, Phicoal, chiS, chiA, LambdaTilde, deltaLambda)).T
                Phicoalderiv = logdLderiv*1j
                
                FisherDerivs2 = onp.vstack((tmpsplit1, logdLderiv[onp.newaxis,:], tmpsplit2, Phicoalderiv[onp.newaxis,:]))
                    
                FisherIntegrands = (onp.conjugate(FisherDerivs2[:,:,onp.newaxis,:])*FisherDerivs2.transpose(1,0,2))
                    
                tmpFisher = onp.zeros((nParams,nParams,len(Mc)))
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
                
                FisherDerivs3 = - (FisherDerivs1 + FisherDerivs2)
                    
                FisherIntegrands = (onp.conjugate(FisherDerivs3[:,:,onp.newaxis,:])*FisherDerivs3.transpose(1,0,2))
                    
                tmpFisher = onp.zeros((nParams,nParams,len(Mc)))
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
            
        return Fisher
    
    
    

        

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
            
            
