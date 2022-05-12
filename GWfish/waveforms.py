#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import jax


from jax.config import config
config.update("jax_enable_x64", True)

# We use both the original numpy, denoted as onp, and the JAX implementation of numpy, denoted as np
import numpy as onp
#import numpy as np
import jax.numpy as np
from jax import custom_jvp

from abc import ABC, abstractmethod
import os
import sys
import h5py

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(SCRIPT_DIR)


import fisherGlobals as glob
import fisherUtils as utils

class WaveFormModel(ABC):
    '''
    Abstract class to compute waveforms
    '''
    
    def __init__(self, objType, fcutPar, is_newtonian=False, is_tidal=False, is_HigherModes=False, is_chi1chi2=False):
        # The kind of system the wf model is made for, can be 'BBH', 'BNS' or 'NSBH'
        self.objType = objType 
        # The cut frequency factor of the waveform, in Hz, to be divided by Mtot (in units of Msun). The method fcut can be redefined, as e.g. in the IMRPhenomD implementation, and fcutPar can be passed as an adimensional frequency (Mf)
        self.fcutPar = fcutPar
        # Note that the Fisher is computed for chiS and chiA, but the waveforms accept as input only chi1z and chi2z
        self.ParNums = {'Mc':0, 'eta':1, 'dL':2, 'theta':3, 'phi':4, 'iota':5, 'psi':6, 'tcoal':7, 'Phicoal':8, 'chiS':9,  'chiA':10}
        self.is_newtonian=is_newtonian
        self.is_tidal=is_tidal
        self.is_HigherModes = is_HigherModes
        self.nParams = 11
        self.is_chi1chi2 = is_chi1chi2
        
        if is_newtonian:
            # In the Newtonian case eta and the spins are not included in the Fisher, since they do not enter the signal
            self.ParNums = {'Mc':0, 'dL':1, 'theta':2, 'phi':3, 'iota':4, 'psi':5, 'tcoal':6, 'Phicoal':7}
            self.nParams = 8
        if is_tidal:
            # Note that the Fisher is computed for LabdaTilde and deltaLambda, but the waveforms accept as input only Lambda1 and Lambda2
            self.ParNums['LambdaTilde']=11
            self.ParNums['deltaLambda']=12
            self.nParams = 13
        if is_chi1chi2:
            self.ParNums['chi1z'] = self.ParNums['chiS']
            self.ParNums['chi2z'] = self.ParNums['chiA']
            self.ParNums.pop('chiS')
            self.ParNums.pop('chiA')
    @abstractmethod    
    def Phi(self, f, **kwargs): 
        # The frequency of the GW, as a function of frequency
        # With reference to the book M. Maggiore - Gravitational Waves Vol. 1, with Phi we mean only
        # the GW frequency, not the full phase of the signal, given by 
        # Psi+(f) = 2 pi f t_c - Phi0 - pi/4 - Phi(f)
        pass
    
    @abstractmethod
    def Ampl(self, f, **kwargs):
        # The amplitude of the signal as a function of frequency
        pass
        
    def tau_star(self, f, **kwargs):
        # The relation among the time to coalescence (in seconds) and the frequency (in Hz). We use as default 
        # the expression in M. Maggiore - Gravitational Waves Vol. 1 eq. (4.21), valid in Newtonian and restricted PN approximation
        return 2.18567 * ((1.21/kwargs['Mc'])**(5./3.)) * ((100/f)**(8./3.))
    
    def fcut(self, **kwargs):
        # The cut frequency of the waveform. In general this can be approximated as 2f_ISCO, but for complete waveforms
        # the expression is different
        return self.fcutPar/(kwargs['Mc']/(kwargs['eta']**(3./5.)))


class NewtInspiral(WaveFormModel):
    '''
    Leading order (inspiral only) waveform model
    '''
    
    def __init__(self, **kwargs):
        # Cut from M. Maggiore - Gravitational Waves Vol. 2 eq. (14.106)
        # From T. Dietrich et al. Phys. Rev. D 99, 024029, 2019, below eq. (4) (also look at Fig. 1) it seems be that, for BNS in the non-tidal case, the cut frequency should be lowered to (0.04/(2.*np.pi*glob.GMsun_over_c3))/Mtot.
        super().__init__('BBH', 1./(6.*np.pi*np.sqrt(6.)*glob.GMsun_over_c3), is_newtonian=True, **kwargs)
    
    def Phi(self, f, **kwargs):
        phase = 3.*0.25*(glob.GMsun_over_c3*kwargs['Mc']*8.*np.pi*f)**(-5./3.)
        return phase - np.pi*0.25
    
    def Ampl(self, f, **kwargs):
        
        amplitude = np.sqrt(5./24.) * (np.pi**(-2./3.)) * glob.clightGpc/kwargs['dL'] * (glob.GMsun_over_c3*kwargs['Mc'])**(5./6.) * (f**(-7./6.))
        return amplitude


class TaylorF2_RestrictedPN(WaveFormModel):
    '''
    TaylorF2 restricted PN waveform model
    '''
    
    # This waveform model is restricted PN (the amplitude stays as in Newtonian approximation) up to 3.5 PN
    def __init__(self, fHigh=None, is_tidal=False, use_3p5PN_SpinHO=False, phiref_vlso=False, **kwargs):
        
        if fHigh is None:
            fHigh = 1./(6.*np.pi*np.sqrt(6.)*glob.GMsun_over_c3) #Hz
        if is_tidal:
            objectT = 'BNS'
        else:
            objectT = 'BBH'
        self.use_3p5PN_SpinHO = use_3p5PN_SpinHO
        self.phiref_vlso = phiref_vlso
        super().__init__(objectT, fHigh, is_tidal=is_tidal, **kwargs)
    
    def Phi(self, f, **kwargs):
        # From A. Buonanno, B. Iyer, E. Ochsner, Y. Pan, B.S. Sathyaprakash - arXiv:0907.0700 - eq. (3.18) plus spins as in arXiv:1107.1267 eq. (5.3) up to 2.5PN and PhysRevD.93.084054 eq. (6) for 3PN and 3.5PN
        Mtot_sec = kwargs['Mc']*glob.GMsun_over_c3/(kwargs['eta']**(3./5.))
        v = (np.pi*Mtot_sec*f)**(1./3.)
        eta = kwargs['eta']
        eta2 = eta*eta
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        #Seta = np.sqrt(1.0 - 4.0*eta)
        
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        chi_s, chi_a   = 0.5*(chi1 + chi2), 0.5*(chi1 - chi2)
        chi_s2, chi_a2 = chi_s*chi_s, chi_a*chi_a
        chi_sdotchi_a  = chi_s*chi_a
        # flso = 1/6^(3/2)/(pi*M) -> vlso = (pi*M*flso)^(1/3) = (1/6^(3/2))^(1/3)
        vlso = 1./np.sqrt(6.)
        
        TF2coeffs = {}
        TF2OverallAmpl = 3./(128. * eta)
        
        TF2coeffs['zero'] = 1.
        TF2coeffs['one'] = 0.
        TF2coeffs['two'] = 3715./756. + (55.*eta)/9.
        TF2coeffs['three'] = -16.*np.pi + (113.*Seta*chi_a)/3. + (113./3. - (76.*eta)/3.)*chi_s
        TF2coeffs['four'] = 15293365./508032. + (27145.*eta)/504.+ (3085.*eta2)/72. + (-405./8. + 200.*eta)*chi_a2 - (405.*Seta*chi_sdotchi_a)/4. + (-405./8. + (5.*eta)/2.)*chi_s2
        # This part is common to 5 and 5log, avoid recomputing
        TF2_5coeff_tmp = (732985./2268. - 24260.*eta/81. - 340.*eta2/9.)*chi_s + (732985./2268. + 140.*eta/9.)*Seta*chi_a
        if self.phiref_vlso:
            TF2coeffs['five'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)*(1.-3.*np.log(vlso))
            phiR = 0.
        else:
            TF2coeffs['five'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)
            # This pi factor is needed to include LAL fRef rescaling, so to end up with the exact same waveform
            phiR = np.pi
        TF2coeffs['five_log'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)*3.
        TF2coeffs['six'] = 11583231236531./4694215680. - 640./3.*np.pi**2 - 6848./21.*np.euler_gamma + eta*(-15737765635./3048192. + 2255./12.*np.pi**2) + eta2*76055./1728. - eta2*eta*127825./1296. - (6848./21.)*np.log(4.) + np.pi*(2270.*Seta*chi_a/3. + (2270./3. - 520.*eta)*chi_s) + (75515./144. - 8225.*eta/18.)*Seta*chi_sdotchi_a + (75515./288. - 263245.*eta/252. - 480.*eta2)*chi_a2 + (75515./288. - 232415.*eta/504. + 1255.*eta2/9.)*chi_s2
        TF2coeffs['six_log'] = -(6848./21.)
        if self.use_3p5PN_SpinHO:
        # This part includes SS and SSS contributions at 3.5PN, which are not included in LAL
            TF2coeffs['seven'] = 77096675.*np.pi/254016. + 378515.*np.pi*eta/1512.- 74045.*np.pi*eta2/756. + (-25150083775./3048192. + 10566655595.*eta/762048. - 1042165.*eta2/3024. + 5345.*eta2*eta/36. + (14585./8. - 7270.*eta + 80.*eta2)*chi_a2)*chi_s + (14585./24. - 475.*eta/6. + 100.*eta2/3.)*chi_s2*chi_s + Seta*((-25150083775./3048192. + 26804935.*eta/6048. - 1985.*eta2/48.)*chi_a + (14585./24. - 2380.*eta)*chi_a2*chi_a + (14585./8. - 215.*eta/2.)*chi_a*chi_s2)
        else:
            TF2coeffs['seven'] = 77096675.*np.pi/254016. + 378515.*np.pi*eta/1512.- 74045.*np.pi*eta2/756. + (-25150083775./3048192. + 10566655595.*eta/762048. - 1042165.*eta2/3024. + 5345.*eta2*eta/36.)*chi_s + Seta*((-25150083775./3048192. + 26804935.*eta/6048. - 1985.*eta2/48.)*chi_a)

        if self.is_tidal:
            # Add tidal contribution if needed, as in PhysRevD.89.103012
            Lambda1, Lambda2 = kwargs['Lambda1'], kwargs['Lambda2']
            Lam_t, delLam    = utils.Lamt_delLam_from_Lam12(Lambda1, Lambda2, eta)
            
            phi_Tidal = (-0.5*39.*Lam_t)*(v**10.) + (-3115./64.*Lam_t + 6595./364.*Seta*delLam)*(v**12.)
            
            phase = TF2OverallAmpl*(TF2coeffs['zero'] + TF2coeffs['one']*v + TF2coeffs['two']*v*v + TF2coeffs['three']*v**3 + TF2coeffs['four']*v**4 + (TF2coeffs['five'] + TF2coeffs['five_log']*np.log(v))*v**5 + (TF2coeffs['six'] + TF2coeffs['six_log']*np.log(v))*v**6 + TF2coeffs['seven']*v**7 + phi_Tidal)/(v**5.)
            
        else:
            phase = TF2OverallAmpl*(TF2coeffs['zero'] + TF2coeffs['one']*v + TF2coeffs['two']*v*v + TF2coeffs['three']*v**3 + TF2coeffs['four']*v**4 + (TF2coeffs['five'] + TF2coeffs['five_log']*np.log(v))*v**5 + (TF2coeffs['six'] + TF2coeffs['six_log']*np.log(v))*v**6 + TF2coeffs['seven']*v**7)/(v**5.)
        
        return phase + phiR - np.pi*0.25

    def Ampl(self, f, **kwargs):
        # In the restricted PN approach the amplitude is the same as for the Newtonian approximation, so this term is equivalent
        amplitude = np.sqrt(5./24.) * (np.pi**(-2./3.)) * glob.clightGpc/kwargs['dL'] * (glob.GMsun_over_c3*kwargs['Mc'])**(5./6.) * (f**(-7./6.))
        return amplitude
    
    def tau_star(self, f, **kwargs):
        # We use the expression in arXiv:0907.0700 eq. (3.8b)
        Mtot_sec = kwargs['Mc']*glob.GMsun_over_c3/(kwargs['eta']**(3./5.))
        v = (np.pi*Mtot_sec*f)**(1./3.)
        eta = kwargs['eta']
        eta2 = eta*eta
        
        OverallFac = 5./256 * Mtot_sec/(eta*(v**8.))
        
        t05 = 1. + (743./252. + 11./3.*eta)*(v*v) - 32./5.*np.pi*(v*v*v) + (3058673./508032. + 5429./504.*eta + 617./72.*eta2)*(v**4) - (7729./252. - 13./3.*eta)*np.pi*(v**5)
        t6  = (- 10052469856691./23471078400. + 128./3.*np.pi*np.pi + 6848./105.*np.euler_gamma + (3147553127./3048192. - 451./12.*np.pi*np.pi)*eta - 15211./1728.*eta2 + 25565./1296.*eta2*eta + 3424./105.*np.log(16.*v*v))*(v**6)
        t7  = (- 15419335./127008. - 75703./756.*eta + 14809./378.*eta2)*np.pi*(v**7)
        
        return OverallFac*(t05 + t6 + t7)

class IMRPhenomD(WaveFormModel):
    '''
    IMRPhenomD waveform model
    '''
    # All is taken from LALSimulation and arXiv:1508.07250, arXiv:1508.07253
    def __init__(self, **kwargs):
        # Dimensionless frequency (Mf) at which the inspiral amplitude switches to the intermediate amplitude
        self.AMP_fJoin_INS = 0.014
        # Dimensionless frequency (Mf) at which the inspiral phase switches to the intermediate phase
        self.PHI_fJoin_INS = 0.018
        # Dimensionless frequency (Mf) at which we define the end of the waveform
        fcutPar = 0.2
         
        super().__init__('BBH', fcutPar, **kwargs)
        
        self.QNMgrid_a     = onp.loadtxt(os.path.join(glob.WFfilesPath, 'QNMData_a.txt'))
        self.QNMgrid_fring = onp.loadtxt(os.path.join(glob.WFfilesPath, 'QNMData_fring.txt'))
        self.QNMgrid_fdamp = onp.loadtxt(os.path.join(glob.WFfilesPath, 'QNMData_fdamp.txt'))
        
    def Phi(self, f, **kwargs):
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        eta2 = eta*eta # These can speed up a bit, we call them multiple times
        etaInv = 1./eta
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        
        QuadMon1, QuadMon2 = np.ones(M.shape), np.ones(M.shape)
        
        chi12, chi22 = chi1*chi1, chi2*chi2
        chi1dotchi2  = chi1*chi2
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        
        SetaPlus1 = 1.0 + Seta
        chi_s = 0.5 * (chi1 + chi2)
        chi_a = 0.5 * (chi1 - chi2)
        q = 0.5*(1.0 + Seta - 2.0*eta)/eta
        chi_s2, chi_a2 = chi_s*chi_s, chi_a*chi_a
        chi1dotchi2    = chi1*chi2
        chi_sdotchi_a  = chi_s*chi_a
        # These are m1/Mtot and m2/Mtot
        m1ByM = 0.5 * (1.0 + Seta)
        m2ByM = 0.5 * (1.0 - Seta)
        # We work in dimensionless frequency M*f, not f
        fgrid = M*glob.GMsun_over_c3*f
        # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
        chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
        xi = - 1.0 + chiPN
        # Compute final spin and radiated energy
        aeff = self._finalspin(eta, chi1, chi2)
        Erad = self._radiatednrg(eta, chi1, chi2)
        # Compute ringdown and damping frequencies from interpolators
        fring = np.interp(aeff, self.QNMgrid_a, self.QNMgrid_fring) / (1.0 - Erad)
        fdamp = np.interp(aeff, self.QNMgrid_a, self.QNMgrid_fdamp) / (1.0 - Erad)
        
        # Compute sigma coefficients appearing in arXiv:1508.07253 eq. (28)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        sigma1 = 2096.551999295543 + 1463.7493168261553*eta + (1312.5493286098522 + 18307.330017082117*eta - 43534.1440746107*eta2 + (-833.2889543511114 + 32047.31997183187*eta - 108609.45037520859*eta2)*xi + (452.25136398112204 + 8353.439546391714*eta - 44531.3250037322*eta2)*xi*xi)*xi
        sigma2 = -10114.056472621156 - 44631.01109458185*eta + (-6541.308761668722 - 266959.23419307504*eta + 686328.3229317984*eta2 + (3405.6372187679685 - 437507.7208209015*eta + 1.6318171307344697e6*eta2)*xi + (-7462.648563007646 - 114585.25177153319*eta + 674402.4689098676*eta2)*xi*xi)*xi
        sigma3 = 22933.658273436497 + 230960.00814979506*eta + (14961.083974183695 + 1.1940181342318142e6*eta - 3.1042239693052764e6*eta2 + (-3038.166617199259 + 1.8720322849093592e6*eta - 7.309145012085539e6*eta2)*xi + (42738.22871475411 + 467502.018616601*eta - 3.064853498512499e6*eta2)*xi*xi)*xi
        sigma4 = -14621.71522218357 - 377812.8579387104*eta + (-9608.682631509726 - 1.7108925257214056e6*eta + 4.332924601416521e6*eta2 + (-22366.683262266528 - 2.5019716386377467e6*eta + 1.0274495902259542e7*eta2)*xi + (-85360.30079034246 - 570025.3441737515*eta + 4.396844346849777e6*eta2)*xi*xi)*xi
        
        # Compute beta coefficients appearing in arXiv:1508.07253 eq. (16)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        beta1 = 97.89747327985583 - 42.659730877489224*eta + (153.48421037904913 - 1417.0620760768954*eta + 2752.8614143665027*eta2 + (138.7406469558649 - 1433.6585075135881*eta + 2857.7418952430758*eta2)*xi + (41.025109467376126 - 423.680737974639*eta + 850.3594335657173*eta2)*xi*xi)*xi
        beta2 = -3.282701958759534 - 9.051384468245866*eta + (-12.415449742258042 + 55.4716447709787*eta - 106.05109938966335*eta2 + (-11.953044553690658 + 76.80704618365418*eta - 155.33172948098394*eta2)*xi + (-3.4129261592393263 + 25.572377569952536*eta - 54.408036707740465*eta2)*xi*xi)*xi
        beta3 = -0.000025156429818799565 + 0.000019750256942201327*eta + (-0.000018370671469295915 + 0.000021886317041311973*eta + 0.00008250240316860033*eta2 + (7.157371250566708e-6 - 0.000055780000112270685*eta + 0.00019142082884072178*eta2)*xi + (5.447166261464217e-6 - 0.00003220610095021982*eta + 0.00007974016714984341*eta2)*xi*xi)*xi
        
        # Compute alpha coefficients appearing in arXiv:1508.07253 eq. (14)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        alpha1 = 43.31514709695348 + 638.6332679188081*eta + (-32.85768747216059 + 2415.8938269370315*eta - 5766.875169379177*eta2 + (-61.85459307173841 + 2953.967762459948*eta - 8986.29057591497*eta2)*xi + (-21.571435779762044 + 981.2158224673428*eta - 3239.5664895930286*eta2)*xi*xi)*xi
        alpha2 = -0.07020209449091723 - 0.16269798450687084*eta + (-0.1872514685185499 + 1.138313650449945*eta - 2.8334196304430046*eta2 + (-0.17137955686840617 + 1.7197549338119527*eta - 4.539717148261272*eta2)*xi + (-0.049983437357548705 + 0.6062072055948309*eta - 1.682769616644546*eta2)*xi*xi)*xi
        alpha3 = 9.5988072383479 - 397.05438595557433*eta + (16.202126189517813 - 1574.8286986717037*eta + 3600.3410843831093*eta2 + (27.092429659075467 - 1786.482357315139*eta + 5152.919378666511*eta2)*xi + (11.175710130033895 - 577.7999423177481*eta + 1808.730762932043*eta2)*xi*xi)*xi
        alpha4 = -0.02989487384493607 + 1.4022106448583738*eta + (-0.07356049468633846 + 0.8337006542278661*eta + 0.2240008282397391*eta2 + (-0.055202870001177226 + 0.5667186343606578*eta + 0.7186931973380503*eta2)*xi + (-0.015507437354325743 + 0.15750322779277187*eta + 0.21076815715176228*eta2)*xi*xi)*xi
        alpha5 = 0.9974408278363099 - 0.007884449714907203*eta + (-0.059046901195591035 + 1.3958712396764088*eta - 4.516631601676276*eta2 + (-0.05585343136869692 + 1.7516580039343603*eta - 5.990208965347804*eta2)*xi + (-0.017945336522161195 + 0.5965097794825992*eta - 2.0608879367971804*eta2)*xi*xi)*xi
        
        # Compute the TF2 phase coefficients and put them in a dictionary (spin effects are included up to 3.5PN)
        TF2coeffs = {}
        TF2OverallAmpl = 3./(128. * eta)
        
        TF2coeffs['zero'] = 1.
        TF2coeffs['one'] = 0.
        TF2coeffs['two'] = 3715./756. + (55.*eta)/9.
        TF2coeffs['three'] = -16.*np.pi + (113.*Seta*chi_a)/3. + (113./3. - (76.*eta)/3.)*chi_s
        # For 2PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['four'] = 5.*(3058.673/7.056 + 5429./7.*eta+617.*eta2)/72. + 247./4.8*eta*chi1dotchi2 -721./4.8*eta*chi1dotchi2 + (-720./9.6*QuadMon1 + 1./9.6)*m1ByM*m1ByM*chi12 + (-720./9.6*QuadMon2 + 1./9.6)*m2ByM*m2ByM*chi22 + (240./9.6*QuadMon1 - 7./9.6)*m1ByM*m1ByM*chi12 + (240./9.6*QuadMon2 - 7./9.6)*m2ByM*m2ByM*chi22
        # This part is common to 5 and 5log, avoid recomputing
        TF2_5coeff_tmp = (732985./2268. - 24260.*eta/81. - 340.*eta2/9.)*chi_s + (732985./2268. + 140.*eta/9.)*Seta*chi_a
        TF2coeffs['five'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)
        TF2coeffs['five_log'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)*3.
        # For 3PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['six'] = 11583.231236531/4.694215680 - 640./3.*np.pi*np.pi - 684.8/2.1*np.euler_gamma + eta*(-15737.765635/3.048192 + 225.5/1.2*np.pi*np.pi) + eta2*76.055/1.728 - eta2*eta*127.825/1.296 - np.log(4.)*684.8/2.1 + np.pi*chi1*m1ByM*(1490./3. + m1ByM*260.) + np.pi*chi2*m2ByM*(1490./3. + m2ByM*260.) + (326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + (4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM)*m1ByM*m1ByM*QuadMon1*chi12 + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM)*m1ByM*m1ByM*chi12 + (4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM)*m2ByM*m2ByM*QuadMon2*chi22 + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM)*m2ByM*m2ByM*chi22
        TF2coeffs['six_log'] = -6848./21.
        TF2coeffs['seven'] = 77096675.*np.pi/254016. + 378515.*np.pi*eta/1512.- 74045.*np.pi*eta2/756. + (-25150083775./3048192. + 10566655595.*eta/762048. - 1042165.*eta2/3024. + 5345.*eta2*eta/36.)*chi_s + Seta*((-25150083775./3048192. + 26804935.*eta/6048. - 1985.*eta2/48.)*chi_a)
        # Remove this part since it was not available when IMRPhenomD was tuned
        TF2coeffs['six'] = TF2coeffs['six'] - ((326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + ((4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM) + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM))*m1ByM*m1ByM*chi12 + ((4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM) + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM))*m2ByM*m2ByM*chi22)
        # Now translate into inspiral coefficients, label with the power in front of which they appear
        PhiInspcoeffs = {}
        
        PhiInspcoeffs['initial_phasing'] = TF2coeffs['five']*TF2OverallAmpl
        PhiInspcoeffs['two_thirds'] = TF2coeffs['seven']*TF2OverallAmpl*(np.pi**(2./3.))
        PhiInspcoeffs['third'] = TF2coeffs['six']*TF2OverallAmpl*(np.pi**(1./3.))
        PhiInspcoeffs['third_log'] = TF2coeffs['six_log']*TF2OverallAmpl*(np.pi**(1./3.))
        PhiInspcoeffs['log'] = TF2coeffs['five_log']*TF2OverallAmpl
        PhiInspcoeffs['min_third'] = TF2coeffs['four']*TF2OverallAmpl*(np.pi**(-1./3.))
        PhiInspcoeffs['min_two_thirds'] = TF2coeffs['three']*TF2OverallAmpl*(np.pi**(-2./3.))
        PhiInspcoeffs['min_one'] = TF2coeffs['two']*TF2OverallAmpl/np.pi
        PhiInspcoeffs['min_four_thirds'] = TF2coeffs['one']*TF2OverallAmpl*(np.pi**(-4./3.))
        PhiInspcoeffs['min_five_thirds'] = TF2coeffs['zero']*TF2OverallAmpl*(np.pi**(-5./3.))
        PhiInspcoeffs['one'] = sigma1
        PhiInspcoeffs['four_thirds'] = sigma2 * 0.75
        PhiInspcoeffs['five_thirds'] = sigma3 * 0.6
        PhiInspcoeffs['two'] = sigma4 * 0.5
        
        #Now compute the coefficients to align the three parts
        
        fInsJoin = self.PHI_fJoin_INS
        fMRDJoin = 0.5*fring
        
        # First the Inspiral - Intermediate: we compute C1Int and C2Int coeffs
        # Equations to solve for to get C(1) continuous join
        # PhiIns (f)  =   PhiInt (f) + C1Int + C2Int f
        # Joining at fInsJoin
        # PhiIns (fInsJoin)  =   PhiInt (fInsJoin) + C1Int + C2Int fInsJoin
        # PhiIns'(fInsJoin)  =   PhiInt'(fInsJoin) + C2Int
        # This is the first derivative wrt f of the inspiral phase computed at fInsJoin, first add the PN contribution and then the higher order calibrated terms
        DPhiIns = (2.0*TF2coeffs['seven']*TF2OverallAmpl*((np.pi*fInsJoin)**(7./3.)) + (TF2coeffs['six']*TF2OverallAmpl + TF2coeffs['six_log']*TF2OverallAmpl * (1.0 + np.log(np.pi*fInsJoin)/3.))*((np.pi*fInsJoin)**(2.)) + TF2coeffs['five_log']*TF2OverallAmpl*((np.pi*fInsJoin)**(5./3.)) - TF2coeffs['four']*TF2OverallAmpl*((np.pi*fInsJoin)**(4./3.)) - 2.*TF2coeffs['three']*TF2OverallAmpl*(np.pi*fInsJoin) - 3.*TF2coeffs['two']*TF2OverallAmpl*((np.pi*fInsJoin)**(2./3.)) - 4.*TF2coeffs['one']*TF2OverallAmpl*((np.pi*fInsJoin)**(1./3.)) - 5.*TF2coeffs['zero']*TF2OverallAmpl)*np.pi/(3.*((np.pi*fInsJoin)**(8./3.)))
        DPhiIns = DPhiIns + (sigma1 + sigma2*(fInsJoin**(1./3.)) + sigma3*(fInsJoin**(2./3.)) + sigma4*fInsJoin)/eta
        # This is the first derivative of the Intermediate phase computed at fInsJoin
        DPhiInt = (beta1 + beta3/(fInsJoin**4) + beta2/fInsJoin)/eta
        
        C2Int = DPhiIns - DPhiInt
        
        # This is the inspiral phase computed at fInsJoin
        PhiInsJoin = PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fInsJoin**(2./3.)) + PhiInspcoeffs['third']*(fInsJoin**(1./3.)) + PhiInspcoeffs['third_log']*(fInsJoin**(1./3.))*np.log(np.pi*fInsJoin)/3. + PhiInspcoeffs['log']*np.log(np.pi*fInsJoin)/3. + PhiInspcoeffs['min_third']*(fInsJoin**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fInsJoin**(-2./3.)) + PhiInspcoeffs['min_one']/fInsJoin + PhiInspcoeffs['min_four_thirds']*(fInsJoin**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fInsJoin**(-5./3.)) + (PhiInspcoeffs['one']*fInsJoin + PhiInspcoeffs['four_thirds']*(fInsJoin**(4./3.)) + PhiInspcoeffs['five_thirds']*(fInsJoin**(5./3.)) + PhiInspcoeffs['two']*fInsJoin*fInsJoin)/eta
        # This is the Intermediate phase computed at fInsJoin
        PhiIntJoin = beta1*fInsJoin - beta3/(3.*fInsJoin*fInsJoin*fInsJoin) + beta2*np.log(fInsJoin)
        
        C1Int = PhiInsJoin - PhiIntJoin/eta - C2Int*fInsJoin
        
        # Now the same for Intermediate - Merger-Ringdown: we also need a temporary Intermediate Phase function
        PhiIntTempVal  = (beta1*fMRDJoin - beta3/(3.*fMRDJoin*fMRDJoin*fMRDJoin) + beta2*np.log(fMRDJoin))/eta + C1Int + C2Int*fMRDJoin
        DPhiIntTempVal = C2Int + (beta1 + beta3/(fMRDJoin**4) + beta2/fMRDJoin)/eta
        DPhiMRDVal     = (alpha1 + alpha2/(fMRDJoin*fMRDJoin) + alpha3/(fMRDJoin**(1./4.)) + alpha4/(fdamp*(1. + (fMRDJoin - alpha5*fring)*(fMRDJoin - alpha5*fring)/(fdamp*fdamp))))/eta
        PhiMRJoinTemp  = -(alpha2/fMRDJoin) + (4.0/3.0) * (alpha3 * (fMRDJoin**(3./4.))) + alpha1 * fMRDJoin + alpha4 * np.arctan((fMRDJoin - alpha5 * fring)/fdamp)
        
        C2MRD = DPhiIntTempVal - DPhiMRDVal
        C1MRD = PhiIntTempVal - PhiMRJoinTemp/eta - C2MRD*fMRDJoin
        
        # Time shift so that peak amplitude is approximately at t=0
        gamma2 = 1.010344404799477 + 0.0008993122007234548*eta + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2 + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi*xi)*xi
        gamma3 = 1.3081615607036106 - 0.005537729694807678*eta +(-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2 + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi*xi)*xi
        fpeak = np.where(gamma2 >= 1.0, np.fabs(fring - (fdamp*gamma3)/gamma2), np.fabs(fring + (fdamp*(-1.0 + np.sqrt(1.0 - gamma2*gamma2))*gamma3)/gamma2))
        
        t0 = (alpha1 + alpha2/(fpeak*fpeak) + alpha3/(fpeak**(1./4.)) + alpha4/(fdamp*(1. + (fpeak - alpha5*fring)*(fpeak - alpha5*fring)/(fdamp*fdamp))))/eta
        
        # LAL sets fRef as the minimum frequency, do the same
        fRef   = np.amin(fgrid, axis=0)
        phiRef = np.where(fRef < self.PHI_fJoin_INS, PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fRef**(2./3.)) + PhiInspcoeffs['third']*(fRef**(1./3.)) + PhiInspcoeffs['third_log']*(fRef**(1./3.))*np.log(np.pi*fRef)/3. + PhiInspcoeffs['log']*np.log(np.pi*fRef)/3. + PhiInspcoeffs['min_third']*(fRef**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fRef**(-2./3.)) + PhiInspcoeffs['min_one']/fRef + PhiInspcoeffs['min_four_thirds']*(fRef**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fRef**(-5./3.)) + (PhiInspcoeffs['one']*fRef + PhiInspcoeffs['four_thirds']*(fRef**(4./3.)) + PhiInspcoeffs['five_thirds']*(fRef**(5./3.)) + PhiInspcoeffs['two']*fRef*fRef)/eta, np.where(fRef<fMRDJoin, (beta1*fRef - beta3/(3.*fRef*fRef*fRef) + beta2*np.log(fRef))/eta + C1Int + C2Int*fRef, np.where(fRef < self.fcutPar, (-(alpha2/fRef) + (4.0/3.0) * (alpha3 * (fRef**(3./4.))) + alpha1 * fRef + alpha4 * np.arctan((fRef - alpha5 * fring)/fdamp))/eta + C1MRD + C2MRD*fRef,0.)))

        phis = np.where(fgrid < self.PHI_fJoin_INS, PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fgrid**(2./3.)) + PhiInspcoeffs['third']*(fgrid**(1./3.)) + PhiInspcoeffs['third_log']*(fgrid**(1./3.))*np.log(np.pi*fgrid)/3. + PhiInspcoeffs['log']*np.log(np.pi*fgrid)/3. + PhiInspcoeffs['min_third']*(fgrid**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fgrid**(-2./3.)) + PhiInspcoeffs['min_one']/fgrid + PhiInspcoeffs['min_four_thirds']*(fgrid**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fgrid**(-5./3.)) + (PhiInspcoeffs['one']*fgrid + PhiInspcoeffs['four_thirds']*(fgrid**(4./3.)) + PhiInspcoeffs['five_thirds']*(fgrid**(5./3.)) + PhiInspcoeffs['two']*fgrid*fgrid)/eta, np.where(fgrid<fMRDJoin, (beta1*fgrid - beta3/(3.*fgrid*fgrid*fgrid) + beta2*np.log(fgrid))/eta + C1Int + C2Int*fgrid, np.where(fgrid < self.fcutPar, (-(alpha2/fgrid) + (4.0/3.0) * (alpha3 * (fgrid**(3./4.))) + alpha1 * fgrid + alpha4 * np.arctan((fgrid - alpha5 * fring)/fdamp))/eta + C1MRD + C2MRD*fgrid,0.)))
        
        return phis + np.where(fgrid < self.fcutPar, - t0*(fgrid - fRef) - phiRef, 0.)
        
    def Ampl(self, f, **kwargs):
        # Useful quantities
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        eta2 = eta*eta # This can speed up a bit, we call it multiple times
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        chi12, chi22 = chi1*chi1, chi2*chi2
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        
        SetaPlus1 = 1.0 + Seta
        chi_s     = 0.5 * (chi1 + chi2)
        chi_a     = 0.5 * (chi1 - chi2)
        # We work in dimensionless frequency M*f, not f
        fgrid = M*glob.GMsun_over_c3*f
        # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
        chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
        xi = -1.0 + chiPN
        # Compute final spin and radiated energy
        aeff = self._finalspin(eta, chi1, chi2)
        Erad = self._radiatednrg(eta, chi1, chi2)
        # Compute ringdown and damping frequencies from interpolators
        fring = np.interp(aeff, self.QNMgrid_a, self.QNMgrid_fring) / (1.0 - Erad)
        fdamp = np.interp(aeff, self.QNMgrid_a, self.QNMgrid_fdamp) / (1.0 - Erad)
        # Compute coefficients gamma appearing in arXiv:1508.07253 eq. (19), the numerical coefficients are in Tab. 5
        gamma1 = 0.006927402739328343 + 0.03020474290328911*eta + (0.006308024337706171 - 0.12074130661131138*eta + 0.26271598905781324*eta2 + (0.0034151773647198794 - 0.10779338611188374*eta + 0.27098966966891747*eta2)*xi+ (0.0007374185938559283 - 0.02749621038376281*eta + 0.0733150789135702*eta2)*xi*xi)*xi
        gamma2 = 1.010344404799477 + 0.0008993122007234548*eta + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2 + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi*xi)*xi
        gamma3 = 1.3081615607036106 - 0.005537729694807678*eta +(-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2 + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi*xi)*xi
        # Compute fpeak, from arXiv:1508.07253 eq. (20), we remove the square root term in case it is complex
        fpeak = np.where(gamma2 >= 1.0, np.fabs(fring - (fdamp*gamma3)/gamma2), fring + (fdamp*(-1.0 + np.sqrt(1.0 - gamma2*gamma2))*gamma3)/gamma2)
        # Compute coefficients rho appearing in arXiv:1508.07253 eq. (30), the numerical coefficients are in Tab. 5
        rho1 = 3931.8979897196696 - 17395.758706812805*eta + (3132.375545898835 + 343965.86092361377*eta - 1.2162565819981997e6*eta2 + (-70698.00600428853 + 1.383907177859705e6*eta - 3.9662761890979446e6*eta2)*xi + (-60017.52423652596 + 803515.1181825735*eta - 2.091710365941658e6*eta2)*xi*xi)*xi
        rho2 = -40105.47653771657 + 112253.0169706701*eta + (23561.696065836168 - 3.476180699403351e6*eta + 1.137593670849482e7*eta2 + (754313.1127166454 - 1.308476044625268e7*eta + 3.6444584853928134e7*eta2)*xi + (596226.612472288 - 7.4277901143564405e6*eta + 1.8928977514040343e7*eta2)*xi*xi)*xi
        rho3 = 83208.35471266537 - 191237.7264145924*eta + (-210916.2454782992 + 8.71797508352568e6*eta - 2.6914942420669552e7*eta2 + (-1.9889806527362722e6 + 3.0888029960154563e7*eta - 8.390870279256162e7*eta2)*xi + (-1.4535031953446497e6 + 1.7063528990822166e7*eta - 4.2748659731120914e7*eta2)*xi*xi)*xi
        # Compute coefficients delta appearing in arXiv:1508.07253 eq. (21)
        f1Interm = self.AMP_fJoin_INS
        f3Interm = fpeak
        dfInterm = 0.5*(f3Interm - f1Interm)
        f2Interm = f1Interm + dfInterm
        # First write the inspiral coefficients, we put them in a dictionary and label with the power in front of which they appear
        amp0 = np.sqrt(2.0*eta/3.0)*(np.pi**(-1./6.))
        Acoeffs = {}
        Acoeffs['two_thirds'] = ((-969. + 1804.*eta)*(np.pi**(2./3.)))/672.
        Acoeffs['one'] = ((chi1*(81.*SetaPlus1 - 44.*eta) + chi2*(81. - 81.*Seta - 44.*eta))*np.pi)/48.
        Acoeffs['four_thirds'] = ((-27312085.0 - 10287648.*chi22 - 10287648.*chi12*SetaPlus1 + 10287648.*chi22*Seta+ 24.*(-1975055. + 857304.*chi12 - 994896.*chi1*chi2 + 857304.*chi22)*eta+ 35371056*eta2)* (np.pi**(4./3.)))/8.128512e6
        Acoeffs['five_thirds'] = ((np.pi**(5./3.)) * (chi2*(-285197.*(-1. + Seta) + 4.*(-91902. + 1579.*Seta)*eta - 35632.*eta2) + chi1*(285197.*SetaPlus1 - 4.*(91902. + 1579.*Seta)*eta - 35632.*eta2) + 42840.*(-1.0 + 4.*eta)*np.pi)) / 32256.
        Acoeffs['two'] = - ((np.pi**2.)*(-336.*(-3248849057.0 + 2943675504.*chi12 - 3339284256.*chi1*chi2 + 2943675504.*chi22)*eta2 - 324322727232.*eta2*eta - 7.*(-177520268561. + 107414046432.*chi22 + 107414046432.*chi12*SetaPlus1 - 107414046432.*chi22*Seta + 11087290368.*(chi1 + chi2 + chi1*Seta - chi2*Seta)*np.pi ) + 12.*eta*(-545384828789. - 176491177632.*chi1*chi2 + 202603761360.*chi22 + 77616.*chi12*(2610335. + 995766.*Seta) - 77287373856.*chi22*Seta + 5841690624.*(chi1 + chi2)*np.pi + 21384760320.*np.pi*np.pi)))/6.0085960704e10
        Acoeffs['seven_thirds'] = rho1
        Acoeffs['eight_thirds'] = rho2
        Acoeffs['three'] = rho3
        # v1 is the inspiral model evaluated at f1Interm
        v1 = 1. + (f1Interm**(2./3.))*Acoeffs['two_thirds'] + (f1Interm**(4./3.)) * Acoeffs['four_thirds'] + (f1Interm**(5./3.)) *  Acoeffs['five_thirds'] + (f1Interm**(7./3.)) * Acoeffs['seven_thirds'] + (f1Interm**(8./3.)) * Acoeffs['eight_thirds'] + f1Interm * (Acoeffs['one'] + f1Interm * Acoeffs['two'] + f1Interm*f1Interm * Acoeffs['three'])
        # d1 is the derivative of the inspiral model evaluated at f1
        d1 = ((-969. + 1804.*eta)*(np.pi**(2./3.)))/(1008.*(f1Interm**(1./3.))) + ((chi1*(81.*SetaPlus1 - 44.*eta) + chi2*(81. - 81.*Seta - 44.*eta))*np.pi)/48. + ((-27312085. - 10287648.*chi22 - 10287648.*chi12*SetaPlus1 + 10287648.*chi22*Seta + 24.*(-1975055. + 857304.*chi12 - 994896.*chi1*chi2 + 857304.*chi22)*eta + 35371056.*eta2)*(f1Interm**(1./3.))*(np.pi**(4./3.)))/6.096384e6 + (5.*(f1Interm**(2./3.))*(np.pi**(5./3.))*(chi2*(-285197.*(-1 + Seta)+ 4.*(-91902. + 1579.*Seta)*eta - 35632.*eta2) + chi1*(285197.*SetaPlus1- 4.*(91902. + 1579.*Seta)*eta - 35632.*eta2) + 42840.*(-1 + 4*eta)*np.pi))/96768.- (f1Interm*np.pi*np.pi*(-336.*(-3248849057.0 + 2943675504.*chi12 - 3339284256.*chi1*chi2 + 2943675504.*chi22)*eta2 - 324322727232.*eta2*eta - 7.*(-177520268561. + 107414046432.*chi22 + 107414046432.*chi12*SetaPlus1 - 107414046432.*chi22*Seta+ 11087290368*(chi1 + chi2 + chi1*Seta - chi2*Seta)*np.pi)+ 12.*eta*(-545384828789.0 - 176491177632.*chi1*chi2 + 202603761360.*chi22 + 77616.*chi12*(2610335. + 995766.*Seta)- 77287373856.*chi22*Seta + 5841690624.*(chi1 + chi2)*np.pi + 21384760320*np.pi*np.pi)))/3.0042980352e10+ (7.0/3.0)*(f1Interm**(4./3.))*rho1 + (8.0/3.0)*(f1Interm**(5./3.))*rho2 + 3.*(f1Interm*f1Interm)*rho3
        # v3 is the merger-ringdown model (eq. (19) of arXiv:1508.07253) evaluated at f3
        v3 = np.exp(-(f3Interm - fring)*gamma2/(fdamp*gamma3))* (fdamp*gamma3*gamma1) / ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3)
        # d2 is the derivative of the merger-ringdown model evaluated at f3
        d2 = ((-2.*fdamp*(f3Interm - fring)*gamma3*gamma1) / ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3) - (gamma2*gamma1))/(np.exp((f3Interm - fring)*gamma2/(fdamp*gamma3)) * ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3))
        # v2 is the value of the amplitude evaluated at f2. They come from the fit of the collocation points in the intermediate region
        v2 = 0.8149838730507785 + 2.5747553517454658*eta + (1.1610198035496786 - 2.3627771785551537*eta + 6.771038707057573*eta2 + (0.7570782938606834 - 2.7256896890432474*eta + 7.1140380397149965*eta2)*xi + (0.1766934149293479 - 0.7978690983168183*eta + 2.1162391502005153*eta2)*xi*xi)*xi
        # Now some definitions to speed up
        f1  = f1Interm
        f2  = f2Interm
        f3  = f3Interm
        f12 = f1Interm*f1Interm
        f13 = f1Interm*f12;
        f14 = f1Interm*f13;
        f15 = f1Interm*f14;
        f22 = f2Interm*f2Interm;
        f23 = f2Interm*f22;
        f24 = f2Interm*f23;
        f32 = f3Interm*f3Interm;
        f33 = f3Interm*f32;
        f34 = f3Interm*f33;
        f35 = f3Interm*f34;
        # Finally conpute the deltas
        delta0 = -((d2*f15*f22*f3 - 2.*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32 - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33 + 2.*d1*f13*f22*f33 - 2.*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33 - d1*f13*f2*f34 - d1*f12*f22*f34 + 2.*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35 + 4.*f12*f23*f32*v1 - 3.*f1*f24*f32*v1 - 8.*f12*f22*f33*v1 + 4.*f1*f23*f33*v1 + f24*f33*v1 + 4.*f12*f2*f34*v1 + f1*f22*f34*v1 - 2.*f23*f34*v1 - 2.*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2 + 3.*f14*f33*v2 - 3.*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2.*f14*f23*v3 - f13*f24*v3 + 2.*f15*f2*f3*v3 - f14*f22*f3*v3 - 4.*f13*f23*f3*v3 + 3.*f12*f24*f3*v3 - 4.*f14*f2*f32*v3 + 8.*f13*f22*f32*v3 - 4.*f12*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta0 = -((d2*f15*f22*f3 - 2.*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32 - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33 + 2*d1*f13*f22*f33 - 2*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33 - d1*f13*f2*f34 - d1*f12*f22*f34 + 2*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35 + 4*f12*f23*f32*v1 - 3*f1*f24*f32*v1 - 8*f12*f22*f33*v1 + 4*f1*f23*f33*v1 + f24*f33*v1 + 4*f12*f2*f34*v1 + f1*f22*f34*v1 - 2*f23*f34*v1 - 2*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2 + 3*f14*f33*v2 - 3*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2*f14*f23*v3 - f13*f24*v3 + 2*f15*f2*f3*v3 - f14*f22*f3*v3 - 4*f13*f23*f3*v3 + 3*f12*f24*f3*v3 - 4*f14*f2*f32*v3 + 8*f13*f22*f32*v3 - 4*f12*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3-f2)*(f3-f2)))
        delta1 = -((-(d2*f15*f22) + 2.*d2*f14*f23 - d2*f13*f24 - d2*f14*f22*f3 + 2.*d1*f13*f23*f3 + 2.*d2*f13*f23*f3 - 2*d1*f12*f24*f3 - d2*f12*f24*f3 + d2*f15*f32 - 3*d1*f13*f22*f32 - d2*f13*f22*f32 + 2*d1*f12*f23*f32 - 2*d2*f12*f23*f32 + d1*f1*f24*f32 + 2*d2*f1*f24*f32 - d2*f14*f33 + d1*f12*f22*f33 + 3*d2*f12*f22*f33 - 2*d1*f1*f23*f33 - 2*d2*f1*f23*f33 + d1*f24*f33 + d1*f13*f34 + d1*f1*f22*f34 - 2*d1*f23*f34 - d1*f12*f35 + d1*f22*f35 - 8*f12*f23*f3*v1 + 6*f1*f24*f3*v1 + 12*f12*f22*f32*v1 - 8*f1*f23*f32*v1 - 4*f12*f34*v1 + 2*f1*f35*v1 + 2*f15*f3*v2 - 4*f14*f32*v2 + 4*f12*f34*v2 - 2*f1*f35*v2 - 2*f15*f3*v3 + 8*f12*f23*f3*v3 - 6*f1*f24*f3*v3 + 4*f14*f32*v3 - 12*f12*f22*f32*v3 + 8*f1*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta2 = -((d2*f15*f2 - d1*f13*f23 - 3*d2*f13*f23 + d1*f12*f24 + 2.*d2*f12*f24 - d2*f15*f3 + d2*f14*f2*f3 - d1*f12*f23*f3 + d2*f12*f23*f3 + d1*f1*f24*f3 - d2*f1*f24*f3 - d2*f14*f32 + 3*d1*f13*f2*f32 + d2*f13*f2*f32 - d1*f1*f23*f32 + d2*f1*f23*f32 - 2*d1*f24*f32 - d2*f24*f32 - 2*d1*f13*f33 + 2*d2*f13*f33 - d1*f12*f2*f33 - 3*d2*f12*f2*f33 + 3*d1*f23*f33 + d2*f23*f33 + d1*f12*f34 - d1*f1*f2*f34 + d1*f1*f35 - d1*f2*f35 + 4*f12*f23*v1 - 3*f1*f24*v1 + 4*f1*f23*f3*v1 - 3*f24*f3*v1 - 12*f12*f2*f32*v1 + 4*f23*f32*v1 + 8*f12*f33*v1 - f1*f34*v1 - f35*v1 - f15*v2 - f14*f3*v2 + 8*f13*f32*v2 - 8*f12*f33*v2 + f1*f34*v2 + f35*v2 + f15*v3 - 4*f12*f23*v3 + 3*f1*f24*v3 + f14*f3*v3 - 4*f1*f23*f3*v3 + 3*f24*f3*v3 - 8*f13*f32*v3 + 12*f12*f2*f32*v3 - 4*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta3 = -((-2.*d2*f14*f2 + d1*f13*f22 + 3*d2*f13*f22 - d1*f1*f24 - d2*f1*f24 + 2*d2*f14*f3 - 2.*d1*f13*f2*f3 - 2*d2*f13*f2*f3 + d1*f12*f22*f3 - d2*f12*f22*f3 + d1*f24*f3 + d2*f24*f3 + d1*f13*f32 - d2*f13*f32 - 2*d1*f12*f2*f32 + 2*d2*f12*f2*f32 + d1*f1*f22*f32 - d2*f1*f22*f32 + d1*f12*f33 - d2*f12*f33 + 2*d1*f1*f2*f33 + 2*d2*f1*f2*f33 - 3*d1*f22*f33 - d2*f22*f33 - 2*d1*f1*f34 + 2*d1*f2*f34 - 4*f12*f22*v1 + 2*f24*v1 + 8*f12*f2*f3*v1 - 4*f1*f22*f3*v1 - 4*f12*f32*v1 + 8*f1*f2*f32*v1 - 4*f22*f32*v1 - 4*f1*f33*v1 + 2*f34*v1 + 2*f14*v2 - 4*f13*f3*v2 + 4*f1*f33*v2 - 2*f34*v2 - 2*f14*v3 + 4*f12*f22*v3 - 2*f24*v3 + 4*f13*f3*v3 - 8*f12*f2*f3*v3 + 4*f1*f22*f3*v3 + 4*f12*f32*v3 - 8*f1*f2*f32*v3 + 4*f22*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta4 = -((d2*f13*f2 - d1*f12*f22 - 2*d2*f12*f22 + d1*f1*f23 + d2*f1*f23 - d2*f13*f3 + 2.*d1*f12*f2*f3 + d2*f12*f2*f3 - d1*f1*f22*f3 + d2*f1*f22*f3 - d1*f23*f3 - d2*f23*f3 - d1*f12*f32 + d2*f12*f32 - d1*f1*f2*f32 - 2*d2*f1*f2*f32 + 2*d1*f22*f32 + d2*f22*f32 + d1*f1*f33 - d1*f2*f33 + 3*f1*f22*v1 - 2*f23*v1 - 6*f1*f2*f3*v1 + 3*f22*f3*v1 + 3*f1*f32*v1 - f33*v1 - f13*v2 + 3*f12*f3*v2 - 3*f1*f32*v2 + f33*v2 + f13*v3 - 3*f1*f22*v3 + 2*f23*v3 - 3*f12*f3*v3 + 6*f1*f2*f3*v3 - 3*f22*f3*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        
        # Defined as in LALSimulation - LALSimIMRPhenomD.c line 332. Final units are correctly Hz^-1
        Overallamp = 2. * np.sqrt(5./(64.*np.pi)) * M * glob.GMsun_over_c2_Gpc * M * glob.GMsun_over_c3 / kwargs['dL']
        
        amplitudeIMR = np.where(fgrid < self.AMP_fJoin_INS, 1. + (fgrid**(2./3.))*Acoeffs['two_thirds'] + (fgrid**(4./3.)) * Acoeffs['four_thirds'] + (fgrid**(5./3.)) *  Acoeffs['five_thirds'] + (fgrid**(7./3.)) * Acoeffs['seven_thirds'] + (fgrid**(8./3.)) * Acoeffs['eight_thirds'] + fgrid * (Acoeffs['one'] + fgrid * Acoeffs['two'] + fgrid*fgrid * Acoeffs['three']), np.where(fgrid < fpeak, delta0 + fgrid*delta1 + fgrid*fgrid*(delta2 + fgrid*delta3 + fgrid*fgrid*delta4), np.where(fgrid < self.fcutPar,np.exp(-(fgrid - fring)*gamma2/(fdamp*gamma3))* (fdamp*gamma3*gamma1) / ((fgrid - fring)*(fgrid - fring) + fdamp*gamma3*fdamp*gamma3), 0.)))
        
        return Overallamp*amp0*(fgrid**(-7./6.))*amplitudeIMR
        
    def _finalspin(self, eta, chi1, chi2):
        # Compute the spin of the final object, as in LALSimIMRPhenomD_internals.c line 161 and 142,
        # which is taken from arXiv:1508.07250 eq. (3.6)
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        s = (m1*m1 * chi1 + m2*m2 * chi2)
        
        af1 = eta*(3.4641016151377544 - 4.399247300629289*eta + 9.397292189321194*eta*eta - 13.180949901606242*eta*eta*eta)
        af2 = eta*(s*((1.0/eta - 0.0850917821418767 - 5.837029316602263*eta) + (0.1014665242971878 - 2.0967746996832157*eta)*s))
        af3 = eta*(s*((-1.3546806617824356 + 4.108962025369336*eta)*s*s + (-0.8676969352555539 + 2.064046835273906*eta)*s*s*s))
        return af1 + af2 + af3
        
    def _radiatednrg(self, eta, chi1, chi2):
        # Predict the total radiated energy, from arXiv:1508.07250 eq (3.7) and (3.8)
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        s = (m1*m1 * chi1 + m2*m2 * chi2) / (m1*m1 + m2*m2)
        
        EradNS = eta * (0.055974469826360077 + 0.5809510763115132 * eta - 0.9606726679372312 * eta*eta + 3.352411249771192 * eta*eta*eta)
        
        return (EradNS * (1. + (-0.0030302335878845507 - 2.0066110851351073 * eta + 7.7050567802399215 * eta*eta) * s)) / (1. + (-0.6714403054720589 - 1.4756929437702908 * eta + 7.304676214885011 * eta*eta) * s)
    
    def tau_star(self, f, **kwargs):
        # For complex waveforms we use the expression in arXiv:0907.0700 eq. (3.8b)
        Mtot_sec = kwargs['Mc']*glob.GMsun_over_c3/(kwargs['eta']**(3./5.))
        v = (np.pi*Mtot_sec*f)**(1./3.)
        eta = kwargs['eta']
        eta2 = eta*eta
        
        OverallFac = 5./256 * Mtot_sec/(eta*(v**8.))
        
        t05 = 1. + (743./252. + 11./3.*eta)*(v*v) - 32./5.*np.pi*(v*v*v) + (3058673./508032. + 5429./504.*eta + 617./72.*eta2)*(v**4) - (7729./252. - 13./3.*eta)*np.pi*(v**5)
        t6  = (-10052469856691./23471078400. + 128./3.*np.pi*np.pi + 6848./105.*np.euler_gamma + (3147553127./3048192. - 451./12.*np.pi*np.pi)*eta - 15211./1728.*eta2 + 25565./1296.*eta2*eta + 3424./105.*np.log(16.*v*v))*(v**6)
        t7  = (- 15419335./127008. - 75703./756.*eta + 14809./378.*eta2)*np.pi*(v**7)
        
        return OverallFac*(t05 + t6 + t7)
    
    def fcut(self, **kwargs):
        
        return self.fcutPar/(kwargs['Mc']*glob.GMsun_over_c3/(kwargs['eta']**(3./5.)))
    
class IMRPhenomD_NRTidalv2(WaveFormModel):
    '''
    IMRPhenomD_NRTidal waveform model
    '''
    # All is taken from LALSimulation and arXiv:1508.07250, arXiv:1508.07253, arXiv:1905.06011
    def __init__(self, **kwargs):
        # Dimensionless frequency (Mf) at which the inspiral amplitude switches to the intermediate amplitude
        self.AMP_fJoin_INS = 0.014
        # Dimensionless frequency (Mf) at which the inspiral phase switches to the intermediate phase
        self.PHI_fJoin_INS = 0.018
        # Dimensionless frequency (Mf) at which we define the end of the waveform
        fcutPar = 0.2
         
        super().__init__('BNS', fcutPar, is_tidal=True, **kwargs)
        
        self.QNMgrid_a     = onp.loadtxt(os.path.join(glob.WFfilesPath, 'QNMData_a.txt'))
        self.QNMgrid_fring = onp.loadtxt(os.path.join(glob.WFfilesPath, 'QNMData_fring.txt'))
        self.QNMgrid_fdamp = onp.loadtxt(os.path.join(glob.WFfilesPath, 'QNMData_fdamp.txt'))
        
    def Phi(self, f, **kwargs):
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        eta2 = eta*eta # These can speed up a bit, we call them multiple times
        etaInv = 1./eta
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        if 'Lambda1' in kwargs:
            Lambda1, Lambda2 = kwargs['Lambda1'], kwargs['Lambda2']
            # A non-zero tidal deformability induces a quadrupole moment (for BBH it is 1).
            # The relation between the two is given in arxiv:1608.02582 eq. (15) with coefficients from third row of Table I
            # We also extend the range to 0 <= Lam < 1, as done in LALSimulation in LALSimUniversalRelations.c line 123
            QuadMon1 = np.where(Lambda1 < 1., 1. + Lambda1*(0.427688866723244 + Lambda1*(-0.324336526985068 + Lambda1*0.1107439432180572)), np.exp(0.1940 + 0.09163 * np.log(Lambda1) + 0.04812 * np.log(Lambda1) * np.log(Lambda1) -4.283e-3 * np.log(Lambda1) * np.log(Lambda1) * np.log(Lambda1) + 1.245e-4 * np.log(Lambda1) * np.log(Lambda1) * np.log(Lambda1) * np.log(Lambda1)))
            QuadMon2 = np.where(Lambda2 < 1., 1. + Lambda2*(0.427688866723244 + Lambda2*(-0.324336526985068 + Lambda2*0.1107439432180572)), np.exp(0.1940 + 0.09163 * np.log(Lambda2) + 0.04812 * np.log(Lambda2) * np.log(Lambda2) -4.283e-3 * np.log(Lambda2) * np.log(Lambda2) * np.log(Lambda2) + 1.245e-4 * np.log(Lambda2) * np.log(Lambda2) * np.log(Lambda2) * np.log(Lambda2)))
            
        else:
            Lambda1, Lambda2   = np.zeros(M.shape), np.zeros(M.shape)
            QuadMon1, QuadMon2 = np.ones(M.shape), np.ones(M.shape)
            
        chi12, chi22 = chi1*chi1, chi2*chi2
        chi1dotchi2  = chi1*chi2
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        SetaPlus1 = 1.0 + Seta
        chi_s = 0.5 * (chi1 + chi2)
        chi_a = 0.5 * (chi1 - chi2)
        chi_s2, chi_a2 = chi_s*chi_s, chi_a*chi_a
        chi_sdotchi_a  = chi_s*chi_a
        q = 0.5*(1.0 + Seta - 2.0*eta)/eta
        # These are m1/Mtot and m2/Mtot
        m1ByM = 0.5 * (1.0 + Seta)
        m2ByM = 0.5 * (1.0 - Seta)
        # We work in dimensionless frequency M*f, not f
        fgrid = M*glob.GMsun_over_c3*f
        # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
        chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
        xi = - 1.0 + chiPN
        # Compute final spin and radiated energy
        aeff = self._finalspin(eta, chi1, chi2)
        Erad = self._radiatednrg(eta, chi1, chi2)
        # Compute ringdown and damping frequencies from interpolators
        fring = np.interp(aeff, self.QNMgrid_a, self.QNMgrid_fring) / (1.0 - Erad)
        fdamp = np.interp(aeff, self.QNMgrid_a, self.QNMgrid_fdamp) / (1.0 - Erad)
        
        # Compute sigma coefficients appearing in arXiv:1508.07253 eq. (28)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        sigma1 = 2096.551999295543 + 1463.7493168261553*eta + (1312.5493286098522 + 18307.330017082117*eta - 43534.1440746107*eta2 + (-833.2889543511114 + 32047.31997183187*eta - 108609.45037520859*eta2)*xi + (452.25136398112204 + 8353.439546391714*eta - 44531.3250037322*eta2)*xi*xi)*xi
        sigma2 = -10114.056472621156 - 44631.01109458185*eta + (-6541.308761668722 - 266959.23419307504*eta + 686328.3229317984*eta2 + (3405.6372187679685 - 437507.7208209015*eta + 1.6318171307344697e6*eta2)*xi + (-7462.648563007646 - 114585.25177153319*eta + 674402.4689098676*eta2)*xi*xi)*xi
        sigma3 = 22933.658273436497 + 230960.00814979506*eta + (14961.083974183695 + 1.1940181342318142e6*eta - 3.1042239693052764e6*eta2 + (-3038.166617199259 + 1.8720322849093592e6*eta - 7.309145012085539e6*eta2)*xi + (42738.22871475411 + 467502.018616601*eta - 3.064853498512499e6*eta2)*xi*xi)*xi
        sigma4 = -14621.71522218357 - 377812.8579387104*eta + (-9608.682631509726 - 1.7108925257214056e6*eta + 4.332924601416521e6*eta2 + (-22366.683262266528 - 2.5019716386377467e6*eta + 1.0274495902259542e7*eta2)*xi + (-85360.30079034246 - 570025.3441737515*eta + 4.396844346849777e6*eta2)*xi*xi)*xi
        
        # Compute beta coefficients appearing in arXiv:1508.07253 eq. (16)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        beta1 = 97.89747327985583 - 42.659730877489224*eta + (153.48421037904913 - 1417.0620760768954*eta + 2752.8614143665027*eta2 + (138.7406469558649 - 1433.6585075135881*eta + 2857.7418952430758*eta2)*xi + (41.025109467376126 - 423.680737974639*eta + 850.3594335657173*eta2)*xi*xi)*xi
        beta2 = -3.282701958759534 - 9.051384468245866*eta + (-12.415449742258042 + 55.4716447709787*eta - 106.05109938966335*eta2 + (-11.953044553690658 + 76.80704618365418*eta - 155.33172948098394*eta2)*xi + (-3.4129261592393263 + 25.572377569952536*eta - 54.408036707740465*eta2)*xi*xi)*xi
        beta3 = -0.000025156429818799565 + 0.000019750256942201327*eta + (-0.000018370671469295915 + 0.000021886317041311973*eta + 0.00008250240316860033*eta2 + (7.157371250566708e-6 - 0.000055780000112270685*eta + 0.00019142082884072178*eta2)*xi + (5.447166261464217e-6 - 0.00003220610095021982*eta + 0.00007974016714984341*eta2)*xi*xi)*xi
        
        # Compute alpha coefficients appearing in arXiv:1508.07253 eq. (14)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        alpha1 = 43.31514709695348 + 638.6332679188081*eta + (-32.85768747216059 + 2415.8938269370315*eta - 5766.875169379177*eta2 + (-61.85459307173841 + 2953.967762459948*eta - 8986.29057591497*eta2)*xi + (-21.571435779762044 + 981.2158224673428*eta - 3239.5664895930286*eta2)*xi*xi)*xi
        alpha2 = -0.07020209449091723 - 0.16269798450687084*eta + (-0.1872514685185499 + 1.138313650449945*eta - 2.8334196304430046*eta2 + (-0.17137955686840617 + 1.7197549338119527*eta - 4.539717148261272*eta2)*xi + (-0.049983437357548705 + 0.6062072055948309*eta - 1.682769616644546*eta2)*xi*xi)*xi
        alpha3 = 9.5988072383479 - 397.05438595557433*eta + (16.202126189517813 - 1574.8286986717037*eta + 3600.3410843831093*eta2 + (27.092429659075467 - 1786.482357315139*eta + 5152.919378666511*eta2)*xi + (11.175710130033895 - 577.7999423177481*eta + 1808.730762932043*eta2)*xi*xi)*xi
        alpha4 = -0.02989487384493607 + 1.4022106448583738*eta + (-0.07356049468633846 + 0.8337006542278661*eta + 0.2240008282397391*eta2 + (-0.055202870001177226 + 0.5667186343606578*eta + 0.7186931973380503*eta2)*xi + (-0.015507437354325743 + 0.15750322779277187*eta + 0.21076815715176228*eta2)*xi*xi)*xi
        alpha5 = 0.9974408278363099 - 0.007884449714907203*eta + (-0.059046901195591035 + 1.3958712396764088*eta - 4.516631601676276*eta2 + (-0.05585343136869692 + 1.7516580039343603*eta - 5.990208965347804*eta2)*xi + (-0.017945336522161195 + 0.5965097794825992*eta - 2.0608879367971804*eta2)*xi*xi)*xi
        
        # Compute the TF2 phase coefficients and put them in a dictionary (spin effects are included up to 3.5 PN)
        # First the nonspinning part
        TF2coeffs = {}
        TF2OverallAmpl = 3./(128. * eta)
        
        TF2coeffs['zero'] = 1.
        TF2coeffs['one'] = 0.
        TF2coeffs['two'] = 3715./756. + (55.*eta)/9.
        TF2coeffs['three'] = -16.*np.pi + (113.*Seta*chi_a)/3. + (113./3. - (76.*eta)/3.)*chi_s
        # For 2PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['four'] = 5.*(3058.673/7.056 + 5429./7.*eta+617.*eta2)/72. + 247./4.8*eta*chi1dotchi2 -721./4.8*eta*chi1dotchi2 + (-720./9.6*QuadMon1 + 1./9.6)*m1ByM*m1ByM*chi12 + (-720./9.6*QuadMon2 + 1./9.6)*m2ByM*m2ByM*chi22 + (240./9.6*QuadMon1 - 7./9.6)*m1ByM*m1ByM*chi12 + (240./9.6*QuadMon2 - 7./9.6)*m2ByM*m2ByM*chi22
        # This part is common to 5 and 5log, avoid recomputing
        TF2_5coeff_tmp = (732985./2268. - 24260.*eta/81. - 340.*eta2/9.)*chi_s + (732985./2268. + 140.*eta/9.)*Seta*chi_a
        TF2coeffs['five'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)
        TF2coeffs['five_log'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)*3.
        # For 3PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['six'] = 11583.231236531/4.694215680 - 640./3.*np.pi*np.pi - 684.8/2.1*np.euler_gamma + eta*(-15737.765635/3.048192 + 225.5/1.2*np.pi*np.pi) + eta2*76.055/1.728 - eta2*eta*127.825/1.296 - np.log(4.)*684.8/2.1 + np.pi*chi1*m1ByM*(1490./3. + m1ByM*260.) + np.pi*chi2*m2ByM*(1490./3. + m2ByM*260.) + (326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + (4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM)*m1ByM*m1ByM*QuadMon1*chi12 + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM)*m1ByM*m1ByM*chi12 + (4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM)*m2ByM*m2ByM*QuadMon2*chi22 + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM)*m2ByM*m2ByM*chi22
        TF2coeffs['six_log'] = -6848./21.
        TF2coeffs['seven'] = 77096675.*np.pi/254016. + 378515.*np.pi*eta/1512.- 74045.*np.pi*eta2/756. + (-25150083775./3048192. + 10566655595.*eta/762048. - 1042165.*eta2/3024. + 5345.*eta2*eta/36.)*chi_s + Seta*((-25150083775./3048192. + 26804935.*eta/6048. - 1985.*eta2/48.)*chi_a)
        # Remove this part since it was not available when IMRPhenomD was tuned
        TF2coeffs['six'] = TF2coeffs['six'] - ((326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + ((4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM) + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM))*m1ByM*m1ByM*chi12 + ((4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM) + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM))*m2ByM*m2ByM*chi22)
        
        # Now translate into inspiral coefficients, label with the power in front of which they appear
        PhiInspcoeffs = {}
        
        PhiInspcoeffs['initial_phasing'] = TF2coeffs['five']*TF2OverallAmpl
        PhiInspcoeffs['two_thirds'] = TF2coeffs['seven']*TF2OverallAmpl*(np.pi**(2./3.))
        PhiInspcoeffs['third'] = TF2coeffs['six']*TF2OverallAmpl*(np.pi**(1./3.))
        PhiInspcoeffs['third_log'] = TF2coeffs['six_log']*TF2OverallAmpl*(np.pi**(1./3.))
        PhiInspcoeffs['log'] = TF2coeffs['five_log']*TF2OverallAmpl
        PhiInspcoeffs['min_third'] = TF2coeffs['four']*TF2OverallAmpl*(np.pi**(-1./3.))
        PhiInspcoeffs['min_two_thirds'] = TF2coeffs['three']*TF2OverallAmpl*(np.pi**(-2./3.))
        PhiInspcoeffs['min_one'] = TF2coeffs['two']*TF2OverallAmpl/np.pi
        PhiInspcoeffs['min_four_thirds'] = TF2coeffs['one']*TF2OverallAmpl*(np.pi**(-4./3.))
        PhiInspcoeffs['min_five_thirds'] = TF2coeffs['zero']*TF2OverallAmpl*(np.pi**(-5./3.))
        PhiInspcoeffs['one'] = sigma1
        PhiInspcoeffs['four_thirds'] = sigma2 * 0.75
        PhiInspcoeffs['five_thirds'] = sigma3 * 0.6
        PhiInspcoeffs['two'] = sigma4 * 0.5
        
        #Now compute the coefficients to align the three parts
        
        fInsJoin = self.PHI_fJoin_INS
        fMRDJoin = 0.5*fring
        # First the Inspiral - Intermediate: we compute C1Int and C2Int coeffs
        # Equations to solve for to get C(1) continuous join
        # PhiIns (f)  =   PhiInt (f) + C1Int + C2Int f
        # Joining at fInsJoin
        # PhiIns (fInsJoin)  =   PhiInt (fInsJoin) + C1Int + C2Int fInsJoin
        # PhiIns'(fInsJoin)  =   PhiInt'(fInsJoin) + C2Int
        # This is the first derivative wrt f of the inspiral phase computed at fInsJoin, first add the PN contribution and then the higher order calibrated terms
        DPhiIns = (2.0*TF2coeffs['seven']*TF2OverallAmpl*((np.pi*fInsJoin)**(7./3.)) + (TF2coeffs['six']*TF2OverallAmpl + TF2coeffs['six_log']*TF2OverallAmpl * (1.0 + np.log(np.pi*fInsJoin)/3.))*((np.pi*fInsJoin)**(2.)) + TF2coeffs['five_log']*TF2OverallAmpl*((np.pi*fInsJoin)**(5./3.)) - TF2coeffs['four']*TF2OverallAmpl*((np.pi*fInsJoin)**(4./3.)) - 2.*TF2coeffs['three']*TF2OverallAmpl*(np.pi*fInsJoin) - 3.*TF2coeffs['two']*TF2OverallAmpl*((np.pi*fInsJoin)**(2./3.)) - 4.*TF2coeffs['one']*TF2OverallAmpl*((np.pi*fInsJoin)**(1./3.)) - 5.*TF2coeffs['zero']*TF2OverallAmpl)*np.pi/(3.*((np.pi*fInsJoin)**(8./3.)))
        DPhiIns = DPhiIns + (sigma1 + sigma2*(fInsJoin**(1./3.)) + sigma3*(fInsJoin**(2./3.)) + sigma4*fInsJoin)/eta
        # This is the first derivative of the Intermediate phase computed at fInsJoin
        DPhiInt = (beta1 + beta3/(fInsJoin**4) + beta2/fInsJoin)/eta
        
        C2Int = DPhiIns - DPhiInt
        
        # This is the inspiral phase computed at fInsJoin
        PhiInsJoin = PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fInsJoin**(2./3.)) + PhiInspcoeffs['third']*(fInsJoin**(1./3.)) + PhiInspcoeffs['third_log']*(fInsJoin**(1./3.))*np.log(np.pi*fInsJoin)/3. + PhiInspcoeffs['log']*np.log(np.pi*fInsJoin)/3. + PhiInspcoeffs['min_third']*(fInsJoin**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fInsJoin**(-2./3.)) + PhiInspcoeffs['min_one']/fInsJoin + PhiInspcoeffs['min_four_thirds']*(fInsJoin**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fInsJoin**(-5./3.)) + (PhiInspcoeffs['one']*fInsJoin + PhiInspcoeffs['four_thirds']*(fInsJoin**(4./3.)) + PhiInspcoeffs['five_thirds']*(fInsJoin**(5./3.)) + PhiInspcoeffs['two']*fInsJoin*fInsJoin)/eta
        # This is the Intermediate phase computed at fInsJoin
        PhiIntJoin = beta1*fInsJoin - beta3/(3.*fInsJoin*fInsJoin*fInsJoin) + beta2*np.log(fInsJoin)
        
        C1Int = PhiInsJoin - PhiIntJoin/eta - C2Int*fInsJoin
        
        # Now the same for Intermediate - Merger-Ringdown: we also need a temporary Intermediate Phase function
        PhiIntTempVal  = (beta1*fMRDJoin - beta3/(3.*fMRDJoin*fMRDJoin*fMRDJoin) + beta2*np.log(fMRDJoin))/eta + C1Int + C2Int*fMRDJoin
        DPhiIntTempVal = C2Int + (beta1 + beta3/(fMRDJoin**4) + beta2/fMRDJoin)/eta
        DPhiMRDVal     = (alpha1 + alpha2/(fMRDJoin*fMRDJoin) + alpha3/(fMRDJoin**(1./4.)) + alpha4/(fdamp*(1. + (fMRDJoin - alpha5*fring)*(fMRDJoin - alpha5*fring)/(fdamp*fdamp))))/eta
        PhiMRJoinTemp  = -(alpha2/fMRDJoin) + (4.0/3.0) * (alpha3 * (fMRDJoin**(3./4.))) + alpha1 * fMRDJoin + alpha4 * np.arctan((fMRDJoin - alpha5 * fring)/fdamp)
        
        C2MRD = DPhiIntTempVal - DPhiMRDVal
        C1MRD = PhiIntTempVal - PhiMRJoinTemp/eta - C2MRD*fMRDJoin
        
        # Time shift so that peak amplitude is approximately at t=0
        gamma2 = 1.010344404799477 + 0.0008993122007234548*eta + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2 + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi*xi)*xi
        gamma3 = 1.3081615607036106 - 0.005537729694807678*eta +(-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2 + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi*xi)*xi
        fpeak = np.where(gamma2 >= 1.0, np.fabs(fring - (fdamp*gamma3)/gamma2), np.fabs(fring + (fdamp*(-1.0 + np.sqrt(1.0 - gamma2*gamma2))*gamma3)/gamma2))
        
        t0 = (alpha1 + alpha2/(fpeak*fpeak) + alpha3/(fpeak**(1./4.)) + alpha4/(fdamp*(1. + (fpeak - alpha5*fring)*(fpeak - alpha5*fring)/(fdamp*fdamp))))/eta
        
        # LAL sets fRef as the minimum frequency, do the same
        fRef   = np.amin(fgrid, axis=0)
        phiRef = np.where(fRef < self.PHI_fJoin_INS, PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fRef**(2./3.)) + PhiInspcoeffs['third']*(fRef**(1./3.)) + PhiInspcoeffs['third_log']*(fRef**(1./3.))*np.log(np.pi*fRef)/3. + PhiInspcoeffs['log']*np.log(np.pi*fRef)/3. + PhiInspcoeffs['min_third']*(fRef**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fRef**(-2./3.)) + PhiInspcoeffs['min_one']/fRef + PhiInspcoeffs['min_four_thirds']*(fRef**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fRef**(-5./3.)) + (PhiInspcoeffs['one']*fRef + PhiInspcoeffs['four_thirds']*(fRef**(4./3.)) + PhiInspcoeffs['five_thirds']*(fRef**(5./3.)) + PhiInspcoeffs['two']*fRef*fRef)/eta, np.where(fRef<fMRDJoin, (beta1*fRef - beta3/(3.*fRef*fRef*fRef) + beta2*np.log(fRef))/eta + C1Int + C2Int*fRef, np.where(fRef < self.fcutPar, (-(alpha2/fRef) + (4.0/3.0) * (alpha3 * (fRef**(3./4.))) + alpha1 * fRef + alpha4 * np.arctan((fRef - alpha5 * fring)/fdamp))/eta + C1MRD + C2MRD*fRef,0.)))
        
        phis = np.where(fgrid < self.PHI_fJoin_INS, PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fgrid**(2./3.)) + PhiInspcoeffs['third']*(fgrid**(1./3.)) + PhiInspcoeffs['third_log']*(fgrid**(1./3.))*np.log(np.pi*fgrid)/3. + PhiInspcoeffs['log']*np.log(np.pi*fgrid)/3. + PhiInspcoeffs['min_third']*(fgrid**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fgrid**(-2./3.)) + PhiInspcoeffs['min_one']/fgrid + PhiInspcoeffs['min_four_thirds']*(fgrid**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fgrid**(-5./3.)) + (PhiInspcoeffs['one']*fgrid + PhiInspcoeffs['four_thirds']*(fgrid**(4./3.)) + PhiInspcoeffs['five_thirds']*(fgrid**(5./3.)) + PhiInspcoeffs['two']*fgrid*fgrid)/eta, np.where(fgrid<fMRDJoin, (beta1*fgrid - beta3/(3.*fgrid*fgrid*fgrid) + beta2*np.log(fgrid))/eta + C1Int + C2Int*fgrid, np.where(fgrid < self.fcutPar, (-(alpha2/fgrid) + (4.0/3.0) * (alpha3 * (fgrid**(3./4.))) + alpha1 * fgrid + alpha4 * np.arctan((fgrid - alpha5 * fring)/fdamp))/eta + C1MRD + C2MRD*fgrid,0.)))
        
        # Add the tidal contribution to the phase, as in arXiv:1905.06011
        # Compute the tidal coupling constant, arXiv:1905.06011 eq. (8) using Lambda = 2/3 k_2/C^5 (eq. (10))

        kappa2T = (3.0/13.0) * ((1.0 + 12.0*m2ByM/m1ByM)*(m1ByM**5)*Lambda1 + (1.0 + 12.0*m1ByM/m2ByM)*(m2ByM**5)*Lambda2)
        
        c_Newt   = 2.4375
        n_1      = -12.615214237993088
        n_3over2 =  19.0537346970349
        n_2      = -21.166863146081035
        n_5over2 =  90.55082156324926
        n_3      = -60.25357801943598
        d_1      = -15.11120782773667
        d_3over2 =  22.195327350624694
        d_2      =   8.064109635305156
    
        numTidal = 1.0 + (n_1 * ((np.pi*fgrid)**(2./3.))) + (n_3over2 * np.pi*fgrid) + (n_2 * ((np.pi*fgrid)**(4./3.))) + (n_5over2 * ((np.pi*fgrid)**(5./3.))) + (n_3 * np.pi*fgrid*np.pi*fgrid)
        denTidal = 1.0 + (d_1 * ((np.pi*fgrid)**(2./3.))) + (d_3over2 * np.pi*fgrid) + (d_2 * ((np.pi*fgrid)**(4./3.)))
        
        tidal_phase = - kappa2T * c_Newt / (m1ByM * m2ByM) * ((np.pi*fgrid)**(5./3.)) * numTidal / denTidal
        
        # In the NRTidalv2 extension also 3.5PN spin-squared and 3.5PN spin-cubed terms are included, see eq. (27) of arXiv:1905.06011
        # This is needed to account for spin-induced quadrupole moments
        # Compute octupole moment and emove -1 to account for BBH baseline
        
        OctMon1 = - 1. + np.exp(0.003131 + 2.071 * np.log(QuadMon1)  - 0.7152 * np.log(QuadMon1) * np.log(QuadMon1) + 0.2458 * np.log(QuadMon1) * np.log(QuadMon1) * np.log(QuadMon1) - 0.03309 * np.log(QuadMon1) * np.log(QuadMon1) * np.log(QuadMon1) * np.log(QuadMon1))
        OctMon2 = - 1. + np.exp(0.003131 + 2.071 * np.log(QuadMon2)  - 0.7152 * np.log(QuadMon2) * np.log(QuadMon2) + 0.2458 * np.log(QuadMon2) * np.log(QuadMon2) * np.log(QuadMon2) - 0.03309 * np.log(QuadMon2) * np.log(QuadMon2) * np.log(QuadMon2) * np.log(QuadMon2))

        SS_3p5PN  = - 400.*np.pi*(QuadMon1-1.)*chi12*m1ByM*m1ByM - 400.*np.pi*(QuadMon2-1.)*chi22*m2ByM*m2ByM
        SSS_3p5PN = 10.*((m1ByM*m1ByM+308./3.*m1ByM)*chi1+(m2ByM*m2ByM-89./3.*m2ByM)*chi2)*(QuadMon1-1.)*m1ByM*m1ByM*chi12 + 10.*((m2ByM*m2ByM+308./3.*m2ByM)*chi2+(m1ByM*m1ByM-89./3.*m1ByM)*chi1)*(QuadMon2-1.)*m2ByM*m2ByM*chi22 - 440.*OctMon1*m1ByM*m1ByM*m1ByM*chi12*chi1 - 440.*OctMon2*m2ByM*m2ByM*m2ByM*chi22*chi2
        
        return phis + np.where(fgrid < self.fcutPar, - t0*(fgrid - fRef) - phiRef + tidal_phase + (SS_3p5PN + SSS_3p5PN)*TF2OverallAmpl*((np.pi*fgrid)**(2./3.)), 0.)
        
    def Ampl(self, f, **kwargs):
        # Useful quantities
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        eta2 = eta*eta # This can speed up a bit, we call it multiple times
        chi1, chi2   = kwargs['chi1z'], kwargs['chi2z']
        chi12, chi22 = chi1*chi1, chi2*chi2
        if 'Lambda1' in kwargs:
            Lambda1, Lambda2 = kwargs['Lambda1'], kwargs['Lambda2']
        else:
            Lambda1, Lambda2 = np.zeros(M.shape), np.zeros(M.shape)
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        SetaPlus1 = 1.0 + Seta
        q = 0.5*(1.0 + Seta - 2.0*eta)/eta
        chi_s = 0.5 * (chi1 + chi2)
        chi_a = 0.5 * (chi1 - chi2)
        # We work in dimensionless frequency M*f, not f
        fgrid = M*glob.GMsun_over_c3*f
        # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
        chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
        xi = -1.0 + chiPN
        # Compute final spin and radiated energy
        aeff = self._finalspin(eta, chi1, chi2)
        Erad = self._radiatednrg(eta, chi1, chi2)
        # Compute ringdown and damping frequencies from interpolators
        fring = np.interp(aeff, self.QNMgrid_a, self.QNMgrid_fring) / (1.0 - Erad)
        fdamp = np.interp(aeff, self.QNMgrid_a, self.QNMgrid_fdamp) / (1.0 - Erad)
        # Compute coefficients gamma appearing in arXiv:1508.07253 eq. (19), the numerical coefficients are in Tab. 5
        gamma1 = 0.006927402739328343 + 0.03020474290328911*eta + (0.006308024337706171 - 0.12074130661131138*eta + 0.26271598905781324*eta2 + (0.0034151773647198794 - 0.10779338611188374*eta + 0.27098966966891747*eta2)*xi+ (0.0007374185938559283 - 0.02749621038376281*eta + 0.0733150789135702*eta2)*xi*xi)*xi
        gamma2 = 1.010344404799477 + 0.0008993122007234548*eta + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2 + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi*xi)*xi
        gamma3 = 1.3081615607036106 - 0.005537729694807678*eta +(-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2 + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi*xi)*xi
        # Compute fpeak, from arXiv:1508.07253 eq. (20), we remove the square root term in case it is complex
        fpeak = np.where(gamma2 >= 1.0, np.fabs(fring - (fdamp*gamma3)/gamma2), fring + (fdamp*(-1.0 + np.sqrt(1.0 - gamma2*gamma2))*gamma3)/gamma2)
        # Compute coefficients rho appearing in arXiv:1508.07253 eq. (30), the numerical coefficients are in Tab. 5
        rho1 = 3931.8979897196696 - 17395.758706812805*eta + (3132.375545898835 + 343965.86092361377*eta - 1.2162565819981997e6*eta2 + (-70698.00600428853 + 1.383907177859705e6*eta - 3.9662761890979446e6*eta2)*xi + (-60017.52423652596 + 803515.1181825735*eta - 2.091710365941658e6*eta2)*xi*xi)*xi
        rho2 = -40105.47653771657 + 112253.0169706701*eta + (23561.696065836168 - 3.476180699403351e6*eta + 1.137593670849482e7*eta2 + (754313.1127166454 - 1.308476044625268e7*eta + 3.6444584853928134e7*eta2)*xi + (596226.612472288 - 7.4277901143564405e6*eta + 1.8928977514040343e7*eta2)*xi*xi)*xi
        rho3 = 83208.35471266537 - 191237.7264145924*eta + (-210916.2454782992 + 8.71797508352568e6*eta - 2.6914942420669552e7*eta2 + (-1.9889806527362722e6 + 3.0888029960154563e7*eta - 8.390870279256162e7*eta2)*xi + (-1.4535031953446497e6 + 1.7063528990822166e7*eta - 4.2748659731120914e7*eta2)*xi*xi)*xi
        # Compute coefficients delta appearing in arXiv:1508.07253 eq. (21)
        f1Interm = self.AMP_fJoin_INS
        f3Interm = fpeak
        dfInterm = 0.5*(f3Interm - f1Interm)
        f2Interm = f1Interm + dfInterm
        # First write the inspiral coefficients, we put them in a dictionary and label with the power in front of which they appear
        amp0 = np.sqrt(2.0*eta/3.0)*(np.pi**(-1./6.))
        Acoeffs = {}
        Acoeffs['two_thirds'] = ((-969. + 1804.*eta)*(np.pi**(2./3.)))/672.
        Acoeffs['one'] = ((chi1*(81.*SetaPlus1 - 44.*eta) + chi2*(81. - 81.*Seta - 44.*eta))*np.pi)/48.
        Acoeffs['four_thirds'] = ((-27312085.0 - 10287648.*chi22 - 10287648.*chi12*SetaPlus1 + 10287648.*chi22*Seta+ 24.*(-1975055. + 857304.*chi12 - 994896.*chi1*chi2 + 857304.*chi22)*eta+ 35371056*eta2)* (np.pi**(4./3.)))/8.128512e6
        Acoeffs['five_thirds'] = ((np.pi**(5./3.)) * (chi2*(-285197.*(-1. + Seta) + 4.*(-91902. + 1579.*Seta)*eta - 35632.*eta2) + chi1*(285197.*SetaPlus1 - 4.*(91902. + 1579.*Seta)*eta - 35632.*eta2) + 42840.*(-1.0 + 4.*eta)*np.pi)) / 32256.
        Acoeffs['two'] = - ((np.pi**2.)*(-336.*(-3248849057.0 + 2943675504.*chi12 - 3339284256.*chi1*chi2 + 2943675504.*chi22)*eta2 - 324322727232.*eta2*eta - 7.*(-177520268561. + 107414046432.*chi22 + 107414046432.*chi12*SetaPlus1 - 107414046432.*chi22*Seta + 11087290368.*(chi1 + chi2 + chi1*Seta - chi2*Seta)*np.pi ) + 12.*eta*(-545384828789. - 176491177632.*chi1*chi2 + 202603761360.*chi22 + 77616.*chi12*(2610335. + 995766.*Seta) - 77287373856.*chi22*Seta + 5841690624.*(chi1 + chi2)*np.pi + 21384760320.*np.pi*np.pi)))/6.0085960704e10
        Acoeffs['seven_thirds'] = rho1
        Acoeffs['eight_thirds'] = rho2
        Acoeffs['three'] = rho3
        # v1 is the inspiral model evaluated at f1Interm
        v1 = 1. + (f1Interm**(2./3.))*Acoeffs['two_thirds'] + (f1Interm**(4./3.)) * Acoeffs['four_thirds'] + (f1Interm**(5./3.)) *  Acoeffs['five_thirds'] + (f1Interm**(7./3.)) * Acoeffs['seven_thirds'] + (f1Interm**(8./3.)) * Acoeffs['eight_thirds'] + f1Interm * (Acoeffs['one'] + f1Interm * Acoeffs['two'] + f1Interm*f1Interm * Acoeffs['three'])
        # d1 is the derivative of the inspiral model evaluated at f1
        d1 = ((-969. + 1804.*eta)*(np.pi**(2./3.)))/(1008.*(f1Interm**(1./3.))) + ((chi1*(81.*SetaPlus1 - 44.*eta) + chi2*(81. - 81.*Seta - 44.*eta))*np.pi)/48. + ((-27312085. - 10287648.*chi22 - 10287648.*chi12*SetaPlus1 + 10287648.*chi22*Seta + 24.*(-1975055. + 857304.*chi12 - 994896.*chi1*chi2 + 857304.*chi22)*eta + 35371056.*eta2)*(f1Interm**(1./3.))*(np.pi**(4./3.)))/6.096384e6 + (5.*(f1Interm**(2./3.))*(np.pi**(5./3.))*(chi2*(-285197.*(-1 + Seta)+ 4.*(-91902. + 1579.*Seta)*eta - 35632.*eta2) + chi1*(285197.*SetaPlus1- 4.*(91902. + 1579.*Seta)*eta - 35632.*eta2) + 42840.*(-1 + 4*eta)*np.pi))/96768.- (f1Interm*np.pi*np.pi*(-336.*(-3248849057.0 + 2943675504.*chi12 - 3339284256.*chi1*chi2 + 2943675504.*chi22)*eta2 - 324322727232.*eta2*eta - 7.*(-177520268561. + 107414046432.*chi22 + 107414046432.*chi12*SetaPlus1 - 107414046432.*chi22*Seta+ 11087290368*(chi1 + chi2 + chi1*Seta - chi2*Seta)*np.pi)+ 12.*eta*(-545384828789.0 - 176491177632.*chi1*chi2 + 202603761360.*chi22 + 77616.*chi12*(2610335. + 995766.*Seta)- 77287373856.*chi22*Seta + 5841690624.*(chi1 + chi2)*np.pi + 21384760320*np.pi*np.pi)))/3.0042980352e10+ (7.0/3.0)*(f1Interm**(4./3.))*rho1 + (8.0/3.0)*(f1Interm**(5./3.))*rho2 + 3.*(f1Interm*f1Interm)*rho3
        # v3 is the merger-ringdown model (eq. (19) of arXiv:1508.07253) evaluated at f3
        v3 = np.exp(-(f3Interm - fring)*gamma2/(fdamp*gamma3))* (fdamp*gamma3*gamma1) / ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3)
        # d2 is the derivative of the merger-ringdown model evaluated at f3
        d2 = ((-2.*fdamp*(f3Interm - fring)*gamma3*gamma1) / ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3) - (gamma2*gamma1))/(np.exp((f3Interm - fring)*gamma2/(fdamp*gamma3)) * ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3))
        # v2 is the value of the amplitude evaluated at f2. They come from the fit of the collocation points in the intermediate region
        v2 = 0.8149838730507785 + 2.5747553517454658*eta + (1.1610198035496786 - 2.3627771785551537*eta + 6.771038707057573*eta2 + (0.7570782938606834 - 2.7256896890432474*eta + 7.1140380397149965*eta2)*xi + (0.1766934149293479 - 0.7978690983168183*eta + 2.1162391502005153*eta2)*xi*xi)*xi
        # Now some definitions to speed up
        f1  = f1Interm
        f2  = f2Interm
        f3  = f3Interm
        f12 = f1Interm*f1Interm
        f13 = f1Interm*f12;
        f14 = f1Interm*f13;
        f15 = f1Interm*f14;
        f22 = f2Interm*f2Interm;
        f23 = f2Interm*f22;
        f24 = f2Interm*f23;
        f32 = f3Interm*f3Interm;
        f33 = f3Interm*f32;
        f34 = f3Interm*f33;
        f35 = f3Interm*f34;
        # Finally conpute the deltas
        delta0 = -((d2*f15*f22*f3 - 2.*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32 - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33 + 2.*d1*f13*f22*f33 - 2.*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33 - d1*f13*f2*f34 - d1*f12*f22*f34 + 2.*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35 + 4.*f12*f23*f32*v1 - 3.*f1*f24*f32*v1 - 8.*f12*f22*f33*v1 + 4.*f1*f23*f33*v1 + f24*f33*v1 + 4.*f12*f2*f34*v1 + f1*f22*f34*v1 - 2.*f23*f34*v1 - 2.*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2 + 3.*f14*f33*v2 - 3.*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2.*f14*f23*v3 - f13*f24*v3 + 2.*f15*f2*f3*v3 - f14*f22*f3*v3 - 4.*f13*f23*f3*v3 + 3.*f12*f24*f3*v3 - 4.*f14*f2*f32*v3 + 8.*f13*f22*f32*v3 - 4.*f12*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta0 = -((d2*f15*f22*f3 - 2.*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32 - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33 + 2*d1*f13*f22*f33 - 2*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33 - d1*f13*f2*f34 - d1*f12*f22*f34 + 2*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35 + 4*f12*f23*f32*v1 - 3*f1*f24*f32*v1 - 8*f12*f22*f33*v1 + 4*f1*f23*f33*v1 + f24*f33*v1 + 4*f12*f2*f34*v1 + f1*f22*f34*v1 - 2*f23*f34*v1 - 2*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2 + 3*f14*f33*v2 - 3*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2*f14*f23*v3 - f13*f24*v3 + 2*f15*f2*f3*v3 - f14*f22*f3*v3 - 4*f13*f23*f3*v3 + 3*f12*f24*f3*v3 - 4*f14*f2*f32*v3 + 8*f13*f22*f32*v3 - 4*f12*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3-f2)*(f3-f2)))
        delta1 = -((-(d2*f15*f22) + 2.*d2*f14*f23 - d2*f13*f24 - d2*f14*f22*f3 + 2.*d1*f13*f23*f3 + 2.*d2*f13*f23*f3 - 2*d1*f12*f24*f3 - d2*f12*f24*f3 + d2*f15*f32 - 3*d1*f13*f22*f32 - d2*f13*f22*f32 + 2*d1*f12*f23*f32 - 2*d2*f12*f23*f32 + d1*f1*f24*f32 + 2*d2*f1*f24*f32 - d2*f14*f33 + d1*f12*f22*f33 + 3*d2*f12*f22*f33 - 2*d1*f1*f23*f33 - 2*d2*f1*f23*f33 + d1*f24*f33 + d1*f13*f34 + d1*f1*f22*f34 - 2*d1*f23*f34 - d1*f12*f35 + d1*f22*f35 - 8*f12*f23*f3*v1 + 6*f1*f24*f3*v1 + 12*f12*f22*f32*v1 - 8*f1*f23*f32*v1 - 4*f12*f34*v1 + 2*f1*f35*v1 + 2*f15*f3*v2 - 4*f14*f32*v2 + 4*f12*f34*v2 - 2*f1*f35*v2 - 2*f15*f3*v3 + 8*f12*f23*f3*v3 - 6*f1*f24*f3*v3 + 4*f14*f32*v3 - 12*f12*f22*f32*v3 + 8*f1*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta2 = -((d2*f15*f2 - d1*f13*f23 - 3*d2*f13*f23 + d1*f12*f24 + 2.*d2*f12*f24 - d2*f15*f3 + d2*f14*f2*f3 - d1*f12*f23*f3 + d2*f12*f23*f3 + d1*f1*f24*f3 - d2*f1*f24*f3 - d2*f14*f32 + 3*d1*f13*f2*f32 + d2*f13*f2*f32 - d1*f1*f23*f32 + d2*f1*f23*f32 - 2*d1*f24*f32 - d2*f24*f32 - 2*d1*f13*f33 + 2*d2*f13*f33 - d1*f12*f2*f33 - 3*d2*f12*f2*f33 + 3*d1*f23*f33 + d2*f23*f33 + d1*f12*f34 - d1*f1*f2*f34 + d1*f1*f35 - d1*f2*f35 + 4*f12*f23*v1 - 3*f1*f24*v1 + 4*f1*f23*f3*v1 - 3*f24*f3*v1 - 12*f12*f2*f32*v1 + 4*f23*f32*v1 + 8*f12*f33*v1 - f1*f34*v1 - f35*v1 - f15*v2 - f14*f3*v2 + 8*f13*f32*v2 - 8*f12*f33*v2 + f1*f34*v2 + f35*v2 + f15*v3 - 4*f12*f23*v3 + 3*f1*f24*v3 + f14*f3*v3 - 4*f1*f23*f3*v3 + 3*f24*f3*v3 - 8*f13*f32*v3 + 12*f12*f2*f32*v3 - 4*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta3 = -((-2.*d2*f14*f2 + d1*f13*f22 + 3*d2*f13*f22 - d1*f1*f24 - d2*f1*f24 + 2*d2*f14*f3 - 2.*d1*f13*f2*f3 - 2*d2*f13*f2*f3 + d1*f12*f22*f3 - d2*f12*f22*f3 + d1*f24*f3 + d2*f24*f3 + d1*f13*f32 - d2*f13*f32 - 2*d1*f12*f2*f32 + 2*d2*f12*f2*f32 + d1*f1*f22*f32 - d2*f1*f22*f32 + d1*f12*f33 - d2*f12*f33 + 2*d1*f1*f2*f33 + 2*d2*f1*f2*f33 - 3*d1*f22*f33 - d2*f22*f33 - 2*d1*f1*f34 + 2*d1*f2*f34 - 4*f12*f22*v1 + 2*f24*v1 + 8*f12*f2*f3*v1 - 4*f1*f22*f3*v1 - 4*f12*f32*v1 + 8*f1*f2*f32*v1 - 4*f22*f32*v1 - 4*f1*f33*v1 + 2*f34*v1 + 2*f14*v2 - 4*f13*f3*v2 + 4*f1*f33*v2 - 2*f34*v2 - 2*f14*v3 + 4*f12*f22*v3 - 2*f24*v3 + 4*f13*f3*v3 - 8*f12*f2*f3*v3 + 4*f1*f22*f3*v3 + 4*f12*f32*v3 - 8*f1*f2*f32*v3 + 4*f22*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta4 = -((d2*f13*f2 - d1*f12*f22 - 2*d2*f12*f22 + d1*f1*f23 + d2*f1*f23 - d2*f13*f3 + 2.*d1*f12*f2*f3 + d2*f12*f2*f3 - d1*f1*f22*f3 + d2*f1*f22*f3 - d1*f23*f3 - d2*f23*f3 - d1*f12*f32 + d2*f12*f32 - d1*f1*f2*f32 - 2*d2*f1*f2*f32 + 2*d1*f22*f32 + d2*f22*f32 + d1*f1*f33 - d1*f2*f33 + 3*f1*f22*v1 - 2*f23*v1 - 6*f1*f2*f3*v1 + 3*f22*f3*v1 + 3*f1*f32*v1 - f33*v1 - f13*v2 + 3*f12*f3*v2 - 3*f1*f32*v2 + f33*v2 + f13*v3 - 3*f1*f22*v3 + 2*f23*v3 - 3*f12*f3*v3 + 6*f1*f2*f3*v3 - 3*f22*f3*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        
        # Defined as in LALSimulation - LALSimIMRPhenomD.c line 332. Final units are correctly Hz^-1
        Overallamp = 2. * np.sqrt(5./(64.*np.pi)) * M * glob.GMsun_over_c2_Gpc * M * glob.GMsun_over_c3 / kwargs['dL']
        
        amplitudeIMR = np.where(fgrid < self.AMP_fJoin_INS, 1. + (fgrid**(2./3.))*Acoeffs['two_thirds'] + (fgrid**(4./3.)) * Acoeffs['four_thirds'] + (fgrid**(5./3.)) *  Acoeffs['five_thirds'] + (fgrid**(7./3.)) * Acoeffs['seven_thirds'] + (fgrid**(8./3.)) * Acoeffs['eight_thirds'] + fgrid * (Acoeffs['one'] + fgrid * Acoeffs['two'] + fgrid*fgrid * Acoeffs['three']), np.where(fgrid < fpeak, delta0 + fgrid*delta1 + fgrid*fgrid*(delta2 + fgrid*delta3 + fgrid*fgrid*delta4), np.where(fgrid < self.fcutPar,np.exp(-(fgrid - fring)*gamma2/(fdamp*gamma3))* (fdamp*gamma3*gamma1) / ((fgrid - fring)*(fgrid - fring) + fdamp*gamma3*fdamp*gamma3), 0.)))
        
        # Now add the tidal amplitude as in arXiv:1905.06011
        # Compute the tidal coupling constant, arXiv:1905.06011 eq. (8) using Lambda = 2/3 k_2/C^5 (eq. (10))
        Xa = 0.5 * (1.0 + Seta)
        Xb = 0.5 * (1.0 - Seta)
        kappa2T = (3.0/13.0) * ((1.0 + 12.0*Xb/Xa)*(Xa**5)*Lambda1 + (1.0 + 12.0*Xa/Xb)*(Xb**5)*Lambda2)
        
        # Now compute the amplitude modification as in arXiv:1905.06011 eq. (24)
        xTidal = (np.pi * fgrid)**(2./3.)
        n1T    = 4.157407407407407
        n289T  = 2519.111111111111
        dTidal = 13477.8073677
        polyTidal = (1.0 + n1T*xTidal + n289T*(xTidal**(2.89)))/(1.+dTidal*(xTidal**4))
        ampTidal = -9.0*kappa2T*(xTidal**3.25)*polyTidal
        
        # Compute the dimensionless merger frequency (Mf) for the Planck taper filtering
        a_0 = 0.3586
        n_1 = 3.35411203e-2
        n_2 = 4.31460284e-5
        d_1 = 7.54224145e-2
        d_2 = 2.23626859e-4
        kappa2T2 = kappa2T*kappa2T
        
        numPT = 1.0 + n_1*kappa2T + n_2*kappa2T2
        denPT = 1.0 + d_1*kappa2T + d_2*kappa2T2
        Q_0 = a_0 / np.sqrt(q)
        f_merger = Q_0 * (numPT / denPT) / (2.*np.pi)
        # The derivative of the Planck taper filter can return NaN in some points because of numerical issues, we declare it explicitly to avoid the issue
        @custom_jvp
        def planck_taper_fun(x, y):
            # Terminate the waveform at 1.2 times the merger frequency
            a=1.2
            yp = a*y
            planck_taper = np.where(x < y, 1., np.where(x > yp, 0., 1. - 1./(np.exp((yp - y)/(x - y) + (yp - y)/(x - yp)) + 1.)))

            return planck_taper

        def planck_taper_fun_der(x,y):
            # Terminate the waveform at 1.2 times the merger frequency
            a=1.2
            yp = a*y
            tangent_out = np.where(x < y, 0., np.where(x > yp, 0., np.exp((yp - y)/(x - y) + (yp - y)/(x - yp))*((-1.+a)/(x-y) + (-1.+a)/(x-yp) + (-y+yp)/((x-y)**2) + 1.2*(-y+yp)/((x-yp)**2))/((np.exp((yp - y)/(x - y) + (yp - y)/(x - yp)) + 1.)**2)))
            tangent_out = np.nan_to_num(tangent_out)
            return tangent_out
        
        planck_taper_fun.defjvps(None, lambda y_dot, primal_out, x, y: planck_taper_fun_der(x,y) * y_dot)
        # Now compute tha Planck taper series
        # This filter causes big numerical issues at the cut when computing derivatives and the last element is very small but not 0. We fix it "by hand" with this nan_to_num which assigns 0 in place of NaN. We performed extensive checks and this does not affect any other part of the computation, only the very last point of the frequency grid in some random and rare cases.
        planck_taper = np.nan_to_num(planck_taper_fun(fgrid, f_merger))
        
        return Overallamp*(amp0*(fgrid**(-7./6.))*amplitudeIMR + 2*np.sqrt(np.pi/5.)*ampTidal)*planck_taper
    
        
    def _finalspin(self, eta, chi1, chi2):
        # Compute the spin of the final object, as in LALSimIMRPhenomD_internals.c line 161 and 142,
        # which is taken from arXiv:1508.07250 eq. (3.6)
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        s  = (m1*m1 * chi1 + m2*m2 * chi2)
        af1 = eta*(3.4641016151377544 - 4.399247300629289*eta + 9.397292189321194*eta*eta - 13.180949901606242*eta*eta*eta)
        af2 = eta*(s*((1.0/eta - 0.0850917821418767 - 5.837029316602263*eta) + (0.1014665242971878 - 2.0967746996832157*eta)*s))
        af3 = eta*(s*((-1.3546806617824356 + 4.108962025369336*eta)*s*s + (-0.8676969352555539 + 2.064046835273906*eta)*s*s*s))
        return af1 + af2 + af3
        
    def _radiatednrg(self, eta, chi1, chi2):
        # Predict the total radiated energy, from arXiv:1508.07250 eq (3.7) and (3.8)
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        s  = (m1*m1 * chi1 + m2*m2 * chi2) / (m1*m1 + m2*m2)
        
        EradNS = eta * (0.055974469826360077 + 0.5809510763115132 * eta - 0.9606726679372312 * eta*eta + 3.352411249771192 * eta*eta*eta)
        
        return (EradNS * (1. + (-0.0030302335878845507 - 2.0066110851351073 * eta + 7.7050567802399215 * eta*eta) * s)) / (1. + (-0.6714403054720589 - 1.4756929437702908 * eta + 7.304676214885011 * eta*eta) * s)
    
    def tau_star(self, f, **kwargs):
        # For complex waveforms we use the expression in arXiv:0907.0700 eq. (3.8b)
        Mtot_sec = kwargs['Mc']*glob.GMsun_over_c3/(kwargs['eta']**(3./5.))
        v = (np.pi*Mtot_sec*f)**(1./3.)
        eta = kwargs['eta']
        eta2 = eta*eta
        
        OverallFac = 5./256 * Mtot_sec/(eta*(v**8.))
        
        t05 = 1. + (743./252. + 11./3.*eta)*(v*v) - 32./5.*np.pi*(v*v*v) + (3058673./508032. + 5429./504.*eta + 617./72.*eta2)*(v**4) - (7729./252. - 13./3.*eta)*np.pi*(v**5)
        t6  = (-10052469856691./23471078400. + 128./3.*np.pi*np.pi + 6848./105.*np.euler_gamma + (3147553127./3048192. - 451./12.*np.pi*np.pi)*eta - 15211./1728.*eta2 + 25565./1296.*eta2*eta + 3424./105.*np.log(16.*v*v))*(v**6)
        t7  = (- 15419335./127008. - 75703./756.*eta + 14809./378.*eta2)*np.pi*(v**7)
        
        return OverallFac*(t05 + t6 + t7)
    
    def fcut(self, **kwargs):
        # We cut the waveform slightly before the end of the Planck taper filter, for numerical stability
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        q = 0.5*(1.0 + Seta - 2.0*eta)/eta
        if 'Lambda1' in kwargs:
            Lambda1, Lambda2 = kwargs['Lambda1'], kwargs['Lambda2']
        else:
            Lambda1, Lambda2 = np.zeros(M.shape), np.zeros(M.shape)
        Xa = 0.5 * (1.0 + Seta)
        Xb = 0.5 * (1.0 - Seta)
        kappa2T = (3.0/13.0) * ((1.0 + 12.0*Xb/Xa)*(Xa**5)*Lambda1 + (1.0 + 12.0*Xa/Xb)*(Xb**5)*Lambda2)
        
        # Compute the dimensionless merger frequency (Mf) for the Planck taper filtering
        a_0 = 0.3586
        n_1 = 3.35411203e-2
        n_2 = 4.31460284e-5
        d_1 = 7.54224145e-2
        d_2 = 2.23626859e-4
        kappa2T2 = kappa2T*kappa2T
        
        numPT = 1.0 + n_1*kappa2T + n_2*kappa2T2
        denPT = 1.0 + d_1*kappa2T + d_2*kappa2T2
        Q_0 = a_0 / np.sqrt(q)
        f_merger = Q_0 * (numPT / denPT) / (2.*np.pi)
        # Terminate the waveform at 1.2 times the merger frequency
        f_end_taper = 1.2*f_merger

        return f_end_taper/(M*glob.GMsun_over_c3)
    
        
class IMRPhenomHM(WaveFormModel):
    '''
    IMRPhenomHM waveform model
    '''
    # All is taken from LALSimulation and arXiv:1508.07250, arXiv:1508.07253, arXiv:1708.00404, arXiv:1909.10010
    def __init__(self, **kwargs):
        # Dimensionless frequency (Mf) at which the inspiral amplitude switches to the intermediate amplitude
        self.AMP_fJoin_INS = 0.014
        # Dimensionless frequency (Mf) at which the inspiral phase switches to the intermediate phase
        self.PHI_fJoin_INS = 0.018
        # Dimensionless frequency (Mf) at which we define the end of the waveform
        fcutPar = 0.2
         
        super().__init__('BBH', fcutPar, is_HigherModes=True, **kwargs)
        
        # List of phase shifts: the index is the azimuthal number m
        self.complShiftm = np.array([0., np.pi*0.5, 0., -np.pi*0.5, np.pi, np.pi*0.5, 0.])
        
    def Phi(self, f, **kwargs):
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        eta2 = eta*eta # These can speed up a bit, we call them multiple times
        etaInv = 1./eta
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        
        QuadMon1, QuadMon2 = np.ones(M.shape), np.ones(M.shape)
        
        chi12, chi22 = chi1*chi1, chi2*chi2
        chi1dotchi2  = chi1*chi2
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        SetaPlus1 = 1.0 + Seta
        chi_s = 0.5 * (chi1 + chi2)
        chi_a = 0.5 * (chi1 - chi2)
        q = 0.5*(1.0 + Seta - 2.0*eta)/eta
        chi_s2, chi_a2 = chi_s*chi_s, chi_a*chi_a
        chi1dotchi2    = chi1*chi2
        chi_sdotchi_a  = chi_s*chi_a
        # These are m1/Mtot and m2/Mtot
        m1ByM = 0.5 * (1.0 + Seta)
        m2ByM = 0.5 * (1.0 - Seta)
        # We work in dimensionless frequency M*f, not f
        fgrid = M*glob.GMsun_over_c3*f
        # This is MfRef, needed to recover LAL, which sets fRef to f_min if fRef=0
        fRef  = np.amin(fgrid, axis=0)
        # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
        chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
        xi = - 1.0 + chiPN
        # Compute final spin, radiated energy and mass
        aeff = self._finalspin(eta, chi1, chi2)
        Erad = self._radiatednrg(eta, chi1, chi2)
        finMass = 1. - Erad
        
        fring, fdamp = self._RDfreqCalc(finMass, aeff, 2, 2)
        
        # Compute sigma coefficients appearing in arXiv:1508.07253 eq. (28)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        sigma1 = 2096.551999295543 + 1463.7493168261553*eta + (1312.5493286098522 + 18307.330017082117*eta - 43534.1440746107*eta2 + (-833.2889543511114 + 32047.31997183187*eta - 108609.45037520859*eta2)*xi + (452.25136398112204 + 8353.439546391714*eta - 44531.3250037322*eta2)*xi*xi)*xi
        sigma2 = -10114.056472621156 - 44631.01109458185*eta + (-6541.308761668722 - 266959.23419307504*eta + 686328.3229317984*eta2 + (3405.6372187679685 - 437507.7208209015*eta + 1.6318171307344697e6*eta2)*xi + (-7462.648563007646 - 114585.25177153319*eta + 674402.4689098676*eta2)*xi*xi)*xi
        sigma3 = 22933.658273436497 + 230960.00814979506*eta + (14961.083974183695 + 1.1940181342318142e6*eta - 3.1042239693052764e6*eta2 + (-3038.166617199259 + 1.8720322849093592e6*eta - 7.309145012085539e6*eta2)*xi + (42738.22871475411 + 467502.018616601*eta - 3.064853498512499e6*eta2)*xi*xi)*xi
        sigma4 = -14621.71522218357 - 377812.8579387104*eta + (-9608.682631509726 - 1.7108925257214056e6*eta + 4.332924601416521e6*eta2 + (-22366.683262266528 - 2.5019716386377467e6*eta + 1.0274495902259542e7*eta2)*xi + (-85360.30079034246 - 570025.3441737515*eta + 4.396844346849777e6*eta2)*xi*xi)*xi
        
        # Compute beta coefficients appearing in arXiv:1508.07253 eq. (16)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        beta1 = 97.89747327985583 - 42.659730877489224*eta + (153.48421037904913 - 1417.0620760768954*eta + 2752.8614143665027*eta2 + (138.7406469558649 - 1433.6585075135881*eta + 2857.7418952430758*eta2)*xi + (41.025109467376126 - 423.680737974639*eta + 850.3594335657173*eta2)*xi*xi)*xi
        beta2 = -3.282701958759534 - 9.051384468245866*eta + (-12.415449742258042 + 55.4716447709787*eta - 106.05109938966335*eta2 + (-11.953044553690658 + 76.80704618365418*eta - 155.33172948098394*eta2)*xi + (-3.4129261592393263 + 25.572377569952536*eta - 54.408036707740465*eta2)*xi*xi)*xi
        beta3 = -0.000025156429818799565 + 0.000019750256942201327*eta + (-0.000018370671469295915 + 0.000021886317041311973*eta + 0.00008250240316860033*eta2 + (7.157371250566708e-6 - 0.000055780000112270685*eta + 0.00019142082884072178*eta2)*xi + (5.447166261464217e-6 - 0.00003220610095021982*eta + 0.00007974016714984341*eta2)*xi*xi)*xi
        
        # Compute alpha coefficients appearing in arXiv:1508.07253 eq. (14)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        alpha1 = 43.31514709695348 + 638.6332679188081*eta + (-32.85768747216059 + 2415.8938269370315*eta - 5766.875169379177*eta2 + (-61.85459307173841 + 2953.967762459948*eta - 8986.29057591497*eta2)*xi + (-21.571435779762044 + 981.2158224673428*eta - 3239.5664895930286*eta2)*xi*xi)*xi
        alpha2 = -0.07020209449091723 - 0.16269798450687084*eta + (-0.1872514685185499 + 1.138313650449945*eta - 2.8334196304430046*eta2 + (-0.17137955686840617 + 1.7197549338119527*eta - 4.539717148261272*eta2)*xi + (-0.049983437357548705 + 0.6062072055948309*eta - 1.682769616644546*eta2)*xi*xi)*xi
        alpha3 = 9.5988072383479 - 397.05438595557433*eta + (16.202126189517813 - 1574.8286986717037*eta + 3600.3410843831093*eta2 + (27.092429659075467 - 1786.482357315139*eta + 5152.919378666511*eta2)*xi + (11.175710130033895 - 577.7999423177481*eta + 1808.730762932043*eta2)*xi*xi)*xi
        alpha4 = -0.02989487384493607 + 1.4022106448583738*eta + (-0.07356049468633846 + 0.8337006542278661*eta + 0.2240008282397391*eta2 + (-0.055202870001177226 + 0.5667186343606578*eta + 0.7186931973380503*eta2)*xi + (-0.015507437354325743 + 0.15750322779277187*eta + 0.21076815715176228*eta2)*xi*xi)*xi
        alpha5 = 0.9974408278363099 - 0.007884449714907203*eta + (-0.059046901195591035 + 1.3958712396764088*eta - 4.516631601676276*eta2 + (-0.05585343136869692 + 1.7516580039343603*eta - 5.990208965347804*eta2)*xi + (-0.017945336522161195 + 0.5965097794825992*eta - 2.0608879367971804*eta2)*xi*xi)*xi
        
        # Compute the TF2 phase coefficients and put them in a dictionary (spin effects are included up to 3.5PN)
        TF2coeffs = {}
        TF2OverallAmpl = 3./(128. * eta)
        
        TF2coeffs['zero'] = 1.
        TF2coeffs['one'] = 0.
        TF2coeffs['two'] = 3715./756. + (55.*eta)/9.
        TF2coeffs['three'] = -16.*np.pi + (113.*Seta*chi_a)/3. + (113./3. - (76.*eta)/3.)*chi_s
        # For 2PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['four'] = 5.*(3058.673/7.056 + 5429./7.*eta+617.*eta2)/72. + 247./4.8*eta*chi1dotchi2 -721./4.8*eta*chi1dotchi2 + (-720./9.6*QuadMon1 + 1./9.6)*m1ByM*m1ByM*chi12 + (-720./9.6*QuadMon2 + 1./9.6)*m2ByM*m2ByM*chi22 + (240./9.6*QuadMon1 - 7./9.6)*m1ByM*m1ByM*chi12 + (240./9.6*QuadMon2 - 7./9.6)*m2ByM*m2ByM*chi22
        # This part is common to 5 and 5log, avoid recomputing
        TF2_5coeff_tmp = (732985./2268. - 24260.*eta/81. - 340.*eta2/9.)*chi_s + (732985./2268. + 140.*eta/9.)*Seta*chi_a
        TF2coeffs['five'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)
        TF2coeffs['five_log'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)*3.
        # For 3PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['six'] = 11583.231236531/4.694215680 - 640./3.*np.pi*np.pi - 684.8/2.1*np.euler_gamma + eta*(-15737.765635/3.048192 + 225.5/1.2*np.pi*np.pi) + eta2*76.055/1.728 - eta2*eta*127.825/1.296 - np.log(4.)*684.8/2.1 + np.pi*chi1*m1ByM*(1490./3. + m1ByM*260.) + np.pi*chi2*m2ByM*(1490./3. + m2ByM*260.) + (326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + (4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM)*m1ByM*m1ByM*QuadMon1*chi12 + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM)*m1ByM*m1ByM*chi12 + (4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM)*m2ByM*m2ByM*QuadMon2*chi22 + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM)*m2ByM*m2ByM*chi22
        TF2coeffs['six_log'] = -6848./21.
        TF2coeffs['seven'] = 77096675.*np.pi/254016. + 378515.*np.pi*eta/1512.- 74045.*np.pi*eta2/756. + (-25150083775./3048192. + 10566655595.*eta/762048. - 1042165.*eta2/3024. + 5345.*eta2*eta/36.)*chi_s + Seta*((-25150083775./3048192. + 26804935.*eta/6048. - 1985.*eta2/48.)*chi_a)
        # Remove this part since it was not available when IMRPhenomD was tuned
        TF2coeffs['six'] = TF2coeffs['six'] - ((326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + ((4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM) + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM))*m1ByM*m1ByM*chi12 + ((4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM) + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM))*m2ByM*m2ByM*chi22)
        # Now translate into inspiral coefficients, label with the power in front of which they appear
        PhiInspcoeffs = {}
        
        PhiInspcoeffs['initial_phasing'] = TF2coeffs['five']*TF2OverallAmpl
        PhiInspcoeffs['two_thirds'] = TF2coeffs['seven']*TF2OverallAmpl*(np.pi**(2./3.))
        PhiInspcoeffs['third'] = TF2coeffs['six']*TF2OverallAmpl*(np.pi**(1./3.))
        PhiInspcoeffs['third_log'] = TF2coeffs['six_log']*TF2OverallAmpl*(np.pi**(1./3.))
        PhiInspcoeffs['log'] = TF2coeffs['five_log']*TF2OverallAmpl
        PhiInspcoeffs['min_third'] = TF2coeffs['four']*TF2OverallAmpl*(np.pi**(-1./3.))
        PhiInspcoeffs['min_two_thirds'] = TF2coeffs['three']*TF2OverallAmpl*(np.pi**(-2./3.))
        PhiInspcoeffs['min_one'] = TF2coeffs['two']*TF2OverallAmpl/np.pi
        PhiInspcoeffs['min_four_thirds'] = TF2coeffs['one']*TF2OverallAmpl*(np.pi**(-4./3.))
        PhiInspcoeffs['min_five_thirds'] = TF2coeffs['zero']*TF2OverallAmpl*(np.pi**(-5./3.))
        PhiInspcoeffs['one'] = sigma1
        PhiInspcoeffs['four_thirds'] = sigma2 * 0.75
        PhiInspcoeffs['five_thirds'] = sigma3 * 0.6
        PhiInspcoeffs['two'] = sigma4 * 0.5
        
        #Now compute the coefficients to align the three parts
        
        fInsJoin = self.PHI_fJoin_INS
        fMRDJoin = 0.5*fring
        
        # First the Inspiral - Intermediate: we compute C1Int and C2Int coeffs
        # Equations to solve for to get C(1) continuous join
        # PhiIns (f)  =   PhiInt (f) + C1Int + C2Int f
        # Joining at fInsJoin
        # PhiIns (fInsJoin)  =   PhiInt (fInsJoin) + C1Int + C2Int fInsJoin
        # PhiIns'(fInsJoin)  =   PhiInt'(fInsJoin) + C2Int
        # This is the first derivative wrt f of the inspiral phase computed at fInsJoin, first add the PN contribution and then the higher order calibrated terms
        DPhiIns = (2.0*TF2coeffs['seven']*TF2OverallAmpl*((np.pi*fInsJoin)**(7./3.)) + (TF2coeffs['six']*TF2OverallAmpl + TF2coeffs['six_log']*TF2OverallAmpl * (1.0 + np.log(np.pi*fInsJoin)/3.))*((np.pi*fInsJoin)**(2.)) + TF2coeffs['five_log']*TF2OverallAmpl*((np.pi*fInsJoin)**(5./3.)) - TF2coeffs['four']*TF2OverallAmpl*((np.pi*fInsJoin)**(4./3.)) - 2.*TF2coeffs['three']*TF2OverallAmpl*(np.pi*fInsJoin) - 3.*TF2coeffs['two']*TF2OverallAmpl*((np.pi*fInsJoin)**(2./3.)) - 4.*TF2coeffs['one']*TF2OverallAmpl*((np.pi*fInsJoin)**(1./3.)) - 5.*TF2coeffs['zero']*TF2OverallAmpl)*np.pi/(3.*((np.pi*fInsJoin)**(8./3.)))
        DPhiIns = DPhiIns + (sigma1 + sigma2*(fInsJoin**(1./3.)) + sigma3*(fInsJoin**(2./3.)) + sigma4*fInsJoin)/eta
        # This is the first derivative of the Intermediate phase computed at fInsJoin
        DPhiInt = (beta1 + beta3/(fInsJoin**4) + beta2/fInsJoin)/eta
        
        C2Int = DPhiIns - DPhiInt
        
        # This is the inspiral phase computed at fInsJoin
        PhiInsJoin = PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fInsJoin**(2./3.)) + PhiInspcoeffs['third']*(fInsJoin**(1./3.)) + PhiInspcoeffs['third_log']*(fInsJoin**(1./3.))*np.log(np.pi*fInsJoin)/3. + PhiInspcoeffs['log']*np.log(np.pi*fInsJoin)/3. + PhiInspcoeffs['min_third']*(fInsJoin**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fInsJoin**(-2./3.)) + PhiInspcoeffs['min_one']/fInsJoin + PhiInspcoeffs['min_four_thirds']*(fInsJoin**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fInsJoin**(-5./3.)) + (PhiInspcoeffs['one']*fInsJoin + PhiInspcoeffs['four_thirds']*(fInsJoin**(4./3.)) + PhiInspcoeffs['five_thirds']*(fInsJoin**(5./3.)) + PhiInspcoeffs['two']*fInsJoin*fInsJoin)/eta
        # This is the Intermediate phase computed at fInsJoin
        PhiIntJoin = beta1*fInsJoin - beta3/(3.*fInsJoin*fInsJoin*fInsJoin) + beta2*np.log(fInsJoin)
        
        C1Int = PhiInsJoin - PhiIntJoin/eta - C2Int*fInsJoin
        
        # Now the same for Intermediate - Merger-Ringdown: we also need a temporary Intermediate Phase function
        PhiIntTempVal  = (beta1*fMRDJoin - beta3/(3.*fMRDJoin*fMRDJoin*fMRDJoin) + beta2*np.log(fMRDJoin))/eta + C1Int + C2Int*fMRDJoin
        DPhiIntTempVal = C2Int + (beta1 + beta3/(fMRDJoin**4) + beta2/fMRDJoin)/eta
        DPhiMRDVal     = (alpha1 + alpha2/(fMRDJoin*fMRDJoin) + alpha3/(fMRDJoin**(1./4.)) + alpha4/(fdamp*(1. + (fMRDJoin - alpha5*fring)*(fMRDJoin - alpha5*fring)/(fdamp*fdamp))))/eta
        PhiMRJoinTemp  = -(alpha2/fMRDJoin) + (4.0/3.0) * (alpha3 * (fMRDJoin**(3./4.))) + alpha1 * fMRDJoin + alpha4 * np.arctan((fMRDJoin - alpha5 * fring)/fdamp)
        
        C2MRD = DPhiIntTempVal - DPhiMRDVal
        C1MRD = PhiIntTempVal - PhiMRJoinTemp/eta - C2MRD*fMRDJoin
        
        # Time shift so that peak amplitude is approximately at t=0
        gamma2 = 1.010344404799477 + 0.0008993122007234548*eta + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2 + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi*xi)*xi
        gamma3 = 1.3081615607036106 - 0.005537729694807678*eta +(-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2 + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi*xi)*xi
        fpeak = np.where(gamma2 >= 1.0, np.fabs(fring - (fdamp*gamma3)/gamma2), np.fabs(fring + (fdamp*(-1.0 + np.sqrt(1.0 - gamma2*gamma2))*gamma3)/gamma2))
        
        t0 = (alpha1 + alpha2/(fpeak*fpeak) + alpha3/(fpeak**(1./4.)) + alpha4/(fdamp*(1. + (fpeak - alpha5*fring)*(fpeak - alpha5*fring)/(fdamp*fdamp))))/eta
        
        def completePhase(infreqs, C1MRDuse, C2MRDuse, RhoUse, TauUse):
            
            return np.where(infreqs < self.PHI_fJoin_INS, PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(infreqs**(2./3.)) + PhiInspcoeffs['third']*(infreqs**(1./3.)) + PhiInspcoeffs['third_log']*(infreqs**(1./3.))*np.log(np.pi*infreqs)/3. + PhiInspcoeffs['log']*np.log(np.pi*infreqs)/3. + PhiInspcoeffs['min_third']*(infreqs**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(infreqs**(-2./3.)) + PhiInspcoeffs['min_one']/infreqs + PhiInspcoeffs['min_four_thirds']*(infreqs**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(infreqs**(-5./3.)) + (PhiInspcoeffs['one']*infreqs + PhiInspcoeffs['four_thirds']*(infreqs**(4./3.)) + PhiInspcoeffs['five_thirds']*(infreqs**(5./3.)) + PhiInspcoeffs['two']*infreqs*infreqs)/eta, np.where(infreqs<fMRDJoin, (beta1*infreqs - beta3/(3.*infreqs*infreqs*infreqs) + beta2*np.log(infreqs))/eta + C1Int + C2Int*infreqs, np.where(infreqs < self.fcutPar, (-(alpha2/infreqs) + (4.0/3.0) * (alpha3 * (infreqs**(3./4.))) + alpha1 * infreqs + alpha4 * RhoUse * np.arctan((infreqs - alpha5 * fring)/(fdamp * RhoUse * TauUse)))/eta + C1MRDuse + C2MRDuse*infreqs,0.)))
 
        phiRef = completePhase(fRef, C1MRD, C2MRD, 1., 1.)
        phi0   = 0.5*phiRef
        
        # Now compute the other modes, they are 5, we loop
        phislmp = {}
        
        for ell in (2,3,4):
            for mm in (ell-1, ell):
                # Compute ringdown and damping frequencies from fits for the various modes
                fringlm, fdamplm = self._RDfreqCalc(finMass, aeff, ell, mm)
                Rholm, Taulm = fring/fringlm, fdamplm/fdamp
                
                # Rholm and Taulm only figure in the MRD part, the rest of the coefficients is the same, recompute only this
                DPhiMRDVal    = (alpha1 + alpha2/(fMRDJoin*fMRDJoin) + alpha3/(fMRDJoin**(1./4.)) + alpha4/(fdamp*Taulm*(1. + (fMRDJoin - alpha5*fring)*(fMRDJoin - alpha5*fring)/(fdamp*Taulm*Rholm*fdamp*Taulm*Rholm))))/eta
                PhiMRJoinTemp = -(alpha2/fMRDJoin) + (4.0/3.0) * (alpha3 * (fMRDJoin**(3./4.))) + alpha1 * fMRDJoin + alpha4 * Rholm* np.arctan((fMRDJoin - alpha5 * fring)/(fdamp*Rholm*Taulm))
                C2MRDHM = DPhiIntTempVal - DPhiMRDVal
                C1MRDHM = PhiIntTempVal - PhiMRJoinTemp/eta - C2MRDHM*fMRDJoin
                    
                # Compute mapping coefficinets
                Map_fl = self.PHI_fJoin_INS
                Map_fi = Map_fl / Rholm
                Map_fr = fringlm
                
                Map_ai, Map_bi = 2./mm, 0.
                    
                Map_Trd = Map_fr * Rholm
                Map_Ti  = 2. * Map_fi / mm
                Map_am  = (Map_Trd - Map_Ti) / (Map_fr - Map_fi)
                Map_bm  = Map_Ti - Map_fi * Map_am
                    
                Map_ar, Map_br = Rholm, 0.
                # Now compute the needed constants
                tmpMf = Map_am * Map_fi + Map_bm
                PhDBconst = completePhase(tmpMf, C1MRDHM, C2MRDHM, Rholm, Taulm) / Map_am
                    
                tmpMf = Map_ar * Map_fr + Map_br
                PhDCconst = completePhase(tmpMf, C1MRDHM, C2MRDHM, Rholm, Taulm) / Map_ar
            
                tmpMf = Map_ai * Map_fi + Map_bi
                PhDBAterm = completePhase(tmpMf, C1MRDHM, C2MRDHM, Rholm, Taulm) / Map_ai
                
                tmpMf = Map_am * Map_fr + Map_bm
                tmpphaseC = - PhDBconst + PhDBAterm + completePhase(tmpMf, C1MRDHM, C2MRDHM, Rholm, Taulm) / Map_am
                
                phitmp = np.where(fgrid < Map_fi, completePhase(fgrid*Map_ai + Map_bi, C1MRDHM, C2MRDHM, Rholm, Taulm)/Map_ai, np.where(fgrid < Map_fr, - PhDBconst + PhDBAterm + completePhase(fgrid*Map_am + Map_bm, C1MRDHM, C2MRDHM, Rholm, Taulm)/Map_am, - PhDCconst + tmpphaseC + completePhase(fgrid*Map_ar + Map_br, C1MRDHM, C2MRDHM, Rholm, Taulm)/Map_ar))
                
                phislmp[str(ell)+str(mm)] =  phitmp - t0*(fgrid - fRef) - mm*phi0 + self.complShiftm[mm]
                
        return phislmp
        
    
    def Ampl(self, f, **kwargs):
        # Useful quantities
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        eta2 = eta*eta # This can speed up a bit, we call it multiple times
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        chi12, chi22 = chi1*chi1, chi2*chi2
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        SetaPlus1 = 1.0 + Seta
        chi_s = 0.5 * (chi1 + chi2)
        chi_a = 0.5 * (chi1 - chi2)
        # We work in dimensionless frequency M*f, not f
        fgrid = M*glob.GMsun_over_c3*f
        # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
        chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
        xi = -1.0 + chiPN
        # Compute final spin and radiated energy
        aeff = self._finalspin(eta, chi1, chi2)
        Erad = self._radiatednrg(eta, chi1, chi2)
        finMass = 1. - Erad
        
        fring, fdamp = self._RDfreqCalc(finMass, aeff, 2, 2)
        
        # Compute coefficients gamma appearing in arXiv:1508.07253 eq. (19), the numerical coefficients are in Tab. 5
        gamma1 = 0.006927402739328343 + 0.03020474290328911*eta + (0.006308024337706171 - 0.12074130661131138*eta + 0.26271598905781324*eta2 + (0.0034151773647198794 - 0.10779338611188374*eta + 0.27098966966891747*eta2)*xi+ (0.0007374185938559283 - 0.02749621038376281*eta + 0.0733150789135702*eta2)*xi*xi)*xi
        gamma2 = 1.010344404799477 + 0.0008993122007234548*eta + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2 + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi*xi)*xi
        gamma3 = 1.3081615607036106 - 0.005537729694807678*eta +(-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2 + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi*xi)*xi
        # Compute fpeak, from arXiv:1508.07253 eq. (20), we remove the square root term in case it is complex
        fpeak = np.where(gamma2 >= 1.0, np.fabs(fring - (fdamp*gamma3)/gamma2), fring + (fdamp*(-1.0 + np.sqrt(1.0 - gamma2*gamma2))*gamma3)/gamma2)
        # Compute coefficients rho appearing in arXiv:1508.07253 eq. (30), the numerical coefficients are in Tab. 5
        rho1 = 3931.8979897196696 - 17395.758706812805*eta + (3132.375545898835 + 343965.86092361377*eta - 1.2162565819981997e6*eta2 + (-70698.00600428853 + 1.383907177859705e6*eta - 3.9662761890979446e6*eta2)*xi + (-60017.52423652596 + 803515.1181825735*eta - 2.091710365941658e6*eta2)*xi*xi)*xi
        rho2 = -40105.47653771657 + 112253.0169706701*eta + (23561.696065836168 - 3.476180699403351e6*eta + 1.137593670849482e7*eta2 + (754313.1127166454 - 1.308476044625268e7*eta + 3.6444584853928134e7*eta2)*xi + (596226.612472288 - 7.4277901143564405e6*eta + 1.8928977514040343e7*eta2)*xi*xi)*xi
        rho3 = 83208.35471266537 - 191237.7264145924*eta + (-210916.2454782992 + 8.71797508352568e6*eta - 2.6914942420669552e7*eta2 + (-1.9889806527362722e6 + 3.0888029960154563e7*eta - 8.390870279256162e7*eta2)*xi + (-1.4535031953446497e6 + 1.7063528990822166e7*eta - 4.2748659731120914e7*eta2)*xi*xi)*xi
        # Compute coefficients delta appearing in arXiv:1508.07253 eq. (21)
        f1Interm = self.AMP_fJoin_INS
        f3Interm = fpeak
        dfInterm = 0.5*(f3Interm - f1Interm)
        f2Interm = f1Interm + dfInterm
        # First write the inspiral coefficients, we put them in a dictionary and label with the power in front of which they appear
        amp0 = np.sqrt(2.0*eta/3.0)*(np.pi**(-1./6.))
        Acoeffs = {}
        Acoeffs['two_thirds'] = ((-969. + 1804.*eta)*(np.pi**(2./3.)))/672.
        Acoeffs['one'] = ((chi1*(81.*SetaPlus1 - 44.*eta) + chi2*(81. - 81.*Seta - 44.*eta))*np.pi)/48.
        Acoeffs['four_thirds'] = ((-27312085.0 - 10287648.*chi22 - 10287648.*chi12*SetaPlus1 + 10287648.*chi22*Seta+ 24.*(-1975055. + 857304.*chi12 - 994896.*chi1*chi2 + 857304.*chi22)*eta+ 35371056*eta2)* (np.pi**(4./3.)))/8.128512e6
        Acoeffs['five_thirds'] = ((np.pi**(5./3.)) * (chi2*(-285197.*(-1. + Seta) + 4.*(-91902. + 1579.*Seta)*eta - 35632.*eta2) + chi1*(285197.*SetaPlus1 - 4.*(91902. + 1579.*Seta)*eta - 35632.*eta2) + 42840.*(-1.0 + 4.*eta)*np.pi)) / 32256.
        Acoeffs['two'] = - ((np.pi**2.)*(-336.*(-3248849057.0 + 2943675504.*chi12 - 3339284256.*chi1*chi2 + 2943675504.*chi22)*eta2 - 324322727232.*eta2*eta - 7.*(-177520268561. + 107414046432.*chi22 + 107414046432.*chi12*SetaPlus1 - 107414046432.*chi22*Seta + 11087290368.*(chi1 + chi2 + chi1*Seta - chi2*Seta)*np.pi ) + 12.*eta*(-545384828789. - 176491177632.*chi1*chi2 + 202603761360.*chi22 + 77616.*chi12*(2610335. + 995766.*Seta) - 77287373856.*chi22*Seta + 5841690624.*(chi1 + chi2)*np.pi + 21384760320.*np.pi*np.pi)))/6.0085960704e10
        Acoeffs['seven_thirds'] = rho1
        Acoeffs['eight_thirds'] = rho2
        Acoeffs['three'] = rho3
        # v1 is the inspiral model evaluated at f1Interm
        v1 = 1. + (f1Interm**(2./3.))*Acoeffs['two_thirds'] + (f1Interm**(4./3.)) * Acoeffs['four_thirds'] + (f1Interm**(5./3.)) *  Acoeffs['five_thirds'] + (f1Interm**(7./3.)) * Acoeffs['seven_thirds'] + (f1Interm**(8./3.)) * Acoeffs['eight_thirds'] + f1Interm * (Acoeffs['one'] + f1Interm * Acoeffs['two'] + f1Interm*f1Interm * Acoeffs['three'])
        # d1 is the derivative of the inspiral model evaluated at f1
        d1 = ((-969. + 1804.*eta)*(np.pi**(2./3.)))/(1008.*(f1Interm**(1./3.))) + ((chi1*(81.*SetaPlus1 - 44.*eta) + chi2*(81. - 81.*Seta - 44.*eta))*np.pi)/48. + ((-27312085. - 10287648.*chi22 - 10287648.*chi12*SetaPlus1 + 10287648.*chi22*Seta + 24.*(-1975055. + 857304.*chi12 - 994896.*chi1*chi2 + 857304.*chi22)*eta + 35371056.*eta2)*(f1Interm**(1./3.))*(np.pi**(4./3.)))/6.096384e6 + (5.*(f1Interm**(2./3.))*(np.pi**(5./3.))*(chi2*(-285197.*(-1 + Seta)+ 4.*(-91902. + 1579.*Seta)*eta - 35632.*eta2) + chi1*(285197.*SetaPlus1- 4.*(91902. + 1579.*Seta)*eta - 35632.*eta2) + 42840.*(-1 + 4*eta)*np.pi))/96768.- (f1Interm*np.pi*np.pi*(-336.*(-3248849057.0 + 2943675504.*chi12 - 3339284256.*chi1*chi2 + 2943675504.*chi22)*eta2 - 324322727232.*eta2*eta - 7.*(-177520268561. + 107414046432.*chi22 + 107414046432.*chi12*SetaPlus1 - 107414046432.*chi22*Seta+ 11087290368*(chi1 + chi2 + chi1*Seta - chi2*Seta)*np.pi)+ 12.*eta*(-545384828789.0 - 176491177632.*chi1*chi2 + 202603761360.*chi22 + 77616.*chi12*(2610335. + 995766.*Seta)- 77287373856.*chi22*Seta + 5841690624.*(chi1 + chi2)*np.pi + 21384760320*np.pi*np.pi)))/3.0042980352e10+ (7.0/3.0)*(f1Interm**(4./3.))*rho1 + (8.0/3.0)*(f1Interm**(5./3.))*rho2 + 3.*(f1Interm*f1Interm)*rho3
        # v3 is the merger-ringdown model (eq. (19) of arXiv:1508.07253) evaluated at f3
        v3 = np.exp(-(f3Interm - fring)*gamma2/(fdamp*gamma3))* (fdamp*gamma3*gamma1) / ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3)
        # d2 is the derivative of the merger-ringdown model evaluated at f3
        d2 = ((-2.*fdamp*(f3Interm - fring)*gamma3*gamma1) / ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3) - (gamma2*gamma1))/(np.exp((f3Interm - fring)*gamma2/(fdamp*gamma3)) * ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3))
        # v2 is the value of the amplitude evaluated at f2. They come from the fit of the collocation points in the intermediate region
        v2 = 0.8149838730507785 + 2.5747553517454658*eta + (1.1610198035496786 - 2.3627771785551537*eta + 6.771038707057573*eta2 + (0.7570782938606834 - 2.7256896890432474*eta + 7.1140380397149965*eta2)*xi + (0.1766934149293479 - 0.7978690983168183*eta + 2.1162391502005153*eta2)*xi*xi)*xi
        # Now some definitions to speed up
        f1  = f1Interm
        f2  = f2Interm
        f3  = f3Interm
        f12 = f1Interm*f1Interm
        f13 = f1Interm*f12;
        f14 = f1Interm*f13;
        f15 = f1Interm*f14;
        f22 = f2Interm*f2Interm;
        f23 = f2Interm*f22;
        f24 = f2Interm*f23;
        f32 = f3Interm*f3Interm;
        f33 = f3Interm*f32;
        f34 = f3Interm*f33;
        f35 = f3Interm*f34;
        # Finally conpute the deltas
        delta0 = -((d2*f15*f22*f3 - 2.*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32 - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33 + 2.*d1*f13*f22*f33 - 2.*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33 - d1*f13*f2*f34 - d1*f12*f22*f34 + 2.*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35 + 4.*f12*f23*f32*v1 - 3.*f1*f24*f32*v1 - 8.*f12*f22*f33*v1 + 4.*f1*f23*f33*v1 + f24*f33*v1 + 4.*f12*f2*f34*v1 + f1*f22*f34*v1 - 2.*f23*f34*v1 - 2.*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2 + 3.*f14*f33*v2 - 3.*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2.*f14*f23*v3 - f13*f24*v3 + 2.*f15*f2*f3*v3 - f14*f22*f3*v3 - 4.*f13*f23*f3*v3 + 3.*f12*f24*f3*v3 - 4.*f14*f2*f32*v3 + 8.*f13*f22*f32*v3 - 4.*f12*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta0 = -((d2*f15*f22*f3 - 2.*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32 - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33 + 2*d1*f13*f22*f33 - 2*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33 - d1*f13*f2*f34 - d1*f12*f22*f34 + 2*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35 + 4*f12*f23*f32*v1 - 3*f1*f24*f32*v1 - 8*f12*f22*f33*v1 + 4*f1*f23*f33*v1 + f24*f33*v1 + 4*f12*f2*f34*v1 + f1*f22*f34*v1 - 2*f23*f34*v1 - 2*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2 + 3*f14*f33*v2 - 3*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2*f14*f23*v3 - f13*f24*v3 + 2*f15*f2*f3*v3 - f14*f22*f3*v3 - 4*f13*f23*f3*v3 + 3*f12*f24*f3*v3 - 4*f14*f2*f32*v3 + 8*f13*f22*f32*v3 - 4*f12*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3-f2)*(f3-f2)))
        delta1 = -((-(d2*f15*f22) + 2.*d2*f14*f23 - d2*f13*f24 - d2*f14*f22*f3 + 2.*d1*f13*f23*f3 + 2.*d2*f13*f23*f3 - 2*d1*f12*f24*f3 - d2*f12*f24*f3 + d2*f15*f32 - 3*d1*f13*f22*f32 - d2*f13*f22*f32 + 2*d1*f12*f23*f32 - 2*d2*f12*f23*f32 + d1*f1*f24*f32 + 2*d2*f1*f24*f32 - d2*f14*f33 + d1*f12*f22*f33 + 3*d2*f12*f22*f33 - 2*d1*f1*f23*f33 - 2*d2*f1*f23*f33 + d1*f24*f33 + d1*f13*f34 + d1*f1*f22*f34 - 2*d1*f23*f34 - d1*f12*f35 + d1*f22*f35 - 8*f12*f23*f3*v1 + 6*f1*f24*f3*v1 + 12*f12*f22*f32*v1 - 8*f1*f23*f32*v1 - 4*f12*f34*v1 + 2*f1*f35*v1 + 2*f15*f3*v2 - 4*f14*f32*v2 + 4*f12*f34*v2 - 2*f1*f35*v2 - 2*f15*f3*v3 + 8*f12*f23*f3*v3 - 6*f1*f24*f3*v3 + 4*f14*f32*v3 - 12*f12*f22*f32*v3 + 8*f1*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta2 = -((d2*f15*f2 - d1*f13*f23 - 3*d2*f13*f23 + d1*f12*f24 + 2.*d2*f12*f24 - d2*f15*f3 + d2*f14*f2*f3 - d1*f12*f23*f3 + d2*f12*f23*f3 + d1*f1*f24*f3 - d2*f1*f24*f3 - d2*f14*f32 + 3*d1*f13*f2*f32 + d2*f13*f2*f32 - d1*f1*f23*f32 + d2*f1*f23*f32 - 2*d1*f24*f32 - d2*f24*f32 - 2*d1*f13*f33 + 2*d2*f13*f33 - d1*f12*f2*f33 - 3*d2*f12*f2*f33 + 3*d1*f23*f33 + d2*f23*f33 + d1*f12*f34 - d1*f1*f2*f34 + d1*f1*f35 - d1*f2*f35 + 4*f12*f23*v1 - 3*f1*f24*v1 + 4*f1*f23*f3*v1 - 3*f24*f3*v1 - 12*f12*f2*f32*v1 + 4*f23*f32*v1 + 8*f12*f33*v1 - f1*f34*v1 - f35*v1 - f15*v2 - f14*f3*v2 + 8*f13*f32*v2 - 8*f12*f33*v2 + f1*f34*v2 + f35*v2 + f15*v3 - 4*f12*f23*v3 + 3*f1*f24*v3 + f14*f3*v3 - 4*f1*f23*f3*v3 + 3*f24*f3*v3 - 8*f13*f32*v3 + 12*f12*f2*f32*v3 - 4*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta3 = -((-2.*d2*f14*f2 + d1*f13*f22 + 3*d2*f13*f22 - d1*f1*f24 - d2*f1*f24 + 2*d2*f14*f3 - 2.*d1*f13*f2*f3 - 2*d2*f13*f2*f3 + d1*f12*f22*f3 - d2*f12*f22*f3 + d1*f24*f3 + d2*f24*f3 + d1*f13*f32 - d2*f13*f32 - 2*d1*f12*f2*f32 + 2*d2*f12*f2*f32 + d1*f1*f22*f32 - d2*f1*f22*f32 + d1*f12*f33 - d2*f12*f33 + 2*d1*f1*f2*f33 + 2*d2*f1*f2*f33 - 3*d1*f22*f33 - d2*f22*f33 - 2*d1*f1*f34 + 2*d1*f2*f34 - 4*f12*f22*v1 + 2*f24*v1 + 8*f12*f2*f3*v1 - 4*f1*f22*f3*v1 - 4*f12*f32*v1 + 8*f1*f2*f32*v1 - 4*f22*f32*v1 - 4*f1*f33*v1 + 2*f34*v1 + 2*f14*v2 - 4*f13*f3*v2 + 4*f1*f33*v2 - 2*f34*v2 - 2*f14*v3 + 4*f12*f22*v3 - 2*f24*v3 + 4*f13*f3*v3 - 8*f12*f2*f3*v3 + 4*f1*f22*f3*v3 + 4*f12*f32*v3 - 8*f1*f2*f32*v3 + 4*f22*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta4 = -((d2*f13*f2 - d1*f12*f22 - 2*d2*f12*f22 + d1*f1*f23 + d2*f1*f23 - d2*f13*f3 + 2.*d1*f12*f2*f3 + d2*f12*f2*f3 - d1*f1*f22*f3 + d2*f1*f22*f3 - d1*f23*f3 - d2*f23*f3 - d1*f12*f32 + d2*f12*f32 - d1*f1*f2*f32 - 2*d2*f1*f2*f32 + 2*d1*f22*f32 + d2*f22*f32 + d1*f1*f33 - d1*f2*f33 + 3*f1*f22*v1 - 2*f23*v1 - 6*f1*f2*f3*v1 + 3*f22*f3*v1 + 3*f1*f32*v1 - f33*v1 - f13*v2 + 3*f12*f3*v2 - 3*f1*f32*v2 + f33*v2 + f13*v3 - 3*f1*f22*v3 + 2*f23*v3 - 3*f12*f3*v3 + 6*f1*f2*f3*v3 - 3*f22*f3*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        
        # Defined as in LALSimulation - LALSimIMRPhenomUtils.c line 70. Final units are correctly Hz^-1
        # there is a 2 * sqrt(5/(64*pi)) missing w.r.t the standard coefficient, which comes from the (2,2) shperical harmonic
        Overallamp = M * glob.GMsun_over_c2_Gpc * M * glob.GMsun_over_c3 / kwargs['dL']
        
        def completeAmpl(infreqs):
            
            return Overallamp*amp0*(infreqs**(-7./6.))*np.where(infreqs < self.AMP_fJoin_INS, 1. + (infreqs**(2./3.))*Acoeffs['two_thirds'] + (infreqs**(4./3.)) * Acoeffs['four_thirds'] + (infreqs**(5./3.)) *  Acoeffs['five_thirds'] + (infreqs**(7./3.)) * Acoeffs['seven_thirds'] + (infreqs**(8./3.)) * Acoeffs['eight_thirds'] + infreqs * (Acoeffs['one'] + infreqs * Acoeffs['two'] + infreqs*infreqs * Acoeffs['three']), np.where(infreqs < fpeak, delta0 + infreqs*delta1 + infreqs*infreqs*(delta2 + infreqs*delta3 + infreqs*infreqs*delta4), np.where(infreqs < self.fcutPar,np.exp(-(infreqs - fring)*gamma2/(fdamp*gamma3))* (fdamp*gamma3*gamma1) / ((infreqs - fring)*(infreqs - fring) + fdamp*gamma3*fdamp*gamma3), 0.)))
        
        def OnePointFiveSpinPN(infreqs, l, m, ChiS, ChiA):
            # PN amplitudes function, needed to scale
            
            v  = (2.*np.pi*infreqs/m)**(1./3.)
            v2 = v*v
            v3 = v2*v
            
            if (2 == l) and (2 == m):
                Hlm = 1.
            
            elif (2 == l) and (1 == m):
                Hlm = (np.sqrt(2.0) / 3.0) * (v * Seta - v2 * 1.5 * (ChiA + Seta * ChiS) + v3 * Seta * ((335.0 / 672.0) + (eta * 117.0 / 56.0)) + v3*v * (ChiA * (3427.0 / 1344. - eta * 2101.0 / 336.) + Seta * ChiS * (3427.0 / 1344 - eta * 965 / 336) + Seta * (-1j * 0.5 - np.pi - 2 * 1j * 0.69314718056)))
            
            elif (3 == l) and (3 == m):
                Hlm = 0.75 * np.sqrt(5.0 / 7.0) * (v * Seta)
            
            elif (3 == l) and (2 == m):
                Hlm = (1.0 / 3.0) * np.sqrt(5.0 / 7.0) * (v2 * (1.0 - 3.0 * eta))
            
            elif (4 == l) and (4 == m):
                Hlm = (4.0 / 9.0) * np.sqrt(10.0 / 7.0) * v2 * (1.0 - 3.0 * eta)
            
            elif (4 == l) and (3 == m):
                Hlm = 0.75 * np.sqrt(3.0 / 35.0) * v3 * Seta * (1.0 - 2.0 * eta)
            
            else:
                raise ValueError('Mode not present in IMRPhenomHM waveform model.')
            
            # Compute the final PN Amplitude at Leading Order in Mf
            
            return np.pi * np.sqrt(eta * 2. / 3.) * (v**(-3.5)) * abs(Hlm)
            
        ampllm = {}
        
        for ell in (2,3,4):
            for mm in (ell-1, ell):
                # Compute ringdown and damping frequencies from fits for the various modes
                fringlm, fdamplm = self._RDfreqCalc(finMass, aeff, ell, mm)
                Rholm, Taulm = fring/fringlm, fdamplm/fdamp
                
                # Scale input frequencies according to PhenomHM model
                # Compute mapping coefficinets
                Map_fl = self.AMP_fJoin_INS
                Map_fi = Map_fl / Rholm
                Map_fr = fringlm
                
                Map_ai, Map_bi = 2./mm, 0.
                    
                Map_Trd = Map_fr - fringlm + fring
                Map_Ti  = 2. * Map_fi / mm
                Map_am  = (Map_Trd - Map_Ti) / (Map_fr - Map_fi)
                Map_bm  = Map_Ti - Map_fi * Map_am
                    
                Map_ar, Map_br = 1., - Map_fr + fring
                
                # Now scale as f -> f*a+b for each regime
                fgridScaled = np.where(fgrid < Map_fi, fgrid*Map_ai + Map_bi, np.where(fgrid < Map_fr, fgrid*Map_am + Map_bm, fgrid*Map_ar + Map_br))
                
                # Map the ampliude's range
                # We divide by the leading order l=m=2 behavior, and then scale in the expected PN behavior for the multipole of interest.
                
                beta_term1  = OnePointFiveSpinPN(fgrid, ell, mm, chi_s, chi_a)
                beta_term2  = OnePointFiveSpinPN(2.*fgrid/mm, ell, mm, chi_s, chi_a)
                
                HMamp_term1 = OnePointFiveSpinPN(fgridScaled, ell, mm, chi_s, chi_a)
                HMamp_term2 = OnePointFiveSpinPN(fgridScaled, 2, 2, 0., 0.)
                
                # The (3,3) and (4,3) modes vanish if eta=0.25 (equal mass case) and the (2,1) mode vanishes if both eta=0.25 and chi1z=chi2z
                # This results in NaNs having 0/0, correct for this using np.nan_to_num()
                
                ampllm[str(ell)+str(mm)] = np.nan_to_num(completeAmpl(fgridScaled) * (beta_term1 / beta_term2) * HMamp_term1 / HMamp_term2)
                
        return ampllm
    
    def hphc(self, f, **kwargs):
        # This function retuns directly the full plus and cross polarisations, avoiding for loops over the modes
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        eta2 = eta*eta # These can speed up a bit, we call them multiple times
        etaInv = 1./eta
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        iota = kwargs['iota']
        QuadMon1, QuadMon2 = np.ones(M.shape), np.ones(M.shape)
        
        chi12, chi22 = chi1*chi1, chi2*chi2
        chi1dotchi2  = chi1*chi2
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        SetaPlus1 = 1.0 + Seta
        chi_s = 0.5 * (chi1 + chi2)
        chi_a = 0.5 * (chi1 - chi2)
        q = 0.5*(1.0 + Seta - 2.0*eta)/eta
        chi_s2, chi_a2 = chi_s*chi_s, chi_a*chi_a
        chi1dotchi2    = chi1*chi2
        chi_sdotchi_a  = chi_s*chi_a
        # These are m1/Mtot and m2/Mtot
        m1ByM = 0.5 * (1.0 + Seta)
        m2ByM = 0.5 * (1.0 - Seta)
        # We work in dimensionless frequency M*f, not f
        fgrid = M*glob.GMsun_over_c3*f
        # This is MfRef, needed to recover LAL, which sets fRef to f_min if fRef=0
        fRef  = np.amin(fgrid, axis=0)
        # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
        chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
        xi = - 1.0 + chiPN
        # Compute final spin, radiated energy and mass
        aeff = self._finalspin(eta, chi1, chi2)
        Erad = self._radiatednrg(eta, chi1, chi2)
        finMass = 1. - Erad
    
        # Compute the real and imag parts of the complex ringdown frequency for the (l,m) mode as in LALSimIMRPhenomHM.c line 189
        # These are all fits of the different modes. We directly exploit the fact that the relevant HM in this WF are 6
        modes = np.array([21,22,32,33,43,44])
        ells = np.floor(modes/10).astype('int')
        mms = modes - ells*10
        # Domain mapping for dimnesionless BH spin
        alphaRDfr = np.log(2. - aeff) / np.log(3.)
        # beta = 1. / (2. + l - abs(m))
        betaRDfr = np.where(modes==21, 1./3., np.where(modes==22, 0.5, np.where(modes==32, 1./3., np.where(modes==33, 0.5, np.where(modes==43, 1./3., 0.5)))))
        kappaRDfr  = np.expand_dims(alphaRDfr,len(alphaRDfr.shape))**betaRDfr
        kappaRDfr2 = kappaRDfr*kappaRDfr
        kappaRDfr3 = kappaRDfr*kappaRDfr2
        kappaRDfr4 = kappaRDfr*kappaRDfr3
        
        tmpRDfr = np.where(modes==21, 0.589113 * np.exp(0.043525 * 1j) + 0.18896353 * np.exp(2.289868 * 1j) * kappaRDfr + 1.15012965 * np.exp(5.810057 * 1j) * kappaRDfr2 + 6.04585476 * np.exp(2.741967 * 1j) * kappaRDfr3 + 11.12627777 * np.exp(5.844130 * 1j) * kappaRDfr4 + 9.34711461 * np.exp(2.669372 * 1j) * kappaRDfr4*kappaRDfr + 3.03838318 * np.exp(5.791518 * 1j) * kappaRDfr4*kappaRDfr2, np.where(modes==22, 1.0 + kappaRDfr * (1.557847 * np.exp(2.903124 * 1j) + 1.95097051 * np.exp(5.920970 * 1j) * kappaRDfr + 2.09971716 * np.exp(2.760585 * 1j) * kappaRDfr2 + 1.41094660 * np.exp(5.914340 * 1j) * kappaRDfr3 + 0.41063923 * np.exp(2.795235 * 1j) * kappaRDfr4), np.where(modes==32, 1.022464 * np.exp(0.004870 * 1j) + 0.24731213 * np.exp(0.665292 * 1j) * kappaRDfr + 1.70468239 * np.exp(3.138283 * 1j) * kappaRDfr2 + 0.94604882 * np.exp(0.163247 * 1j) * kappaRDfr3 + 1.53189884 * np.exp(5.703573 * 1j) * kappaRDfr4 + 2.28052668 * np.exp(2.685231 * 1j) * kappaRDfr4*kappaRDfr + 0.92150314 * np.exp(5.841704 * 1j) * kappaRDfr4*kappaRDfr2, np.where(modes==33, 1.5 + kappaRDfr * (2.095657 * np.exp(2.964973 * 1j) + 2.46964352 * np.exp(5.996734 * 1j) * kappaRDfr + 2.66552551 * np.exp(2.817591 * 1j) * kappaRDfr2 + 1.75836443 * np.exp(5.932693 * 1j) * kappaRDfr3 + 0.49905688 * np.exp(2.781658 * 1j) * kappaRDfr4), np.where(modes==43, 1.5 + kappaRDfr * (0.205046 * np.exp(0.595328 * 1j) + 3.10333396 * np.exp(3.016200 * 1j) * kappaRDfr + 4.23612166 * np.exp(6.038842 * 1j) * kappaRDfr2 + 3.02890198 * np.exp(2.826239 * 1j) * kappaRDfr3 + 0.90843949 * np.exp(5.915164 * 1j) * kappaRDfr4), 2.0 + kappaRDfr * (2.658908 * np.exp(3.002787 * 1j) + 2.97825567 * np.exp(6.050955 * 1j) * kappaRDfr + 3.21842350 * np.exp(2.877514 * 1j) * kappaRDfr2 + 2.12764967 * np.exp(5.989669 * 1j) * kappaRDfr3 + 0.60338186 * np.exp(2.830031 * 1j) * kappaRDfr4))))))

        fringlm = (np.real(tmpRDfr)/(2.*np.pi*np.expand_dims(finMass, len(finMass.shape))))
        fdamplm = (np.imag(tmpRDfr)/(2.*np.pi*np.expand_dims(finMass, len(finMass.shape))))
        
        # This recomputation is needed for JAX derivatives
        betaRDfr = 0.5
        kappaRDfr  = alphaRDfr**betaRDfr
        kappaRDfr2 = kappaRDfr*kappaRDfr
        kappaRDfr3 = kappaRDfr*kappaRDfr2
        kappaRDfr4 = kappaRDfr*kappaRDfr3
        
        tmpRDfr = 1.0 + kappaRDfr * (1.557847 * np.exp(2.903124 * 1j) + 1.95097051 * np.exp(5.920970 * 1j) * kappaRDfr + 2.09971716 * np.exp(2.760585 * 1j) * kappaRDfr2 + 1.41094660 * np.exp(5.914340 * 1j) * kappaRDfr3 + 0.41063923 * np.exp(2.795235 * 1j) * kappaRDfr4)
        
        fring = (np.real(tmpRDfr)/(2.*np.pi*finMass))
        fdamp = (np.imag(tmpRDfr)/(2.*np.pi*finMass))

        # Compute sigma coefficients appearing in arXiv:1508.07253 eq. (28)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        sigma1 = 2096.551999295543 + 1463.7493168261553*eta + (1312.5493286098522 + 18307.330017082117*eta - 43534.1440746107*eta2 + (-833.2889543511114 + 32047.31997183187*eta - 108609.45037520859*eta2)*xi + (452.25136398112204 + 8353.439546391714*eta - 44531.3250037322*eta2)*xi*xi)*xi
        sigma2 = -10114.056472621156 - 44631.01109458185*eta + (-6541.308761668722 - 266959.23419307504*eta + 686328.3229317984*eta2 + (3405.6372187679685 - 437507.7208209015*eta + 1.6318171307344697e6*eta2)*xi + (-7462.648563007646 - 114585.25177153319*eta + 674402.4689098676*eta2)*xi*xi)*xi
        sigma3 = 22933.658273436497 + 230960.00814979506*eta + (14961.083974183695 + 1.1940181342318142e6*eta - 3.1042239693052764e6*eta2 + (-3038.166617199259 + 1.8720322849093592e6*eta - 7.309145012085539e6*eta2)*xi + (42738.22871475411 + 467502.018616601*eta - 3.064853498512499e6*eta2)*xi*xi)*xi
        sigma4 = -14621.71522218357 - 377812.8579387104*eta + (-9608.682631509726 - 1.7108925257214056e6*eta + 4.332924601416521e6*eta2 + (-22366.683262266528 - 2.5019716386377467e6*eta + 1.0274495902259542e7*eta2)*xi + (-85360.30079034246 - 570025.3441737515*eta + 4.396844346849777e6*eta2)*xi*xi)*xi
        
        # Compute beta coefficients appearing in arXiv:1508.07253 eq. (16)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        beta1 = 97.89747327985583 - 42.659730877489224*eta + (153.48421037904913 - 1417.0620760768954*eta + 2752.8614143665027*eta2 + (138.7406469558649 - 1433.6585075135881*eta + 2857.7418952430758*eta2)*xi + (41.025109467376126 - 423.680737974639*eta + 850.3594335657173*eta2)*xi*xi)*xi
        beta2 = -3.282701958759534 - 9.051384468245866*eta + (-12.415449742258042 + 55.4716447709787*eta - 106.05109938966335*eta2 + (-11.953044553690658 + 76.80704618365418*eta - 155.33172948098394*eta2)*xi + (-3.4129261592393263 + 25.572377569952536*eta - 54.408036707740465*eta2)*xi*xi)*xi
        beta3 = -0.000025156429818799565 + 0.000019750256942201327*eta + (-0.000018370671469295915 + 0.000021886317041311973*eta + 0.00008250240316860033*eta2 + (7.157371250566708e-6 - 0.000055780000112270685*eta + 0.00019142082884072178*eta2)*xi + (5.447166261464217e-6 - 0.00003220610095021982*eta + 0.00007974016714984341*eta2)*xi*xi)*xi
        
        # Compute alpha coefficients appearing in arXiv:1508.07253 eq. (14)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        alpha1 = 43.31514709695348 + 638.6332679188081*eta + (-32.85768747216059 + 2415.8938269370315*eta - 5766.875169379177*eta2 + (-61.85459307173841 + 2953.967762459948*eta - 8986.29057591497*eta2)*xi + (-21.571435779762044 + 981.2158224673428*eta - 3239.5664895930286*eta2)*xi*xi)*xi
        alpha2 = -0.07020209449091723 - 0.16269798450687084*eta + (-0.1872514685185499 + 1.138313650449945*eta - 2.8334196304430046*eta2 + (-0.17137955686840617 + 1.7197549338119527*eta - 4.539717148261272*eta2)*xi + (-0.049983437357548705 + 0.6062072055948309*eta - 1.682769616644546*eta2)*xi*xi)*xi
        alpha3 = 9.5988072383479 - 397.05438595557433*eta + (16.202126189517813 - 1574.8286986717037*eta + 3600.3410843831093*eta2 + (27.092429659075467 - 1786.482357315139*eta + 5152.919378666511*eta2)*xi + (11.175710130033895 - 577.7999423177481*eta + 1808.730762932043*eta2)*xi*xi)*xi
        alpha4 = -0.02989487384493607 + 1.4022106448583738*eta + (-0.07356049468633846 + 0.8337006542278661*eta + 0.2240008282397391*eta2 + (-0.055202870001177226 + 0.5667186343606578*eta + 0.7186931973380503*eta2)*xi + (-0.015507437354325743 + 0.15750322779277187*eta + 0.21076815715176228*eta2)*xi*xi)*xi
        alpha5 = 0.9974408278363099 - 0.007884449714907203*eta + (-0.059046901195591035 + 1.3958712396764088*eta - 4.516631601676276*eta2 + (-0.05585343136869692 + 1.7516580039343603*eta - 5.990208965347804*eta2)*xi + (-0.017945336522161195 + 0.5965097794825992*eta - 2.0608879367971804*eta2)*xi*xi)*xi
        
        # Compute the TF2 phase coefficients and put them in a dictionary (spin effects are included up to 3.5PN)
        TF2coeffs = {}
        TF2OverallAmpl = 3./(128. * eta)
        
        TF2coeffs['zero'] = 1.
        TF2coeffs['one'] = 0.
        TF2coeffs['two'] = 3715./756. + (55.*eta)/9.
        TF2coeffs['three'] = -16.*np.pi + (113.*Seta*chi_a)/3. + (113./3. - (76.*eta)/3.)*chi_s
        # For 2PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['four'] = 5.*(3058.673/7.056 + 5429./7.*eta+617.*eta2)/72. + 247./4.8*eta*chi1dotchi2 -721./4.8*eta*chi1dotchi2 + (-720./9.6*QuadMon1 + 1./9.6)*m1ByM*m1ByM*chi12 + (-720./9.6*QuadMon2 + 1./9.6)*m2ByM*m2ByM*chi22 + (240./9.6*QuadMon1 - 7./9.6)*m1ByM*m1ByM*chi12 + (240./9.6*QuadMon2 - 7./9.6)*m2ByM*m2ByM*chi22
        # This part is common to 5 and 5log, avoid recomputing
        TF2_5coeff_tmp = (732985./2268. - 24260.*eta/81. - 340.*eta2/9.)*chi_s + (732985./2268. + 140.*eta/9.)*Seta*chi_a
        TF2coeffs['five'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)
        TF2coeffs['five_log'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)*3.
        # For 3PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['six'] = 11583.231236531/4.694215680 - 640./3.*np.pi*np.pi - 684.8/2.1*np.euler_gamma + eta*(-15737.765635/3.048192 + 225.5/1.2*np.pi*np.pi) + eta2*76.055/1.728 - eta2*eta*127.825/1.296 - np.log(4.)*684.8/2.1 + np.pi*chi1*m1ByM*(1490./3. + m1ByM*260.) + np.pi*chi2*m2ByM*(1490./3. + m2ByM*260.) + (326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + (4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM)*m1ByM*m1ByM*QuadMon1*chi12 + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM)*m1ByM*m1ByM*chi12 + (4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM)*m2ByM*m2ByM*QuadMon2*chi22 + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM)*m2ByM*m2ByM*chi22
        TF2coeffs['six_log'] = -6848./21.
        TF2coeffs['seven'] = 77096675.*np.pi/254016. + 378515.*np.pi*eta/1512.- 74045.*np.pi*eta2/756. + (-25150083775./3048192. + 10566655595.*eta/762048. - 1042165.*eta2/3024. + 5345.*eta2*eta/36.)*chi_s + Seta*((-25150083775./3048192. + 26804935.*eta/6048. - 1985.*eta2/48.)*chi_a)
        # Remove this part since it was not available when IMRPhenomD was tuned
        TF2coeffs['six'] = TF2coeffs['six'] - ((326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + ((4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM) + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM))*m1ByM*m1ByM*chi12 + ((4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM) + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM))*m2ByM*m2ByM*chi22)
        # Now translate into inspiral coefficients, label with the power in front of which they appear
        PhiInspcoeffs = {}
        
        PhiInspcoeffs['initial_phasing'] = TF2coeffs['five']*TF2OverallAmpl
        PhiInspcoeffs['two_thirds'] = TF2coeffs['seven']*TF2OverallAmpl*(np.pi**(2./3.))
        PhiInspcoeffs['third'] = TF2coeffs['six']*TF2OverallAmpl*(np.pi**(1./3.))
        PhiInspcoeffs['third_log'] = TF2coeffs['six_log']*TF2OverallAmpl*(np.pi**(1./3.))
        PhiInspcoeffs['log'] = TF2coeffs['five_log']*TF2OverallAmpl
        PhiInspcoeffs['min_third'] = TF2coeffs['four']*TF2OverallAmpl*(np.pi**(-1./3.))
        PhiInspcoeffs['min_two_thirds'] = TF2coeffs['three']*TF2OverallAmpl*(np.pi**(-2./3.))
        PhiInspcoeffs['min_one'] = TF2coeffs['two']*TF2OverallAmpl/np.pi
        PhiInspcoeffs['min_four_thirds'] = TF2coeffs['one']*TF2OverallAmpl*(np.pi**(-4./3.))
        PhiInspcoeffs['min_five_thirds'] = TF2coeffs['zero']*TF2OverallAmpl*(np.pi**(-5./3.))
        PhiInspcoeffs['one'] = sigma1
        PhiInspcoeffs['four_thirds'] = sigma2 * 0.75
        PhiInspcoeffs['five_thirds'] = sigma3 * 0.6
        PhiInspcoeffs['two'] = sigma4 * 0.5
        
        #Now compute the coefficients to align the three parts
        
        fInsJoinPh = self.PHI_fJoin_INS
        fMRDJoinPh = 0.5*fring
        
        # First the Inspiral - Intermediate: we compute C1Int and C2Int coeffs
        # Equations to solve for to get C(1) continuous join
        # PhiIns (f)  =   PhiInt (f) + C1Int + C2Int f
        # Joining at fInsJoin
        # PhiIns (fInsJoin)  =   PhiInt (fInsJoin) + C1Int + C2Int fInsJoin
        # PhiIns'(fInsJoin)  =   PhiInt'(fInsJoin) + C2Int
        # This is the first derivative wrt f of the inspiral phase computed at fInsJoin, first add the PN contribution and then the higher order calibrated terms
        DPhiIns = (2.0*TF2coeffs['seven']*TF2OverallAmpl*((np.pi*fInsJoinPh)**(7./3.)) + (TF2coeffs['six']*TF2OverallAmpl + TF2coeffs['six_log']*TF2OverallAmpl * (1.0 + np.log(np.pi*fInsJoinPh)/3.))*((np.pi*fInsJoinPh)**(2.)) + TF2coeffs['five_log']*TF2OverallAmpl*((np.pi*fInsJoinPh)**(5./3.)) - TF2coeffs['four']*TF2OverallAmpl*((np.pi*fInsJoinPh)**(4./3.)) - 2.*TF2coeffs['three']*TF2OverallAmpl*(np.pi*fInsJoinPh) - 3.*TF2coeffs['two']*TF2OverallAmpl*((np.pi*fInsJoinPh)**(2./3.)) - 4.*TF2coeffs['one']*TF2OverallAmpl*((np.pi*fInsJoinPh)**(1./3.)) - 5.*TF2coeffs['zero']*TF2OverallAmpl)*np.pi/(3.*((np.pi*fInsJoinPh)**(8./3.)))
        DPhiIns = DPhiIns + (sigma1 + sigma2*(fInsJoinPh**(1./3.)) + sigma3*(fInsJoinPh**(2./3.)) + sigma4*fInsJoinPh)/eta
        # This is the first derivative of the Intermediate phase computed at fInsJoin
        DPhiInt = (beta1 + beta3/(fInsJoinPh**4) + beta2/fInsJoinPh)/eta
        
        C2Int = DPhiIns - DPhiInt
        
        # This is the inspiral phase computed at fInsJoin
        PhiInsJoin = PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fInsJoinPh**(2./3.)) + PhiInspcoeffs['third']*(fInsJoinPh**(1./3.)) + PhiInspcoeffs['third_log']*(fInsJoinPh**(1./3.))*np.log(np.pi*fInsJoinPh)/3. + PhiInspcoeffs['log']*np.log(np.pi*fInsJoinPh)/3. + PhiInspcoeffs['min_third']*(fInsJoinPh**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fInsJoinPh**(-2./3.)) + PhiInspcoeffs['min_one']/fInsJoinPh + PhiInspcoeffs['min_four_thirds']*(fInsJoinPh**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fInsJoinPh**(-5./3.)) + (PhiInspcoeffs['one']*fInsJoinPh + PhiInspcoeffs['four_thirds']*(fInsJoinPh**(4./3.)) + PhiInspcoeffs['five_thirds']*(fInsJoinPh**(5./3.)) + PhiInspcoeffs['two']*fInsJoinPh*fInsJoinPh)/eta
        # This is the Intermediate phase computed at fInsJoin
        PhiIntJoin = beta1*fInsJoinPh - beta3/(3.*fInsJoinPh*fInsJoinPh*fInsJoinPh) + beta2*np.log(fInsJoinPh)
        
        C1Int = PhiInsJoin - PhiIntJoin/eta - C2Int*fInsJoinPh
        
        # Now the same for Intermediate - Merger-Ringdown: we also need a temporary Intermediate Phase function
        PhiIntTempVal  = (beta1*fMRDJoinPh - beta3/(3.*fMRDJoinPh*fMRDJoinPh*fMRDJoinPh) + beta2*np.log(fMRDJoinPh))/eta + C1Int + C2Int*fMRDJoinPh
        DPhiIntTempVal = C2Int + (beta1 + beta3/(fMRDJoinPh**4) + beta2/fMRDJoinPh)/eta
        DPhiMRDVal     = (alpha1 + alpha2/(fMRDJoinPh*fMRDJoinPh) + alpha3/(fMRDJoinPh**(1./4.)) + alpha4/(fdamp*(1. + (fMRDJoinPh - alpha5*fring)*(fMRDJoinPh - alpha5*fring)/(fdamp*fdamp))))/eta
        PhiMRJoinTemp  = -(alpha2/fMRDJoinPh) + (4.0/3.0) * (alpha3 * (fMRDJoinPh**(3./4.))) + alpha1 * fMRDJoinPh + alpha4 * np.arctan((fMRDJoinPh - alpha5 * fring)/fdamp)
        
        C2MRD = DPhiIntTempVal - DPhiMRDVal
        C1MRD = PhiIntTempVal - PhiMRJoinTemp/eta - C2MRD*fMRDJoinPh
        
        # Compute coefficients gamma appearing in arXiv:1508.07253 eq. (19), the numerical coefficients are in Tab. 5
        gamma1 = 0.006927402739328343 + 0.03020474290328911*eta + (0.006308024337706171 - 0.12074130661131138*eta + 0.26271598905781324*eta2 + (0.0034151773647198794 - 0.10779338611188374*eta + 0.27098966966891747*eta2)*xi+ (0.0007374185938559283 - 0.02749621038376281*eta + 0.0733150789135702*eta2)*xi*xi)*xi
        gamma2 = 1.010344404799477 + 0.0008993122007234548*eta + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2 + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi*xi)*xi
        gamma3 = 1.3081615607036106 - 0.005537729694807678*eta +(-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2 + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi*xi)*xi
        # Compute fpeak, from arXiv:1508.07253 eq. (20), we remove the square root term in case it is complex
        fpeak = np.where(gamma2 >= 1.0, np.fabs(fring - (fdamp*gamma3)/gamma2), fring + (fdamp*(-1.0 + np.sqrt(1.0 - gamma2*gamma2))*gamma3)/gamma2)
        # Compute coefficients rho appearing in arXiv:1508.07253 eq. (30), the numerical coefficients are in Tab. 5
        rho1 = 3931.8979897196696 - 17395.758706812805*eta + (3132.375545898835 + 343965.86092361377*eta - 1.2162565819981997e6*eta2 + (-70698.00600428853 + 1.383907177859705e6*eta - 3.9662761890979446e6*eta2)*xi + (-60017.52423652596 + 803515.1181825735*eta - 2.091710365941658e6*eta2)*xi*xi)*xi
        rho2 = -40105.47653771657 + 112253.0169706701*eta + (23561.696065836168 - 3.476180699403351e6*eta + 1.137593670849482e7*eta2 + (754313.1127166454 - 1.308476044625268e7*eta + 3.6444584853928134e7*eta2)*xi + (596226.612472288 - 7.4277901143564405e6*eta + 1.8928977514040343e7*eta2)*xi*xi)*xi
        rho3 = 83208.35471266537 - 191237.7264145924*eta + (-210916.2454782992 + 8.71797508352568e6*eta - 2.6914942420669552e7*eta2 + (-1.9889806527362722e6 + 3.0888029960154563e7*eta - 8.390870279256162e7*eta2)*xi + (-1.4535031953446497e6 + 1.7063528990822166e7*eta - 4.2748659731120914e7*eta2)*xi*xi)*xi
        # Compute coefficients delta appearing in arXiv:1508.07253 eq. (21)
        f1Interm = self.AMP_fJoin_INS
        f3Interm = fpeak
        dfInterm = 0.5*(f3Interm - f1Interm)
        f2Interm = f1Interm + dfInterm
        # First write the inspiral coefficients, we put them in a dictionary and label with the power in front of which they appear
        amp0 = np.sqrt(2.0*eta/3.0)*(np.pi**(-1./6.))
        Acoeffs = {}
        Acoeffs['two_thirds'] = ((-969. + 1804.*eta)*(np.pi**(2./3.)))/672.
        Acoeffs['one'] = ((chi1*(81.*SetaPlus1 - 44.*eta) + chi2*(81. - 81.*Seta - 44.*eta))*np.pi)/48.
        Acoeffs['four_thirds'] = ((-27312085.0 - 10287648.*chi22 - 10287648.*chi12*SetaPlus1 + 10287648.*chi22*Seta+ 24.*(-1975055. + 857304.*chi12 - 994896.*chi1*chi2 + 857304.*chi22)*eta+ 35371056*eta2)* (np.pi**(4./3.)))/8.128512e6
        Acoeffs['five_thirds'] = ((np.pi**(5./3.)) * (chi2*(-285197.*(-1. + Seta) + 4.*(-91902. + 1579.*Seta)*eta - 35632.*eta2) + chi1*(285197.*SetaPlus1 - 4.*(91902. + 1579.*Seta)*eta - 35632.*eta2) + 42840.*(-1.0 + 4.*eta)*np.pi)) / 32256.
        Acoeffs['two'] = - ((np.pi**2.)*(-336.*(-3248849057.0 + 2943675504.*chi12 - 3339284256.*chi1*chi2 + 2943675504.*chi22)*eta2 - 324322727232.*eta2*eta - 7.*(-177520268561. + 107414046432.*chi22 + 107414046432.*chi12*SetaPlus1 - 107414046432.*chi22*Seta + 11087290368.*(chi1 + chi2 + chi1*Seta - chi2*Seta)*np.pi ) + 12.*eta*(-545384828789. - 176491177632.*chi1*chi2 + 202603761360.*chi22 + 77616.*chi12*(2610335. + 995766.*Seta) - 77287373856.*chi22*Seta + 5841690624.*(chi1 + chi2)*np.pi + 21384760320.*np.pi*np.pi)))/6.0085960704e10
        Acoeffs['seven_thirds'] = rho1
        Acoeffs['eight_thirds'] = rho2
        Acoeffs['three'] = rho3
        # v1 is the inspiral model evaluated at f1Interm
        v1 = 1. + (f1Interm**(2./3.))*Acoeffs['two_thirds'] + (f1Interm**(4./3.)) * Acoeffs['four_thirds'] + (f1Interm**(5./3.)) *  Acoeffs['five_thirds'] + (f1Interm**(7./3.)) * Acoeffs['seven_thirds'] + (f1Interm**(8./3.)) * Acoeffs['eight_thirds'] + f1Interm * (Acoeffs['one'] + f1Interm * Acoeffs['two'] + f1Interm*f1Interm * Acoeffs['three'])
        # d1 is the derivative of the inspiral model evaluated at f1
        d1 = ((-969. + 1804.*eta)*(np.pi**(2./3.)))/(1008.*(f1Interm**(1./3.))) + ((chi1*(81.*SetaPlus1 - 44.*eta) + chi2*(81. - 81.*Seta - 44.*eta))*np.pi)/48. + ((-27312085. - 10287648.*chi22 - 10287648.*chi12*SetaPlus1 + 10287648.*chi22*Seta + 24.*(-1975055. + 857304.*chi12 - 994896.*chi1*chi2 + 857304.*chi22)*eta + 35371056.*eta2)*(f1Interm**(1./3.))*(np.pi**(4./3.)))/6.096384e6 + (5.*(f1Interm**(2./3.))*(np.pi**(5./3.))*(chi2*(-285197.*(-1 + Seta)+ 4.*(-91902. + 1579.*Seta)*eta - 35632.*eta2) + chi1*(285197.*SetaPlus1- 4.*(91902. + 1579.*Seta)*eta - 35632.*eta2) + 42840.*(-1 + 4*eta)*np.pi))/96768.- (f1Interm*np.pi*np.pi*(-336.*(-3248849057.0 + 2943675504.*chi12 - 3339284256.*chi1*chi2 + 2943675504.*chi22)*eta2 - 324322727232.*eta2*eta - 7.*(-177520268561. + 107414046432.*chi22 + 107414046432.*chi12*SetaPlus1 - 107414046432.*chi22*Seta+ 11087290368*(chi1 + chi2 + chi1*Seta - chi2*Seta)*np.pi)+ 12.*eta*(-545384828789.0 - 176491177632.*chi1*chi2 + 202603761360.*chi22 + 77616.*chi12*(2610335. + 995766.*Seta)- 77287373856.*chi22*Seta + 5841690624.*(chi1 + chi2)*np.pi + 21384760320*np.pi*np.pi)))/3.0042980352e10+ (7.0/3.0)*(f1Interm**(4./3.))*rho1 + (8.0/3.0)*(f1Interm**(5./3.))*rho2 + 3.*(f1Interm*f1Interm)*rho3
        # v3 is the merger-ringdown model (eq. (19) of arXiv:1508.07253) evaluated at f3
        v3 = np.exp(-(f3Interm - fring)*gamma2/(fdamp*gamma3))* (fdamp*gamma3*gamma1) / ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3)
        # d2 is the derivative of the merger-ringdown model evaluated at f3
        d2 = ((-2.*fdamp*(f3Interm - fring)*gamma3*gamma1) / ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3) - (gamma2*gamma1))/(np.exp((f3Interm - fring)*gamma2/(fdamp*gamma3)) * ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3))
        # v2 is the value of the amplitude evaluated at f2. They come from the fit of the collocation points in the intermediate region
        v2 = 0.8149838730507785 + 2.5747553517454658*eta + (1.1610198035496786 - 2.3627771785551537*eta + 6.771038707057573*eta2 + (0.7570782938606834 - 2.7256896890432474*eta + 7.1140380397149965*eta2)*xi + (0.1766934149293479 - 0.7978690983168183*eta + 2.1162391502005153*eta2)*xi*xi)*xi
        # Now some definitions to speed up
        f1  = f1Interm
        f2  = f2Interm
        f3  = f3Interm
        f12 = f1Interm*f1Interm
        f13 = f1Interm*f12;
        f14 = f1Interm*f13;
        f15 = f1Interm*f14;
        f22 = f2Interm*f2Interm;
        f23 = f2Interm*f22;
        f24 = f2Interm*f23;
        f32 = f3Interm*f3Interm;
        f33 = f3Interm*f32;
        f34 = f3Interm*f33;
        f35 = f3Interm*f34;
        # Finally conpute the deltas
        delta0 = -((d2*f15*f22*f3 - 2.*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32 - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33 + 2.*d1*f13*f22*f33 - 2.*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33 - d1*f13*f2*f34 - d1*f12*f22*f34 + 2.*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35 + 4.*f12*f23*f32*v1 - 3.*f1*f24*f32*v1 - 8.*f12*f22*f33*v1 + 4.*f1*f23*f33*v1 + f24*f33*v1 + 4.*f12*f2*f34*v1 + f1*f22*f34*v1 - 2.*f23*f34*v1 - 2.*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2 + 3.*f14*f33*v2 - 3.*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2.*f14*f23*v3 - f13*f24*v3 + 2.*f15*f2*f3*v3 - f14*f22*f3*v3 - 4.*f13*f23*f3*v3 + 3.*f12*f24*f3*v3 - 4.*f14*f2*f32*v3 + 8.*f13*f22*f32*v3 - 4.*f12*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta0 = -((d2*f15*f22*f3 - 2.*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32 - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33 + 2*d1*f13*f22*f33 - 2*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33 - d1*f13*f2*f34 - d1*f12*f22*f34 + 2*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35 + 4*f12*f23*f32*v1 - 3*f1*f24*f32*v1 - 8*f12*f22*f33*v1 + 4*f1*f23*f33*v1 + f24*f33*v1 + 4*f12*f2*f34*v1 + f1*f22*f34*v1 - 2*f23*f34*v1 - 2*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2 + 3*f14*f33*v2 - 3*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2*f14*f23*v3 - f13*f24*v3 + 2*f15*f2*f3*v3 - f14*f22*f3*v3 - 4*f13*f23*f3*v3 + 3*f12*f24*f3*v3 - 4*f14*f2*f32*v3 + 8*f13*f22*f32*v3 - 4*f12*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3-f2)*(f3-f2)))
        delta1 = -((-(d2*f15*f22) + 2.*d2*f14*f23 - d2*f13*f24 - d2*f14*f22*f3 + 2.*d1*f13*f23*f3 + 2.*d2*f13*f23*f3 - 2*d1*f12*f24*f3 - d2*f12*f24*f3 + d2*f15*f32 - 3*d1*f13*f22*f32 - d2*f13*f22*f32 + 2*d1*f12*f23*f32 - 2*d2*f12*f23*f32 + d1*f1*f24*f32 + 2*d2*f1*f24*f32 - d2*f14*f33 + d1*f12*f22*f33 + 3*d2*f12*f22*f33 - 2*d1*f1*f23*f33 - 2*d2*f1*f23*f33 + d1*f24*f33 + d1*f13*f34 + d1*f1*f22*f34 - 2*d1*f23*f34 - d1*f12*f35 + d1*f22*f35 - 8*f12*f23*f3*v1 + 6*f1*f24*f3*v1 + 12*f12*f22*f32*v1 - 8*f1*f23*f32*v1 - 4*f12*f34*v1 + 2*f1*f35*v1 + 2*f15*f3*v2 - 4*f14*f32*v2 + 4*f12*f34*v2 - 2*f1*f35*v2 - 2*f15*f3*v3 + 8*f12*f23*f3*v3 - 6*f1*f24*f3*v3 + 4*f14*f32*v3 - 12*f12*f22*f32*v3 + 8*f1*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta2 = -((d2*f15*f2 - d1*f13*f23 - 3*d2*f13*f23 + d1*f12*f24 + 2.*d2*f12*f24 - d2*f15*f3 + d2*f14*f2*f3 - d1*f12*f23*f3 + d2*f12*f23*f3 + d1*f1*f24*f3 - d2*f1*f24*f3 - d2*f14*f32 + 3*d1*f13*f2*f32 + d2*f13*f2*f32 - d1*f1*f23*f32 + d2*f1*f23*f32 - 2*d1*f24*f32 - d2*f24*f32 - 2*d1*f13*f33 + 2*d2*f13*f33 - d1*f12*f2*f33 - 3*d2*f12*f2*f33 + 3*d1*f23*f33 + d2*f23*f33 + d1*f12*f34 - d1*f1*f2*f34 + d1*f1*f35 - d1*f2*f35 + 4*f12*f23*v1 - 3*f1*f24*v1 + 4*f1*f23*f3*v1 - 3*f24*f3*v1 - 12*f12*f2*f32*v1 + 4*f23*f32*v1 + 8*f12*f33*v1 - f1*f34*v1 - f35*v1 - f15*v2 - f14*f3*v2 + 8*f13*f32*v2 - 8*f12*f33*v2 + f1*f34*v2 + f35*v2 + f15*v3 - 4*f12*f23*v3 + 3*f1*f24*v3 + f14*f3*v3 - 4*f1*f23*f3*v3 + 3*f24*f3*v3 - 8*f13*f32*v3 + 12*f12*f2*f32*v3 - 4*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta3 = -((-2.*d2*f14*f2 + d1*f13*f22 + 3*d2*f13*f22 - d1*f1*f24 - d2*f1*f24 + 2*d2*f14*f3 - 2.*d1*f13*f2*f3 - 2*d2*f13*f2*f3 + d1*f12*f22*f3 - d2*f12*f22*f3 + d1*f24*f3 + d2*f24*f3 + d1*f13*f32 - d2*f13*f32 - 2*d1*f12*f2*f32 + 2*d2*f12*f2*f32 + d1*f1*f22*f32 - d2*f1*f22*f32 + d1*f12*f33 - d2*f12*f33 + 2*d1*f1*f2*f33 + 2*d2*f1*f2*f33 - 3*d1*f22*f33 - d2*f22*f33 - 2*d1*f1*f34 + 2*d1*f2*f34 - 4*f12*f22*v1 + 2*f24*v1 + 8*f12*f2*f3*v1 - 4*f1*f22*f3*v1 - 4*f12*f32*v1 + 8*f1*f2*f32*v1 - 4*f22*f32*v1 - 4*f1*f33*v1 + 2*f34*v1 + 2*f14*v2 - 4*f13*f3*v2 + 4*f1*f33*v2 - 2*f34*v2 - 2*f14*v3 + 4*f12*f22*v3 - 2*f24*v3 + 4*f13*f3*v3 - 8*f12*f2*f3*v3 + 4*f1*f22*f3*v3 + 4*f12*f32*v3 - 8*f1*f2*f32*v3 + 4*f22*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta4 = -((d2*f13*f2 - d1*f12*f22 - 2*d2*f12*f22 + d1*f1*f23 + d2*f1*f23 - d2*f13*f3 + 2.*d1*f12*f2*f3 + d2*f12*f2*f3 - d1*f1*f22*f3 + d2*f1*f22*f3 - d1*f23*f3 - d2*f23*f3 - d1*f12*f32 + d2*f12*f32 - d1*f1*f2*f32 - 2*d2*f1*f2*f32 + 2*d1*f22*f32 + d2*f22*f32 + d1*f1*f33 - d1*f2*f33 + 3*f1*f22*v1 - 2*f23*v1 - 6*f1*f2*f3*v1 + 3*f22*f3*v1 + 3*f1*f32*v1 - f33*v1 - f13*v2 + 3*f12*f3*v2 - 3*f1*f32*v2 + f33*v2 + f13*v3 - 3*f1*f22*v3 + 2*f23*v3 - 3*f12*f3*v3 + 6*f1*f2*f3*v3 - 3*f22*f3*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        
        # Defined as in LALSimulation - LALSimIMRPhenomUtils.c line 70. Final units are correctly Hz^-1
        # there is a 2 * sqrt(5/(64*pi)) missing w.r.t the standard coefficient, which comes from the (2,2) shperical harmonic
        Overallamp = M * glob.GMsun_over_c2_Gpc * M * glob.GMsun_over_c3 / kwargs['dL']
        
        def completeAmpl(infreqs):
            
            return Overallamp*amp0*(infreqs**(-7./6.))*np.where(infreqs < self.AMP_fJoin_INS, 1. + (infreqs**(2./3.))*Acoeffs['two_thirds'] + (infreqs**(4./3.)) * Acoeffs['four_thirds'] + (infreqs**(5./3.)) *  Acoeffs['five_thirds'] + (infreqs**(7./3.)) * Acoeffs['seven_thirds'] + (infreqs**(8./3.)) * Acoeffs['eight_thirds'] + infreqs * (Acoeffs['one'] + infreqs * Acoeffs['two'] + infreqs*infreqs * Acoeffs['three']), np.where(infreqs < fpeak, delta0 + infreqs*delta1 + infreqs*infreqs*(delta2 + infreqs*delta3 + infreqs*infreqs*delta4), np.where(infreqs < self.fcutPar,np.exp(-(infreqs - fring)*gamma2/(fdamp*gamma3))* (fdamp*gamma3*gamma1) / ((infreqs - fring)*(infreqs - fring) + fdamp*gamma3*fdamp*gamma3), 0.)))
        
        def completePhase(infreqs, C1MRDuse, C2MRDuse, RhoUse, TauUse):
            
            return np.where(infreqs < self.PHI_fJoin_INS, PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(infreqs**(2./3.)) + PhiInspcoeffs['third']*(infreqs**(1./3.)) + PhiInspcoeffs['third_log']*(infreqs**(1./3.))*np.log(np.pi*infreqs)/3. + PhiInspcoeffs['log']*np.log(np.pi*infreqs)/3. + PhiInspcoeffs['min_third']*(infreqs**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(infreqs**(-2./3.)) + PhiInspcoeffs['min_one']/infreqs + PhiInspcoeffs['min_four_thirds']*(infreqs**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(infreqs**(-5./3.)) + (PhiInspcoeffs['one']*infreqs + PhiInspcoeffs['four_thirds']*(infreqs**(4./3.)) + PhiInspcoeffs['five_thirds']*(infreqs**(5./3.)) + PhiInspcoeffs['two']*infreqs*infreqs)/eta, np.where(infreqs<fMRDJoinPh, (beta1*infreqs - beta3/(3.*infreqs*infreqs*infreqs) + beta2*np.log(infreqs))/eta + C1Int + C2Int*infreqs, np.where(infreqs < self.fcutPar, (-(alpha2/infreqs) + (4.0/3.0) * (alpha3 * (infreqs**(3./4.))) + alpha1 * infreqs + alpha4 * RhoUse * np.arctan((infreqs - alpha5 * fring)/(fdamp * RhoUse * TauUse)))/eta + C1MRDuse + C2MRDuse*infreqs,0.)))
 
        def OnePointFiveSpinPN(infreqs, ChiS, ChiA):
            # PN amplitudes function, needed to scale
            
            v  = np.moveaxis((2.*np.pi*infreqs/mms)**(1./3.), len(infreqs.shape)-1, len(infreqs.shape) - 2)
            v2 = v*v
            v3 = v2*v
            
            reshModes = np.expand_dims(modes, len(modes.shape))
            Hlm = np.where(reshModes==21, (np.sqrt(2.0) / 3.0) * (v * Seta - v2 * 1.5 * (ChiA + Seta * ChiS) + v3 * Seta * ((335.0 / 672.0) + (eta * 117.0 / 56.0)) + v3*v * (ChiA * (3427.0 / 1344. - eta * 2101.0 / 336.) + Seta * ChiS * (3427.0 / 1344 - eta * 965 / 336) + Seta * (-1j * 0.5 - np.pi - 2 * 1j * 0.69314718056))), np.where(reshModes==22, 1., np.where(reshModes==32, (1.0 / 3.0) * np.sqrt(5.0 / 7.0) * (v2 * (1.0 - 3.0 * eta)), np.where(reshModes==33, 0.75 * np.sqrt(5.0 / 7.0) * (v * Seta), np.where(reshModes==43, 0.75 * np.sqrt(3.0 / 35.0) * v3 * Seta * (1.0 - 2.0 * eta), (4.0 / 9.0) * np.sqrt(10.0 / 7.0) * v2 * (1.0 - 3.0 * eta))))))
            
            # Compute the final PN Amplitude at Leading Order in Mf
            
            return np.pi * np.sqrt(eta * 2. / 3.) * (v**(-3.5)) * abs(Hlm)
        
        def SpinWeighted_SphericalHarmonic(theta, phi=0.):
            # Taken from arXiv:0709.0093v3 eq. (II.7), (II.8) and LALSimulation for the s=-2 case and up to l=4.
            # We assume already phi=0 and s=-2 to simplify the function
            
            Ylm    = np.where(modes==21, np.sqrt( 5.0 / ( 16.0 * np.pi ) ) * np.sin( theta )*( 1.0 + np.cos( theta )), np.where(modes==22, np.sqrt( 5.0 / ( 64.0 * np.pi ) ) * ( 1.0 + np.cos( theta ))*( 1.0 + np.cos( theta )), np.where(modes==32, np.sqrt(7.0/np.pi)*((np.cos(theta*0.5))**(4.0))*(-2.0 + 3.0*np.cos(theta))*0.5, np.where(modes==33, -np.sqrt(21.0/(2.0*np.pi))*((np.cos(theta/2.0))**(5.0))*np.sin(theta*0.5), np.where(modes==43, -3.0*np.sqrt(7.0/(2.0*np.pi))*((np.cos(theta*0.5))**5.0)*(-1.0 + 2.0*np.cos(theta))*np.sin(theta*0.5), 3.0*np.sqrt(7.0/np.pi)*((np.cos(theta*0.5))**6.0)*(np.sin(theta*0.5)*np.sin(theta*0.5)))))))
            Ylminm = np.where(modes==21, np.sqrt( 5.0 / ( 16.0 * np.pi ) ) * np.sin( theta )*( 1.0 - np.cos( theta )), np.where(modes==22, np.sqrt( 5.0 / ( 64.0 * np.pi ) ) * ( 1.0 - np.cos( theta ))*( 1.0 - np.cos( theta )), np.where(modes==32, np.sqrt(7.0/(4.0*np.pi))*(2.0 + 3.0*np.cos(theta))*((np.sin(theta*0.5))**(4.0)), np.where(modes==33, np.sqrt(21.0/(2.0*np.pi))*np.cos(theta*0.5)*((np.sin(theta*0.5))**(5.)), np.where(modes==43, 3.0*np.sqrt(7.0/(2.0*np.pi))*np.cos(theta*0.5)*(1.0 + 2.0*np.cos(theta))*((np.sin(theta*0.5))**5.0), 3.0*np.sqrt(7.0/np.pi)*(np.cos(theta*0.5)*np.cos(theta*0.5))*((np.sin(theta*0.5))**6.0))))))
            
            return Ylm, Ylminm
        
        # Time shift so that peak amplitude is approximately at t=0
        t0 = (alpha1 + alpha2/(fpeak*fpeak) + alpha3/(fpeak**(1./4.)) + alpha4/(fdamp*(1. + (fpeak - alpha5*fring)*(fpeak - alpha5*fring)/(fdamp*fdamp))))/eta
        
        phiRef = completePhase(fRef, C1MRD, C2MRD, 1., 1.)
        phi0   = 0.5*phiRef
        
        # Now compute all the modes, they are 6, we parallelize
        
        Rholm, Taulm = (fring/fringlm.T), (fdamplm.T/fdamp)
        # Rholm and Taulm only figure in the MRD part, the rest of the coefficients is the same, recompute only this
        DPhiMRDVal    = (alpha1 + alpha2/(fMRDJoinPh*fMRDJoinPh) + alpha3/(fMRDJoinPh**(1./4.)) + alpha4/(fdamp*Taulm*(1. + (fMRDJoinPh - alpha5*fring)*(fMRDJoinPh - alpha5*fring)/(fdamp*Taulm*Rholm*fdamp*Taulm*Rholm))))/eta
        PhiMRJoinTemp = -(alpha2/fMRDJoinPh) + (4.0/3.0) * (alpha3 * (fMRDJoinPh**(3./4.))) + alpha1 * fMRDJoinPh + alpha4 * Rholm* np.arctan((fMRDJoinPh - alpha5 * fring)/(fdamp*Rholm*Taulm))
        C2MRDHM = DPhiIntTempVal - DPhiMRDVal
        C1MRDHM = (PhiIntTempVal - PhiMRJoinTemp/eta - C2MRDHM*fMRDJoinPh).T
        Rholm, Taulm, DPhiMRDVal, PhiMRJoinTemp, C2MRDHM = Rholm.T, Taulm.T, DPhiMRDVal.T, PhiMRJoinTemp.T, C2MRDHM.T
        
        # Scale input frequencies according to PhenomHM model
        # Compute mapping coefficinets
        Map_flPhi = self.PHI_fJoin_INS
        Map_fiPhi = Map_flPhi / Rholm
        Map_flAmp = self.AMP_fJoin_INS
        Map_fiAmp = Map_flAmp / Rholm
        Map_fr = fringlm
        
        Map_ai, Map_bi = 2./mms, 0.

        Map_TrdAmp = Map_fr - fringlm + np.expand_dims(fring, len(fring.shape))
        Map_TiAmp  = 2. * Map_fiAmp / mms
        Map_amAmp  = (Map_TrdAmp - Map_TiAmp) / (Map_fr - Map_fiAmp)
        Map_bmAmp  = Map_TiAmp - Map_fiAmp * Map_amAmp

        Map_TrdPhi = Map_fr * Rholm
        Map_TiPhi  = 2. * Map_fiPhi / mms
        Map_amPhi  = (Map_TrdPhi - Map_TiPhi) / (Map_fr - Map_fiPhi)
        Map_bmPhi  = Map_TiPhi - Map_fiPhi * Map_amPhi

        Map_arAmp, Map_brAmp = 1., - Map_fr + np.expand_dims(fring, len(fring.shape))
        Map_arPhi, Map_brPhi = Rholm, 0.
        
        # Now scale as f -> f*a+b for each regime
        fgrid = np.expand_dims(fgrid, len(fgrid.shape))# Need a new axis to do all the 6 calculations together

        fgridScaled = np.where(fgrid < Map_fiAmp, fgrid*Map_ai + Map_bi, np.where(fgrid < Map_fr, fgrid*Map_amAmp + Map_bmAmp, fgrid*Map_arAmp + Map_brAmp))
        # Map the ampliude's range
        # We divide by the leading order l=m=2 behavior, and then scale in the expected PN behavior for the multipole of interest.
              
        beta_term1  = OnePointFiveSpinPN(fgrid, chi_s, chi_a)
        beta_term2  = OnePointFiveSpinPN(2.*fgrid/mms, chi_s, chi_a)
        HMamp_term1 = OnePointFiveSpinPN(fgridScaled, chi_s, chi_a)
        fgridScaled = np.moveaxis(fgridScaled, len(fgridScaled.shape)-1, len(fgridScaled.shape) - 2)
        #fgridScaled = fgridScaled.transpose(0,2,1)
        HMamp_term2 = np.pi * np.sqrt(eta * 2. / 3.) * ((np.pi*fgridScaled)**(-7./6.))
        
        # The (3,3) and (4,3) modes vanish if eta=0.25 (equal mass case) and the (2,1) mode vanishes if both eta=0.25 and chi1z=chi2z
        # This results in NaNs having 0/0, correct for this using np.nan_to_num()
                
        AmplsAllModes = np.nan_to_num(completeAmpl(fgridScaled) * (beta_term1 / beta_term2) * HMamp_term1 / HMamp_term2)
        AmplsAllModes = np.moveaxis(AmplsAllModes, len(AmplsAllModes.shape)-1, len(AmplsAllModes.shape) - 2)
        #AmplsAllModes = AmplsAllModes.transpose(0,2,1)
        C1MRDHM, C2MRDHM, Rholm, Taulm = C1MRDHM.T, C2MRDHM.T, Rholm.T, Taulm.T
        
        tmpMf = Map_amPhi * Map_fiPhi + Map_bmPhi
        PhDBconst = (completePhase(tmpMf.T, C1MRDHM, C2MRDHM, Rholm, Taulm) / Map_amPhi.T)
                    
        tmpMf = Map_arPhi * Map_fr + Map_brPhi
        PhDCconst = (completePhase(tmpMf.T, C1MRDHM, C2MRDHM, Rholm, Taulm) / Map_arPhi.T)
            
        tmpMf = Map_ai * Map_fiPhi + Map_bi
        PhDBAterm = (completePhase(tmpMf.T, C1MRDHM, C2MRDHM, Rholm, Taulm).T / Map_ai).T
        
        tmpMf = Map_amPhi * Map_fr + Map_bmPhi
        tmpphaseC = (- PhDBconst + PhDBAterm + completePhase(tmpMf.T, C1MRDHM, C2MRDHM, Rholm, Taulm) / Map_amPhi.T)
        
        tmpGridShape = len((fgrid*Map_ai + Map_bi).shape)
                                
        if len(AmplsAllModes.shape)==3:
            PhisAllModes = np.where(fgrid < Map_fiPhi, np.moveaxis(completePhase(np.moveaxis((fgrid*Map_ai + Map_bi), tmpGridShape-1, tmpGridShape-2), C1MRDHM, C2MRDHM, Rholm, Taulm), len(AmplsAllModes.shape)-1, len(AmplsAllModes.shape)-2)/Map_ai, np.where(fgrid < Map_fr, np.moveaxis(- PhDBconst + PhDBAterm + completePhase(np.moveaxis((fgrid*Map_amPhi + Map_bmPhi), tmpGridShape-1, tmpGridShape-2), C1MRDHM, C2MRDHM, Rholm, Taulm)/Map_amPhi.T, len(AmplsAllModes.shape)-1, len(AmplsAllModes.shape)-2), np.moveaxis(- PhDCconst + tmpphaseC + completePhase(np.moveaxis((fgrid*Map_arPhi + Map_brPhi), tmpGridShape-1, tmpGridShape-2), C1MRDHM, C2MRDHM, Rholm, Taulm)/Map_arPhi.T, len(AmplsAllModes.shape)-1, len(AmplsAllModes.shape)-2)))
        else:
            C1MRDHM, C2MRDHM, Rholm, Taulm = C1MRDHM.T, C2MRDHM.T, Rholm.T, Taulm.T
            PhDBconst, PhDCconst, PhDBAterm, tmpphaseC = PhDBconst.T, PhDCconst.T, PhDBAterm.T, tmpphaseC.T
            PhisAllModes = np.where(fgrid < Map_fiPhi, completePhase((fgrid*Map_ai + Map_bi), C1MRDHM, C2MRDHM, Rholm, Taulm)/Map_ai, np.where(fgrid < Map_fr, - PhDBconst + PhDBAterm + completePhase((fgrid*Map_amPhi + Map_bmPhi), C1MRDHM, C2MRDHM, Rholm, Taulm)/Map_amPhi, - PhDCconst + tmpphaseC + completePhase((fgrid*Map_arPhi + Map_brPhi), C1MRDHM, C2MRDHM, Rholm, Taulm)/Map_arPhi))
            
        PhisAllModes = PhisAllModes - np.expand_dims(t0, len(t0.shape))*(fgrid - np.expand_dims(fRef, len(fRef.shape))) - mms*np.expand_dims(phi0, len(phi0.shape)) + self.complShiftm[mms]
        modes = np.expand_dims(modes, len(modes.shape))
        Y, Ymstar = SpinWeighted_SphericalHarmonic(iota)
        Y, Ymstar = Y.T, np.conj(Ymstar).T
        
        hp = np.sum(AmplsAllModes*np.exp(-1j*PhisAllModes)*(0.5*(Y + ((-1)**ells)*Ymstar)), axis=-1)
        hc = np.sum(AmplsAllModes*np.exp(-1j*PhisAllModes)*(-1j* 0.5 * (Y - ((-1)**ells)* Ymstar)), axis=-1)
        
        return hp, hc

        
    def _finalspin(self, eta, chi1, chi2):
        # Compute the spin of the final object, as in LALSimIMRPhenomD_internals.c line 161 and 142,
        # which is taken from arXiv:1508.07250 eq. (3.6)
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        s  = (m1*m1 * chi1 + m2*m2 * chi2)
        af1 = eta*(3.4641016151377544 - 4.399247300629289*eta + 9.397292189321194*eta*eta - 13.180949901606242*eta*eta*eta)
        af2 = eta*(s*((1.0/eta - 0.0850917821418767 - 5.837029316602263*eta) + (0.1014665242971878 - 2.0967746996832157*eta)*s))
        af3 = eta*(s*((-1.3546806617824356 + 4.108962025369336*eta)*s*s + (-0.8676969352555539 + 2.064046835273906*eta)*s*s*s))
        return af1 + af2 + af3
        
    def _radiatednrg(self, eta, chi1, chi2):
        # Predict the total radiated energy, from arXiv:1508.07250 eq (3.7) and (3.8)
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        s  = (m1*m1 * chi1 + m2*m2 * chi2) / (m1*m1 + m2*m2)
        
        EradNS = eta * (0.055974469826360077 + 0.5809510763115132 * eta - 0.9606726679372312 * eta*eta + 3.352411249771192 * eta*eta*eta)
        
        return (EradNS * (1. + (-0.0030302335878845507 - 2.0066110851351073 * eta + 7.7050567802399215 * eta*eta) * s)) / (1. + (-0.6714403054720589 - 1.4756929437702908 * eta + 7.304676214885011 * eta*eta) * s)
    
    def _RDfreqCalc(self, finalmass, finalspin, l, m):
        # Compute the real and imag parts of the complex ringdown frequency for the (l,m) mode as in LALSimIMRPhenomHM.c line 189
        # These are all fits of the different modes
        
        # Domain mapping for dimnesionless BH spin
        alpha = np.log(2. - finalspin) / np.log(3.);
        beta = 1. / (2. + l - abs(m));
        kappa  = alpha**beta
        kappa2 = kappa*kappa
        kappa3 = kappa*kappa2
        kappa4 = kappa*kappa3
        
        if (2 == l) and (2 == m):
            res = 1.0 + kappa * (1.557847 * np.exp(2.903124 * 1j) + 1.95097051 * np.exp(5.920970 * 1j) * kappa + 2.09971716 * np.exp(2.760585 * 1j) * kappa2 + 1.41094660 * np.exp(5.914340 * 1j) * kappa3 + 0.41063923 * np.exp(2.795235 * 1j) * kappa4)
        
        elif (3 == l) and (2 == m):
            res = 1.022464 * np.exp(0.004870 * 1j) + 0.24731213 * np.exp(0.665292 * 1j) * kappa + 1.70468239 * np.exp(3.138283 * 1j) * kappa2 + 0.94604882 * np.exp(0.163247 * 1j) * kappa3 + 1.53189884 * np.exp(5.703573 * 1j) * kappa4 + 2.28052668 * np.exp(2.685231 * 1j) * kappa4*kappa + 0.92150314 * np.exp(5.841704 * 1j) * kappa4*kappa2
        
        elif (4 == l) and (4 == m):
            res = 2.0 + kappa * (2.658908 * np.exp(3.002787 * 1j) + 2.97825567 * np.exp(6.050955 * 1j) * kappa + 3.21842350 * np.exp(2.877514 * 1j) * kappa2 + 2.12764967 * np.exp(5.989669 * 1j) * kappa3 + 0.60338186 * np.exp(2.830031 * 1j) * kappa4)
        
        elif (2 == l) and (1 == m):
            res = 0.589113 * np.exp(0.043525 * 1j) + 0.18896353 * np.exp(2.289868 * 1j) * kappa + 1.15012965 * np.exp(5.810057 * 1j) * kappa2 + 6.04585476 * np.exp(2.741967 * 1j) * kappa3 + 11.12627777 * np.exp(5.844130 * 1j) * kappa4 + 9.34711461 * np.exp(2.669372 * 1j) * kappa4*kappa + 3.03838318 * np.exp(5.791518 * 1j) * kappa4*kappa2
        
        elif (3 == l) and (3 == m):
            res = 1.5 + kappa * (2.095657 * np.exp(2.964973 * 1j) + 2.46964352 * np.exp(5.996734 * 1j) * kappa + 2.66552551 * np.exp(2.817591 * 1j) * kappa2 + 1.75836443 * np.exp(5.932693 * 1j) * kappa3 + 0.49905688 * np.exp(2.781658 * 1j) * kappa4)
        
        elif (4 == l) and (3 == m):
            res = 1.5 + kappa * (0.205046 * np.exp(0.595328 * 1j) + 3.10333396 * np.exp(3.016200 * 1j) * kappa + 4.23612166 * np.exp(6.038842 * 1j) * kappa2 + 3.02890198 * np.exp(2.826239 * 1j) * kappa3 + 0.90843949 * np.exp(5.915164 * 1j) * kappa4)
        
        else:
            raise ValueError('Mode not present in IMRPhenomHM waveform model.')
        
        if m < 0:
            res = -np.conj(res)
        
        fring = np.real(res)/(2.*np.pi*finalmass)
        
        fdamp = np.imag(res)/(2.*np.pi*finalmass)
        
        return fring, fdamp
        
    def tau_star(self, f, **kwargs):
        # For complex waveforms we use the expression in arXiv:0907.0700 eq. (3.8b)
        Mtot_sec = kwargs['Mc']*glob.GMsun_over_c3/(kwargs['eta']**(3./5.))
        v = (np.pi*Mtot_sec*f)**(1./3.)
        eta  = kwargs['eta']
        eta2 = eta*eta
        
        OverallFac = 5./256 * Mtot_sec/(eta*(v**8.))
        
        t05 = 1. + (743./252. + 11./3.*eta)*(v*v) - 32./5.*np.pi*(v*v*v) + (3058673./508032. + 5429./504.*eta + 617./72.*eta2)*(v**4) - (7729./252. - 13./3.*eta)*np.pi*(v**5)
        t6  = (-10052469856691./23471078400. + 128./3.*np.pi*np.pi + 6848./105.*np.euler_gamma + (3147553127./3048192. - 451./12.*np.pi*np.pi)*eta - 15211./1728.*eta2 + 25565./1296.*eta2*eta + 3424./105.*np.log(16.*v*v))*(v**6)
        t7  = (- 15419335./127008. - 75703./756.*eta + 14809./378.*eta2)*np.pi*(v**7)
        
        return OverallFac*(t05 + t6 + t7)
    
    def fcut(self, **kwargs):
        
        return self.fcutPar/(kwargs['Mc']*glob.GMsun_over_c3/(kwargs['eta']**(3./5.)))

class IMRPhenomNSBH(WaveFormModel):
    '''
    IMRPhenomNSBH waveform model
    The inputs labelled as 1 refer to the BH (e.g. chi1z) and with 2 to the NS (e.g. Lambda2)
    
    NOTE: In LAL, to compute the parameter xi_tide in arXiv:1509.00512 eq. (8), the roots are extracted.
          In python this would break the possibility to vectorise so, to circumvent the issue, we compute
          a grid of xi_tide as a function of the compactness, mass ratio and BH spin, and then use a 3D
          interpolator. The first time the code runs, if this interpolator is not already present, it will be
          computed (the base resolution of the grid is 200 pts per parameter, that we find
          sufficient to reproduce LAL waveforms with sufficient precision, given the smooth behaviour of the function,
          but this can be raised if needed. In this case, it is necessary to change tha name of the file assigned to self.path_xiTide_tab and the res parameter passed to _make_xiTide_interpolator())
    '''
    # All is taken from LALSimulation and arXiv:1508.07250, arXiv:1508.07253, arXiv:1509.00512, arXiv:1905.06011
    def __init__(self, verbose=True, **kwargs):
    
        # Dimensionless frequency (Mf) at which the inspiral phase switches to the intermediate phase
        self.PHI_fJoin_INS = 0.018
        # Dimensionless frequency (Mf) at which we define the end of the waveform
        fcutPar = 0.2
        self.verbose=verbose
        super().__init__('NSBH', fcutPar, is_tidal=True, **kwargs)
        
        self.QNMgrid_a       = onp.loadtxt(os.path.join(glob.WFfilesPath, 'QNMData_a.txt'))
        self.QNMgrid_fring   = onp.loadtxt(os.path.join(glob.WFfilesPath, 'QNMData_fring.txt'))
        self.QNMgrid_fdamp   = onp.loadtxt(os.path.join(glob.WFfilesPath, 'QNMData_fdamp.txt'))
        self.path_xiTide_tab = os.path.join(glob.WFfilesPath, 'xiTide_Table_200.h5')
        
        self._make_xiTide_interpolator(res=200)
        
    def Phi(self, f, **kwargs):
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        eta2 = eta*eta # These can speed up a bit, we call them multiple times
        etaInv = 1./eta
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        
        #if np.any(chi2 != 0.): # This if does not work in JAX jacrev
        #    print('WARNING: IMRPhenomNSBH is tuned only for chi_NS = 0')
        
        chi12, chi22 = chi1*chi1, chi2*chi2
        chi1dotchi2 = chi1*chi2
        
        Lambda1, Lambda = kwargs['Lambda1'], kwargs['Lambda2']
        
        #if np.any(Lambda1 != 0.): # This if does not work in JAX jacrev
        #    print('WARNING: BH tidal deformability cannot be different from 0, discarding it')

        del Lambda1
        # A non-zero tidal deformability induces a quadrupole moment (for BBH it is 1).
        # Taken from arXiv:1303.1528 eq. (54) and Tab. I
        QuadMon1, QuadMon2 = 1., np.where(Lambda<1e-5, 1., np.exp(0.194 + 0.0936*np.log(Lambda) + 0.0474*np.log(Lambda)*np.log(Lambda) - 0.00421*np.log(Lambda)*np.log(Lambda)*np.log(Lambda) + 0.000123*np.log(Lambda)*np.log(Lambda)*np.log(Lambda)*np.log(Lambda)))
        
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        SetaPlus1 = 1.0 + Seta
        chi_s = 0.5 * (chi1 + chi2)
        chi_a = 0.5 * (chi1 - chi2)
        q = 0.5*(1.0 + Seta - 2.0*eta)/eta
        chi_s2, chi_a2 = chi_s*chi_s, chi_a*chi_a
        chi1dotchi2 = chi1*chi2
        chi_sdotchi_a = chi_s*chi_a
        # These are m1/Mtot and m2/Mtot
        m1ByM = 0.5 * (1.0 + Seta)
        m2ByM = 0.5 * (1.0 - Seta)
        # We work in dimensionless frequency M*f, not f
        fgrid = M*glob.GMsun_over_c3*f
        # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
        chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
        xi = - 1.0 + chiPN
        # Compute final spin and radiated energy for IMRPhenomNSBH, the rest is equivalent to IMRPhenomD_NRTidalv2
        # Get remnant spin for assumed aligned spin system, from arXiv:1903.11622 Table I and eq. (4), (5) and (6)
        
        p1_remSp = ((-5.44187381e-03*chi1 + 7.91165608e-03) + (2.33362046e-02*chi1 + 2.47764497e-02)*eta)*eta
        p2_remSp = ((-8.56844797e-07*chi1 - 2.81727682e-06) + (6.61290966e-06*chi1 + 4.28979016e-05)*eta)*eta
        p3_remSp = ((-3.04174272e-02*chi1 + 2.54889050e-01) + (1.47549350e-01*chi1 - 4.27905832e-01)*eta)*eta
        
        modelRemSp = (1. + Lambda * p1_remSp + Lambda*Lambda * p2_remSp) / ((1. + Lambda*p3_remSp*p3_remSp)*(1. + Lambda*p3_remSp*p3_remSp))

        modelRemSp = np.where((chi1 < 0.) & (eta < 0.188), 1., modelRemSp)
        modelRemSp = np.where(chi1 < -0.5, 1., modelRemSp)
        modelRemSp = np.where(modelRemSp > 1., 1., modelRemSp)
        
        del p1_remSp, p2_remSp, p3_remSp
        
        # Work with spin variables weighted on square of the BH mass over total mass
        S1BH = chi1 * m1ByM * m1ByM
        Shat = S1BH / (m1ByM*m1ByM + m2ByM*m2ByM) # this would be = (chi1*m1*m1 + chi2*m2*m2)/(m1*m1 + m2*m2), but chi2=0 by assumption
        
        # Compute fit to L_orb in arXiv:1611.00332 eq. (16)
        Lorb = (2.*np.sqrt(3.)*eta + 5.24*3.8326341618708577*eta2 + 1.3*(-9.487364155598392)*eta*eta2)/(1. + 2.88*2.5134875145648374*eta) + ((-0.194)*1.0009563702914628*Shat*(4.409160174224525*eta + 0.5118334706832706*eta2 + (64. - 16.*4.409160174224525 - 4.*0.5118334706832706)*eta2*eta) + 0.0851*0.7877509372255369*Shat*Shat*(8.77367320110712*eta + (-32.060648277652994)*eta2 + (64. - 16.*8.77367320110712 - 4.*(-32.060648277652994))*eta2*eta) + 0.00954*0.6540138407185817*Shat*Shat*Shat*(22.830033250479833*eta + (-153.83722669033995)*eta2 + (64. - 16.*22.830033250479833 - 4.*(-153.83722669033995))*eta2*eta))/(1. + (-0.579)*0.8396665722805308*Shat*(1.8804718791591157 + (-4.770246856212403)*eta + 0.*eta2 + (64. - 64.*1.8804718791591157 - 16.*(-4.770246856212403) - 4.*0.)*eta2*eta)) + 0.3223660562764661*Seta*eta2*(1. + 9.332575956437443*eta)*chi1 + 2.3170397514509933*Shat*Seta*eta2*eta*(1. + (-3.2624649875884852)*eta)*chi1 + (-0.059808322561702126)*eta2*eta*chi12;
        
        chif = (Lorb + S1BH)*modelRemSp

        Erad = self._radiatednrg(eta, chi1, chi2)
        # Compute ringdown and damping frequencies from interpolators
        fring = np.interp(chif, self.QNMgrid_a, self.QNMgrid_fring) / (1.0 - Erad)
        fdamp = np.interp(chif, self.QNMgrid_a, self.QNMgrid_fdamp) / (1.0 - Erad)
        
        # Compute sigma coefficients appearing in arXiv:1508.07253 eq. (28)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        sigma1 = 2096.551999295543 + 1463.7493168261553*eta + (1312.5493286098522 + 18307.330017082117*eta - 43534.1440746107*eta2 + (-833.2889543511114 + 32047.31997183187*eta - 108609.45037520859*eta2)*xi + (452.25136398112204 + 8353.439546391714*eta - 44531.3250037322*eta2)*xi*xi)*xi
        sigma2 = -10114.056472621156 - 44631.01109458185*eta + (-6541.308761668722 - 266959.23419307504*eta + 686328.3229317984*eta2 + (3405.6372187679685 - 437507.7208209015*eta + 1.6318171307344697e6*eta2)*xi + (-7462.648563007646 - 114585.25177153319*eta + 674402.4689098676*eta2)*xi*xi)*xi
        sigma3 = 22933.658273436497 + 230960.00814979506*eta + (14961.083974183695 + 1.1940181342318142e6*eta - 3.1042239693052764e6*eta2 + (-3038.166617199259 + 1.8720322849093592e6*eta - 7.309145012085539e6*eta2)*xi + (42738.22871475411 + 467502.018616601*eta - 3.064853498512499e6*eta2)*xi*xi)*xi
        sigma4 = -14621.71522218357 - 377812.8579387104*eta + (-9608.682631509726 - 1.7108925257214056e6*eta + 4.332924601416521e6*eta2 + (-22366.683262266528 - 2.5019716386377467e6*eta + 1.0274495902259542e7*eta2)*xi + (-85360.30079034246 - 570025.3441737515*eta + 4.396844346849777e6*eta2)*xi*xi)*xi
        
        # Compute beta coefficients appearing in arXiv:1508.07253 eq. (16)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        beta1 = 97.89747327985583 - 42.659730877489224*eta + (153.48421037904913 - 1417.0620760768954*eta + 2752.8614143665027*eta2 + (138.7406469558649 - 1433.6585075135881*eta + 2857.7418952430758*eta2)*xi + (41.025109467376126 - 423.680737974639*eta + 850.3594335657173*eta2)*xi*xi)*xi
        beta2 = -3.282701958759534 - 9.051384468245866*eta + (-12.415449742258042 + 55.4716447709787*eta - 106.05109938966335*eta2 + (-11.953044553690658 + 76.80704618365418*eta - 155.33172948098394*eta2)*xi + (-3.4129261592393263 + 25.572377569952536*eta - 54.408036707740465*eta2)*xi*xi)*xi
        beta3 = -0.000025156429818799565 + 0.000019750256942201327*eta + (-0.000018370671469295915 + 0.000021886317041311973*eta + 0.00008250240316860033*eta2 + (7.157371250566708e-6 - 0.000055780000112270685*eta + 0.00019142082884072178*eta2)*xi + (5.447166261464217e-6 - 0.00003220610095021982*eta + 0.00007974016714984341*eta2)*xi*xi)*xi
        
        # Compute alpha coefficients appearing in arXiv:1508.07253 eq. (14)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        alpha1 = 43.31514709695348 + 638.6332679188081*eta + (-32.85768747216059 + 2415.8938269370315*eta - 5766.875169379177*eta2 + (-61.85459307173841 + 2953.967762459948*eta - 8986.29057591497*eta2)*xi + (-21.571435779762044 + 981.2158224673428*eta - 3239.5664895930286*eta2)*xi*xi)*xi
        alpha2 = -0.07020209449091723 - 0.16269798450687084*eta + (-0.1872514685185499 + 1.138313650449945*eta - 2.8334196304430046*eta2 + (-0.17137955686840617 + 1.7197549338119527*eta - 4.539717148261272*eta2)*xi + (-0.049983437357548705 + 0.6062072055948309*eta - 1.682769616644546*eta2)*xi*xi)*xi
        alpha3 = 9.5988072383479 - 397.05438595557433*eta + (16.202126189517813 - 1574.8286986717037*eta + 3600.3410843831093*eta2 + (27.092429659075467 - 1786.482357315139*eta + 5152.919378666511*eta2)*xi + (11.175710130033895 - 577.7999423177481*eta + 1808.730762932043*eta2)*xi*xi)*xi
        alpha4 = -0.02989487384493607 + 1.4022106448583738*eta + (-0.07356049468633846 + 0.8337006542278661*eta + 0.2240008282397391*eta2 + (-0.055202870001177226 + 0.5667186343606578*eta + 0.7186931973380503*eta2)*xi + (-0.015507437354325743 + 0.15750322779277187*eta + 0.21076815715176228*eta2)*xi*xi)*xi
        alpha5 = 0.9974408278363099 - 0.007884449714907203*eta + (-0.059046901195591035 + 1.3958712396764088*eta - 4.516631601676276*eta2 + (-0.05585343136869692 + 1.7516580039343603*eta - 5.990208965347804*eta2)*xi + (-0.017945336522161195 + 0.5965097794825992*eta - 2.0608879367971804*eta2)*xi*xi)*xi
        
        # Compute the TF2 phase coefficients and put them in a dictionary (spin effects are included up to 3.5PN)
        TF2coeffs = {}
        TF2OverallAmpl = 3./(128. * eta)
        
        TF2coeffs['zero'] = 1.
        TF2coeffs['one'] = 0.
        TF2coeffs['two'] = 3715./756. + (55.*eta)/9.
        TF2coeffs['three'] = -16.*np.pi + (113.*Seta*chi_a)/3. + (113./3. - (76.*eta)/3.)*chi_s
        # For 2PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['four'] = 5.*(3058.673/7.056 + 5429./7.*eta+617.*eta2)/72. + 247./4.8*eta*chi1dotchi2 -721./4.8*eta*chi1dotchi2 + (-720./9.6*QuadMon1 + 1./9.6)*m1ByM*m1ByM*chi12 + (-720./9.6*QuadMon2 + 1./9.6)*m2ByM*m2ByM*chi22 + (240./9.6*QuadMon1 - 7./9.6)*m1ByM*m1ByM*chi12 + (240./9.6*QuadMon2 - 7./9.6)*m2ByM*m2ByM*chi22
        # This part is common to 5 and 5log, avoid recomputing
        TF2_5coeff_tmp = (732985./2268. - 24260.*eta/81. - 340.*eta2/9.)*chi_s + (732985./2268. + 140.*eta/9.)*Seta*chi_a
        TF2coeffs['five'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)
        TF2coeffs['five_log'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)*3.
        # For 3PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['six'] = 11583.231236531/4.694215680 - 640./3.*np.pi*np.pi - 684.8/2.1*np.euler_gamma + eta*(-15737.765635/3.048192 + 225.5/1.2*np.pi*np.pi) + eta2*76.055/1.728 - eta2*eta*127.825/1.296 - np.log(4.)*684.8/2.1 + np.pi*chi1*m1ByM*(1490./3. + m1ByM*260.) + np.pi*chi2*m2ByM*(1490./3. + m2ByM*260.) + (326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + (4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM)*m1ByM*m1ByM*QuadMon1*chi12 + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM)*m1ByM*m1ByM*chi12 + (4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM)*m2ByM*m2ByM*QuadMon2*chi22 + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM)*m2ByM*m2ByM*chi22
        TF2coeffs['six_log'] = -6848./21.
        TF2coeffs['seven'] = 77096675.*np.pi/254016. + 378515.*np.pi*eta/1512.- 74045.*np.pi*eta2/756. + (-25150083775./3048192. + 10566655595.*eta/762048. - 1042165.*eta2/3024. + 5345.*eta2*eta/36.)*chi_s + Seta*((-25150083775./3048192. + 26804935.*eta/6048. - 1985.*eta2/48.)*chi_a)
        # Remove this part since it was not available when IMRPhenomD was tuned
        TF2coeffs['six'] = TF2coeffs['six'] - ((326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + ((4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM) + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM))*m1ByM*m1ByM*chi12 + ((4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM) + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM))*m2ByM*m2ByM*chi22)
        # Now translate into inspiral coefficients, label with the power in front of which they appear
        PhiInspcoeffs = {}
        
        PhiInspcoeffs['initial_phasing'] = TF2coeffs['five']*TF2OverallAmpl
        PhiInspcoeffs['two_thirds'] = TF2coeffs['seven']*TF2OverallAmpl*(np.pi**(2./3.))
        PhiInspcoeffs['third'] = TF2coeffs['six']*TF2OverallAmpl*(np.pi**(1./3.))
        PhiInspcoeffs['third_log'] = TF2coeffs['six_log']*TF2OverallAmpl*(np.pi**(1./3.))
        PhiInspcoeffs['log'] = TF2coeffs['five_log']*TF2OverallAmpl
        PhiInspcoeffs['min_third'] = TF2coeffs['four']*TF2OverallAmpl*(np.pi**(-1./3.))
        PhiInspcoeffs['min_two_thirds'] = TF2coeffs['three']*TF2OverallAmpl*(np.pi**(-2./3.))
        PhiInspcoeffs['min_one'] = TF2coeffs['two']*TF2OverallAmpl/np.pi
        PhiInspcoeffs['min_four_thirds'] = TF2coeffs['one']*TF2OverallAmpl*(np.pi**(-4./3.))
        PhiInspcoeffs['min_five_thirds'] = TF2coeffs['zero']*TF2OverallAmpl*(np.pi**(-5./3.))
        PhiInspcoeffs['one'] = sigma1
        PhiInspcoeffs['four_thirds'] = sigma2 * 0.75
        PhiInspcoeffs['five_thirds'] = sigma3 * 0.6
        PhiInspcoeffs['two'] = sigma4 * 0.5
        
        #Now compute the coefficients to align the three parts
        
        fInsJoin = self.PHI_fJoin_INS
        fMRDJoin = 0.5*fring
        
        # First the Inspiral - Intermediate: we compute C1Int and C2Int coeffs
        # Equations to solve for to get C(1) continuous join
        # PhiIns (f)  =   PhiInt (f) + C1Int + C2Int f
        # Joining at fInsJoin
        # PhiIns (fInsJoin)  =   PhiInt (fInsJoin) + C1Int + C2Int fInsJoin
        # PhiIns'(fInsJoin)  =   PhiInt'(fInsJoin) + C2Int
        # This is the first derivative wrt f of the inspiral phase computed at fInsJoin, first add the PN contribution and then the higher order calibrated terms
        DPhiIns = (2.0*TF2coeffs['seven']*TF2OverallAmpl*((np.pi*fInsJoin)**(7./3.)) + (TF2coeffs['six']*TF2OverallAmpl + TF2coeffs['six_log']*TF2OverallAmpl * (1.0 + np.log(np.pi*fInsJoin)/3.))*((np.pi*fInsJoin)**(2.)) + TF2coeffs['five_log']*TF2OverallAmpl*((np.pi*fInsJoin)**(5./3.)) - TF2coeffs['four']*TF2OverallAmpl*((np.pi*fInsJoin)**(4./3.)) - 2.*TF2coeffs['three']*TF2OverallAmpl*(np.pi*fInsJoin) - 3.*TF2coeffs['two']*TF2OverallAmpl*((np.pi*fInsJoin)**(2./3.)) - 4.*TF2coeffs['one']*TF2OverallAmpl*((np.pi*fInsJoin)**(1./3.)) - 5.*TF2coeffs['zero']*TF2OverallAmpl)*np.pi/(3.*((np.pi*fInsJoin)**(8./3.)))
        DPhiIns = DPhiIns + (sigma1 + sigma2*(fInsJoin**(1./3.)) + sigma3*(fInsJoin**(2./3.)) + sigma4*fInsJoin)/eta
        # This is the first derivative of the Intermediate phase computed at fInsJoin
        DPhiInt = (beta1 + beta3/(fInsJoin**4) + beta2/fInsJoin)/eta
        
        C2Int = DPhiIns - DPhiInt
        
        # This is the inspiral phase computed at fInsJoin
        PhiInsJoin = PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fInsJoin**(2./3.)) + PhiInspcoeffs['third']*(fInsJoin**(1./3.)) + PhiInspcoeffs['third_log']*(fInsJoin**(1./3.))*np.log(np.pi*fInsJoin)/3. + PhiInspcoeffs['log']*np.log(np.pi*fInsJoin)/3. + PhiInspcoeffs['min_third']*(fInsJoin**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fInsJoin**(-2./3.)) + PhiInspcoeffs['min_one']/fInsJoin + PhiInspcoeffs['min_four_thirds']*(fInsJoin**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fInsJoin**(-5./3.)) + (PhiInspcoeffs['one']*fInsJoin + PhiInspcoeffs['four_thirds']*(fInsJoin**(4./3.)) + PhiInspcoeffs['five_thirds']*(fInsJoin**(5./3.)) + PhiInspcoeffs['two']*fInsJoin*fInsJoin)/eta
        # This is the Intermediate phase computed at fInsJoin
        PhiIntJoin = beta1*fInsJoin - beta3/(3.*fInsJoin*fInsJoin*fInsJoin) + beta2*np.log(fInsJoin)
        
        C1Int = PhiInsJoin - PhiIntJoin/eta - C2Int*fInsJoin
        
        # Now the same for Intermediate - Merger-Ringdown: we also need a temporary Intermediate Phase function
        PhiIntTempVal  = (beta1*fMRDJoin - beta3/(3.*fMRDJoin*fMRDJoin*fMRDJoin) + beta2*np.log(fMRDJoin))/eta + C1Int + C2Int*fMRDJoin
        DPhiIntTempVal = C2Int + (beta1 + beta3/(fMRDJoin**4) + beta2/fMRDJoin)/eta
        DPhiMRDVal     = (alpha1 + alpha2/(fMRDJoin*fMRDJoin) + alpha3/(fMRDJoin**(1./4.)) + alpha4/(fdamp*(1. + (fMRDJoin - alpha5*fring)*(fMRDJoin - alpha5*fring)/(fdamp*fdamp))))/eta
        PhiMRJoinTemp  = -(alpha2/fMRDJoin) + (4.0/3.0) * (alpha3 * (fMRDJoin**(3./4.))) + alpha1 * fMRDJoin + alpha4 * np.arctan((fMRDJoin - alpha5 * fring)/fdamp)
        
        C2MRD = DPhiIntTempVal - DPhiMRDVal
        C1MRD = PhiIntTempVal - PhiMRJoinTemp/eta - C2MRD*fMRDJoin
        
        fpeak = np.amax(fgrid, axis=0) # In LAL the maximum of the grid is used to rescale
        
        t0 = (alpha1 + alpha2/(fpeak*fpeak) + alpha3/(fpeak**(1./4.)) + alpha4/(fdamp*(1. + (fpeak - alpha5*fring)*(fpeak - alpha5*fring)/(fdamp*fdamp))))/eta
        
        # LAL sets fRef as the minimum frequency, do the same
        fRef   = np.amin(fgrid, axis=0)
        phiRef = np.where(fRef < self.PHI_fJoin_INS, PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fRef**(2./3.)) + PhiInspcoeffs['third']*(fRef**(1./3.)) + PhiInspcoeffs['third_log']*(fRef**(1./3.))*np.log(np.pi*fRef)/3. + PhiInspcoeffs['log']*np.log(np.pi*fRef)/3. + PhiInspcoeffs['min_third']*(fRef**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fRef**(-2./3.)) + PhiInspcoeffs['min_one']/fRef + PhiInspcoeffs['min_four_thirds']*(fRef**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fRef**(-5./3.)) + (PhiInspcoeffs['one']*fRef + PhiInspcoeffs['four_thirds']*(fRef**(4./3.)) + PhiInspcoeffs['five_thirds']*(fRef**(5./3.)) + PhiInspcoeffs['two']*fRef*fRef)/eta, np.where(fRef<fMRDJoin, (beta1*fRef - beta3/(3.*fRef*fRef*fRef) + beta2*np.log(fRef))/eta + C1Int + C2Int*fRef, np.where(fRef < self.fcutPar, (-(alpha2/fRef) + (4.0/3.0) * (alpha3 * (fRef**(3./4.))) + alpha1 * fRef + alpha4 * np.arctan((fRef - alpha5 * fring)/fdamp))/eta + C1MRD + C2MRD*fRef,0.)))

        phis = np.where(fgrid < self.PHI_fJoin_INS, PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fgrid**(2./3.)) + PhiInspcoeffs['third']*(fgrid**(1./3.)) + PhiInspcoeffs['third_log']*(fgrid**(1./3.))*np.log(np.pi*fgrid)/3. + PhiInspcoeffs['log']*np.log(np.pi*fgrid)/3. + PhiInspcoeffs['min_third']*(fgrid**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fgrid**(-2./3.)) + PhiInspcoeffs['min_one']/fgrid + PhiInspcoeffs['min_four_thirds']*(fgrid**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fgrid**(-5./3.)) + (PhiInspcoeffs['one']*fgrid + PhiInspcoeffs['four_thirds']*(fgrid**(4./3.)) + PhiInspcoeffs['five_thirds']*(fgrid**(5./3.)) + PhiInspcoeffs['two']*fgrid*fgrid)/eta, np.where(fgrid<fMRDJoin, (beta1*fgrid - beta3/(3.*fgrid*fgrid*fgrid) + beta2*np.log(fgrid))/eta + C1Int + C2Int*fgrid, np.where(fgrid < self.fcutPar, (-(alpha2/fgrid) + (4.0/3.0) * (alpha3 * (fgrid**(3./4.))) + alpha1 * fgrid + alpha4 * np.arctan((fgrid - alpha5 * fring)/fdamp))/eta + C1MRD + C2MRD*fgrid,0.)))
        
        # Add the tidal contribution to the phase, as in arXiv:1905.06011
        # Compute the tidal coupling constant, arXiv:1905.06011 eq. (8) using Lambda = 2/3 k_2/C^5 (eq. (10))

        kappa2T = (3.0/13.0) * ((1.0 + 12.0*m1ByM/m2ByM)*(m2ByM**5)*Lambda)
        
        c_Newt   = 2.4375
        n_1      = -12.615214237993088
        n_3over2 =  19.0537346970349
        n_2      = -21.166863146081035
        n_5over2 =  90.55082156324926
        n_3      = -60.25357801943598
        d_1      = -15.11120782773667
        d_3over2 =  22.195327350624694
        d_2      =   8.064109635305156
    
        numTidal = 1.0 + (n_1 * ((np.pi*fgrid)**(2./3.))) + (n_3over2 * np.pi*fgrid) + (n_2 * ((np.pi*fgrid)**(4./3.))) + (n_5over2 * ((np.pi*fgrid)**(5./3.))) + (n_3 * np.pi*fgrid*np.pi*fgrid)
        denTidal = 1.0 + (d_1 * ((np.pi*fgrid)**(2./3.))) + (d_3over2 * np.pi*fgrid) + (d_2 * ((np.pi*fgrid)**(4./3.)))
        
        tidal_phase = - kappa2T * c_Newt / (m1ByM * m2ByM) * ((np.pi*fgrid)**(5./3.)) * numTidal / denTidal
        
        # This pi factor is needed to include LAL fRef rescaling, so to end up with the exact same waveform
        return phis + np.where(fgrid < self.fcutPar, - t0*(fgrid - fRef) - phiRef + np.pi +  tidal_phase, 0.)
        
    def Ampl(self, f, **kwargs):
        # Useful quantities
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        eta2 = eta*eta # This can speed up a bit, we call it multiple times
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        #if np.any(chi2 != 0.): # This if does not work in JAX jacrev
        #    print('WARNING: IMRPhenomNSBH is tuned only for chi_NS = 0')
        chi12, chi22 = chi1*chi1, chi2*chi2
        
        Lambda1, Lambda = kwargs['Lambda1'], kwargs['Lambda2']
        
        #if np.any(Lambda1 != 0.): # This if does not work in JAX jacrev
        #    print('WARNING: BH tidal deformability cannot be different from 0, discarding it')
        
        del Lambda1
        
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        q = 0.5*(1.0 + Seta - 2.0*eta)/eta
        SetaPlus1 = 1.0 + Seta
        # We work in dimensionless frequency M*f, not f
        fgrid = M*glob.GMsun_over_c3*f
        # These are m1/Mtot and m2/Mtot
        m1ByM = 0.5 * (1.0 + Seta)
        m2ByM = 0.5 * (1.0 - Seta)
        # As in arXiv:0909.2867
        chieff = m1ByM * chi1 + m2ByM * chi2
        chisum = 2.*chieff
        chiprod = chieff*chieff
        
        # compute needed IMRPhenomC attributes
        # First the SPA part, LALSimIMRPhenomC_internals.c line 38
        # Frequency-domain Amplitude coefficients
        xdotaN = 64.*eta/5.
        xdota2 = -7.43/3.36 - 11.*eta/4.
        xdota3 = 4.*np.pi - 11.3*chieff/1.2 + 19.*eta*chisum/6.
        xdota4 = 3.4103/1.8144 + 5*chiprod + eta*(13.661/2.016 - chiprod/8.) + 5.9*eta2/1.8
        xdota5 = -np.pi*(41.59/6.72 + 189.*eta/8.) - chieff*(31.571/1.008 - 116.5*eta/2.4) + chisum*(21.863*eta/1.008 - 79.*eta2/6.) - 3*chieff*chiprod/4. + 9.*eta*chieff*chiprod/4.
        xdota6 = 164.47322263/1.39708800 - 17.12*np.euler_gamma/1.05 + 16.*np.pi*np.pi/3 - 8.56*np.log(16.)/1.05 + eta*(45.1*np.pi*np.pi/4.8 - 561.98689/2.17728) + 5.41*eta2/8.96 - 5.605*eta*eta2/2.592 - 80.*np.pi*chieff/3. + eta*chisum*(20.*np.pi/3. - 113.5*chieff/3.6) + chiprod*(64.153/1.008 - 45.7*eta/3.6) - chiprod*(7.87*eta/1.44 - 30.37*eta2/1.44)
        xdota6log = -856./105.
        xdota7 = -np.pi*(4.415/4.032 - 358.675*eta/6.048 - 91.495*eta2/1.512) - chieff*(252.9407/2.7216 - 845.827*eta/6.048 + 415.51*eta2/8.64) + chisum*(158.0239*eta/5.4432 - 451.597*eta2/6.048 + 20.45*eta2*eta/4.32 + 107.*eta*chiprod/6. - 5.*eta2*chiprod/24.) + 12.*np.pi*chiprod - chiprod*chieff*(150.5/2.4 + eta/8.) + chieff*chiprod*(10.1*eta/2.4 + 3.*eta2/8.)
        # Time-domain amplitude coefficients, which also enters the fourier amplitude in this model
        AN = 8.*eta*np.sqrt(np.pi/5.)
        A2 = (-107. + 55.*eta)/42.
        A3 = 2.*np.pi - 4.*chieff/3. + 2.*eta*chisum/3.
        A4 = -2.173/1.512 - eta*(10.69/2.16 - 2.*chiprod) + 2.047*eta2/1.512
        A5 = -10.7*np.pi/2.1 + eta*(3.4*np.pi/2.1)
        A5imag = -24.*eta
        A6 = 270.27409/6.46800 - 8.56*np.euler_gamma/1.05 + 2.*np.pi*np.pi/3. + eta*(4.1*np.pi*np.pi/9.6 - 27.8185/3.3264) - 20.261*eta2/2.772 + 11.4635*eta*eta2/9.9792 - 4.28*np.log(16.)/1.05
        A6log = -428./105.
        A6imag = 4.28*np.pi/1.05
        
        z701, z702, z711, z710, z720 = 4.149e+00, -4.070e+00, -8.752e+01, -4.897e+01, 6.665e+02
        z801, z802, z811, z810, z820 = -5.472e-02, 2.094e-02, 3.554e-01, 1.151e-01, 9.640e-01
        z901, z902, z911, z910, z920 = -1.235e+00, 3.423e-01, 6.062e+00, 5.949e+00, -1.069e+01
        
        g1 = z701 * chieff + z702 * chiprod + z711 * eta * chieff + z710 * eta + z720 * eta2
        g1 = np.where(g1 < 0., 0., g1)
        
        del1 = z801 * chieff + z802 * chiprod + z811 * eta * chieff + z810 * eta + z820 * eta2
        del2 = z901 * chieff + z902 * chiprod + z911 * eta * chieff + z910 * eta + z920 * eta2
        del1 = np.where(del1 < 0., 0., del1)
        del2 = np.where(del2 < 1.0e-4, 1.0e-4, del2)
        
        d0 = 0.015
        
        # All the other coefficients from IMRPhenomC are not needed
        
        # Now compute NSBH coefficients
        # Get NS compactness and baryonic mass, see arXiv:1608.02582 eq. (78)
        a0Comp = 0.360
        a1Comp = -0.0355
        a2Comp = 0.000705
        
        Comp = np.where(Lambda > 1., a0Comp + a1Comp*np.log(Lambda) + a2Comp*np.log(Lambda)*np.log(Lambda), 0.5 + (3.*a0Comp-a1Comp-1.5)*Lambda*Lambda + (-2.*a0Comp+a1Comp+1.)*Lambda*Lambda*Lambda)
        
        # Get baryonic mass of the torus remnant of a BH-NS merger in units of the NS baryonic mass,
        # see arXiv:1509.00512 eq. (11)
        alphaTor = 0.296
        betaTor = 0.171
        # In LAL the relation is inverted each time, but this would break the vectorisation,
        # we use an interpolator on a grid of Comp, q, chi instead. Already with 100 pts per parameter the
        # agreement we find with LAL waveforms is at machine precision
        
        xiTide = self.xiTide_interp(np.asarray((np.asarray(Comp), np.asarray(q), np.asarray(chi1))).T)
        
        # Compute Kerr BH ISCO radius
        Z1_ISCO = 1.0 + ((1.0 - chi1*chi1)**(1./3.))*((1.0+chi1)**(1./3.) + (1.0-chi1)**(1./3.))
        Z2_ISCO = np.sqrt(3.0*chi1*chi1 + Z1_ISCO*Z1_ISCO)
        r_ISCO  = np.where(chi1>0., 3.0 + Z2_ISCO - np.sqrt((3.0 - Z1_ISCO)*(3.0 + Z1_ISCO + 2.0*Z2_ISCO)), 3.0 + Z2_ISCO + np.sqrt((3.0 - Z1_ISCO)*(3.0 + Z1_ISCO + 2.0*Z2_ISCO)))
        
        tmpMtorus = alphaTor * xiTide * (1.0-2.0*Comp) - betaTor * q*Comp * r_ISCO
        
        Mtorus = np.where(tmpMtorus>0., tmpMtorus, 0.)
        
        del tmpMtorus
        
        # Get remnant spin for assumed aligned spin system, from arXiv:1903.11622 Table I and eq. (4), (5) and (6)
        
        p1_remSp = ((-5.44187381e-03*chi1 + 7.91165608e-03) + (2.33362046e-02*chi1 + 2.47764497e-02)*eta)*eta
        p2_remSp = ((-8.56844797e-07*chi1 - 2.81727682e-06) + (6.61290966e-06*chi1 + 4.28979016e-05)*eta)*eta
        p3_remSp = ((-3.04174272e-02*chi1 + 2.54889050e-01) + (1.47549350e-01*chi1 - 4.27905832e-01)*eta)*eta
        
        modelRemSp = (1. + Lambda * p1_remSp + Lambda*Lambda * p2_remSp) / ((1. + Lambda*p3_remSp*p3_remSp)*(1. + Lambda*p3_remSp*p3_remSp))

        modelRemSp = np.where((chi1 < 0.) & (eta < 0.188), 1., modelRemSp)
        modelRemSp = np.where(chi1 < -0.5, 1., modelRemSp)
        modelRemSp = np.where(modelRemSp > 1., 1., modelRemSp)
        
        del p1_remSp, p2_remSp, p3_remSp
        
        # Work with spin variables weighted on square of the BH mass over total mass
        S1BH = chi1 * m1ByM * m1ByM
        Shat = S1BH / (m1ByM*m1ByM + m2ByM*m2ByM) # this would be = (chi1*m1*m1 + chi2*m2*m2)/(m1*m1 + m2*m2), but chi2=0 by assumption
        
        # Compute fit to L_orb in arXiv:1611.00332 eq. (16)
        Lorb = (2.*np.sqrt(3.)*eta + 5.24*3.8326341618708577*eta2 + 1.3*(-9.487364155598392)*eta*eta2)/(1. + 2.88*2.5134875145648374*eta) + ((-0.194)*1.0009563702914628*Shat*(4.409160174224525*eta + 0.5118334706832706*eta2 + (64. - 16.*4.409160174224525 - 4.*0.5118334706832706)*eta2*eta) + 0.0851*0.7877509372255369*Shat*Shat*(8.77367320110712*eta + (-32.060648277652994)*eta2 + (64. - 16.*8.77367320110712 - 4.*(-32.060648277652994))*eta2*eta) + 0.00954*0.6540138407185817*Shat*Shat*Shat*(22.830033250479833*eta + (-153.83722669033995)*eta2 + (64. - 16.*22.830033250479833 - 4.*(-153.83722669033995))*eta2*eta))/(1. + (-0.579)*0.8396665722805308*Shat*(1.8804718791591157 + (-4.770246856212403)*eta + 0.*eta2 + (64. - 64.*1.8804718791591157 - 16.*(-4.770246856212403) - 4.*0.)*eta2*eta)) + 0.3223660562764661*Seta*eta2*(1. + 9.332575956437443*eta)*chi1 + 2.3170397514509933*Shat*Seta*eta2*eta*(1. + (-3.2624649875884852)*eta)*chi1 + (-0.059808322561702126)*eta2*eta*chi12;
        
        chif = (Lorb + S1BH)*modelRemSp
        
        # Get remnant mass scaled to a total (initial) mass of 1
        
        p1_remM = ((-1.83417425e-03*chi1 + 2.39226041e-03) + (4.29407902e-03*chi1 + 9.79775571e-03)*eta)*eta
        p2_remM = ((2.33868869e-07*chi1 - 8.28090025e-07) + (-1.64315549e-06*chi1 + 8.08340931e-06)*eta)*eta
        p3_remM = ((-2.00726981e-02*chi1 + 1.31986011e-01) + (6.50754064e-02*chi1 - 1.42749961e-01)*eta)*eta

        modelRemM = (1. + Lambda * p1_remM + Lambda*Lambda * p2_remM) / ((1. + Lambda*p3_remM*p3_remM)*(1. + Lambda*p3_remM*p3_remM))
        modelRemM = np.where((chi1 < 0.) & (eta < 0.188), 1., modelRemM)
        modelRemM = np.where(chi1 < -0.5, 1., modelRemM)
        modelRemM = np.where(modelRemM > 1., 1., modelRemM)
        
        del p1_remM, p2_remM, p3_remM
        
        # Compute the radiated-energy fit from arXiv:1611.00332 eq. (27)
        EradNSBH = (((1. + -2.0/3.0*np.sqrt(2.))*eta + 0.5609904135313374*eta2 + (-0.84667563764404)*eta2*eta + 3.145145224278187*eta2*eta2)*(1. + 0.346*(-0.2091189048177395)*Shat*(1.8083565298668276 + 15.738082204419655*eta + (16. - 16.*1.8083565298668276 - 4.*15.738082204419655)*eta2) + 0.211*(-0.19709136361080587)*Shat*Shat*(4.271313308472851 + 0.*eta + (16. - 16.*4.271313308472851 - 4.*0.)*eta2) + 0.128*(-0.1588185739358418)*Shat*Shat*Shat*(31.08987570280556 + (-243.6299258830685)*eta + (16. - 16.*31.08987570280556 - 4.*(-243.6299258830685))*eta2)))/(1. + (-0.212)*2.9852925538232014*Shat*(1.5673498395263061 + (-0.5808669012986468)*eta + (16. - 16.*1.5673498395263061 - 4.*(-0.5808669012986468))*eta2)) + (-0.09803730445895877)*Seta*eta2*(1. + (-3.2283713377939134)*eta)*chi1 + (-0.01978238971523653)*Shat*Seta*eta*(1. + (-4.91667749015812)*eta)*chi1 + 0.01118530335431078*eta2*eta*chi12
        finalMass = (1.-EradNSBH)*modelRemM
        
        # Compute 22 quasi-normal mode dimensionless frequency
        kappaOm = np.sqrt(np.log(2.-chif)/np.log(3.))
        omega_tilde = (1.0 + kappaOm*(1.5578*np.exp(1j*2.9031) + 1.9510*np.exp(1j*5.9210)*kappaOm + 2.0997*np.exp(1j*2.7606)*kappaOm*kappaOm + 1.4109*np.exp(1j*5.9143)*kappaOm*kappaOm*kappaOm + 0.4106*np.exp(1j*2.7952)*(kappaOm**4)))
        
        fring = 0.5*np.real(omega_tilde)/np.pi/finalMass
        
        rtide = xiTide * (1.0 - 2.0 * Comp) / (q*Comp)
        
        q_factor = 0.5*np.real(omega_tilde)/np.imag(omega_tilde)
        
        ftide = abs(1.0/(np.pi*(chi1 + np.sqrt(rtide*rtide*rtide)))*(1.0 + 1.0 / q))
        
        # Now compute last amplitude quantities
        fring_tilde = 0.99 * 0.98 * fring
        
        gamma_correction = np.where(Lambda > 1.0, 1.25, 1.0 + 0.5*Lambda - 0.25*Lambda*Lambda)
        delta_2_prime = np.where(Lambda > 1.0, 1.62496*0.25*(1. + np.tanh(4.0*((ftide/fring_tilde - 1.)-0.0188092)/0.338737)), del2 - 2.*(del2 - 0.81248)*Lambda + (del2 - 0.81248)*Lambda*Lambda)
        
        sigma = delta_2_prime * fring / q_factor
        
        # Determine the type of merger we see and determine coefficients
        epsilon_tide = np.where(ftide < fring, 0., 2.*0.25*(1 + np.tanh(4.0*(((ftide/fring_tilde - 1.)*(ftide/fring_tilde - 1.) - 0.571505*Comp - 0.00508451*chi1)+0.0796251)/0.0801192)))
        
        epsilon_ins  = np.where(ftide < fring, np.where(1.29971 - 1.61724 * (Mtorus + 0.424912*Comp + 0.363604*np.sqrt(eta) - 0.0605591*chi1)>1., 1., 1.29971 - 1.61724 * (Mtorus + 0.424912*Comp + 0.363604*np.sqrt(eta) - 0.0605591*chi1)), np.where(Mtorus > 0., 1.29971 - 1.61724 * (Mtorus + 0.424912*Comp + 0.363604*np.sqrt(eta) - 0.0605591*chi1), 1.))
        
        sigma_tide   = np.where(ftide < fring, np.where(Mtorus>0., 0.137722 - 0.293237*(Mtorus - 0.132754*Comp + 0.576669*np.sqrt(eta) - 0.0603749*chi1 - 0.0601185*chi1*chi1 - 0.0729134*chi1*chi1*chi1), 0.5*(0.137722 - 0.293237*(Mtorus - 0.132754*Comp + 0.576669*np.sqrt(eta) - 0.0603749*chi1 - 0.0601185*chi1*chi1 - 0.0729134*chi1*chi1*chi1) + 0.5*(1. - np.tanh(4.0*(((ftide/fring_tilde - 1.)*(ftide/fring_tilde - 1.) - 0.657424*Comp - 0.0259977*chi1)+0.206465)/0.226844)))),0.5*(1. - np.tanh(4.0*(((ftide/fring_tilde - 1.)*(ftide/fring_tilde - 1.) - 0.657424*Comp - 0.0259977*chi1)+0.206465)/0.226844)))
        
        f0_tilde_PN  = np.where(ftide < fring, np.where(Mtorus>0., ftide / (M*glob.GMsun_over_c3), ((1.0 - 1.0 / q) * fring_tilde + epsilon_ins * ftide / q)/(M*glob.GMsun_over_c3)), np.where(Lambda>1., fring_tilde/(M*glob.GMsun_over_c3), ((1.0 - 0.02*Lambda + 0.01*Lambda*Lambda)*0.98*fring)/(M*glob.GMsun_over_c3)))
        
        f0_tilde_PM  = np.where(ftide < fring, np.where(Mtorus>0., ftide / (M*glob.GMsun_over_c3), ((1.0 - 1.0 / q) * fring_tilde + ftide/q)/(M*glob.GMsun_over_c3)), np.where(Lambda>1., fring_tilde/(M*glob.GMsun_over_c3), ((1.0 - 0.02*Lambda + 0.01*Lambda*Lambda)*0.98*fring)/(M*glob.GMsun_over_c3)))
        
        f0_tilde_RD  = np.where(ftide < fring, 0., np.where(Lambda>1., fring_tilde/(M*glob.GMsun_over_c3), ((1.0 - 0.02*Lambda + 0.01*Lambda*Lambda)*0.98*fring)/(M*glob.GMsun_over_c3)))
        
        # This can be used to output the merger type if needed
        #merger_type = onp.where(ftide < fring, onp.where(Mtorus>0., 'DISRUPTIVE', 'MILDLY_DISRUPTIVE_NO_TORUS_REMNANT'), onp.where(Mtorus>0.,'MILDLY_DISRUPTIVE_TORUS_REMNANT', 'NON_DISRUPTIVE'))
        
        v = (fgrid*np.pi)**(1./3.)

        xdot = xdotaN*(v**10)*(1. + xdota2*v*v + xdota3 * fgrid*np.pi + xdota4 * fgrid*np.pi*v + xdota5 * v*v*fgrid*np.pi + (xdota6 + xdota6log * 2.*np.log(v)) * fgrid*np.pi*fgrid*np.pi + xdota7 * v*fgrid*np.pi*fgrid*np.pi)
        ampfacTime = np.sqrt(abs(np.pi / (1.5 * v * xdot)))
        
        AmpPNre = ampfacTime * AN * v*v * (1. + A2*v*v + A3 * fgrid*np.pi + A4 * v*fgrid*np.pi + A5 * v*v*fgrid*np.pi + (A6 + A6log * 2.*np.log(v)) * fgrid*np.pi*fgrid*np.pi)
        AmpPNim = ampfacTime * AN * v*v * (A5imag * v*v*fgrid*np.pi + A6imag * fgrid*np.pi*fgrid*np.pi)
        
        aPN = np.sqrt(AmpPNre * AmpPNre + AmpPNim * AmpPNim)
        aPM = (gamma_correction * g1 * (fgrid**(5./6.)))
        
        LRD = sigma*sigma / ((fgrid - fring) * (fgrid - fring) + sigma*sigma*0.25)
        aRD = epsilon_tide * del1 * LRD * (fgrid**(-7./6.))
        
        wMinusf0_PN = 0.5 * (1. - np.tanh(4.*(fgrid - (epsilon_ins * f0_tilde_PN)*M*glob.GMsun_over_c3)/(d0 + sigma_tide)))
        wMinusf0_PM = 0.5 * (1. - np.tanh(4.*(fgrid - f0_tilde_PM*M*glob.GMsun_over_c3)/(d0 + sigma_tide)))
        wPlusf0     = 0.5 * (1. + np.tanh(4.*(fgrid - f0_tilde_RD*M*glob.GMsun_over_c3)/(d0 + sigma_tide)))
        
        amplitudeIMR = np.where(fgrid < self.fcutPar, (aPN * wMinusf0_PN + aPM * wMinusf0_PM + aRD * wPlusf0), 0.)
        
        # Defined as in LALSimulation - LALSimIMRPhenomD.c line 332. Final units are correctly Hz^-1
        Overallamp = 2. * np.sqrt(5./(64.*np.pi)) * M * glob.GMsun_over_c2_Gpc * M * glob.GMsun_over_c3 / kwargs['dL']
        
        return Overallamp*amplitudeIMR
    
    def _radiatednrg(self, eta, chi1, chi2):
        # Predict the total radiated energy, from arXiv:1508.07250 eq (3.7) and (3.8)
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        s = (m1*m1 * chi1 + m2*m2 * chi2) / (m1*m1 + m2*m2)
        
        EradNS = eta * (0.055974469826360077 + 0.5809510763115132 * eta - 0.9606726679372312 * eta*eta + 3.352411249771192 * eta*eta*eta)
        
        return (EradNS * (1. + (-0.0030302335878845507 - 2.0066110851351073 * eta + 7.7050567802399215 * eta*eta) * s)) / (1. + (-0.6714403054720589 - 1.4756929437702908 * eta + 7.304676214885011 * eta*eta) * s)
    
    def tau_star(self, f, **kwargs):
        # For complex waveforms we use the expression in arXiv:0907.0700 eq. (3.8b)
        Mtot_sec = kwargs['Mc']*glob.GMsun_over_c3/(kwargs['eta']**(3./5.))
        v = (np.pi*Mtot_sec*f)**(1./3.)
        eta = kwargs['eta']
        eta2 = eta*eta
        
        OverallFac = 5./256 * Mtot_sec/(eta*(v**8.))
        
        t05 = 1. + (743./252. + 11./3.*eta)*(v*v) - 32./5.*np.pi*(v*v*v) + (3058673./508032. + 5429./504.*eta + 617./72.*eta2)*(v**4) - (7729./252. - 13./3.*eta)*np.pi*(v**5)
        t6  = (-10052469856691./23471078400. + 128./3.*np.pi*np.pi + 6848./105.*np.euler_gamma + (3147553127./3048192. - 451./12.*np.pi*np.pi)*eta - 15211./1728.*eta2 + 25565./1296.*eta2*eta + 3424./105.*np.log(16.*v*v))*(v**6)
        t7  = (- 15419335./127008. - 75703./756.*eta + 14809./378.*eta2)*np.pi*(v**7)
        
        return OverallFac*(t05 + t6 + t7)
    
    def fcut(self, **kwargs):
        
        return self.fcutPar/(kwargs['Mc']*glob.GMsun_over_c3/(kwargs['eta']**(3./5.)))
    
    def _tabulate_xiTide(self, res=200, store=True, Compmin=.1, qmax=100.):
        '''
        The ranges are chosen to cover LAL's tuning range:
            - Compactness in [0.1, 0.5] (LAL is tuned up to Lambda=5000, corresponding to C = 0.109);
            - mass ratio, q, in [1, 100];
            - chi_BH in [-1, 1].
        They can easily be changed if needed
        '''
        Compgrid = onp.linspace(Compmin, .5, res)
        qgrid = onp.linspace(1., qmax, res)
        chigrid = onp.linspace(-1.,1.,res)

        def sqrtxifun(Comp, q, chi):
            # Coefficients of eq. (8) of arXiv:1509.00512, using as variable sqrt(xi) (so order 10 polynomial)
            mu = q*Comp
            return onp.array([1., 0., -3.*mu,  2.*chi*(mu**(3./2.)), 0., 0., -3.*q, 0., 6.*q*mu, 0., -3.*q*mu*chi*mu*chi])

        xires = onp.zeros((res,res,res))
        in_time=time.time()
        for i,Comp in enumerate(Compgrid):
            for j,q in enumerate(qgrid):
                for k,chi in enumerate(chigrid):
                    tmpcoeffs = sqrtxifun(Comp, q, chi)
                    tmproots = onp.roots(tmpcoeffs)
                    # We select only real and positive solutions and take the maximum of the squares
                    tmproots_rp = onp.real(tmproots[(abs(onp.imag(tmproots))<1e-5) & (onp.real(tmproots)>0.)])
                    tmpres = max(tmproots_rp*tmproots_rp)
                    xires[i,j,k] = tmpres

        print('Done in %.2fs \n' %(time.time() - in_time))
        if store:
            print('Saving result...')
            if not os.path.isdir(os.path.join(glob.WFfilesPath)):
                os.mkdir(os.path.join(glob.WFfilesPath))

            with h5py.File(os.path.join(glob.WFfilesPath, 'xiTide_Table_'+str(res)+'.h5'), 'w') as out:
                out.create_dataset('Compactness', data=Compgrid, compression='gzip', shuffle=True)
                out.create_dataset('q', data=qgrid, compression='gzip', shuffle=True)
                out.create_dataset('chi', data=chigrid, compression='gzip', shuffle=True)
                out.create_dataset('xiTide', data=xires, compression='gzip', shuffle=True)
                out.attrs['npoints'] = res
                out.attrs['Compactness_min'] = Compmin
                out.attrs['q_max'] = qmax
            print('Done...')

        return xires, Compgrid, qgrid, chigrid

    def _make_xiTide_interpolator(self, res=200):

        #from scipy.interpolate import RegularGridInterpolator
        if self.path_xiTide_tab is not None:
            if os.path.exists(self.path_xiTide_tab):
                if self.verbose:
                    print('Pre-computed xi_tide grid is present. Loading...')
                with h5py.File(self.path_xiTide_tab, 'r') as inp:
                    Comps = np.array(inp['Compactness'])
                    qs = np.array(inp['q'])
                    chis = np.array(inp['chi'])
                    xiTides = np.array(inp['xiTide'])
                    if self.verbose:
                        print('Attributes of pre-computed grid: ')
                        print([(k, inp.attrs[k]) for k in inp.attrs.keys()])
                        self.verbose=False
            else:
                print('Tabulating xi_tide...')
                xiTides, Comps, qs, chis = self._tabulate_xiTide(res=res)

        else:
            print('Tabulating xi_tide...')
            xiTides, Comps, qs, chis = self._tabulate_xiTide(res=res)

        self.xiTide_interp = utils.RegularGridInterpolator_JAX((Comps, qs, chis), xiTides, bounds_error=False)

