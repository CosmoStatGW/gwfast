#
#    Copyright (c) 2024 Francesco Iacovelli <francesco.iacovelli@unige.ch>, Michele Mancarella <michele.mancarella@unige.ch>, Niccolo Muttoni <niccolo.muttoni@unige.ch>, Enis Belgacem <enis.belgacem@unige.ch>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.


import numpy as np
import copy
from scipy.optimize import minimize
import warnings
# This is needed for the maximizer in powerlaw_integrated_sensitivity
warnings.filterwarnings("ignore", category=RuntimeWarning)
from gwfast import gwfastUtils as utils
from gwfast import gwfastGlobals as glob

##############################################################################
# ORF
##############################################################################

def overlap_reduction_function(f, det1, det2, det1_name='det1', det2_name='det2', arm_length_1=None, arm_length_2=None):
    """
    Compute the overlap reduction function (*ORF*) as a function of frequency given a pair of detectors (see e.g. `arXiv:astro-ph/9305029 <https://arxiv.org/abs/astro-ph/9305029v1>`_).

    :param array or float f: The frequency(ies) at which to evaluate the ORF, in :math:`\\rm Hz`.
    :param dict(float, float, float, str) det1: Dictionary containing the latitude, ``'lat'``, longitude, ``'long'``, orientation, ``'xax'`` (all in degrees), and shape, ``'shape'``, of the first detector, as in :py:data:`gwfast.gwfastGlobals.detectors`.
    :param dict(float, float, float, str) det2: Dictionary containing the latitude, ``'lat'``, longitude, ``'long'``, orientation, ``'xax'`` (all in degrees), and shape, ``'shape'``, of the second detector, as in :py:data:`gwfast.gwfastGlobals.detectors`.
    :param str, optional det1_name: The name to assign to the first detector. If not provided, it will default to ``det1``.
    :param str, optional det2_name: The name to assign to the second detector. If not provided, it will default to ``det2``.
    :param float, optional arm_length_1: Length of the arms of the first detector (needed in the triangular configuration), in :math:`\\rm km`.
    :param float, optional arm_length_2: Length of the arms of the second detector (needed in the triangular configuration), in :math:`\\rm km`.

    :return: Overlap reduction function for the given configuration. Each key in the output dictionary corresponds to a given combination of detectors (1 for two L-shaped detectors, 6 for an L-shaped detector and a triangle and 15 for two triangles).
    :rtype: dict(array, array, ...)

    """
    def get_ORF(alpha, beta, delta, Delta, angbtwArms1, angbtwArms2):
    # For the following functions see arXiv:astro-ph/9305029 Sect. 4 and App. B.
        def g1(alpha):
            return (5./16.)*(-9.*alpha*np.cos(alpha) - 6.*alpha*alpha*alpha*np.cos(alpha) + 9.*np.sin(alpha) + 3.*alpha*alpha*np.sin(alpha) + (alpha**4)*np.sin(alpha))/(alpha**5)
        def g2(alpha):
            return (5./16.)*(45.*alpha*np.cos(alpha) + 6.*alpha*alpha*alpha*np.cos(alpha)  - 45.*np.sin(alpha) + 9.*alpha*alpha*np.sin(alpha) + 3*(alpha**4)*np.sin(alpha))/(alpha**5)
        def g3(alpha):
            return (5./4.)*(15.*alpha*np.cos(alpha) - 4.*alpha*alpha*alpha*np.cos(alpha) - 15.*np.sin(alpha) + 9.*alpha*alpha*np.sin(alpha) - (alpha**4)*np.sin(alpha))/(alpha**5)

        def Theta1(alpha, beta):
            return ((np.cos(beta*0.5))**4)*g1(alpha)

        def Theta2(alpha, beta):
            g2Use = g2(alpha)
            return ((np.cos(beta*0.5))**4)*g2Use + g3(alpha) - ((np.sin(beta*0.5))**4)*(g2Use+g1(alpha))

        return (np.cos(4.*delta)*Theta1(alpha, beta) + np.cos(4.*Delta)*Theta2(alpha, beta))*np.sin(angbtwArms1)*np.sin(angbtwArms2)

    if det2 is not None:
        lat1, lat2   = np.deg2rad(det1['lat']), np.deg2rad(det2['lat'])
        long1, long2 = np.deg2rad(det1['long']), np.deg2rad(det2['long'])

        def initial_course(lat1, lat2, long1, long2):
            # Compute the course at the initial point given two points
            # See http://www.edwilliams.org/avform147.htm#Crs or https://en.wikipedia.org/wiki/Great-circle_navigation
            a = np.sin(long2-long1)*np.cos(lat2)
            b = np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(long2-long1)

            # If the initial point is a pole we need a "fix"
            return np.rad2deg(np.where(np.isclose(np.cos(lat1), 0.), np.where(lat1 > 0., np.pi, 2.*np.pi), np.arctan2(a,b)))

        def final_course(lat1, lat2, long1, long2):
            # Compute the course at the final point given two points
            # See http://www.edwilliams.org/avform147.htm#Crs or https://en.wikipedia.org/wiki/Great-circle_navigation
            a = np.sin(long2-long1)*np.cos(lat1)
            b = -np.cos(lat2)*np.sin(lat1)+np.sin(lat2)*np.cos(lat1)*np.cos(long2-long1)

            # If the final point is a pole we need a "fix"
            return np.rad2deg(np.where(np.isclose(np.cos(lat2), 0.), np.where(lat2 > 0., np.pi, 2.*np.pi), np.arctan2(a,b)))

        # Compute the course at the first detector
        ang1 = (initial_course(lat1, lat2, long1, long2) - 90.)
        # Compute the course at the second detector
        ang2 = (final_course(lat1, lat2, long1, long2) - 90.)

        if det1['shape']=='L':
            angbtwArms1 = np.pi*0.5
        else:
            angbtwArms1 = np.pi/3.
            if arm_length_1 is None:
                raise ValueError('In the triangular case the arm length has to be specified. Missing arm length for the first detector.')

        if det2['shape']=='L':
            angbtwArms2 = np.pi*0.5
        else:
            angbtwArms2 = np.pi/3.
            if arm_length_2 is None:
                raise ValueError('In the triangular case the arm length has to be specified. Missing arm length for the second detector.')

        out_ORF = {}

        if (det1['shape']=='L') and (det2['shape']=='L'):

            delta = np.deg2rad((det1['xax']+ang1 - det2['xax']-ang2)*0.5)
            Delta = np.deg2rad((det1['xax']+ang1 + det2['xax']+ang2)*0.5)

            dist = utils.dist_btw_dets_Chord(det1, det2)

            alpha = 2.*np.pi*f*dist/glob.clight
            beta  = 2.*np.arcsin(dist*0.5/glob.REarth)

            out_ORF[det1_name+'-'+det2_name] = np.where(alpha > 2e-3, get_ORF(alpha, beta, delta, Delta, angbtwArms1=angbtwArms1, angbtwArms2=angbtwArms2), np.cos(4.*delta)*np.sin(angbtwArms1)*np.sin(angbtwArms1))

        elif (det1['shape']=='T') and (det2['shape']=='L'):
            dist = utils.dist_btw_dets_Chord(det1, det2)
            alpha = 2.*np.pi*f*dist/glob.clight
            beta  = 2.*np.arcsin(dist*0.5/glob.REarth)

            for i in range(3):
                delta = np.deg2rad((det1['xax']+i*60.+ang1 - det2['xax']-ang2)*0.5)
                Delta = np.deg2rad((det1['xax']+i*60.+ang1 + det2['xax']+ang2)*0.5)

                out_ORF[det1_name+'_'+str(i)+'-'+det2_name] = get_ORF(alpha, beta, delta, Delta, angbtwArms1=angbtwArms1, angbtwArms2=angbtwArms2)

            ang2_same = ang1
            dist = arm_length_1
            alpha = 2.*np.pi*f*dist/glob.clight
            beta  = 2.*np.arcsin(dist*0.5/glob.REarth)
            for i in range(3):
                for j in [1,2]:
                    if j>i:
                        delta = np.deg2rad((det1['xax']+i*60.+ang1 - det1['xax']-j*60.-ang2_same)*0.5)
                        Delta = np.deg2rad((det1['xax']+i*60.+ang1 + det1['xax']+j*60.+ang2_same)*0.5)

                        out_ORF[det1_name+'_'+str(i)+'-'+det1_name+'_'+str(j)] = np.where(alpha>2e-3, get_ORF(alpha, beta, delta, Delta, angbtwArms1=angbtwArms1, angbtwArms2=angbtwArms1), np.cos(4.*delta)*np.sin(angbtwArms1)*np.sin(angbtwArms1))

        elif (det1['shape']=='L') and (det2['shape']=='T'):
            dist = utils.dist_btw_dets_Chord(det1, det2)
            alpha = 2.*np.pi*f*dist/glob.clight
            beta  = 2.*np.arcsin(dist*0.5/glob.REarth)

            for i in range(3):
                delta = np.deg2rad((det1['xax']+ang1 - det2['xax']-i*60.-ang2)*0.5)
                Delta = np.deg2rad((det1['xax']+ang1 + det2['xax']+i*60.+ang2)*0.5)

                out_ORF[det1_name+'-'+det2_name+'_'+str(i)] = get_ORF(alpha, beta, delta, Delta, angbtwArms1=angbtwArms1, angbtwArms2=angbtwArms2)

            ang1_same = ang2
            dist = arm_length_2
            alpha = 2.*np.pi*f*dist/glob.clight
            beta  = 2.*np.arcsin(dist*0.5/glob.REarth)
            for i in range(3):
                for j in [1,2]:
                    if j>i:
                        delta = np.deg2rad((det2['xax']+i*60.+ang1_same - det2['xax']-j*60.-ang2)*0.5)
                        Delta = np.deg2rad((det2['xax']+i*60.+ang1_same + det2['xax']+j*60.+ang2)*0.5)

                        out_ORF[det2_name+'_'+str(i)+'-'+det2_name+'_'+str(j)] = np.where(alpha>2e-3, get_ORF(alpha, beta, delta, Delta, angbtwArms1=angbtwArms2, angbtwArms2=angbtwArms2), np.cos(4.*delta)*np.sin(angbtwArms2)*np.sin(angbtwArms2))

        elif (det1['shape']=='T') and (det2['shape']=='T'):
            dist = utils.dist_btw_dets_Chord(det1, det2)
            alpha = 2.*np.pi*f*dist/glob.clight
            beta  = 2.*np.arcsin(dist*0.5/glob.REarth)

            for i in range(3):
                for j in range(3):
                    delta = np.deg2rad((det1['xax']+i*60.+ang1 - det2['xax']-j*60.-ang2)*0.5)
                    Delta = np.deg2rad((det1['xax']+i*60.+ang1 + det2['xax']+j*60.+ang2)*0.5)

                    out_ORF[det1_name+'_'+str(i)+'-'+det2_name+'_'+str(j)] = get_ORF(alpha, beta, delta, Delta, angbtwArms1=angbtwArms1, angbtwArms2=angbtwArms2)


            ang2_same = ang1
            dist = arm_length_1
            alpha = 2.*np.pi*f*dist/glob.clight
            beta  = 2.*np.arcsin(dist*0.5/glob.REarth)
            for i in range(3):
                for j in [1,2]:
                    if j>i:
                        delta = np.deg2rad((det1['xax']+i*60.+ang1 - det1['xax']-j*60.-ang2_same)*0.5)
                        Delta = np.deg2rad((det1['xax']+i*60.+ang1 + det1['xax']+j*60.+ang2_same)*0.5)

                        out_ORF[det1_name+'_'+str(i)+'-'+det1_name+'_'+str(j)] = np.where(alpha>2e-3, get_ORF(alpha, beta, delta, Delta, angbtwArms1=angbtwArms1, angbtwArms2=angbtwArms1), np.cos(4.*delta)*np.sin(angbtwArms1)*np.sin(angbtwArms1))

            ang1_same = ang2
            dist = arm_length_2
            alpha = 2.*np.pi*f*dist/glob.clight
            beta  = 2.*np.arcsin(dist*0.5/glob.REarth)
            for i in range(3):
                for j in [1,2]:
                    if j>i:
                        delta = np.deg2rad((det2['xax']+i*60.+ang1_same - det2['xax']-j*60.-ang2)*0.5)
                        Delta = np.deg2rad((det2['xax']+i*60.+ang1_same + det2['xax']+j*60.+ang2)*0.5)

                        out_ORF[det2_name+'_'+str(i)+'-'+det2_name+'_'+str(j)] = np.where(alpha>2e-3, get_ORF(alpha, beta, delta, Delta, angbtwArms1=angbtwArms2, angbtwArms2=angbtwArms2), np.cos(4.*delta)*np.sin(angbtwArms2)*np.sin(angbtwArms2))
    elif (det2 is None) and (det1['shape'] =='T'):
        angbtwArms1 = np.pi/3.
        out_ORF = {}
        dist = arm_length_1
        alpha = 2.*np.pi*f*dist/glob.clight
        beta  = 2.*np.arcsin(dist*0.5/glob.REarth)
        for i in range(3):
            for j in [1,2]:
                if j>i:
                    delta = np.deg2rad((det1['xax']+i*60. - det1['xax']-j*60.)*0.5)
                    Delta = np.deg2rad((det1['xax']+i*60. + det1['xax']+j*60.)*0.5)

                    out_ORF[det1_name+'_'+str(i)+'-'+det1_name+'_'+str(j)] = np.where(alpha>2e-3, get_ORF(alpha, beta, delta, Delta, angbtwArms1=angbtwArms1, angbtwArms2=angbtwArms1), np.cos(4.*delta)*np.sin(angbtwArms1)*np.sin(angbtwArms1))

    return out_ORF

##############################################################################
# PLS
##############################################################################

def powerlaw_integrated_sensitivity_TR(f, Net, H0=67.66, Tobs=1., SNRval=1., betamin=-50., betamax=50., betares=5000, is_ASD=True, return_all=False, use_maximizer=False):
    """
    Compute the power-law integrated sensitivity curve for a detector network (see e.g. `arXiv:1310.5300 <https://arxiv.org/abs/1310.5300>`_).

    :param array or float f: The frequency(ies) at which to evaluate the power-law integrated sensitivity curve, in :math:`\\rm Hz`.
    :param dict(dict, dict, ...) Net: Dictionary of dictionaries containing the latitude, ``'lat'``, longitude, ``'long'``, orientation, ``'xax'`` (all in degrees), shape, ``'shape'``, path to the ASD data, ``'psd_path'``, and arm length in the triangular case, ``'arm_length'`` (in :math:`\\rm km`), for each detector in the desired network, as in :py:data:`gwfast.gwfastGlobals.detectors`.
    :param float, optional H0: The present-day value of the Hubble parameter to use in the estimation, in :math:`\\rm km \, s^{-1}\, Mpc^{-1}`.
    :param float, optional Tobs: The observational time to consider, in :math:`\\rm yr`.
    :param float, optional SNRval: The value of integrated signa-to-noise ratio time to consider.
    :param float, optional betamin: The minimum value of power-law index to consider.
    :param float, optional betamax: The maximum value of power-law index to consider.
    :param int, optional betares: The resolution of the power-law indices grid to consider.
    :param bool, optional is_ASD: Boolean specifying if the provided sensitivity files for each detector a PSD or an ASD.
    :param bool, optional return_all: Boolean specifying if returning the PLS for all the pairs involved in the network.
    :param bool, optional use_maximizer: Boolean specifying if computing the PLS for the full network using a maximizer in each frequency bin.

    :return: Power-law integrated sensitivity curve for the chosen detector network as a function of frequency.
    :rtype: dict

    """
    print("WARNING: Results are H0 dependent. Consider tuning H0 to a value of your choice if needed. Default is H0 = 67.66 km s^-1 Mpc^-1.")
    if use_maximizer:
        print("WARNING: use_maximizer=True applies for the network PLS only.")
    for key in Net.keys():
        if (Net[key]['shape']=='T') and ('arm_length' not in Net[key].keys()):
            raise ValueError('For a triangle the arm length has to be provided.')
        if (Net[key]['shape']=='L') and ('arm_length' not in Net[key].keys()):
            Net[key]["arm_length"] = None

    betagrid = np.linspace(betamin, betamax, betares)
    fg = np.repeat(f[:,None], len(betagrid), axis=1)
    # H0_to_h = (1e2 * 1e3 / (1e-3*glob.uGpc))
    # Seff_to_hsqOmeff  = ((10.*(np.pi**2.)/3.) * H0_to_h**(-2.))**(-1.)
    Seff_to_Omeff  = ((H0/(glob.uGpc/1e6))**2.)*3./10./np.pi**2.
    Seff_to_Omeff_sqr  = Seff_to_Omeff**2.
    # Seff_to_hsqOmeff_sqr  = Seff_to_hsqOmeff**2.

    PSD_interps = {}
    detsList = list(Net.keys())
    for key in Net.keys():
        tmpASD = np.loadtxt(Net[key]['psd_path'], usecols=(0,1))
        if is_ASD:
            PSD_interps[key] = np.interp(f, tmpASD[:,0], tmpASD[:,1]**2, left=1., right=1.)
        else:
            PSD_interps[key] = np.interp(f, tmpASD[:,0], tmpASD[:,1], left=1., right=1.)

    NetOrdered = {k: v for k, v in sorted(Net.items(), key=lambda item: item[1]['shape'])}
    detsList = list(NetOrdered.keys())
    individual_dets = copy.deepcopy(detsList)
    T_dets = [det for det in detsList if Net[det]["shape"] == "T"]
    N_T_dets = len(T_dets)
    for det in detsList:
        if Net[det]["shape"] == "T":
            individual_dets.pop(individual_dets.index(det))
            individual_dets.insert(0, det+"_0")
            individual_dets.insert(1, det+"_1")
            individual_dets.insert(2, det+"_2")

    det_pairs = [
        d1+"-"+d2
        for d1 in individual_dets
        for d2 in individual_dets
        if individual_dets.index(d1) < individual_dets.index(d2)
    ]

    orf_aux = {}
    for pair in det_pairs:
        orf_aux[pair] = {}
        det1 = pair.split("-")[0].split("_")[0]
        det2 = pair.split("-")[-1].split("_")[0]
        orf_aux[pair] = overlap_reduction_function(f, Net[det1], Net[det2], det1_name=det1, det2_name=det2, arm_length_1=Net[det1]["arm_length"], arm_length_2=Net[det2]["arm_length"])

    orf = {pair: orf_aux[pair][pair] for pair in det_pairs}
    pls_ab = {}
    sum_over_pairs_of_inv_omega_eff_ab_sq = np.zeros_like(f)
    sum_over_pairs_of_inv_omega_eff_ab_sq_T_dets = {Tdet: np.zeros_like(f) for Tdet in T_dets}
    for pair in det_pairs:

        det1 = pair.split("-")[0].split("_")[0]
        det2 = pair.split("-")[-1].split("_")[0]

        if np.any(PSD_interps[det1] == 1.):
            fstart_idx = np.where(PSD_interps[det1] != 1.)[0][0]
        if np.any(PSD_interps[det2] == 1.):
            fstart_idx = np.where(PSD_interps[det2] != 1.)[0][0]
        else:
            fstart_idx = 0

        one_over_S_eff_n_ab_sq = (PSD_interps[det1]*PSD_interps[det2])**(-1.)
        inv_omega_eff_ab_sq = ((f**-6.) * (orf[pair]**2.) * one_over_S_eff_n_ab_sq) * Seff_to_Omeff_sqr
        sum_over_pairs_of_inv_omega_eff_ab_sq += inv_omega_eff_ab_sq

        if (Net[det1] == Net[det2]) and (Net[det1]["shape"] == "T"):

            sum_over_pairs_of_inv_omega_eff_ab_sq_T_dets[det1] += inv_omega_eff_ab_sq

        if return_all:

            integral = np.trapz(inv_omega_eff_ab_sq[fstart_idx:][:, None]*(fg[fstart_idx:]**(2.*betagrid[None, :])), x=fg[fstart_idx:], axis=0)
            pls_ab[pair] = ((fg**betagrid[None,:])*SNRval/np.sqrt(2.*Tobs*glob.seconds_in_year*integral[None,:])).max(axis=1)
            pls_ab[pair][:fstart_idx] = 1.

    if N_T_dets > 0 and return_all:
        for Tdet in T_dets:

            integral = np.trapz(sum_over_pairs_of_inv_omega_eff_ab_sq_T_dets[Tdet][:, None]*(fg**(2.*betagrid[None, :])), x=fg, axis=0)
            pls_ab[Tdet] = ((fg**betagrid[None,:])*SNRval/np.sqrt(2.*Tobs*glob.seconds_in_year*integral[None,:])).max(axis=1)

    if not use_maximizer:

        integral = np.trapz(sum_over_pairs_of_inv_omega_eff_ab_sq[:, None]*(fg**(2.*betagrid[None, :])), x=fg, axis=0)
        pls_ab["net"] = ((fg**betagrid[None,:])*SNRval/np.sqrt(2.*Tobs*glob.seconds_in_year*integral[None,:])).max(axis=1)

    else:

        fref = 100.
        minOmegaGWpow = lambda beta, fst : -(fst/fref)**beta/np.sqrt(np.trapz((f/fref)**(2*beta)*sum_over_pairs_of_inv_omega_eff_ab_sq, f, axis=0))*SNRval/np.sqrt(2*Tobs*glob.seconds_in_year)
        # Find the maximum value of OmegaGWpow for each frequency
        betas = np.zeros_like(f)
        OmegaGWpowMax = np.zeros_like(f)
        for i, fi in enumerate(f):
            betas[i] = minimize(minOmegaGWpow, 0., bounds=[(-np.inf,np.inf)], method='Nelder-Mead', args=fi, tol=1e-6).x
            OmegaGWpowMax[i] = -minOmegaGWpow(betas[i], fi)
        # Return the maximum value of Omega_GW over the power-law indices for each frequency
        pls_ab["net"] = OmegaGWpowMax

    min_f_net_idx = len(f)
    for pair in det_pairs:

        det1 = pair.split("-")[0].split("_")[0]
        det2 = pair.split("-")[-1].split("_")[0]

        min_f_det1_idx = np.where(PSD_interps[det1] != 1.)[0][0]
        min_f_det2_idx = np.where(PSD_interps[det2] != 1.)[0][0]
        # min_f_pair = np.min([f[min_f_det1_idx], f[min_f_det2_idx]])
        min_f_pair_idx = np.min([min_f_det1_idx, min_f_det2_idx])
        if min_f_pair_idx <= min_f_net_idx:
            min_f_net_idx = min_f_pair_idx

    if min_f_net_idx > 0:

        pls_ab["net"][:min_f_pair_idx] = 1.

    return pls_ab


def powerlaw_integrated_sensitivity(fmin, fmax, Net, fres=2000, H0=67.66, Tobs=1., SNRval=1., betamin=-50., betamax=50., betares=5000, is_ASD=True, return_all=False, verbose=False):
    """
    Compute the power-law integrated sensitivity curve for a detector network and all the detector pairs within.

    :param float fmin: Minimum frequency to consider in the estimation, in :math:`\\rm Hz`.
    :param float fmax: Maximum frequency to consider in the estimation, in :math:`\\rm Hz`.
    :param dict(dict, dict, ...) Net: Dictionary of dictionaries containing the latitude, ``'lat'``, longitude, ``'long'``, orientation, ``'xax'`` (all in degrees), shape, ``'shape'``, path to the ASD data, ``'psd_path'``, and arm length in the triangular case, ``'arm_length'`` (in :math:`\\rm km`), for each detector in the desired network, as in :py:data:`gwfast.gwfastGlobals.detectors`.
    :param int, optional fres: The desired frequency resolution.
    :param float, optional Tobs: The observational time to consider, in :math:`\\rm yr`.
    :param float, optional SNRval: The value of integrated signa-to-noise ratio time to consider.
    :param float, optional betamin: The minimum value of power-law index to consider.
    :param float, optional betamax: The maximum value of power-law index to consider.
    :param int, optional betares: The resolution of the power-law indices grid to consider.
    :param bool, optional is_ASD: Boolean specifying if the provided sensitivity files for each detector is a PSD or an ASD.
    :param bool, optional return_all: Boolean specifying if returning the PLS for all the pairs involved in the network.

    :return: frequency grid and Power-law integrated sensitivity curve for the chosen detector network and all the detector pairs within.
    :rtype: tuple(dict, dict)

    """
    print("WARNING: Results are H0 dependent. Consider tuning H0 to a value of your choice if needed. Default is H0 = 67.66 km s^-1 Mpc^-1.")
    for key in Net.keys():
        if (Net[key]['shape']=='T') and ('arm_length' not in Net[key].keys()):
            raise ValueError('For a triangle the arm length has to be provided.')
        if (Net[key]['shape']=='L') and ('arm_length' not in Net[key].keys()):
            Net[key]["arm_length"] = None

    # Constants and normalization
    # H0_to_h = (1e2 * 1e3 / (1e-3*glob.uGpc))
    # Seff_to_hsqOmeff  = ((10.*(np.pi**2.)/3.) * H0_to_h**(-2.))**(-1.)
    # Seff_to_hsqOmeff_sqr  = Seff_to_hsqOmeff**2.
    Seff_to_Omeff  = ((H0/(glob.uGpc/1e6))**2.)*3./10./np.pi**2.
    Seff_to_Omeff_sqr  = Seff_to_Omeff**2.

    # define the grid of power-law indices beta and expand the grid of frequency for vectorization
    betagrid = np.linspace(betamin, betamax, betares)
    f = np.geomspace(fmin, fmax, fres)
    fg = np.repeat(f[:,None], len(betagrid), axis=1)

    PSD_interps = {}

    for key in Net.keys():
        # Load PSD
        tmpASD = np.loadtxt(Net[key]['psd_path'], usecols=(0,1))
        if is_ASD:
            PSD_interps[key] = np.interp(f, tmpASD[:,0], tmpASD[:,1]**2, left=1., right=1.)
        else:
            PSD_interps[key] = np.interp(f, tmpASD[:,0], tmpASD[:,1], left=1., right=1.)

    # Sort detectors by shape to simplify the loop, first the L-shaped ones and then the triangular ones
    NetOrdered = {k: v for k, v in sorted(Net.items(), key=lambda item: item[1]['shape'])}
    detsList = list(NetOrdered.keys())
    individual_dets = copy.deepcopy(detsList)
    T_dets = [det for det in detsList if Net[det]["shape"] == "T"]
    N_T_dets = len(T_dets)
    for det in detsList:
        if Net[det]["shape"] == "T":
            individual_dets.pop(individual_dets.index(det))
            individual_dets.insert(0, det+"_0")
            individual_dets.insert(1, det+"_1")
            individual_dets.insert(2, det+"_2")

    det_pairs = [
        d1+"-"+d2
        for d1 in individual_dets
        for d2 in individual_dets
        if individual_dets.index(d1) < individual_dets.index(d2)
    ]

    orf_aux = {}
    for pair in det_pairs:
        orf_aux[pair] = {}
        det1 = pair.split("-")[0].split("_")[0]
        det2 = pair.split("-")[-1].split("_")[0]
        orf_aux[pair] = overlap_reduction_function(f, Net[det1], Net[det2], det1_name=det1, det2_name=det2, arm_length_1=Net[det1]["arm_length"], arm_length_2=Net[det2]["arm_length"])

    orf = {pair: orf_aux[pair][pair] for pair in det_pairs}
    del orf_aux
    pls_ab = {}
    fgrid_finer_ab = {}
    sum_over_pairs_of_inv_omega_eff_ab_sq = np.zeros_like(f)
    sum_over_pairs_of_inv_omega_eff_ab_sq_T_dets = {Tdet: np.zeros_like(f) for Tdet in T_dets}
    for pair in det_pairs:

        det1 = pair.split("-")[0].split("_")[0]
        det2 = pair.split("-")[-1].split("_")[0]

        if np.any(PSD_interps[det1] == 1.):
            fstart_idx = np.where(PSD_interps[det1] != 1.)[0][0]
        if np.any(PSD_interps[det2] == 1.):
            fstart_idx = np.where(PSD_interps[det2] != 1.)[0][0]
        else:
            fstart_idx = 0

        one_over_S_eff_n_ab_sq = (PSD_interps[det1]*PSD_interps[det2])**(-1.)
        inv_omega_eff_ab_sq = ((f**-6.) * (orf[pair]**2.) * one_over_S_eff_n_ab_sq) * Seff_to_Omeff_sqr
        sum_over_pairs_of_inv_omega_eff_ab_sq += inv_omega_eff_ab_sq

        if (det1 == det2) and (det1 in T_dets):

            sum_over_pairs_of_inv_omega_eff_ab_sq_T_dets[det1] += inv_omega_eff_ab_sq

        if return_all:

            n2 = np.trapz(inv_omega_eff_ab_sq[fstart_idx:][:, None]*np.log(fg[fstart_idx:])*(fg[fstart_idx:]**(2.*betagrid[None, :])), x=fg[fstart_idx:], axis=0)
            d2 = np.trapz(inv_omega_eff_ab_sq[fstart_idx:][:, None]*(fg[fstart_idx:]**(2.*betagrid[None, :])), x=fg[fstart_idx:], axis=0)

            fgrid_finer_ab_aux = np.exp(n2/d2)
            pls_ab_aux = (SNRval/np.sqrt(2.*Tobs*glob.seconds_in_year*d2))*np.exp(betagrid*n2/d2)
            regular = np.where((pls_ab_aux != 0.) & (np.isfinite(pls_ab_aux)) & (np.isfinite(fgrid_finer_ab_aux)))[0]
            pls_ab[pair] = pls_ab_aux[regular]
            fgrid_finer_ab[pair] = fgrid_finer_ab_aux[regular]

    if N_T_dets > 0 and return_all:
        for Tdet in T_dets:

            n2 = np.trapz(sum_over_pairs_of_inv_omega_eff_ab_sq_T_dets[Tdet][:, None]*np.log(fg)*(fg**(2.*betagrid[None, :])), x=fg, axis=0)
            d2 = np.trapz(sum_over_pairs_of_inv_omega_eff_ab_sq_T_dets[Tdet][:, None]*(fg**(2.*betagrid[None, :])), x=fg, axis=0)
            fgrid_finer_ab_aux = np.exp(n2/d2)
            pls_ab_aux = (SNRval/np.sqrt(2.*Tobs*glob.seconds_in_year*d2))*np.exp(betagrid*n2/d2)
            regular = np.where((pls_ab_aux != 0.) & (np.isfinite(pls_ab_aux)) & (np.isfinite(fgrid_finer_ab_aux)))[0]
            pls_ab[Tdet] = pls_ab_aux[regular]
            fgrid_finer_ab[Tdet] = fgrid_finer_ab_aux[regular]

    n2 = np.trapz(sum_over_pairs_of_inv_omega_eff_ab_sq[:, None]*np.log(fg)*(fg**(2.*betagrid[None, :])), x=fg, axis=0)
    d2 = np.trapz(sum_over_pairs_of_inv_omega_eff_ab_sq[:, None]*(fg**(2.*betagrid[None, :])), x=fg, axis=0)

    fgrid_finer_ab_aux = np.exp(n2/d2)
    pls_ab_aux = (SNRval/np.sqrt(2.*Tobs*glob.seconds_in_year*d2))*np.exp(betagrid*n2/d2)
    regular = np.where((pls_ab_aux != 0.) & (np.isfinite(pls_ab_aux)) & (np.isfinite(fgrid_finer_ab_aux)))[0]
    pls_ab["net"] = pls_ab_aux[regular]
    fgrid_finer_ab["net"] = fgrid_finer_ab_aux[regular]

    for configuration in pls_ab.keys():
        if fgrid_finer_ab[configuration].min() > fmin:
            print(f"WARNING: for {configuration}, the minimum frequency of the PLS is {fgrid_finer_ab[configuration].min():.3f} Hz, higher than the desired minimum frequency {fmin} Hz, consider using a larger grid for the power-law indices.")
        elif fgrid_finer_ab[configuration].max() < fmax:
            print(f"WARNING: for {configuration}, the maximum frequency of the PLS is {fgrid_finer_ab[configuration].max():.3f} Hz, lower than the desired maximum frequency {fmax} Hz, consider using a larger grid for the power-law indices.")

    return fgrid_finer_ab, pls_ab


def powerlaw_integrated_sensitivity_minimum(fmin, fmax, Net, fres=2000, H0=67.66, Tobs=1., SNRval=1., is_ASD=True, return_all=False):

    """
    Compute the minimum of the power-law integrated sensitivity curve for a detector network.

    :param float fmin: Minimum frequency to consider in the estimation, in :math:`\\rm Hz`.
    :param float fmax: Maximum frequency to consider in the estimation, in :math:`\\rm Hz`.
    :param dict(dict, dict, ...) Net: Dictionary of dictionaries containing the latitude, ``'lat'``, longitude, ``'long'``, orientation, ``'xax'`` (all in degrees), shape, ``'shape'``, path to the ASD data, ``'psd_path'``, and arm length in the triangular case, ``'arm_length'`` (in :math:`\\rm km`), for each detector in the desired network, as in :py:data:`gwfast.gwfastGlobals.detectors`. In the case of triangular detectors, we also give the possibility to account for onsite correlation among different interferometers. This can be done adding an ``'onsite_corr'`` key to the detector dictionary containing the fraction of correlation (in the interval :math:`[0,\,1)`) given as a scalar (resulting in a frequency-independent correlation), a path to a data file or a 2D array (for a frequency-dependent correlation, the first column will be interpreted as the frequency grid and the second as the fraction of correlation).
    :param float, optional H0: The present-day value of the Hubble parameter to use in the estimation, in :math:`\\rm km \, s^{-1}\, Mpc^{-1}`.
    :param float, optional Tobs: The observational time to consider, in :math:`\\rm yr`.
    :param float, optional SNRval: The value of integrated signa-to-noise ratio time to consider.
    :param bool, optional is_ASD: Boolean specifying if the provided sensitivity files for each detector is a PSD or an ASD.
    :param bool, optional return_all: Boolean specifying if returning the output for all the pairs involved in the network.

    :return: Power-law integrated sensitivity curve minimum frequency and value for the chosen detector network.
    :rtype: tuple(dict, dict)
    """
    print("WARNING: Results are H0 dependent. Consider tuning H0 to a value of your choice if needed. Default is H0 = 67.66 km s^-1 Mpc^-1.")
    for key in Net.keys():
        if (Net[key]['shape']=='T') and ('arm_length' not in Net[key].keys()):
            raise ValueError('For a triangle the arm length has to be provided.')
        if (Net[key]['shape']=='L') and ('arm_length' not in Net[key].keys()):
            Net[key]["arm_length"] = None

    f = np.geomspace(fmin, fmax, fres)
    # H0_to_h = (1e2 * 1e3 / (1e-3*glob.uGpc))
    # Seff_to_hsqOmeff  = ((10.*(np.pi**2.)/3.) * H0_to_h**(-2.))**(-1.)
    # Seff_to_hsqOmeff_sqr  = Seff_to_hsqOmeff**2.
    Seff_to_Omeff  = ((H0/(glob.uGpc/1e6))**2.)*3./10./np.pi**2.
    Seff_to_Omeff_sqr  = Seff_to_Omeff**2.

    PSD_interps = {}
    detsList = list(Net.keys())
    for key in Net.keys():
        tmpASD = np.loadtxt(Net[key]['psd_path'], usecols=(0,1))
        if is_ASD:
            PSD_interps[key] = np.interp(f, tmpASD[:,0], tmpASD[:,1]**2, left=1., right=1.)
        else:
            PSD_interps[key] = np.interp(f, tmpASD[:,0], tmpASD[:,1], left=1., right=1.)

    NetOrdered = {k: v for k, v in sorted(Net.items(), key=lambda item: item[1]['shape'])}
    detsList = list(NetOrdered.keys())
    individual_dets = copy.deepcopy(detsList)
    T_dets = [det for det in detsList if Net[det]["shape"] == "T"]
    for det in detsList:
        if Net[det]["shape"] == "T":
            individual_dets.pop(individual_dets.index(det))
            individual_dets.insert(0, det+"_0")
            individual_dets.insert(1, det+"_1")
            individual_dets.insert(2, det+"_2")

    det_pairs = [
        d1+"-"+d2
        for d1 in individual_dets
        for d2 in individual_dets
        if individual_dets.index(d1) < individual_dets.index(d2)
    ]

    orf_aux = {}
    for pair in det_pairs:
        orf_aux[pair] = {}
        det1 = pair.split("-")[0].split("_")[0]
        det2 = pair.split("-")[-1].split("_")[0]
        orf_aux[pair] = overlap_reduction_function(f, Net[det1], Net[det2], det1_name=det1, det2_name=det2, arm_length_1=Net[det1]["arm_length"], arm_length_2=Net[det2]["arm_length"])

    orf = {pair: orf_aux[pair][pair] for pair in det_pairs}
    pls_min_ab = {}
    f_min_ab = {}
    sum_over_pairs_of_inv_omega_eff_ab_sq = np.zeros_like(f)
    sum_over_pairs_of_inv_omega_eff_ab_sq_T_dets = {Tdet: np.zeros_like(f) for Tdet in T_dets}
    for pair in det_pairs:

        det1 = pair.split("-")[0].split("_")[0]
        det2 = pair.split("-")[-1].split("_")[0]

        if np.any(PSD_interps[det1] == 1.):
            fstart_idx = np.where(PSD_interps[det1] != 1.)[0][0]
        if np.any(PSD_interps[det2] == 1.):
            fstart_idx = np.where(PSD_interps[det2] != 1.)[0][0]
        else:
            fstart_idx = 0

        one_over_S_eff_n_ab_sq = (PSD_interps[det1]*PSD_interps[det2])**(-1.)
        inv_omega_eff_ab_sq = ((f**-6.) * (orf[pair]**2.) * one_over_S_eff_n_ab_sq) * Seff_to_Omeff_sqr
        sum_over_pairs_of_inv_omega_eff_ab_sq += inv_omega_eff_ab_sq

        if (Net[det1] == Net[det2]) and (Net[det1]["shape"] == "T"):

            sum_over_pairs_of_inv_omega_eff_ab_sq_T_dets[det1] += inv_omega_eff_ab_sq

        if return_all:

            integral1 = np.trapz(inv_omega_eff_ab_sq[fstart_idx:], x=f[fstart_idx:])
            integral2 = np.trapz(inv_omega_eff_ab_sq[fstart_idx:]*np.log(f[fstart_idx:]), x=f[fstart_idx:])
            f_min_ab[pair] = np.exp(integral2/integral1)
            pls_min_ab[pair] = SNRval/np.sqrt(2.*Tobs*glob.seconds_in_year*integral1)

    for Tdet in T_dets:

        integral1 = np.trapz(sum_over_pairs_of_inv_omega_eff_ab_sq_T_dets[Tdet], x=f)
        integral2 = np.trapz(sum_over_pairs_of_inv_omega_eff_ab_sq_T_dets[Tdet]*np.log(f), x=f)
        f_min_ab[Tdet] = np.exp(integral2/integral1)
        pls_min_ab[Tdet] = SNRval/np.sqrt(2.*Tobs*glob.seconds_in_year*integral1)

    integral1 = np.trapz(sum_over_pairs_of_inv_omega_eff_ab_sq, x=f, axis=0)
    integral2 = np.trapz(sum_over_pairs_of_inv_omega_eff_ab_sq*np.log(f), x=f)
    f_min_ab["net"] = np.exp(integral2/integral1)
    pls_min_ab["net"] = SNRval/np.sqrt(2.*Tobs*glob.seconds_in_year*integral1)

    return f_min_ab, pls_min_ab


##############################################################################
# SNR
##############################################################################    

def background_SNR_crosscorr(f, omega_gw, Net, H0=67.66, Tobs=1., is_ASD=True, return_all=False):
    """
    Compute the integrated signal-to-noise ratio for a cross-correlation search for a detector network (see e.g. `arXiv:1310.5300 <https://arxiv.org/abs/1310.5300>`_).

    :param array or float f: The frequency(ies) at which the background signal is provided, in :math:`\\rm Hz`.
    :param array or float omega_gw: The adimensional background signal(s) to consider as fractional energy density contribution with respect to the critical energy density of the Universe today.
    :param dict(dict, dict, ...) Net: Dictionary of dictionaries containing the latitude, ``'lat'``, longitude, ``'long'``, orientation, ``'xax'`` (all in degrees), shape, ``'shape'``, path to the ASD data, ``'psd_path'``, and arm length in the triangular case, ``'arm_length'`` (in :math:`\\rm km`), for each detector in the desired network, as in :py:data:`gwfast.gwfastGlobals.detectors`. In the case of triangular detectors, we also give the possibility to account for onsite correlation among different interferometers. This can be done adding an ``'onsite_corr'`` key to the detector dictionary containing the fraction of correlation (in the interval :math:`[0,\,1)`) given as a scalar (resulting in a frequency-independent correlation), a path to a data file or a 2D array (for a frequency-dependent correlation, the first column will be interpreted as the frequency grid and the second as the fraction of correlation).
    :param float, optional H0: The present-day value of the Hubble parameter to use in the estimation, in :math:`\\rm km \, s^{-1}\, Mpc^{-1}`.
    :param float, optional Tobs: The observational time to consider, in :math:`\\rm yr`.
    :param bool, optional is_ASD: Boolean specifying if the provided sensitivity files for each detector is a PSD or an ASD.
    :param bool, optional return_all: Boolean specifying if returning the SNR(s) for all the pairs involved in the network.

    :return: SNR(s) for the provided signal(s) at the chosen network.
    :rtype: dict

    """
    print("WARNING: Results are H0 dependent. Consider tuning H0 to a value of your choice if needed. Default is H0 = 67.66 km s^-1 Mpc^-1.")
    for key in Net.keys():
        if (Net[key]['shape']=='T') and ('arm_length' not in Net[key].keys()):
            raise ValueError('For a triangle the arm length has to be provided.')
        if (Net[key]['shape']=='L') and ('arm_length' not in Net[key].keys()):
            Net[key]["arm_length"] = None

    # H0_to_h = (1e2 * 1e3 / (1e-3*glob.uGpc))
    # Seff_to_hsqOmeff  = ((10.*(np.pi**2.)/3.) * H0_to_h**(-2.))**(-1.)
    # Seff_to_hsqOmeff_sqr  = Seff_to_hsqOmeff**2.

    Seff_to_Omeff  = ((H0/(glob.uGpc/1e6))**2.)*3./10./np.pi**2.
    Seff_to_Omeff_sqr  = Seff_to_Omeff**2.

    if f.ndim < omega_gw.ndim:
        f_integ = np.array([f for _ in range(omega_gw.shape[-1])]).T
    else:
        f_integ = f
 
    PSD_interps = {}
    detsList = list(Net.keys())
    for key in Net.keys():
        tmpASD = np.loadtxt(Net[key]['psd_path'], usecols=(0,1))
        if is_ASD:
            PSD_interps[key] = np.interp(f, tmpASD[:,0], tmpASD[:,1]**2, left=1., right=1.)
        else:
            PSD_interps[key] = np.interp(f, tmpASD[:,0], tmpASD[:,1], left=1., right=1.)

    NetOrdered = {k: v for k, v in sorted(Net.items(), key=lambda item: item[1]['shape'])}
    detsList = list(NetOrdered.keys())
    individual_dets = copy.deepcopy(detsList)
    T_dets = [det for det in detsList if Net[det]["shape"] == "T"]
    N_T_dets = len(T_dets)
    for det in detsList:
        if Net[det]["shape"] == "T":
            individual_dets.pop(individual_dets.index(det))
            individual_dets.insert(0, det+"_0")
            individual_dets.insert(1, det+"_1")
            individual_dets.insert(2, det+"_2")

    det_pairs = [
        d1+"-"+d2
        for d1 in individual_dets
        for d2 in individual_dets
        if individual_dets.index(d1) < individual_dets.index(d2)
    ]

    orf_aux = {}
    for pair in det_pairs:
        orf_aux[pair] = {}
        det1 = pair.split("-")[0].split("_")[0]
        det2 = pair.split("-")[-1].split("_")[0]
        orf_aux[pair] = overlap_reduction_function(f, Net[det1], Net[det2], det1_name=det1, det2_name=det2, arm_length_1=Net[det1]["arm_length"], arm_length_2=Net[det2]["arm_length"])

    orf = {pair: orf_aux[pair][pair] for pair in det_pairs}
    snr = {}
    sum_over_pairs_of_inv_omega_eff_ab_sq = np.zeros_like(f)
    sum_over_pairs_of_inv_omega_eff_ab_sq_T_dets = {Tdet: np.zeros_like(f) for Tdet in T_dets}
    for pair in det_pairs:

        det1 = pair.split("-")[0].split("_")[0]
        det2 = pair.split("-")[-1].split("_")[0]

        if np.any(PSD_interps[det1] == 1.):
            fstart_idx = np.where(PSD_interps[det1] != 1.)[0][0]
        if np.any(PSD_interps[det2] == 1.):
            fstart_idx = np.where(PSD_interps[det2] != 1.)[0][0]
        else:
            fstart_idx = 0

        one_over_S_eff_n_ab_sq = (PSD_interps[det1]*PSD_interps[det2])**(-1.)
        inv_omega_eff_ab_sq = ((f**-6.) * (orf[pair]**2.) * one_over_S_eff_n_ab_sq) * Seff_to_Omeff_sqr
        sum_over_pairs_of_inv_omega_eff_ab_sq += inv_omega_eff_ab_sq

        if (Net[det1] == Net[det2]) and (Net[det1]["shape"] == "T"):

            sum_over_pairs_of_inv_omega_eff_ab_sq_T_dets[det1] += inv_omega_eff_ab_sq

        if return_all:

            if f.ndim < omega_gw.ndim:
                inv_omega_eff_ab_sq = inv_omega_eff_ab_sq[:, np.newaxis]

            integral = np.trapz((omega_gw[fstart_idx:]**2.)*inv_omega_eff_ab_sq[fstart_idx:], f_integ[fstart_idx:], axis=0)
            snr[pair] = np.sqrt(2.*Tobs*glob.seconds_in_year)*np.sqrt(integral)

    if N_T_dets > 0 and return_all:
        for Tdet in T_dets:

            if f.ndim < omega_gw.ndim:
                sum_over_pairs_of_inv_omega_eff_ab_sq_T_dets[Tdet] = sum_over_pairs_of_inv_omega_eff_ab_sq_T_dets[Tdet][:, np.newaxis]

            integral = np.trapz((omega_gw**2.)*sum_over_pairs_of_inv_omega_eff_ab_sq_T_dets[Tdet], f_integ, axis=0)
            snr[Tdet] = np.sqrt(2.*Tobs*glob.seconds_in_year)*np.sqrt(integral)

    if f.ndim < omega_gw.ndim:
        sum_over_pairs_of_inv_omega_eff_ab_sq = sum_over_pairs_of_inv_omega_eff_ab_sq[:, np.newaxis]

    integral = np.trapz((omega_gw**2.)*sum_over_pairs_of_inv_omega_eff_ab_sq, f_integ, axis=0)
    snr["net"] = np.sqrt(2.*Tobs*glob.seconds_in_year)*np.sqrt(integral)

    return snr


##############################################################################
# OMEGA_GW
##############################################################################

def minimum_detectable_energy_density_singleDet(f, Det, H0=67.66, SNRval=1., is_ASD=True, pattern_Avg=2./5.):
    """
    Compute the minimum detectable background energy density for a given single detector (see e.g. `M. Maggiore -- Gravitational Waves Vol. 1 <https://global.oup.com/academic/product/gravitational-waves-9780198570745?q=Michele%20Maggiore&lang=en&cc=it>`_, section 7.8.2).

    :param array or float f: The frequency(ies) at which to evaluate the power-law integrated sensitivity curve, in :math:`\\rm Hz`.
    :param str psd_path: Path to the PSD data file of the detector.
    :param float, optional H0: The present-day value of the Hubble parameter to use in the estimation, in :math:`\\rm km \, s^{-1}\, Mpc^{-1}`.
    :param float, optional SNRval: The value of integrated signa-to-noise ratio time to consider.
    :param bool, optional is_ASD: Boolean specifying if the provided sensitivity file is a PSD or an ASD.
    :param float, optional pattern_Avg: Angular efficiency factor of the chosen detector :math:`F = \langle F_+^2\rangle + \langle F_{\\times}^2\rangle`. This is :math:`F = 2/5` for an L--shaped interferometer, :math:`F = 3/10` for a triangular--shaped interferometer, and :math:`F = 8/15` for a resonant bar.

    :return: Minimum detectable background energy density for the chosen detector as a function of frequency.
    :rtype: array

    """
    print("WARNING: Results are H0 dependent. Consider tuning H0 to a value of your choice if needed. Default is H0 = 67.66 km s^-1 Mpc^-1.")
    prefac = 4.*np.pi*np.pi/(3.*(H0/(glob.uGpc/1e6))**2.)

    tmpPSD = np.loadtxt(Det["psd_path"], usecols=(0,1))
    if is_ASD:
        PSD_interps = np.interp(f, tmpPSD[:,0], tmpPSD[:,1]**2., left=1., right=1.)
    else:
        PSD_interps = np.interp(f, tmpPSD[:,0], tmpPSD[:,1], left=1., right=1.)

    return prefac*(f**3.)*np.where(PSD_interps==1., 0., PSD_interps)*(SNRval**2.)/pattern_Avg


def GW_energy_density_from_catalog(f, evParams, wf_model, H0=67.66, Tobs=1., SNRs=None, SNRth=12., return_detected=True):
    """
    Compute the gravitational wave energy density :math:`\Omega_{\\rm GW}` on a given frequency grid for a catalog of CBC events.

    :param array or float f: The frequency(ies) at which to evaluate the GW energy density, in :math:`\\rm Hz`.
    :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
    :param WaveFormModel wf_model: Object containing the waveform model.
    :param float, optional H0: The present-day value of the Hubble parameter to use in the estimation, in :math:`\\rm km \, s^{-1}\, Mpc^{-1}`.
    :param float, optional Tobs: The observational time to consider, in :math:`\\rm yr`.
    :param array, optional SNRs: The signal-to-noise ratios of the events in the catalog for a given network of detectors. If provided, only events with SNR above (or below if ``return_detected`` is set to ``False``) ``SNRth`` will be considered.
    :param floar, optional SNRth: The signal-to-noise ratio threshold to use.
    :param bool, optional return_detected: Boolean specifying if the returned GW energy density refers to the detected events or to the undetected ones.

    :return: GW energy density (adimensional) as a function of frequency.
    :rtype: array

    """
    print("WARNING: Results are H0 dependent. Consider tuning H0 to a value of your choice if needed. Default is H0 = 67.66 km s^-1 Mpc^-1.")

    prefac = 4.*np.pi*np.pi/(3.*(H0/(glob.uGpc/1e6))**2.)

    if SNRs is not None:
        if return_detected:
            evParams_use = {k:evParams[k][SNRs>SNRth] for k in evParams.keys()}
        else:
            evParams_use = {k:evParams[k][SNRs<SNRth] for k in evParams.keys()}
    else:
        evParams_use = evParams
    Nevs = len(evParams_use['dL'])
    fgrids = np.repeat(f, Nevs).reshape(f.shape[0], Nevs)

    if not ((wf_model.is_HigherModes) or (wf_model.is_Precessing)):
        Ampls = wf_model.Ampl(fgrids, **evParams_use)
        Aps, Acs = Ampls*0.5*(1.+(np.cos(evParams_use['iota']))**2.), 1j*Ampls*np.cos(evParams_use['iota'])
    else:
        Aps, Acs = wf_model.hphc(fgrids, **evParams_use)

    totAmpl = np.nan_to_num(np.abs(Aps).astype(np.float64)**2 + np.abs(Acs).astype(np.float64)**2)

    OmGW = prefac*totAmpl.sum(axis=-1)*(f**3.)/(Tobs*glob.seconds_in_year)

    return OmGW
