#
#    Copyright (c) 2022 Francesco Iacovelli <francesco.iacovelli@unige.ch>, Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.


import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
import jax
jax.devices('cpu')
from jax.config import config
config.update("jax_enable_x64", True)

# We use both the original numpy, denoted as onp, and the JAX implementation of numpy, denoted as np
import numpy as onp
import copy
import mpmath
import scipy

try:
    onp.float128(1.)
    typeuse='float128'
except AttributeError:
    print('WARNING: numpy float128 type not supported on this machine, resorting to float64, precision might be lower.')
    typeuse='float64'

##############################################################################
# INVERSION AND SANITY CHECKS
##############################################################################
def CovMatr(FisherMatrix, 
            invMethodIn='cho', 
            condNumbMax=1e50, 
            truncate=False, svals_thresh=1e-15,  
            verbose=False,
            alt_method = 'svd'
            ):
    """
    Invert the Fisher matrix(ces), obtaining the covariance matrix(ces).
    
    :param array FisherMatrix: Array containing the Fisher matrix(ces) to invert, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :param str invMethodIn: Inversion method to use. To be chosen among ``'inv'``, ``'cho'``, ``'svd'``, ``'svd_reg'`` and ``'lu'``.
    :param float condNumbMax: Maximum allowed condition number, above which the inverse matrix is not computed. The default value is 1e50, so the code will try to invert every matrix, irrespectively of the conditioning.
    :param bool, optional truncate: Boolean specifying if, when using the ``'svd'`` method, the function has to truncate the smallest singular values to the minimum allowed numerical precision.
    :param float svals_thresh: Threshold value to truncate the singular values when using the ``'svd'`` method, or to exclude the singular values from the inversion when using the ``'svd_reg'`` method.
    :param bool, optional verbose: Boolean specifying if the code has to print additional details during execution.
    :param str alt_method: Inversion method to use in case the inverison with ``invMethodIn`` fails. To be chosen among ``'inv'``, ``'cho'``, ``'svd'``, ``'svd_reg'`` and ``'lu'``. It has to be different from ``invMethodIn``.
    :return: Covariance matrix(ces) (3-D array) and inversion error(s) (1-D array). The covariance matrix(ces) have the same shape of ``FisherMatrix``, i.e. :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :rtype: tuple(array, array)
    
    """
    FisherMatrixOr = copy.deepcopy(FisherMatrix)
    
    reweighted=False
    FisherM = FisherMatrix.astype(typeuse)
    CovMatr = onp.zeros(FisherMatrix.shape).astype(typeuse)
    
    cho_failed = 0
    for k in range(FisherM.shape[-1]):
        
        
        if onp.all(onp.isnan(FisherM[:, :, k])):
            if verbose:
                print('Fisher is nan at position %s. ' %k)
            CovMatr[:, :, k] = onp.full( FisherM[:, :, k].shape , onp.nan)
        else:
            # go to mpmath
            ff = mpmath.matrix( FisherM[:, :, k].astype(typeuse))
            try:
                # Conditioning of the original Fisher
                E, _ = mpmath.eigh(ff)
                E = onp.array(E.tolist(), dtype=typeuse)
                if onp.any(E<0) and verbose:
                    print('Matrix is not positive definite!')
    
                cond = onp.max(onp.abs(E))/onp.min(onp.abs(E))
                if verbose:
                    print('Condition of original matrix: %s' %cond)
                
                try:
                    # Normalize by the diagonal
                    ws =  mpmath.diag([ 1/mpmath.sqrt(ff[i, i]) for i in range(FisherM.shape[-2]) ])
                    FisherM_ = ws*ff*ws
                    # Conditioning of the new Fisher
                    EE, _ = mpmath.eigh(FisherM_)
                    E = onp.array(EE.tolist(), dtype=typeuse)
                    cond = onp.max(onp.abs(E))/onp.min(onp.abs(E))
                    if verbose:
                        print('Condition of the new matrix: %s' %cond)
                    reweighted=True
                except ZeroDivisionError:
                    print('The Fisher matrix has a zero element on the diagonal at position %s. The normalization procedure will not be applied. Consider using a prior.' %k)
                    FisherM_ = ff
                
                
                invMethod = invMethodIn
                if onp.any(E<0):
                    if verbose:
                        print('Matrix is not positive definite at position %s!' %k)
                    if invMethodIn=='cho':
                        cho_failed+=1
                        invMethod=alt_method
                        if verbose:
                            print('Cholesky decomposition not usable. Using method %s' %invMethod)
                elif invMethod=='cho':
                    try:
                        # In rare cases, the choleski decomposition still fails even if the eigenvalues are positive...
                        # likely for very small eigenvalues
                        c = (mpmath.cholesky(FisherM_))**-1
                    except Exception as e:
                        print(e)
                        invMethod=alt_method
                        print('Cholesky decomposition not usable. Eigenvalues seem ok but cholesky decomposition failed. Using method %s' %invMethod)
                        #print('Eigenvalues: %s' %str(E))
                        cho_failed+=1
    
                if invMethod=='inv':
                        cc = FisherM_**-1
                elif invMethod=='cho':
                        #c = cF**-1
                        cc = c.T*c
                elif invMethod=='svd':
                        U, Sm, V = mpmath.svd_r(FisherM_)
                        S = onp.array(Sm.tolist(), dtype=typeuse)
                        if ((truncate) and (onp.abs(cond)>condNumbMax)):
                            if verbose:
                                print('Truncating singular values below %s' %svals_thresh)
                           
                            maxev = onp.max(onp.abs(S))
                            Sinv = mpmath.matrix(onp.array([1/s if onp.abs(s)/maxev>svals_thresh else 1/(maxev*svals_thresh) for s in S ]).astype(typeuse))
                            St = mpmath.matrix(onp.array([s if onp.abs(s)/maxev>svals_thresh else maxev*svals_thresh for s in S ]).astype(typeuse))
                            
                            # Also copute truncated Fisher to quantify inversion error consistently
                            truncFisher = U*mpmath.diag([s for s in St])*V
                            truncFisher = (truncFisher+truncFisher.T)/2
                            FisherMatrixOr[:, :, k] = onp.array(truncFisher.tolist(), dtype=typeuse)
                            
                            if verbose:
                                truncated = onp.abs(S)/maxev<svals_thresh #onp.array([1 if onp.abs(s)/maxev>svals_thresh else 0 for s in S ]
                                print('%s singular values truncated' %(truncated.sum()))
                        else:
                            Sinv = mpmath.matrix(onp.array([1/s for s in S ]).astype(typeuse))
                            St = S
                        
                        cc=V.T*mpmath.diag([s for s in Sinv])*U.T
                        
                elif invMethod=='svd_reg':
                    
                        U, Sm, V = mpmath.svd_r(FisherM_)
                        
                        S = onp.squeeze(onp.array(Sm.tolist(), dtype=typeuse))
                        Um = onp.array(U.tolist(), dtype=typeuse)
                        Vm = onp.array(V.tolist(), dtype=typeuse)
  
                        kVal = sum(S > svals_thresh)
                                                    
                        Sinv = mpmath.matrix(onp.array([1/s  for s in S ]).astype(typeuse))
                        cc = mpmath.matrix(Um[:, 0:kVal] @ onp.diag(1. / S[0:kVal]) @ Vm[0:kVal, :])
                                                
                        
                elif invMethod=='lu':
                        P, L, U = mpmath.lu(FisherM_)
                        ll = P*L
                        llinv = ll**-1
                        uinv=U**-1
                        cc = uinv*llinv
                
                
                # Enforce symmetry.
                cc = (cc+cc.T)/2
                
                if reweighted:
                    # Undo the reweighting
                    CovMatr_ = ws*cc*ws
                else:
                    CovMatr_ = cc
    
                CovMatr[:, :, k] =  onp.array(CovMatr_.tolist(), dtype=typeuse)
                if verbose:
                    print()
            
            except Exception as e:
                # Eigenvalue decomposition failed
                print(e)
                print('Inversion failed!')
                CovMatr[:, :, k] = onp.full( FisherM[:, :, k].shape , onp.nan)

    
    eps = compute_inversion_error(FisherMatrixOr, CovMatr)

    if verbose:
            print('Error with %s: %s\n' %(invMethod, eps))
            print(' Inversion error with method %s: min=%s, max=%s, mean=%s, std=%s ' %(invMethodIn, onp.min(eps), onp.max(eps), onp.mean(eps), onp.std(eps)) )
            print('Method %s not possible on %s non-positive definite matrices, %s was used in those cases. ' %(invMethodIn, cho_failed, alt_method))
    return CovMatr , eps

    
def compute_inversion_error(Fisher, Cov):
    """
    Compute the inversion error given the Fisher and covariance matrices.
    
    :param array Fisher: Array containing the Fisher matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :param array Cov: Array containing the covariance matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    
    :return: Inversion error for the given matrices.
    :rtype: 1-D array
    
    """
    return onp.array([ onp.max( onp.abs(Cov[:, :, i]@Fisher[:, :, i]-onp.eye(Fisher.shape[0]))) for i in range(Fisher.shape[-1]) ])
    



            
def CheckFisher(FisherM, condNumbMax=1.0e15, use_mpmath=True, verbose=False):
    """
    Perform some sanity checks on the Fisher matrix, in particular:
    
        - compute the eigenvalues and eigenvectors;
        - compute the condition number (ratio of the largest to smallest eigenvalue) and check this is not large;
    
    :param array FisherM: Array containing the Fisher matrix(ces) to check, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :param float condNumbMax: Maximum allowed condition number, depending on the machine precision.
    :param bool, optional use_mpmath: Boolean specifying if the checks have to be performed using the `mpmath library <https://mpmath.org>`_.
    :param bool, optional verbose: Boolean specifying if the code has to print additional details during execution.
    
    :return: Eigenvalues, eigenvectors and condition number(s) of the input Fisher matrix(ces).
    :rtype: tuple(array, array, array)
    
    """
    
    # Being the Fisher symmetric by definition, we can use the numpy.linalg function 'eigh', to speed up a bit
    # The input has size (Npar,Npar,Nev), so we have to swap
    
    if not use_mpmath:
        evals, evecs = scipy.linalg.eigh(FisherM.transpose(2,0,1))
    else:
        evals = onp.zeros(FisherM.shape[1:][::-1])
        evecs = onp.zeros(FisherM.shape[::-1])
        for k in range(FisherM.shape[-1]):
            
            if onp.all(onp.isnan(FisherM[:, :, k])):
                if verbose:
                    print('Fisher is nan at position %s. ' %k)
                evals[k, :]=onp.full( FisherM.shape[0] , onp.nan)
                evecs[k, :, :]=onp.full( FisherM[:, :, k].shape , onp.nan)
            else:
                try:
                    aam = mpmath.matrix(FisherM[:,:,k].astype(typeuse))
                    E, ER = mpmath.eigh(aam)
                    evals[k, :] = onp.array(E, dtype=typeuse)
                    evecs[k, :, :] = onp.array(ER.tolist(), dtype=typeuse)
                except Exception as e:
                    print(e)
                    print('Trying with scipy')
                    try:
                        evals[k, :], evecs[k, :, :] = scipy.linalg.eigh(FisherM[:,:,k])
                    except Exception as e:
                        print(e)
                        print('Event is number %s' %k)
                        evals[k, :], evecs[k, :, :] = onp.full(FisherM.shape[0], onp.nan, ), onp.full((FisherM.shape[0], FisherM.shape[0]), onp.nan, )
                        #condNumber = None
                        print(FisherM[:,:,k])
                    
                
    if onp.any(evals <= 0.):
        print('WARNING: one or more eigenvalues are negative at position(s) %s' %str( onp.unique(onp.where(evals<0)[0]) ))
    

    condNumber = onp.abs(evals).max(axis=1)/onp.abs(evals).min(axis=1)
    
    if onp.any(condNumber>condNumbMax) and verbose:
                    print('WARNING: the condition number is too large (%s>%s)'%(condNumber,condNumbMax) )
                    print('Unreliable covariance at positions ' +str(condNumber>condNumbMax))
    elif verbose:
                    print('Condition number= %s . Ok. '%condNumber)
    
    return evals, evecs, condNumber


def perturb_Fisher(totF, eps=1e-10, **kwargs):
    """
    Add random small perturbations to the FIM to a specified decimal and prints the relative errors, to check if the inversion remains stable.
    
    :param array totF: Array containing the Fisher matrix(ces) to check, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :param float eps: Decimal at which to add the random perturbation.
    :param kwargs: Optional arguments to be passed to :py:class:`gwfast.fisherTools.CovMatr`, such as ``invMethodIn``.
    
    """
    Cov_base, _ = CovMatr(totF, **kwargs)


    totF_random = totF + onp.random.rand(*totF.shape)*eps
    Cov, _ = CovMatr(totF_random, **kwargs)
    
    
    epsErr = [onp.linalg.norm( Cov_base[i]/Cov[i]-1, ord=onp.inf) for i in range(Cov.shape[-1])]
    print('Relative errors when perturbing at the %s level: %s' %(eps, epsErr))


def check_covariance(FisherM, Cov, tol=1e-10):
    """
    Compute the inversion error, print the difference between the product of Fisher and covariance matrices and the identity matrix on the diagonal, and print the off–diagonal elements of the product of Fisher and covariance matrices higher than a threshold.
    
    :param array FisherM: Array containing the Fisher matrix(ces) to check, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :param array Cov: Array containing the covariance matrix(ces) to check, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :param float tol: Threshold above which to print the off–diagonal elements of the product of Fisher and covariance matrices.
    
    :return: Product of Fisher and covariance matrices, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :rtype: 3-D array
    
    """
    recovered_Ids = [ Cov[:, :, i]@FisherM[:, :, i] for i in range(Cov.shape[-1])]
    
    #
    epsErr = compute_inversion_error(FisherM, Cov)
    print('Inversion errors: %s' %epsErr)
    
    # 
    diag_diff = [recovered_Ids[i].diagonal()-1 for i in range(Cov.shape[-1])]
    print('diagonal-1 = %s' %str(diag_diff) )
    
    # 
    offDiag = [ recovered_Ids[i][onp.matrix(~onp.eye(recovered_Ids[i].shape[0],dtype=bool))] for i in range(Cov.shape[-1])] 
    
    print('Max off diagonal: %s' % str([ max(offDiag[i]) for i in range(Cov.shape[-1])] ) )
    
    print('\nmask: where F*S(off-diagonal)>%s (--> problematic if True off diagonal)' %tol)
    print([recovered_Ids[i]>tol for i in range(Cov.shape[-1])])
    
    return recovered_Ids
    
##############################################################################
# ADDING PRIOR, ELIMINATING ROWS
##############################################################################

    
    
def fixParams(MatrIn, ParNums_inp, ParMarg):
    """
    Fix one or multiple parameters to their fiducial values in the Fisher matrix.
    
    :param array MatrIn: Array containing the Fisher matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :param dict(int) ParNums_inp: Dictionary specifying the position of each parameter in the input Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param list(str) ParMarg: List of the names of parameters to fix.
    
    :return: Fisher matrix with parameters fixed, of shape :math:`(\\tilde{N}_{\\rm parameters}`, :math:`\\tilde{N}_{\\rm parameters}`, :math:`N_{\\rm events})`, and dictionary specifying the position of each parameter in the new Fisher matrix. :math:`\\tilde{N}_{\\rm parameters}` is the original :math:`N_{\\rm parameters}` minus the number of parameters that have been fixed.
    :rtype: tuple(array, dict(int))
    
    """
    import copy
    ParNums = copy.deepcopy(ParNums_inp)
    
    IdxMarg = onp.sort(onp.array([ParNums[par] for par in ParMarg]))
    newdim = MatrIn.shape[0]-len(IdxMarg)
    
    NewMatr = onp.full( (newdim, newdim, MatrIn.shape[-1]), onp.NaN )
    
    for k in range(MatrIn.shape[-1]):
    
        Matr = onp.delete(MatrIn[:, :, k], IdxMarg, 0)
        Matr = onp.delete(Matr[:, :], IdxMarg, 1)
        NewMatr[:, :, k] = Matr
        
    # Given that we deleted some rows and columns, 
    # the meaning of the numbers of the remaining ones changes
    
    for pm in ParMarg:
        for k in ParNums.keys():
            if ParNums[k]>ParNums[pm]:
                ParNums[k] -= 1
        ParNums.pop(pm, None)

    return NewMatr, ParNums


def addPrior(Matr, vals, ParNums, ParAdd):
    """
    Add a Gaussian priors to the Fisher matrix on one or multiple parameters.
    
    :param array Matr: Array containing the Fisher matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :param list(float) vals: List of values to be added on the diagonal of the Fisher matrix.
    :param dict(int) ParNums: Dictionary specifying the position of each parameter in the Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param list(str) ParAdd: List of the names of parameters on which the prior should be added.
    
    :return: Fisher matrix with Gaussian priors added at the chosen positions, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :rtype: 3-D array
    
    """
    IdxAdd = onp.sort(onp.array([ParNums[par] for par in ParAdd]))
    
    pp = onp.zeros((Matr.shape[0], Matr.shape[1]))
    
    diag = onp.zeros(Matr.shape[0])
    diag[IdxAdd] = vals
    
    onp.fill_diagonal(pp, diag)
    
    if Matr.ndim==2:
        return pp+Matr
    else:
        return pp[:,:,onp.newaxis]+Matr


##############################################################################
# DERIVATIVES AND JACOBIANS
##############################################################################

def log_dL_to_dL_derivative_cov(or_matrix, ParNums, evParams):
    """
    Change variables in the covariance matrix from :math:`{\\rm log}(d_L)` to :math:`d_L`.
    
    :param array or_matrix: Array containing the covariance matrix, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :param dict(int) ParNums: Dictionary specifying the position of each parameter in the Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
    
    :return: Covariance matrix in :math:`d_L`, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :rtype: 2-D array
    
    """
    matrix = copy.deepcopy(or_matrix)
    #for i in range(matrix.shape[-1]):
        # This has to be vectorised
    try:
            matrix = matrix.at[:, ParNums['dL'], : ].set(matrix[:, ParNums['dL'], :]* evParams['dL'])
            matrix = matrix.at[ ParNums['dL'], : , : ].set(matrix[ParNums['dL'], :, :]* evParams['dL'])
    except AttributeError:
            matrix = matrix.astype(typeuse)
            matrix[:, ParNums['dL'], :] *= evParams['dL'].astype(typeuse)
            matrix[ ParNums['dL'], :, :] *= evParams['dL'].astype(typeuse)
    return matrix


def log_dL_to_dL_derivative_fish(or_matrix, ParNums, evParams):
    """
    Change variables in the Fisher matrix from :math:`{\\rm log}(d_L)` to :math:`d_L`.
    
    :param array or_matrix: Array containing the Fisher matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :param dict(int) ParNums: Dictionary specifying the position of each parameter in the Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
    
    :return: Fisher matrix in :math:`d_L`, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :rtype: 2-D array
    
    """
    matrix = copy.deepcopy(or_matrix)
    #for i in range(matrix.shape[-1]):
        # This has to be vectorised
    try:
            matrix = matrix.at[:, ParNums['dL'], : ].set(matrix[:, ParNums['dL'], :]/ evParams['dL'])
            matrix = matrix.at[ ParNums['dL'], : , : ].set(matrix[ ParNums['dL'], :, :]/ evParams['dL'])
    except AttributeError:
            matrix = matrix.astype(typeuse)
            matrix[:, ParNums['dL'], :] /= evParams['dL'].astype(typeuse)
            matrix[ ParNums['dL'], :, :] /= evParams['dL'].astype(typeuse)
    return matrix
    
    

def dm1_dMc(eta):
    """
    Compute the derivative of :math:`m_1` with respect to :math:`{\cal M}_c`.
    
    :param array eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    
    :return: :math:`\partial m_1/\partial {\cal M}_c`.
    :rtype: 1-D array
    
    """
    return (1+onp.sqrt(1-4*eta) )*eta**(-3./5.)/2

def dm2_dMc(eta):
    """
    Compute the derivative of :math:`m_2` with respect to :math:`{\cal M}_c`.
    
    :param array eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    
    :return: :math:`\partial m_2/\partial {\cal M}_c`.
    :rtype: 1-D array
    
    """
    return (1-onp.sqrt(1-4*eta) )*eta**(-3./5.)/2

def dm1_deta(Mc, eta):
    """
    Compute the derivative of :math:`m_1` with respect to :math:`\eta`.
    
    :param array or float Mc: Chirp mass of the binary, :math:`{\cal M}_c`, in units of :math:`\\rm M_{\odot}`.
    :param array eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    
    :return: :math:`\partial m_1/\partial \eta`.
    :rtype: 1-D array
    
    """
    return -Mc*(3-2*eta+3*onp.sqrt(1-4*eta))/(10*onp.sqrt(1-4*eta)*eta**(8./5.))
#(1-onp.sqrt(1-4*eta) )*eta**(-3./5.)/2

def dm2_deta(Mc, eta):
    """
    Compute the derivative of :math:`m_2` with respect to :math:`\eta`.
    
    :param array or float Mc: Chirp mass of the binary, :math:`{\cal M}_c`, in units of :math:`\\rm M_{\odot}`.
    :param array eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    
    :return: :math:`\partial m_2/\partial \eta`.
    :rtype: 1-D array
    
    """
    return -Mc*(-3+2*eta+3*onp.sqrt(1-4*eta))/(10*onp.sqrt(1-4*eta)*eta**(8./5.))


def dMc_dm1(m1, m2):
    """
    Compute the derivative of :math:`{\cal M}_c` with respect to :math:`m_1`.
    
    :param array or float m1: Mass of the first object, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
    :param array or float m2: Mass of the second object, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.
    
    :return: :math:`\partial {\cal M}_c/\partial m_1`.
    :rtype: 1-D array
    
    """
    return m2*(2*m1+3*m2)/(5*(m1*m2)**(2/5)*(m1+m2)**(6/5))

def dMc_dm2(m1, m2):
    """
    Compute the derivative of :math:`{\cal M}_c` with respect to :math:`m_2`.
    
    :param array or float m1: Mass of the first object, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
    :param array or float m2: Mass of the second object, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.
    
    :return: :math:`\partial {\cal M}_c/\partial m_2`.
    :rtype: 1-D array
    
    """
    return dMc_dm1(m2, m1)

def deta_dm1(m1, m2):
    """
    Compute the derivative of :math:`\eta` with respect to :math:`m_1`.
    
    :param array or float m1: Mass of the first object, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
    :param array or float m2: Mass of the second object, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.
    
    :return: :math:`\partial \eta/\partial m_1`.
    :rtype: 1-D array
    
    """
    return m2*(m2-m1)/(m1+m2)**3

def deta_dm2(m1, m2):
    """
    Compute the derivative of :math:`\eta` with respect to :math:`m_2`.
    
    :param array or float m1: Mass of the first object, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
    :param array or float m2: Mass of the second object, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.
    
    :return: :math:`\partial \eta/\partial m_2`.
    :rtype: 1-D array
    
    """
    return deta_dm1(m2, m1)


def J_m1m2_Mceta(Mc, eta):
    """
    Compute the Jacobian matrix from :math:`{\cal M}_c` and :math:`\eta` to :math:`m_1` and :math:`m_2`.
    
    :param array or float Mc: Chirp mass of the binary, :math:`{\cal M}_c`, in units of :math:`\\rm M_{\odot}`.
    :param array eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    
    :return: :math:`\partial (m_1, m_2)/\partial ({\cal M}_c, \eta)`.
    :rtype: 2-D array
    
    """
    return onp.array( [[dm1_dMc(eta), dm1_deta(Mc, eta)],[dm2_dMc(eta), dm2_deta(Mc, eta)]] )

def J_Mceta_m1m2(m1, m2):
    """
    Compute the Jacobian matrix from :math:`m_1` and :math:`m_2` to :math:`{\cal M}_c` and :math:`\eta`.
    
    :param array or float m1: Mass of the first object, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
    :param array or float m2: Mass of the second object, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.
    
    :return: :math:`\partial ({\cal M}_c, \eta)/\partial (m_1, m_2)`.
    :rtype: 2-D array
    
    """
    return onp.array( [[dMc_dm1(m1, m2), dMc_dm2( m1, m2)],[deta_dm1(m1, m2), deta_dm2(m1, m2)]] )

def m1m2_from_Mceta(Mc, eta):
    """
    Compute the component masses of a binary given its chirp mass and symmetric mass ratio.
    
    :param array or float Mc: Chirp mass of the binary, :math:`{\cal M}_c`.
    :param array or float eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    :return: :math:`m_1` and :math:`m_2`.
    :rtype: tuple(array, array) or tuple(float, float)
    
    """
    delta = 1-4*eta
    return (1+onp.sqrt(delta))/2*Mc/eta**(3./5.), (1-onp.sqrt(delta))/2*Mc/eta**(3./5.)


def Mceta_from_m1m2(m1, m2):
    """
    Compute the chirp mass and symmetric mass ratio of a binary given its component masses.
    
    :param array or float m1: Mass of the first object, :math:`m_1`.
    :param array or float m2: Mass of the second object, :math:`m_2`.
    :return: :math:`{\cal M}_c` and :math:`\eta`.
    :rtype: tuple(array, array) or tuple(float, float)
    
    """
    Mc = (m1*m2)**3/5/(m1+m2)**1/5
    eta = (m1*m2)/(m1+m2)**2
    return Mc, eta

def m1m2_to_Mceta_fish(or_matrix, ParNums, evParams):
    """
    Change variables in the Fisher matrix from :math:`m_1` and :math:`m_2` to :math:`{\cal M}_c` and :math:`\eta`.
    
    :param array or_matrix: Array containing the Fisher matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :param dict(int) ParNums: Dictionary specifying the position of each parameter in the Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
    
    :return: Fisher matrix in :math:`{\cal M}_c` and :math:`\eta`, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :rtype: 2-D array
    
    """
    nparams=len(list(ParNums.keys()))
    
    rotMatrix = onp.identity(nparams)
    
    rotMatrix[onp.ix_([ParNums['Mc'],ParNums['eta']],[ParNums['Mc'],ParNums['eta']])] = J_m1m2_Mceta(evParams['Mc'], evParams['eta'])
    
    matrix = rotMatrix@or_matrix@rotMatrix
    
    return matrix


def m1m2_to_Mceta_cov(or_matrix, ParNums, evParams):
    """
    Change variables in the covariance matrix from :math:`m_1` and :math:`m_2` to :math:`{\cal M}_c` and :math:`\eta`.
    
    :param array or_matrix: Array containing the covariance matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :param dict(int) ParNums: Dictionary specifying the position of each parameter in the Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
    
    :return: Covariance matrix in :math:`{\cal M}_c` and :math:`\eta`, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :rtype: 2-D array
    
    """
    nparams=len(list(ParNums.keys()))
    
    rotMatrix = onp.identity(nparams)
    
    rotMatrix[onp.ix_([ParNums['Mc'],ParNums['eta']],[ParNums['Mc'],ParNums['eta']])] = J_Mceta_m1m2(*m1m2_from_Mceta(evParams['Mc'], evParams['eta']))
    
    matrix = rotMatrix@or_matrix@rotMatrix
    
    return matrix


def Mceta_to_m1m2_fish(or_matrix, ParNums, evParams):
    """
    Change variables in the Fisher matrix from :math:`{\cal M}_c` and :math:`\eta` to :math:`m_1` and :math:`m_2`.
    
    :param array or_matrix: Array containing the Fisher matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :param dict(int) ParNums: Dictionary specifying the position of each parameter in the Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
    
    :return: Fisher matrix in :math:`m_1` and :math:`m_2`, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :rtype: 2-D array
    
    """
    nparams=or_matrix.shape[0] #len(list(ParNums.keys()))
    
    rotMatrix = onp.identity(nparams)
    
    m1, m2 = m1m2_from_Mceta(evParams['Mc'], evParams['eta'])
    
    rotMatrix[onp.ix_([ParNums['Mc'],ParNums['eta']],[ParNums['Mc'],ParNums['eta']])] = J_Mceta_m1m2(m1, m2)
    
    matrix = rotMatrix.T@or_matrix@rotMatrix
    
    return matrix

def Mceta_to_m1m2_cov(or_matrix, ParNums, evParams):
    """
    Change variables in the covariance matrix from :math:`{\cal M}_c` and :math:`\eta` to :math:`m_1` and :math:`m_2`.
    
    :param array or_matrix: Array containing the covariance matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :param dict(int) ParNums: Dictionary specifying the position of each parameter in the Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
    
    :return: Covariance matrix in :math:`m_1` and :math:`m_2`, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :rtype: 2-D array
    
    """
    nparams=or_matrix.shape[0] #len(list(ParNums.keys()))
    
    rotMatrix = onp.identity(nparams)
    
    rotMatrix[onp.ix_([ParNums['Mc'],ParNums['eta']],[ParNums['Mc'],ParNums['eta']])] = J_m1m2_Mceta(evParams['Mc'], evParams['eta'])
    
    matrix = rotMatrix.T@or_matrix@rotMatrix
    
    return matrix

def dchi1_dchieff(m1, m2):
    """
    Compute the derivative of :math:`\chi_{1,z}` with respect to :math:`\chi_{\\rm eff}`.
    
    :param array or float m1: Mass of the first object, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
    :param array or float m2: Mass of the second object, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.
    
    :return: :math:`\partial \chi_{1,z}/\partial \chi_{\\rm eff}`.
    :rtype: 1-D array
    
    """
    return 1.

def dchi2_dchieff(m1, m2):
    """
    Compute the derivative of :math:`\chi_{2,z}` with respect to :math:`\chi_{\\rm eff}`.
    
    :param array or float m1: Mass of the first object, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
    :param array or float m2: Mass of the second object, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.
    
    :return: :math:`\partial \chi_{2,z}/\partial \chi_{\\rm eff}`.
    :rtype: 1-D array
    
    """
    return 1.

def dchi1_dDelchi(m1, m2):
    """
    Compute the derivative of :math:`\chi_{1,z}` with respect to :math:`\Delta\chi`.
    
    :param array or float m1: Mass of the first object, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
    :param array or float m2: Mass of the second object, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.
    
    :return: :math:`\partial \chi_{1,z}/\partial \Delta\chi`.
    :rtype: 1-D array
    
    """
    return m2/(m1+m2)

def dchi2_dDelchi(m1, m2):
    """
    Compute the derivative of :math:`\chi_{2,z}` with respect to :math:`\Delta\chi`.
    
    :param array or float m1: Mass of the first object, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
    :param array or float m2: Mass of the second object, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.
    
    :return: :math:`\partial \chi_{2,z}/\partial \Delta\chi`.
    :rtype: 1-D array
    
    """
    return -m1/(m1+m2)

def J_chi1chi2_chieffDeltachi(m1, m2):
    """
    Compute the Jacobian matrix from :math:`\chi_{\\rm eff}` and :math:`\Delta\chi`` to :math:`\chi_{1,z}` and :math:`\chi_{2,z}`.
    
    :param array or float m1: Mass of the first object, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
    :param array or float m2: Mass of the second object, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.
    
    :return: :math:`\partial (\chi_{1,z}, \chi_{2,z})/\partial (\chi_{\\rm eff}, \Delta\chi)`.
    :rtype: 2-D array
    
    """
    return onp.array( [[1., dchi1_dDelchi( m1, m2)],[1., dchi2_dDelchi(m1, m2)]] )

def chi1chi2_to_chieffDeltachi_fish(or_matrix, ParNums, evParams):
    """
    Change variables in the Fisher matrix from :math:`\chi_{1,z}` and :math:`\chi_{2,z}` to :math:`\chi_{\\rm eff}` and :math:`\Delta\chi`.
    
    :param array or_matrix: Array containing the Fisher matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :param dict(int) ParNums: Dictionary specifying the position of each parameter in the Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
    
    :return: Fisher matrix in :math:`\chi_{\\rm eff}` and :math:`\Delta\chi`, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :rtype: 2-D array
    
    """
    nparams=len(list(ParNums.keys()))
    
    rotMatrix = onp.identity(nparams)
    
    rotMatrix[onp.ix_([ParNums['chi1z'],ParNums['chi2z']],[ParNums['chi1z'],ParNums['chi2z']])] = J_chi1chi2_chieffDeltachi(*m1m2_from_Mceta(evParams['Mc'], evParams['eta']))
    
    matrix = rotMatrix.T@or_matrix@rotMatrix
    
    return matrix


def chiSchiA_to_chi1chi2_fish(or_matrix, ParNums, evParams):
    """
    Change variables in the Fisher matrix from :math:`\chi_s` and :math:`\chi_a` to :math:`\chi_{1,z}` and :math:`\chi_{2,z}`.
    
    :param array or_matrix: Array containing the Fisher matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :param dict(int) ParNums: Dictionary specifying the position of each parameter in the Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
    
    :return: Fisher matrix in :math:`\chi_{1,z}` and :math:`\chi_{2,z}`, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :rtype: 2-D array
    
    """
    nparams=len(list(ParNums.keys()))
    
    rotMatrix = onp.identity(nparams)
    
    J_chiSchiA_chi1chi2 = onp.array([[.5, .5], [.5, -0.5]])
    
    rotMatrix[onp.ix_([ParNums['chiS'],ParNums['chiA']],[ParNums['chiS'],ParNums['chiA']])] = J_chiSchiA_chi1chi2
    
    matrix = rotMatrix@or_matrix@rotMatrix
    
    return matrix


##############################################################################
# LOCALIZATION REGION
##############################################################################


def compute_localization_region(Cov, parNum, thFid, perc_level=90, units='SqDeg'):
    """
    Compute the localisation region of one or multiple events.
    
    :param array Cov: Array containing the covariance matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :param dict(int) parNum: Dictionary specifying the position of each parameter in the Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param array thFid: Array containing the :math:`\\theta` sky position angle(s) of the event(s), in :math:`\\rm rad`.
    :param float perc_level: The percent level at which to compute the localisation region, from 0 to 100.
    :param str units: The units to use for the output, to choose among square degrees, ``'SqDeg'``, or steradians, ``'Sterad'``.
    :return: Localisation region(s) of the event(s).
    :rtype: 1-D array
    
    """
    #Cov_th_ph = Cov[ [parNum['theta'], parNum['phi']] ][:, [parNum['theta'], parNum['phi']] ]
    
    DelThSq  = Cov[parNum['theta'], parNum['theta']]
    DelPhiSq  = Cov[parNum['phi'], parNum['phi']]
    DelThDelPhi  = Cov[parNum['phi'], parNum['theta']]
    
    # From Barak, Cutler, PRD 69, 082005 (2004), gr-qc/0310125
    DelOmegaSr_base = 2*onp.pi*onp.sqrt(DelThSq*DelPhiSq-DelThDelPhi**2)*onp.abs(onp.sin(thFid))
    

    DelOmegaSr =  - DelOmegaSr_base*onp.log(1-perc_level/100)
    
    if units=='Sterad':
        return DelOmegaSr
    elif units=='SqDeg':
        return  (180/onp.pi)**2*DelOmegaSr
    
    


##############################################################################
# PLOTTING TOOLS (ELLIPSES)
##############################################################################



def plot_contours(Covariance, plot_vars, plot_idxs, event, my_scales, plt_labels):
    
    # Example scales: scales = {'dL': lambda x: np.round(x*1000, 0), 'theta': theta_to_dec_degminsec ,'phi':phi_to_ra_hrms, 'iota':np.cos}
    # Example plt_labels:  plt_labels = [('RA', 'dec'), (r'$d_L[Mpc]$', r'$cos(\iota)$') ]

    
    
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(1, len(plot_vars), figsize=(15, 5))
    for ax, plotvar, plot_idx in zip(axs, plot_vars, plot_idxs):
    
        print(plotvar)
        print(plot_idx)
   # print(plotvar[1])
    
        confidence_ellipse(Covariance[onp.ix_(plot_idx, plot_idx)], ax, 
                       event[plotvar[0]], event[plotvar[1]], 
                       edgecolor='red',
                      n_std=2.0)
    
        ax.scatter(event[plotvar[0]], event[plotvar[1]], c='red', s=3)
    #ax.set_title(title)
        ax.set_xlabel(plotvar[0], fontsize=15)
        ax.set_ylabel(plotvar[1], fontsize=15)
    
        if plotvar[1]=='theta':
            ax.set_ylim(ax.get_ylim()[::-1])
        if plotvar[0]=='phi':
            ax.set_xlim(ax.get_xlim()[::-1])
          

    plt.show()

    for ax, plotvar, plot_idx, plot_label in zip(axs, plot_vars, plot_idxs, plt_labels):
        if plotvar[0] in my_scales.keys():
        # transform scale on x axis
            old_labels= onp.array([ax.get_xticklabels()[k].get_position()[0] for k in  range(len(ax.get_xticklabels())) ])
            print(old_labels)
        
            labels = my_scales[plotvar[0]]( onp.array([ax.get_xticklabels()[k].get_position()[0] for k in  range(len(ax.get_xticklabels())) ])  )
            print(labels)
        
            ax.set_xticklabels(labels)
        
        if plotvar[1] in my_scales.keys():
        # transform scale on y axis
            labels= onp.array([ax.get_yticklabels()[k].get_position()[1] for k in  range(len(ax.get_yticklabels())) ])
            print(labels)
        
            new_labels = my_scales[plotvar[1]]( onp.array([ax.get_yticklabels()[k].get_position()[1] for k in  range(len(ax.get_yticklabels())) ])  )
            print(new_labels)
        
            ax.set_yticklabels(new_labels)
    
    
            ax.set_xlabel(plot_label[0], fontsize=15)
            ax.set_ylabel(plot_label[1], fontsize=15)

#fig = plt.gcf()

    return fig



# From https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(cov, ax, mean_x, mean_y, n_std=3.0, facecolor='none', **kwargs):
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    #if x.size != y.size:
    #    raise ValueError("x and y must be the same size")

    #cov = np.cov(x, y)
    pearson = cov[0, 1]/onp.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = onp.sqrt(1 + pearson)
    ell_radius_y = onp.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = onp.sqrt(cov[0, 0]) * n_std
    #mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = onp.sqrt(cov[1, 1]) * n_std
    #mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
