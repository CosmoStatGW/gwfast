#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
import jax
jax.devices('cpu')
from jax.config import config
config.update("jax_enable_x64", True)

# We use both the original numpy, denoted as onp, and the JAX implementation of numpy, denoted as np
import numpy as onp
#import numpy as np
#import jax.numpy as np
import copy
import GWfish.fisherUtils as utils
import mpmath
import scipy

##############################################################################
# INVERSION AND SANITY CHECKS
##############################################################################
def CovMatr(FisherMatrix, evParams, 
            invMethodIn='cho', 
            condNumbMax=1e50, 
            evals_trunc=1e-15, 
            truncate=False, verbose=False):
    
        utils.check_evparams(evParams)
    
        FisherMatrix = FisherMatrix.astype('float128')

        ws_dL= onp.array([onp.identity(FisherMatrix.shape[0]) for _ in range(FisherMatrix.shape[-1])])
        ws_dL[:, 2,2] = evParams['dL']
        FisherM = (ws_dL@((FisherMatrix.T@ws_dL).T).T).T


        ws_mc= onp.array([onp.identity(FisherMatrix.shape[0]) for _ in range(FisherMatrix.shape[-1])])
        ws_mc[:, 0,0] = evParams['Mc']
        FisherM = (ws_mc@((FisherM.T@ws_mc).T).T).T

        ws_i= onp.array([onp.identity(FisherMatrix.shape[0]) for _ in range(FisherMatrix.shape[-1])])
        ws_i[:, 5,5] = 1/onp.cos(evParams['iota'])
        FisherM = (ws_i@((FisherM.T@ws_i).T).T).T

        # sin(delta)
        ws_del= onp.array([onp.identity(FisherMatrix.shape[0]) for _ in range(FisherMatrix.shape[-1])])
        ws_del[:, 3,3] = -1/onp.cos(evParams['theta'])
        FisherM = (ws_del@((FisherM.T@ws_del).T).T).T
        
        CovMatr = onp.zeros(FisherMatrix.shape).astype('float128')
        cho_failed = 0
        for k in range(FisherM.shape[-1]): 
            
            E, _ = scipy.linalg.eigh(FisherMatrix[:, :, k])
            if onp.any(E<0) and verbose:
                print('Matrix is not positive definite!')

            cond_or = onp.max(E)/onp.min(E)
            if verbose:
                print('Condition of the original matrix: %s' %cond_or)
        

            ff = mpmath.matrix( FisherM[:, :, k].astype('float128'))
            ws =  mpmath.diag([ 1/mpmath.sqrt(ff[i, i]) for i in range(FisherM.shape[-2]) ])
            FisherM_ = ws*ff*ws

            E, ER = mpmath.eigh(ff)
            E = onp.array(E.tolist(), dtype=onp.float128)

            if onp.any(E<0) and verbose:
                print('Matrix is not positive definite!')

            cond = onp.max(E)/onp.min(E)
            if verbose:
                print('Condition of matrix in logdL, logMc, cosiota: %s' %cond)

            EE, ER = mpmath.eigh(FisherM_)
            E = onp.array(EE.tolist(), dtype=onp.float128)
            cond = onp.max(E)/onp.min(E)
            if verbose:
                print('Condition of the new matrix: %s' %cond)
            
            invMethod = invMethodIn
            if onp.any(E<0):
                if verbose:
                    print('Matrix is not positive definite!')
                if invMethodIn=='cho':
                    cho_failed+=1
                    invMethod='svd'
                    if verbose:
                        print('Cholesky decomposition not usable. Using method %s' %invMethod)
            elif invMethod=='cho':
                try:
                    c = (mpmath.cholesky(FisherM_))**-1 
                except Exception as e:
                    print(e)
                    invMethod='svd'
                    print('Cholesky decomposition not usable. Eigenvalues seem ok but cholesky decomposition failed. Using method %s' %invMethod)
                    cho_failed+=1

            if invMethod=='inv':
                    cc = FisherM_**-1
                    cc = (cc+cc.T)/2
            elif invMethod=='cho':
                    #c = cF**-1 
                    cc = c.T*c
            elif invMethod=='svd':
                    U, S, V = mpmath.svd_r(FisherM_)
                    if ((truncate) and (onp.abs(cond_or)>condNumbMax)):
                        if verbose:
                            print('Truncating singular values below %s' %evals_trunc)
                        S = onp.array(S.tolist(), dtype=onp.float128)
                        maxev = onp.max(S)
                        Sinv = mpmath.matrix(onp.array([1/s if onp.abs(s/maxev)>evals_trunc else 0 for s in S ]).astype('float128'))
                        SinvVec = onp.array(Sinv.tolist())
                        if verbose:
                            print('%s singular values truncated' %SinvVec[SinvVec==0].shape)
                    else:
                        Sinv = mpmath.matrix(onp.array([1/s for s in S ]).astype('float128'))
                    cc=V.T*mpmath.diag([s for s in Sinv])*U.T
            elif invMethod=='lu':
                    P, L, U = mpmath.lu(FisherM_)
                    ll = P*L
                    llinv = ll**-1
                    uinv=U**-1
                    cc = uinv*llinv
            elif invMethod=='evals':
                maxe = onp.max(E)
                Lamb = mpmath.matrix(onp.array([e if onp.abs(e/maxe)>evals_trunc else evals_trunc for e in E ]).astype('float128'))
                Lamb = onp.array(Lamb.tolist())
                LambInv = mpmath.matrix(onp.array([1/e if onp.abs(e/maxe)>evals_trunc else 1/evals_trunc for e in E ]).astype('float128'))
                if verbose:
                    print('%s eigenvalues truncated' %Lamb[Lamb==evals_trunc].shape)
                LambInv = mpmath.diag(LambInv)
                cc = ER*LambInv*ER.T
                cc = (cc+cc.T)/2

            CovMatr_ = ws*cc*ws

            CovMatr[:, :, k] =  onp.array(CovMatr_.tolist(), dtype=onp.float128)
            if verbose:
                print()
        
        CovMatr = (ws_i@((CovMatr.T@ws_i).T).T).T
        CovMatr = (ws_mc@((CovMatr.T@ws_mc).T).T).T
        CovMatr = (ws_dL@((CovMatr.T@ws_dL).T).T).T
        CovMatr = (ws_del@((CovMatr.T@ws_del).T).T).T

        
        eps = onp.array([ onp.max( onp.abs(CovMatr[:, :, i]@FisherMatrix[:, :, i]-onp.eye(FisherMatrix.shape[0]))) for i in range(FisherMatrix.shape[-1]) ])


        if verbose:
                print('Error with %s: %s\n' %(invMethod, eps))
        print(' Inversion error with method %s: min=%s, max=%s, mean=%s, std=%s ' %(invMethodIn, onp.min(eps), onp.max(eps), onp.mean(eps), onp.std(eps)) )
        print('Method %s not possible on %s non-positive definite matrices, %s was used in those cases. ' %(invMethodIn, cho_failed, 'svd'))
        return CovMatr, eps




    
def compute_inversion_error(Fisher, Cov):
    diff = onp.array([Cov[:, :, i]@Fisher[:, :, i]-onp.identity(Cov.shape[0]) for i in range(Fisher.shape[-1])])
    #eps = [ onp.linalg.norm(mm, ord=onp.inf) for mm in diff]
    maxnorm = [ onp.max(mm) for mm in diff]
    #eps = [ onp.linalg.norm(Cov[:, :, i]@Fisher[:, :, i]-onp.identity(Cov.shape[0]), ord=onp.inf) for i in range(Fisher.shape[-1])]
    #print('Inversion error (inf norm): %s ' %str(eps))
    print('Inversion error (max): %s ' %str(maxnorm))
    return maxnorm#, maxnorm



def inverse_vect(A, invMethod, wp=None):
        # Compute the inverse of matrices in an array of shape (N,N,M)
            if invMethod=='inv':
                return onp.linalg.inv(A.transpose(2,0,1)).transpose(1,2,0)
            elif invMethod=='pinv':
                return onp.linalg.pinv(A.transpose(2,0,1)).transpose(1,2,0)
            elif invMethod=='svd':
                allInvs=onp.zeros(A.shape)
                for k in range(A.shape[-1]):
                    u,s,v=onp.linalg.svd(A[:, :, k])
                    Ainv=onp.dot(v.transpose(),onp.dot(onp.diag(s**-1),u.transpose()))
                    allInvs[:, :, k] = Ainv
                return allInvs
            elif invMethod=='solve':
                allInvs=onp.zeros(A.shape)
                for k in range(A.shape[-1]):
                    Ainv=onp.linalg.solve(A[:, :, k], onp.identity(A.shape[0]))
                    allInvs[:, :, k] = Ainv
                return allInvs  
            elif invMethod=='lu':
                allInvs=onp.zeros(A.shape)
                from scipy.linalg import lu
                for k in range(A.shape[-1]):
                    p,l,u = lu(A[:, :, k], permute_l = False)
                    l = onp.dot(p,l) 
                    l_inv = onp.linalg.inv(l)
                    u_inv = onp.linalg.inv(u)
                    Ainv = onp.dot(u_inv,l_inv)
                    allInvs[:, :, k] = Ainv
                return allInvs
            elif invMethod=='cho':
                allInvs=onp.zeros(A.shape)
                for k in range(A.shape[-1]):
                    c = onp.linalg.inv(onp.linalg.cholesky(A[:, :, k]))
                    Ainv = onp.dot(c.T,c)
                    allInvs[:, :, k] = Ainv
                return allInvs
            elif invMethod=='mpmath_cho':
                allInvs=onp.zeros(A.shape)
                for k in range(A.shape[-1]):
                    c = (mpmath.cholesky(mpmath.matrix(A[:, :, k].astype('float128'))))**-1 #np.linalg.inv(np.linalg.cholesky(A[:, :, k]))
                    Ainv = c.T*c
                    allInvs[:, :, k] = onp.array(Ainv.tolist(), dtype=onp.float128)
                return allInvs
            elif invMethod=='mpmath':
                allInvs=onp.zeros(A.shape)
                for k in range(A.shape[-1]):
                    #c = (mpmath.cholesky(A[:, :, k]))**-1 #np.linalg.inv(np.linalg.cholesky(A[:, :, k]))
                    Ainv = mpmath.matrix(A[:, :, k].astype('float128'))**-1
                    allInvs[:, :, k] = onp.array(Ainv.tolist(), dtype=onp.float128)
                return allInvs
            elif invMethod=='evals':
                allInvs=onp.zeros(A.shape)
                for k in range(A.shape[-1]):
                    fm = mpmath.matrix(A[:,:,k].astype('float128'))
                    E, ER = mpmath.eig(fm)
                    Eem = mpmath.diag([1/e for e in E])
                    Ainv = ER*(Eem)*ER.T
                    allInvs[:, :, k] = onp.array(Ainv.tolist(), dtype=onp.float128)
                return allInvs
            elif invMethod=='evals_conditioned':
                allInvs=onp.zeros(A.shape)
                for k in range(A.shape[-1]):
                    fm = mpmath.matrix(A[:,:,k].astype('float128'))
                    E, ER = mpmath.eig(fm)
                    #Eem = mpmath.diag([1/e for e in E])
                    Eem = mpmath.diag([1/e if (onp.abs(e)>wp ) else mpmath.ctx_mp_python.mpf(1/onp.abs(wp)) for e in E])
                    Ainv = ER*(Eem)*ER.T
                    allInvs[:, :, k] = onp.array(Ainv.tolist(), dtype=onp.float128)
                return allInvs
            
            elif invMethod=='mpmath_svd':
                allInvs=onp.zeros(A.shape)
                for k in range(A.shape[-1]):
                    fm = mpmath.matrix(A[:,:,k].astype('float128'))
                    U, S, V = mpmath.svd(fm)
                    Ainv=V.T*mpmath.diag([1/s for s in S])*U.T
                    allInvs[:, :, k] = onp.array(Ainv.tolist(), dtype=onp.float128)
                return allInvs
            elif invMethod=='mpmath_lu':
                allInvs=onp.zeros(A.shape)
                for k in range(A.shape[-1]):
                    fm = mpmath.matrix(A[:,:,k].astype('float128'))
                    P, L, U = mpmath.lu(fm)
                    ll = P*L
                    llinv = ll**-1
                    uinv=U**-1
                    Ainv = uinv*llinv
                    allInvs[:, :, k] = onp.array(Ainv.tolist(), dtype=onp.float128)
                return allInvs
                
            
            
def CheckFisher(FisherM, condNumbMax=1.0e15, use_mpmath=True, verbose=False):
        # Perform some sanity checks on the Fisher matrix, in particular:
        # - compute the eigenvalues and eigenvectors
        # - compute the condition number (ratio of the largest to smallest eigenvalue) and check this is not large
        
        # Being the Fisher symmetric by definition, we can use the numpy.linalg function 'eigh', to speed up a bit
        # The input has size (Npar,Npar,Nev), so we have to swap
        if not use_mpmath:
            evals, evecs = scipy.linalg.eigh(FisherM.transpose(2,0,1))
        else:
            evals = onp.zeros(FisherM.shape[1:][::-1])
            evecs = onp.zeros(FisherM.shape[::-1])
            for k in range(FisherM.shape[-1]):
                try:
                    aam = mpmath.matrix(FisherM[:,:,k].astype('float128'))
                    E, ER = mpmath.eigh(aam)
                    evals[k, :] = onp.array(E, dtype=onp.float128)
                    evecs[k, :, :] = onp.array(ER.tolist(), dtype=onp.float128)
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
            print('WARNING: one or more eigenvalues are negative.')
        
        #print('Evals: %s ' %str(evals))
        
        #print('Max eval: %s ' %str(onp.abs(evals).max(axis=1)))
        #print('Min  eval: %s ' %str(onp.abs(evals).min(axis=1)))
        condNumber = onp.abs(evals).max(axis=1)/onp.abs(evals).min(axis=1)
        
        if onp.any(condNumber>condNumbMax) and verbose:
            print('WARNING: the condition number is too large (%s>%s)'%(condNumber,condNumbMax) )
            print('Unreliable covariance at positions ' +str(condNumber>condNumbMax))
        elif verbose:
            print('Condition number= %s . Ok. '%condNumber)
        
        return evals, evecs, condNumber


def perturb_Fisher(totF, myNet, events, eps=1e-10, **kwargs):
    
    #_, parnums_ = myNet.signals[list(myNet.signals.keys())[0]].CovMatr(events)
    Cov_base,  _, _ = myNet.CovMatr(events, FisherM=totF, get_individual=False, **kwargs)
    #print('Forecasted dL error, true Fisher:')
    #print(onp.sqrt(Cov_base[1, 1])*1000)
    
    #DelOmegaDegSq_base = compute_localization_region(Cov_base, parnums_, units='SqDeg')
    #print('Forecasted Localization error, true Fisher:')
    #print(DelOmegaDegSq_base)

    totF_random = totF + onp.random.rand(*totF.shape)*eps
    Cov,  _, _ = myNet.CovMatr(events, FisherM=totF_random, get_individual=False, **kwargs)
    
    #print('Forecasted dL error:')
    #print(onp.sqrt(Cov[1, 1])*1000)
    
    
    #DelOmegaDegSq = compute_localization_region(Cov, parnums_, units='SqDeg')
    #print('Forecasted Localization error:')
    #print(DelOmegaDegSq)
    
    epsErr = [onp.linalg.norm( Cov_base[i]/Cov[i]-1, ord=onp.inf) for i in range(Cov.shape[-1])]
    print('Relative errors: %s' %epsErr)


def check_covariance(FisherM, Cov, tol=1e-10):
    
    recovered_Ids = [ Cov[:, :, i]@FisherM[:, :, i] for i in range(Cov.shape[-1])]
    
    #
    epsErr = [onp.linalg.norm( recovered_Ids[i]-onp.identity(Cov.shape[0]), ord=onp.inf) for i in range(Cov.shape[-1])]
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
    import copy
    ParNums = copy.deepcopy(ParNums_inp)
    
    IdxMarg = onp.sort(onp.array([ParNums[par] for par in ParMarg]))
    newdim = MatrIn.shape[0]-len(IdxMarg)
    
    NewMatr = onp.zeros( (newdim, newdim, MatrIn.shape[-1]) )
    
    for k in range(MatrIn.shape[-1]):
    
        Matr = onp.delete(MatrIn[:, :, k], IdxMarg, 0)
        Matr = onp.delete(Matr[:, :], IdxMarg, 1)
        NewMatr[:, :, k] = Matr
        
    # Given that we deleted some rows and columns, 
    # the meaning of the numbers of the remaining ones changes
        
    OrKeys = [key for key in ParNums.keys()]
    for i,key in enumerate(OrKeys):
                if key in ParMarg:
                    for tmp in OrKeys[i::1]:
                        ParNums[tmp] -= 1
                        ParNums.pop(key, None)
    
    return NewMatr, ParNums


def addPrior(Matr, vals, ParNums, ParAdd):
    
    IdxAdd = onp.sort(onp.array([ParNums[par] for par in ParAdd]))
    
    pp = onp.zeros(Matr.shape)
    
    diag = onp.zeros(Matr.shape[0])
    diag[IdxAdd] = vals
    #diag = diag.at[IdxAdd].set(vals)
    
    onp.fill_diagonal(pp, diag)
    
    return pp+Matr


##############################################################################
# DERIVATIVES AND JACOBIANS
##############################################################################

def log_dL_to_dL_derivative_cov(or_matrix, ParNums, evParams):
    
    matrix = copy.deepcopy(or_matrix)
    #for i in range(matrix.shape[-1]):
        # This has to be vectorised
    try:
            matrix = matrix.at[:, ParNums['dL'], : ].set(matrix[:, ParNums['dL'], :]* evParams['dL'])
            matrix = matrix.at[ ParNums['dL'], : , : ].set(matrix[ParNums['dL'], :, :]* evParams['dL'])
    except AttributeError:
            matrix = matrix.astype('float128')
            matrix[:, ParNums['dL'], :] *= evParams['dL'].astype('float128')
            matrix[ ParNums['dL'], :, :] *= evParams['dL'].astype('float128')
    return matrix


def log_dL_to_dL_derivative_fish(or_matrix, ParNums, evParams):
    matrix = copy.deepcopy(or_matrix)
    #for i in range(matrix.shape[-1]):
        # This has to be vectorised
    try:
            matrix = matrix.at[:, ParNums['dL'], : ].set(matrix[:, ParNums['dL'], :]/ evParams['dL'])
            matrix = matrix.at[ ParNums['dL'], : , : ].set(matrix[ ParNums['dL'], :, :]/ evParams['dL'])
    except AttributeError:
            matrix = matrix.astype('float128')
            matrix[:, ParNums['dL'], :] /= evParams['dL'].astype('float128')
            matrix[ ParNums['dL'], :, :] /= evParams['dL'].astype('float128')
    return matrix
    
    

def dm1_dMc(eta):
    return (1+onp.sqrt(1-4*eta) )*eta**(-3./5.)/2

def dm2_dMc(eta):
    return (1-onp.sqrt(1-4*eta) )*eta**(-3./5.)/2

def dm1_deta(Mc, eta):
    return -Mc*(3-2*eta+3*onp.sqrt(1-4*eta))/(10*onp.sqrt(1-4*eta)*eta**(8./5.))
#(1-onp.sqrt(1-4*eta) )*eta**(-3./5.)/2

def dm2_deta(Mc, eta):
    return -Mc*(-3+2*eta+3*onp.sqrt(1-4*eta))/(10*onp.sqrt(1-4*eta)*eta**(8./5.))


def dMc_dm1(m1, m2): 
    return m2*(2*m1+3*m2)/(5*(m1*m2)**(2/5)*(m1+m2)**(6/5))

def dMc_dm2(m1, m2): 
    return dMc_dm1(m2, m1)

def deta_dm1(m1, m2):
    return m2*(m2-m1)/(m1+m2)**3

def deta_dm2(m1, m2):
    return deta_dm1(m2, m1)


def J_m1m2_Mceta(Mc, eta):
    return onp.array( [[dm1_dMc(eta), dm1_deta(Mc, eta)],[dm2_dMc(eta), dm2_deta(Mc, eta)]] )

def J_Mceta_m1m2(m1, m2):
    return onp.array( [[dMc_dm1(m1, m2), dMc_dm2( m1, m2)],[deta_dm1(m1, m2), deta_dm2(m1, m2)]] )

def m1m2_from_Mceta(Mc, eta):
    delta = 1-4*eta
    return (1+onp.sqrt(delta))/2*Mc*eta**(3./5.), (1-onp.sqrt(delta))/2*Mc*eta**(3./5.)


def m1m2_to_Mceta_fish(or_matrix, ParNums, evParams):
    
    nparams=len(list(ParNums.keys()))
    
    rotMatrix = onp.identity(nparams)
    
    rotMatrix[onp.ix_([ParNums['Mc'],ParNums['eta']],[ParNums['Mc'],ParNums['eta']])] = J_m1m2_Mceta(evParams['Mc'], evParams['eta'])
    
    matrix = rotMatrix@or_matrix@rotMatrix
    
    return matrix


def m1m2_to_Mceta_cov(or_matrix, ParNums, evParams):
    
    nparams=len(list(ParNums.keys()))
    
    rotMatrix = onp.identity(nparams)
    
    rotMatrix[onp.ix_([ParNums['Mc'],ParNums['eta']],[ParNums['Mc'],ParNums['eta']])] = J_Mceta_m1m2(*m1m2_from_Mceta(evParams['Mc'], evParams['eta']))
    
    matrix = rotMatrix@or_matrix@rotMatrix
    
    return matrix


def Mceta_to_m1m2_fish(or_matrix, ParNums, evParams):
    
    nparams=len(list(ParNums.keys()))
    
    rotMatrix = onp.identity(nparams)
    
    m1, m2 = m1m2_from_Mceta(evParams['Mc'], evParams['eta'])
    
    rotMatrix[onp.ix_([ParNums['Mc'],ParNums['eta']],[ParNums['Mc'],ParNums['eta']])] = J_Mceta_m1m2(m1, m2)
    
    matrix = rotMatrix@or_matrix@rotMatrix
    
    return matrix


def dchi1_dchieff(m1, m2):
    return 1.

def dchi2_dchieff(m1, m2):
    return 1.

def dchi1_dDelchi(m1, m2):
    return m2/(m1+m2)

def dchi2_dDelchi(m1, m2):
    return -m1/(m1+m2)

def J_chi1chi2_chieffDeltachi(m1, m2):
    return onp.array( [[1., dchi1_dDelchi( m1, m2)],[1., dchi2_dDelchi(m1, m2)]] )

def chi1chi2_to_chieffDeltachi_fish(or_matrix, ParNums, evParams):
    
    nparams=len(list(ParNums.keys()))
    
    rotMatrix = onp.identity(nparams)
    
    rotMatrix[onp.ix_([ParNums['chi1z'],ParNums['chi2z']],[ParNums['chi1z'],ParNums['chi2z']])] = J_chi1chi2_chieffDeltachi(*m1m2_from_Mceta(evParams['Mc'], evParams['eta']))
    
    matrix = rotMatrix@or_matrix@rotMatrix
    
    return matrix


def chiSchiA_to_chi1chi2_fish(or_matrix, ParNums, evParams):
    
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

    #Cov_th_ph = Cov[ [parNum['theta'], parNum['phi']] ][:, [parNum['theta'], parNum['phi']] ]
    
    DelThSq  = Cov[parNum['theta'], parNum['theta']]
    DelPhiSq  = Cov[parNum['phi'], parNum['phi']]
    DelThDelPhi  = Cov[parNum['phi'], parNum['theta']]
    
    # From Barak, Cutler, PRD 69, 082005 (2004), gr-qc/0310125
    DelOmegaSr_base = 2*onp.pi*onp.sqrt(DelThSq*DelPhiSq-DelThDelPhi**2)*onp.abs(onp.sin(thFid))
    #DelOmegaSr_base = 2*onp.pi*onp.sqrt(np.linalg.det(Cov_th_ph))*onp.abs(onp.sin(thFid))

    DelOmegaSr =  - DelOmegaSr_base*onp.log(1-perc_level/100)
    
    if units=='Sterad':
        return DelOmegaSr
    elif units=='SqDeg':
        return  (180/onp.pi)**2*DelOmegaSr
    
    
def compute_localization_region_1(Cov, parNum, thFid, perc_level=90, units='SqDeg'):

    
    Cov_th_ph = Cov[ [parNum['theta'], parNum['phi']] ][:, [parNum['theta'], parNum['phi']] ]
    #evals, _ = onp.linalg.eigh(Cov_th_ph)
    
    cosDelta = onp.cos(0.5*onp.pi - thFid)
    
    # From Barak, Cutler, PRD 69, 082005 (2004), gr-qc/0310125
    DelOmegaSr_base = onp.pi*onp.abs(cosDelta)*onp.sqrt(onp.linalg.det(Cov_th_ph)) #onp.product(evals))
    DelOmegaSr =  - 2*DelOmegaSr_base*onp.log(1-perc_level/100)
    
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
