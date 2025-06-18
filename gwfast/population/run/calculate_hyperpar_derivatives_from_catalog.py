#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as onp
import os,sys,time,multiprocessing,subprocess,copy,argparse,configparser,h5py
os.environ["OMP_NUM_THREADS"] = "1"
# Needed for mpipool not to stall when trying to write on a file (do not ask me why)
multiprocessing.set_start_method("spawn",force=True)

from astropy.cosmology import Planck18, z_at_value
from astropy import units as uAstro


SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd())))
#sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR,'..')))
sys.path.append(SCRIPT_DIR)
import POPutils as utils
from POPmodels import MassSpinRedshiftIndependent_PopulationModel,MassOnly_PopulationModel,MassRedshiftIndependent_PopulationModel
from popdistributions.massdistribution import TruncatedPowerLaw_modsmooth_MassDistribution, PowerLawPlusPeak_modsmooth_MassDistribution
from popdistributions.ratedistribution import PowerLaw_RateDistribution, MadauDickinson_RateDistribution, MadauDickinsonPLTimeDelta_RateDistribution
from popdistributions.spindistribution import DefaultPrecessing_SpinDistribution, SameFlatNonPrecessing_SpinDistribution, FlatNonPrecessing_SpinDistribution, GaussNonPrecessing_SpinDistribution

#####################################################################################
# GLOBALS
#####################################################################################

clight = 2.99792458e5 # km/s

#RESPATH = os.path.join(SCRIPT_DIR, PACKAGE_PARENT, 'results' )
PNUMS_FIM_PREC_ROT    = {'m1_src':0, 'm2_src':1, 'z':2, 'theta':3, 'phi':4, 'thetaJN':5, 'psi':6, 'tcoal':7, 'Phicoal':8, 'chi1':9,  'chi2':10, 'tilt1':11, 'tilt2':12, 'phiJL':13, 'phi12':14}
PNUMS_FIM_ALIGNED_ROT = {'m1_src':0, 'm2_src':1, 'z':2, 'theta':3, 'phi':4, 'iota':5, 'psi':6, 'tcoal':7, 'Phicoal':8, 'chi1z':9,  'chi2z':10}

PNUMS_FIM_PREC_OR    = {'Mc':0, 'eta':1, 'dL':2, 'theta':3, 'phi':4, 'thetaJN':5, 'psi':6, 'tcoal':7, 'Phicoal':8, 'chi1':9,  'chi2':10, 'tilt1':11, 'tilt2':12, 'phiJL':13, 'phi12':14}
PNUMS_FIM_ALIGNED_OR = {'Mc':0, 'eta':1, 'dL':2, 'theta':3, 'phi':4, 'iota':5, 'psi':6, 'tcoal':7, 'Phicoal':8, 'chi1z':9,  'chi2z':10}

# shortcuts for distribution functions names; have to be used in input
mass_models_dict = {
                    'TruncatedPowerLaw': TruncatedPowerLaw_modsmooth_MassDistribution(),
                    'PowerLawPlusPeak': PowerLawPlusPeak_modsmooth_MassDistribution(),
                    }

rate_models_dict = {'PowerLaw': PowerLaw_RateDistribution(),
                    'MadauDickinson': MadauDickinson_RateDistribution(),
                    'MadauDickinsonPLTimeDelta': MadauDickinsonPLTimeDelta_RateDistribution(),
                    }
                    
spin_models_dict = {'Default': DefaultPrecessing_SpinDistribution(),
                    'SameFlatNonPrecessing': SameFlatNonPrecessing_SpinDistribution(),
                    'FlatNonPrecessing': FlatNonPrecessing_SpinDistribution(),
                    'GaussNonPrecessing': GaussNonPrecessing_SpinDistribution(),
                    }
POPmodel_dict = {'MassSpinRedshiftIndependent': MassSpinRedshiftIndependent_PopulationModel,
                'MassRedshiftIndependent': MassRedshiftIndependent_PopulationModel,
                'MassOnly': MassOnly_PopulationModel,
                    }

#####################################################################################
# input/output logic
#####################################################################################

# Writes output both on std output and on log file
class Logger(object):
    
    def __init__(self, fname):
        self.terminal = sys.__stdout__
        self.log = open(fname, "w+")
        self.log.write('--------- LOG FILE ---------\n')
        print('Logger created log file: %s' %fname)
        #self.write('Logger')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

    def close(self):
        self.log.close
        sys.stdout = sys.__stdout__
        
    def isatty(self):
        return False

def get_pool(mpi=False, threads=None):
    """ Always returns a pool object with a `map()` method. By default,
        returns a `SerialPool()` -- `SerialPool.map()` just calls the built-in
        Python function `map()`. If `mpi=True`, will attempt to import the 
        `MPIPool` implementation from `emcee`. If `threads` is set to a 
        number > 1, it will return a Python multiprocessing pool.
        Parameters
        ----------
        mpi : bool (optional)
            Use MPI or not. If specified, ignores the threads kwarg.
        threads : int (optional)
            If mpi is False and threads is specified, use a Python
            multiprocessing pool with the specified number of threads.
    """

    if mpi:
        from schwimmbad import MPIPool
        print('Using MPI...')
        # Initialize the MPI pool
        pool = MPIPool()

        # Make sure the thread we're running on is the master
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    elif threads > 1:
        print('Using multiprocessing with %s processes...' %(threads-1))
        pool = multiprocessing.Pool(threads-1)

    else:
        raise ValueError('Called get pool with threads=1. No need')
        #pool = SerialPool()

    return pool

def get_indexes(p, all_n_per_pool):
    
    if p==0:
        pin = 0
        pf = all_n_per_pool[p]
    else: 
        pin = sum(all_n_per_pool[:p])
        pf = pin+all_n_per_pool[p]
    return pin, pf

#####################################################################################
# load and save results
#####################################################################################

def to_file(termI_der, termI_hess, termII, termIII, termIV, termV, out_path, suff=''):
    
    print('Saving all to files. Names of termI matrix: %s' %('termI_matr'+suff+'.npy'))
    onp.save(os.path.join(out_path, 'termI_der'+suff+'.npy'), termI_der)
    onp.save(os.path.join(out_path, 'termI_hess'+suff+'.npy'), termI_hess)
    onp.save(os.path.join(out_path, 'termII_matr'+suff+'.npy'), termII)
    onp.save(os.path.join(out_path, 'termIII_matr'+suff+'.npy'), termIII)
    onp.save(os.path.join(out_path, 'termIV_matr'+suff+'.npy'), termIV)
    onp.save(os.path.join(out_path, 'termV_matr'+suff+'.npy'), termV)
    
    print('Saving successful.')
    
    
def from_file(out_path, suff=''):

    print('Loading all to files. Names of termI matrix: %s' %('termI_matr'+suff+'.npy'))
    termI_der  = onp.load(os.path.join(out_path, 'termI_der'+suff+'.npy'))
    termI_hess = onp.load(os.path.join(out_path, 'termI_hess'+suff+'.npy'))
    termII     = onp.load(os.path.join(out_path, 'termII_matr'+suff+'.npy'))
    termIII    = onp.load(os.path.join(out_path, 'termIII_matr'+suff+'.npy'))
    termIV     = onp.load(os.path.join(out_path, 'termIV_matr'+suff+'.npy'))
    termV      = onp.load(os.path.join(out_path, 'termV_matr'+suff+'.npy'))
    
    return termI_der, termI_hess, termII, termIII, termIV, termV
    
def delete_files(idxin, idxf, out_path):
    
    print('Removing unnecessary files from %s to %s... ' %(idxin, idxf))
    suff = '_'+str(idxin)+'_to_'+str(idxf)
    os.remove(os.path.join(out_path, 'termI_der'+suff+'.npy'))
    os.remove(os.path.join(out_path, 'termI_hess'+suff+'.npy'))
    os.remove(os.path.join(out_path, 'termII_matr'+suff+'.npy'))
    os.remove(os.path.join(out_path, 'termIII_matr'+suff+'.npy'))
    os.remove(os.path.join(out_path, 'termIV_matr'+suff+'.npy'))
    os.remove(os.path.join(out_path, 'termV_matr'+suff+'.npy'))


def save_data(fname, data, ):
    """
    Store a dictionary containing the events parameters in ``h5`` file.
    
    :param str fname: The name of the file to store the events in. This has to include the path and the ``h5`` or ``hdf5`` extension.
    :param dict(array, array, ...) data: The dictionary containing the parameters of the events, as in :py:data:`events`.
    
    """
    print('Saving to %s '%fname)
    with h5py.File(fname, 'w') as out:
            
        def cd(n, d):
            d = onp.array(d)
            out.create_dataset(n, data=d, compression='gzip', shuffle=True)
        
        for key in data.keys():
            cd(key, data[key])
            
def load_population(name, nEventsUse=None, keys_skip=[]):
    
    """
    Load a dictionary containing the events parameters in h5 file.
    
    :param str name: The name of the file to load the events from. This has to include the path and the ``h5`` or ``hdf5`` extension.
    :param int or None nEventsUse: Number of the events in the given file to load.
    :type kind: int or None
    :param list(str) keys_skip: Parameters present in the file to skip.
    
    :return: Dictionary containing the loaded events, as in :py:data:`events`.
    :rtype: dict(array, array, ...)
    
    """
    
    events={}
    with h5py.File(name, 'r') as f:
        for key in f.keys():
            if key not in keys_skip:
                events[key] = onp.array(f[key])
            else:
                print('Skipping %s' %key)
        if nEventsUse is not None:
            for key in f.keys():
                events[key]=events[key][:nEventsUse]
    
    return events

#####################################################################################
# FIMS AND DERIVATIVES HANDLING
#####################################################################################

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

def FISHER_McetadL_to_m1srcm2srcz(or_matrix, ParNums, evParams, cosmo=Planck18):
    """
    Change variables in the Fisher matrix from :math:`{\cal M}_c`, :math:`\eta` and :math:`d_L` to :math:`m_1^{\\rm src}`, :math:`m_2^{\\rm src}` and :math:`z`.
    
    :param array or_matrix: Array containing the Fisher matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :param dict(int) ParNums: Dictionary specifying the position of each parameter in the Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
    :param cosmo: Cosmology to use to compute the Hubble parameter and the redshift if not provided in the event parameters.
    
    :return: Fisher matrix in :math:`m_1^{\\rm src}`, :math:`m_2^{\\rm src}` and :math:`z`, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :rtype: 2-D array
    
    """
    nparams=or_matrix.shape[0]
    rotMatrix = onp.identity(nparams)

    # Try to read the redshift, otherwise compute it from the luminosity distance
    try:
        z = evParams['z']
    except KeyError:
        z = z_at_value(cosmo.luminosity_distance, evParams['dL']*uAstro.Gpc).value
    
    m1det, m2det = m1m2_from_Mceta(evParams['Mc'], evParams['eta'])
    m1, m2 = m1det/(1.+z), m2det/(1.+z)

    def J_McetadL_m1srcm2srcz(m1, m2, z, dL=None):
        """
        Jacobian of the transformation from :math:`{\cal M}_c`, :math:`\eta` and :math:`d_L` to :math:`m_1^{\\rm src}`, :math:`m_2^{\\rm src}` and :math:`z`.
        
        :param array or float m1: Source-frame mass of the primary object, :math:`m_1^{\\rm src}`.
        :param array or float m2: Source-frame mass of the secondary object, :math:`m_2^{\\rm src}`.
        :param array or float z: Redshift of the source, :math:`z`.
        :param array or float dL: Luminosity distance of the source, :math:`d_L`.
        :return: Jacobian matrix.
        :rtype: 2-D array
        
        """
        if dL is None:
            dL = cosmo.luminosity_distance(z).value/1000.
        
        dMc_dm1 = m2*(2.*m1+3.*m2)/(5.*(m1*m2)**(2./5.)*(m1+m2)**(6./5.))*(1.+z)
        dMc_dm2 = m1*(3.*m1+2.*m2)/(5.*(m1*m2)**(2./5.)*(m1+m2)**(6./5.))*(1.+z)
        dMc_dz = ((m1*m2)**(3./5.))/((m1+m2)**(1./5.))
        deta_dm1 = m2*(m2-m1)/(m1+m2)**3
        deta_dm2 = m1*(m1-m2)/(m1+m2)**3
        deta_dz = 0.
        ddL_dm1 = 0.
        ddL_dm2 = 0.
        ddL_dz = dL/(1.+z) + clight*(1.+z)/(cosmo.H(z).value*1000.)
        
        return onp.array([[dMc_dm1, dMc_dm2, dMc_dz], [deta_dm1, deta_dm2, deta_dz], [ddL_dm1, ddL_dm2, ddL_dz]])
    
    rotMatrix[onp.ix_([ParNums['Mc'],ParNums['eta'],ParNums['dL']],[ParNums['Mc'],ParNums['eta'],ParNums['dL']])] = J_McetadL_m1srcm2srcz(m1, m2, z, dL=evParams['dL'])
    
    matrix = rotMatrix.T@or_matrix@rotMatrix
    
    return matrix

def DERSNR_McetadL_to_m1srcm2srcz(der_snr, ParNums, evParams, cosmo=Planck18):
    """
    Change variables in the Fisher matrix from :math:`{\cal M}_c`, :math:`\eta` and :math:`d_L` to :math:`m_1^{\\rm src}`, :math:`m_2^{\\rm src}` and :math:`z`.
    
    :param array or_matrix: Array containing the Fisher matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :param dict(int) ParNums: Dictionary specifying the position of each parameter in the Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
    :param cosmo: Cosmology to use to compute the Hubble parameter and the redshift if not provided in the event parameters.
    
    :return: Fisher matrix in :math:`m_1^{\\rm src}`, :math:`m_2^{\\rm src}` and :math:`z`, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :rtype: 2-D array
    
    """
    nparams=der_snr.shape[0]
    
    rotMatrix = onp.identity(nparams)

    # Try to read the redshift, otherwise compute it from the luminosity distance
    try:
        z = evParams['z']
    except KeyError:
        z = z_at_value(cosmo.luminosity_distance, evParams['dL']*uAstro.Gpc).value
    
    m1det, m2det = m1m2_from_Mceta(evParams['Mc'], evParams['eta'])
    m1, m2 = m1det/(1.+z), m2det/(1.+z)

    def J_McetadL_m1srcm2srcz(m1, m2, z, dL=None):
        """
        Jacobian of the transformation from :math:`{\cal M}_c`, :math:`\eta` and :math:`d_L` to :math:`m_1^{\\rm src}`, :math:`m_2^{\\rm src}` and :math:`z`.
        
        :param array or float m1: Source-frame mass of the primary object, :math:`m_1^{\\rm src}`.
        :param array or float m2: Source-frame mass of the secondary object, :math:`m_2^{\\rm src}`.
        :param array or float z: Redshift of the source, :math:`z`.
        :param array or float dL: Luminosity distance of the source, :math:`d_L`.
        :return: Jacobian matrix.
        :rtype: 2-D array
        
        """
        if dL is None:
            dL = cosmo.luminosity_distance(z).value/1000.
        
        dMc_dm1 = m2*(2.*m1+3.*m2)/(5.*(m1*m2)**(2./5.)*(m1+m2)**(6./5.))*(1.+z)
        dMc_dm2 = m1*(3.*m1+2.*m2)/(5.*(m1*m2)**(2./5.)*(m1+m2)**(6./5.))*(1.+z)
        dMc_dz = ((m1*m2)**(3./5.))/((m1+m2)**(1./5.))
        deta_dm1 = m2*(m2-m1)/(m1+m2)**3
        deta_dm2 = m1*(m1-m2)/(m1+m2)**3
        deta_dz = 0.
        ddL_dm1 = 0.
        ddL_dm2 = 0.
        ddL_dz = dL/(1.+z) + clight*(1.+z)/(cosmo.H(z).value*1000.)
        
        return onp.array([[dMc_dm1, dMc_dm2, dMc_dz], [deta_dm1, deta_dm2, deta_dz], [ddL_dm1, ddL_dm2, ddL_dz]])
    
    rotMatrix[onp.ix_([ParNums['Mc'],ParNums['eta'],ParNums['dL']],[ParNums['Mc'],ParNums['eta'],ParNums['dL']])] = J_McetadL_m1srcm2srcz(m1, m2, z, dL=evParams['dL'])
    
    matrix = rotMatrix.T@der_snr
    
    return matrix

#####################################################################################
# ACTUAL COMPUTATIONS OF DERIVATIVES
#####################################################################################

def main(idx, FLAGS):

        idx=idx-1        

        ti=  time.time()

        logidx = '_'+str(FLAGS.idx_in)+'_to_'+str(FLAGS.idx_f) 
        if FLAGS.resume_run==0: 
            logfile = os.path.join(FLAGS.fout, 'logfile'+logidx+'_'+str(idx)+'.txt') #out_path+'logfile.txt'
        else:
            logfile = os.path.join(FLAGS.fout, 'logfile'+logidx+'_'+str(idx)+'_resume_'+str(FLAGS.resume_run)+'.txt')
        myLog = Logger(logfile)
        sys.stdout = myLog
        sys.stderr = myLog
        
        print('\n------------ Mass distribution used:  ------------\n%s' %str(FLAGS.mass_model))
        print('------------------------\n')
        print('------------ Rate distribution used:  ------------\n%s' %str(FLAGS.rate_model))
        print('------------------------\n')
        print('------------ Spin distribution used:  ------------\n%s' %str(FLAGS.spin_model))
        print('------------------------\n')
        print('------------ Model used:  ------------\n%s' %str(FLAGS.POPmodel))
        print('------------------------\n')
        
        MASS_model = mass_models_dict[FLAGS.mass_model]
        if FLAGS.mass_model_params_names:
            for i,par in enumerate(FLAGS.mass_model_params_names):
                MASS_model.update_hyperparameters({par: FLAGS.mass_model_params_values[i]})
        RATE_model = rate_models_dict[FLAGS.rate_model]
        if FLAGS.rate_model_params_names:
            for i,par in enumerate(FLAGS.rate_model_params_names):
                RATE_model.update_hyperparameters({par: FLAGS.rate_model_params_values[i]})
        SPIN_model = spin_models_dict[FLAGS.spin_model]
        if FLAGS.spin_model_params_names:
            for i,par in enumerate(FLAGS.spin_model_params_names):
                SPIN_model.update_hyperparameters({par: FLAGS.spin_model_params_values[i]})
        
        if len(SPIN_model.par_list) == 2:
            parnums    = PNUMS_FIM_ALIGNED_ROT
            parnums_or = PNUMS_FIM_ALIGNED_OR
        elif len(SPIN_model.par_list) == 6:
            parnums    = PNUMS_FIM_PREC_ROT
            parnums_or = PNUMS_FIM_PREC_OR

        popmodel=POPmodel_dict[FLAGS.POPmodel]
    
        if FLAGS.POPmodel=='MassOnly' or FLAGS.POPmodel=='MassRedshiftIndependent':
            POPmodel = popmodel(mass_function=MASS_model, 
                                rate_function=RATE_model, 
                                #spin_function=SPIN_model,
                                verbose=True)
        elif FLAGS.POPmodel=='MassSpinRedshiftIndependent':
            POPmodel = popmodel(mass_function=MASS_model, 
                                rate_function=RATE_model, 
                                spin_function=SPIN_model,
                                verbose=True)
        
    
            
        for it in range( FLAGS.all_n_it_pools[idx] ):
            
            idxin=FLAGS.idxs_lists[str(idx)][it][0] 
            idxf=FLAGS.idxs_lists[str(idx)][it][1]  
            
            ev_chunk = FLAGS.events_lists[str(idx)][it] 
            snr_chunk = FLAGS.snrs_lists[str(idx)][it]
            FIM_chunk = FLAGS.fishers_lists[str(idx)][it]
            snrder_chunk = FLAGS.SNRderivatives_lists[str(idx)][it]
            
            if FLAGS.prior_limits_params_names and FLAGS.prior_limits_params_values:
                parpriors=dict(zip(FLAGS.prior_limits_params_names, FLAGS.prior_limits_params_values))
            else:
                parpriors=None 



            if parpriors is not None:
                diag = onp.array([parpriors[key] if key in parpriors.keys() else 0. for key in parnums_or.keys()])
                #pp   = onp.eye(FIM_use.shape[0])*diag
                pp   = onp.eye(FIM_chunk.shape[0])*diag
                if FIM_chunk.ndim==2:
                    FIM_chunk = pp + FFIM_chunk
                else:
                    FIM_chunk = pp[:,:,onp.newaxis] + FIM_chunk

            
            
            if FLAGS.FIM_rotated==0:
                for i in range(FIM_chunk.shape[-1]):
                    FIM_chunk[:,:,i]  = FISHER_McetadL_to_m1srcm2srcz(FIM_chunk[:,:,i], parnums_or, {k:ev_chunk[k][i] for k in ev_chunk.keys()})
                    snrder_chunk[:,i] = DERSNR_McetadL_to_m1srcm2srcz(snrder_chunk[:,i], parnums_or, {k:ev_chunk[k][i] for k in ev_chunk.keys()})

            Pdetder_chunk = snrder_chunk*onp.sqrt(0.5/onp.pi)*onp.exp(-(snr_chunk-FLAGS.snr_th)**2/(2*FLAGS.Pdet_sigma**2))/FLAGS.Pdet_sigma

            nevents_chunk = len(ev_chunk['dL'])
            
            i_in = idxin+FLAGS.idx_in
            i_f = idxf+FLAGS.idx_in
            suffstr = '_'+str(idxin+FLAGS.idx_in)+'_to_'+str(idxf+FLAGS.idx_in)

            if FLAGS.resume_run>0:
                if os.path.exists(os.path.join(FLAGS.fout, 'termI_der'+suffstr+'.npy')):
                    print('Batch already present, skipping...')
                    continue
            
            print('\nIn this chunk we have %s events, from %s to %s' %(nevents_chunk, i_in,  i_f  ))
    

                
            
            # to name output files
            suffstr = '_'+str(idxin+FLAGS.idx_in)+'_to_'+str(idxf+FLAGS.idx_in)

            print('\n\n\n\n\n\n\n')
            print('EVCHUNCK=',ev_chunk['dL'].shape)
            print('\n\n\n\n\n\n\n')

            termI_ders = POPmodel.pop_function_derivative(ev_chunk, uselog=True)
            termI_hess = POPmodel.pop_function_hessian(ev_chunk, uselog=True)

            print('\n\n\n termI', termI_ders.shape)
            termII     = POPmodel.pop_function_hessian_termII(ev_chunk, FIM_chunk, parnums,ParPrior=None)
            termIII    = POPmodel.pop_function_hessian_termIII(ev_chunk, FIM_chunk, parnums,ParPrior=None)
            termIV     = POPmodel.pop_function_hessian_termIV(ev_chunk, FIM_chunk, Pdetder_chunk, parnums,ParPrior=None)
            termV      = POPmodel.pop_function_hessian_termV(ev_chunk, FIM_chunk, parnums,ParPrior=None)
                
            to_file(termI_ders, termI_hess, termII, termIII, termIV, termV, FLAGS.fout, suff=suffstr)
            
        te=time.time()
        print('------ Total execution time: %s sec.\n\n' %( str((te-ti))))
        myLog.close()
        

def mainMPI(i):
    # Just a workaround to make mpi work without starmap
    return main(i, FLAGS)


parser = argparse.ArgumentParser(prog = 'calculate_hyperpar_derivatives_from_catalog.py', description='Executable to run ``gwfast`` on a catalog of events, with the possibility to parallelize over multiple CPUs, ready to use both on single machines and on clusters.')
parser.add_argument("--fname_obs", default='', type=str, required=True, help='Name of the file containing the catalog.')
parser.add_argument("--fname_evSNRs", default='', type=str, required=True, help='Name of the file containing the SNRs for the events in the catalog.')
parser.add_argument("--fname_evFIMs", default='', type=str, required=True, help='Name of the file containing the FIMs for the events in the catalog.')
parser.add_argument("--fname_evSNRders", default='', type=str, required=True, help='Name of the file containing the SNR derivatives for the events in the catalog.')
parser.add_argument("--fout", default='test_gwfast', type=str, required=True, help='Path to output folder, which has to exist before the script is launched.')


parser.add_argument("--mass_model",  default='TruncatedPowerLaw', type=str, required=False, help='Name of the mass distribution model.')
parser.add_argument("--mass_model_params_names", nargs='+', default=[ ], type=str, required=False, help='Hyperparameters names of the mass distribution model, separated by *single spacing*.')
parser.add_argument("--mass_model_params_values", nargs='+', default=[ ], type=float, required=False, help='Hyperparameters values of the mass distribution model, separated by *single spacing*.')
parser.add_argument("--rate_model",  default='PowerLaw', type=str, required=False, help='Name of the rate distribution model.')
parser.add_argument("--rate_model_params_names", nargs='+', default=[ ], type=str, required=False, help='Hyperparameters names of the rate distribution model, separated by *single spacing*.')
parser.add_argument("--rate_model_params_values", nargs='+', default=[ ], type=float, required=False, help='Hyperparameters values of the rate distribution model, separated by *single spacing*.')
parser.add_argument("--spin_model",  default='GaussNonPrecessing', type=str, required=False, help='Name of the spin distribution model.')
parser.add_argument("--spin_model_params_names", nargs='+', default=[ ], type=str, required=False, help='Hyperparameters names of the spin distribution model, separated by *single spacing*.')
parser.add_argument("--spin_model_params_values", nargs='+', default=[ ], type=float, required=False, help='Hyperparameters values of the spin distribution model, separated by *single spacing*.')

parser.add_argument("--POPmodel",  default='MassSpinRedshiftIndependent_PopulationModel', type=str, required=False, help='Name of the spin distribution model.')


parser.add_argument("--prior_limits_params_names", nargs='+', default=[ ], type=str, required=False, help='Names of the parameters where priors limits need to be added to the single-event-fisher, separated by *single spacing*.')
parser.add_argument("--prior_limits_params_values", nargs='+', default=[ ], type=float, required=False, help='Values of the priors limits that need to be added to the single-event-fisher, separated by *single spacing*.')

parser.add_argument("--batch_size", default=1, type=int, required=False, help='Size of the batch to be computed in vectorized form on each process.')
parser.add_argument("--npools", default=1, type=int, required=False, help='Number of parallel processes.')
parser.add_argument("--snr_th", default=12., type=float, required=False, help='Threshold value for the SNR to use in the Pdet function.')
parser.add_argument("--snr_th_FIM", default=0., type=float, required=False, help='Threshold value for the SNR used in the FIM calculation to consider the event detectable.')
parser.add_argument("--idx_in", default=0, type=int, required=False, help='Index of the event in the catalog from which to start the calculation.')
parser.add_argument("--idx_f", default=None, type=int, required=False, help='Index of the event in the catalog at which to end the calculation.')
parser.add_argument("--Pdet_sigma", default=1., type=float, required=False, help='Standard deviation to use in the Pdet function.')
parser.add_argument("--concatenate", default=1, type=int, required=False, help='Int specifying if the results of the individual batches have to be concatenated (``1``) or not (``0``).')
parser.add_argument("--mpi", default=0, type=int, required=False, help='Int specifying if the code has to parallelize using multiprocessing (``0``), or using MPI (``1``), suitable for clusters.')
parser.add_argument("--resume_run", default=0, type=int, required=False, help='Int specifying whether to resume previous run. In this case the parent seed structure is preserved.')
parser.add_argument("--FIM_rotated", default=1, type=int, required=False, help='Int specifying whether the FIMs and SNR derivatives are already in the variables needed for the analysis.')

if __name__ =='__main__':

    FLAGS = parser.parse_args()
        
    print('Input arguments: %s' %str(FLAGS))
    
    ti =  time.time()
    
    #####################################################################################
    # LOAD EVENTS, SNRS AND FISHERS
    #####################################################################################
    
    fname_obs = os.path.join(FLAGS.fname_obs)
    if not os.path.exists(fname_obs):
        raise ValueError('Path to catalog does not exist. Value entered: %s' %fname_obs)
    
    print('Loading events from %s...' %fname_obs)
    events_loaded = load_population(fname_obs)
    
    keylist=list(events_loaded.keys())
    nevents_total = len(events_loaded[keylist[0]])
    print('This catalog has %s events.' %nevents_total)
    
    if FLAGS.idx_f is None:
        events_loaded = {k: events_loaded[k][FLAGS.idx_in:] for k in events_loaded.keys()}
    else:
        events_loaded = {k: events_loaded[k][FLAGS.idx_in:FLAGS.idx_f] for k in events_loaded.keys()}
    
    fname_evSNRs = os.path.join(FLAGS.fname_evSNRs)
    if not os.path.exists(fname_evSNRs):
        raise ValueError('Path to SNRs does not exist. Value entered: %s' %fname_evSNRs)
    print('Loading SNRs from %s...' %fname_evSNRs)
    
    snrs_loaded = onp.loadtxt(fname_evSNRs)
    if FLAGS.idx_f is None:
        snrs_loaded = snrs_loaded[FLAGS.idx_in:]
    else:
        snrs_loaded = snrs_loaded[FLAGS.idx_in:FLAGS.idx_f]
    print(events_loaded.keys())
    print('\n\n', 'SNR SHAPE', snrs_loaded.shape,'\n\n')

    if FLAGS.snr_th != FLAGS.snr_th_FIM:
        print('Using different SNR threshold for Pdet and FIM: %s and %s' %(FLAGS.snr_th, FLAGS.snr_th_FIM))
        detected = snrs_loaded>FLAGS.snr_th_FIM
        events_loaded_use = {k: events_loaded[k][detected] for k in events_loaded.keys()}
        snrs_loaded = snrs_loaded[detected]
    else:
        events_loaded_use = copy.deepcopy(events_loaded)

    
    fname_evFIMs = os.path.join(FLAGS.fname_evFIMs)
    if not os.path.exists(fname_evFIMs):
        raise ValueError('Path to FIMs does not exist. Value entered: %s' %fname_evFIMs)
    print('Loading FIMs from %s...' %fname_evFIMs)
    fishers_loaded = onp.load(fname_evFIMs)

    if FLAGS.idx_f is None:
        fishers_loaded = fishers_loaded[:,:,FLAGS.idx_in:]
    else:
        fishers_loaded = fishers_loaded[:,:,FLAGS.idx_in:FLAGS.idx_f]


    fname_evSNRders = os.path.join(FLAGS.fname_evSNRders)
    if not os.path.exists(fname_evSNRders):
        raise ValueError('Path to SNR derivatives does not exist. Value entered: %s' %fname_evSNRders)
    with h5py.File(fname_evSNRders, 'r') as derivs:
        SNRderivativess_loaded = onp.array(derivs['derivative']['net'])

    if FLAGS.idx_f is None:
        SNRderivativess_loaded = SNRderivativess_loaded[:,FLAGS.idx_in:]
    else:
        SNRderivativess_loaded = SNRderivativess_loaded[:,FLAGS.idx_in:FLAGS.idx_f]
        
    print('\n\n', 'der snr SHAPE', SNRderivativess_loaded.shape[-1],'\n\n')
    print('\n\n', 'FIM SHAPE', fishers_loaded.shape[-1],'\n\n')
    print('\n\n', 'SNR SHAPE', len(snrs_loaded),'\n\n')
    print('\n\n', 'EVENTs SHAPE', len(events_loaded_use['dL']),'\n\n')
    print(len(snrs_loaded))
    

    
    assert SNRderivativess_loaded.shape[-1] == len(events_loaded_use['dL']) == fishers_loaded.shape[-1] == len(snrs_loaded)


    #####################################################################################
    # remove events in which the population model for the recovery is zero
    #####################################################################################

    MASS_model = mass_models_dict[FLAGS.mass_model]
    if FLAGS.mass_model_params_names:
        for i,par in enumerate(FLAGS.mass_model_params_names):
            MASS_model.update_hyperparameters({par: FLAGS.mass_model_params_values[i]})
    RATE_model = rate_models_dict[FLAGS.rate_model]
    if FLAGS.rate_model_params_names:
        for i,par in enumerate(FLAGS.rate_model_params_names):
            RATE_model.update_hyperparameters({par: FLAGS.rate_model_params_values[i]})
    SPIN_model = spin_models_dict[FLAGS.spin_model]
    if FLAGS.spin_model_params_names:
        for i,par in enumerate(FLAGS.spin_model_params_names):
            SPIN_model.update_hyperparameters({par: FLAGS.spin_model_params_values[i]})
    
    if len(SPIN_model.par_list) == 2:
        parnums    = PNUMS_FIM_ALIGNED_ROT
        parnums_or = PNUMS_FIM_ALIGNED_OR
    elif len(SPIN_model.par_list) == 6:
        parnums    = PNUMS_FIM_PREC_ROT
        parnums_or = PNUMS_FIM_PREC_OR
    popmodel=POPmodel_dict[FLAGS.POPmodel]

    if FLAGS.POPmodel=='MassOnly' or FLAGS.POPmodel=='MassRedshiftIndependent':
        POPmodel = popmodel(mass_function=MASS_model, 
                            rate_function=RATE_model, 
                            #spin_function=SPIN_model,
                            verbose=True)
    elif FLAGS.POPmodel=='MassSpinRedshiftIndependent':
        POPmodel = popmodel(mass_function=MASS_model, 
                            rate_function=RATE_model, 
                            spin_function=SPIN_model,
                            verbose=True)


    th=10**-20
    pop_rec=POPmodel.pop_function(events_loaded_use,uselog=False)

    finite_indices=pop_rec>th
    events_loaded_use = {k: events_loaded_use[k][finite_indices] for k in events_loaded_use.keys()}
    selected_indices = onp.where(finite_indices)[0]
    idx_sel=len(selected_indices)
    onp.savetxt(FLAGS.fout+'/selected_indices'+str(FLAGS.idx_in)+'_to_'+str(FLAGS.idx_f)+'.txt',selected_indices)




    #with h5py.File(out_path +f'all_derivatives_SNR_{idxi}_to_{idx_sel}.hdf5', 'w') as f:
    #    derivative_group = f.create_group('derivative')
    #    derivative_group.create_dataset('net', data=SNRderivativess_loaded[:,selected_indices])
    #save_data(os.path.join(out_path,f'Detected_catalog_{idxi}_to_{idx_sel}.h5'), samps_tmp)
    #onp.savetxt(out_path+f'snrs_{idxi}_to_{idx_sel}.txt',snrs_loaded[selected_indices])
    #onp.save(out_path+f'fishers_{idxi}_to_{idx_sel}.npy',fishers_loaded[:,:,selected_indices])
    #nevents_total=len(selected_indices)

    #onp.savetxt('indices_for_catalog.txt',selected_indices)
    #save_data(os.path.join(out_path,f'Detected_catalog_{idxi}_to_{idx_sel}.h5'), samps_tmp)
    #onp.savetxt(out_path+f'snrs_{idxi}_to_{idx_sel}.txt',snrs_loaded[selected_indices])
    #onp.save(out_path+f'fishers_{idxi}_to_{idx_sel}.npy',fishers_loaded[:,:,selected_indices])
    #with h5py.File(out_path +f'all_derivatives_SNR_{idxi}_to_{idx_sel}.hdf5', 'w') as f:
        #derivative_group = f.create_group('derivative')
        #derivative_group.create_dataset('net', data=SNRderivativess_loaded[:,selected_indices])

    #SNRderivativess_loaded=SNRderivativess_loaded[:,selected_indices]
    #fishers_loaded=fishers_loaded[:,:,selected_indices]
    #events_loaded_use=samps_tmp
    #snrs_loaded=snrs_loaded[selected_indices]



    nevents_total = len(events_loaded_use[keylist[0]])
    print('Using events between %s and %s with SNR>%s, total %s events' %(FLAGS.idx_in, FLAGS.idx_f, FLAGS.snr_th_FIM, nevents_total))
    print('Among them, the ones above pop_rec>10**-20 are', (idx_sel))

    
    #####################################################################################
    # SPLIT EVENTS BETWEEN PROCESSES ACCORDING TO BATCH SIZE AND NUMBER OF PROCESSES REQUIRED
    #####################################################################################

    batch_size = FLAGS.batch_size
    npools = FLAGS.npools
    
    n_per_it = batch_size*FLAGS.npools # total events computed simultaneously on all cores
    if n_per_it>nevents_total:
        raise ValueError('The chosen batch size and number of pools are too large (it would take <1 iteration to cover all data). Choose smaller values.')
    
    n_it_per_pool = nevents_total//n_per_it # iterations done on every core
    nev_covered = n_per_it*n_it_per_pool # events done on every core
    #if nev_covered<nevents_total:
        
    nev_miss = nevents_total-nev_covered
    # compute how many pools will have to do one iteration more  
    diff = nevents_total-nev_covered
    
    if diff==0: # all cores do same number of iterations

        all_n_it_pools = [ n_it_per_pool for _ in range(npools) ]
        all_n_per_pool =  [ n_it_per_pool*batch_size for _ in range(npools)]
        full_last_chunk_sizes =  [ batch_size for _ in range(npools) ]
        #last_chunk_sizes = 
        cores_to_add=0
        n_it_per_pool_extra=0
        res=0
        int_part=0
        last_chunk_sizes=[]
    else:
        if diff<batch_size: 
            cores_to_add=1 # only 1 core needs to do one more iteration but with n of events < batch size
            res = 1
            int_part = 0
            n_it_per_pool_extra = 1
        elif diff==batch_size: # only one core needs to do one more iteration with n of events = batch size
            cores_to_add=1 
            res=0
            int_part = 1
            n_it_per_pool_extra = 1
        else:
            evsres = diff%batch_size # number of events left after cores did extra iteration 
            int_part = int((diff-evsres)/batch_size) # number of cores doing exactly one batch more
            if evsres>0:
                res=1
            else: res=0
                
            cores_to_add =  int(int_part+res)
            n_it_per_pool_extra = 1
        print('Cores which do one extra iteration: %s' %cores_to_add)
        print('N of cores doing less than one batch more: %s' %res)
        print('N of cores doing exactly one batch more: %s' %int_part)
            
        last_chunk_sizes = [ batch_size for _ in range(int_part) ]
        if int(nev_miss-batch_size*int_part)>0:
            last_chunk_sizes+=[ int(nev_miss-batch_size*int_part) for _ in range(1)] 
        print('Last chunk size will be %s on last %s cores' %(str(last_chunk_sizes), cores_to_add))
        all_n_it_pools = [ n_it_per_pool for _ in range(npools-cores_to_add) ]+[ n_it_per_pool+n_it_per_pool_extra for _ in range(cores_to_add)]
        all_n_per_pool = [ n_it_per_pool*batch_size for _ in range(npools-cores_to_add)]+[ (n_it_per_pool*batch_size)+last_chunk_sizes[i] for i in range(cores_to_add)] # events computed on every core in total 
        full_last_chunk_sizes = onp.hstack( [onp.zeros(npools-len(last_chunk_sizes)), last_chunk_sizes] ) #.T
    
    print('Number of iterations on every core: %s' %str(all_n_it_pools))     
    print('%s iterations will be used to cover all the %s events.' %(max(all_n_it_pools), nevents_total))
    print('For each iteration, we will parallise on %s cores with max %s events/core.' %(npools, batch_size))
    print('Number of events computed on every core in total: %s' %str(all_n_per_pool))
    
    all_n_it_pools = onp.array(all_n_it_pools)
    
    all_batch_sizes = onp.zeros((npools, max(all_n_it_pools) )).T
    for i in range(npools):
        for j in range(max(all_n_it_pools)):
            #print(i, j)
            is_last = j==max(onp.array(all_n_it_pools)-1)
            #print(is_last)
            if not is_last or onp.all(full_last_chunk_sizes==0):
                all_batch_sizes[j, i] = batch_size
            else:
                all_batch_sizes[j, i] = full_last_chunk_sizes[i]
    
    all_batch_sizes = all_batch_sizes.astype('int')
    print('All batch sizes, last three iterations (shape nbatches x npools ): ')
    print(all_batch_sizes[-3:, :])
    
    assert all_batch_sizes.sum()==nevents_total
    assert onp.array(all_n_per_pool).sum()==nevents_total
    
    events_lists = {str(i):[] for i in range(npools)}
    idxs_lists = {str(i):[] for i in range(npools)}
    FIMs_lists = {str(i):[] for i in range(npools)}
    snrs_lists = {str(i):[] for i in range(npools)}
    SNRderivatives_lists = {str(i):[] for i in range(npools)}
    
    pin=0
    for it in range(all_batch_sizes.shape[0]): # iterations
        for p in range(all_batch_sizes.shape[-1]): # pools
            pf =  pin+all_batch_sizes[it, p]
            if pf>pin:
                events_lists[str(p)].append({k: events_loaded_use[k][pin:pf] for k in events_loaded_use.keys()})
                idxs_lists[str(p)].append( (pin, pf) )
                FIMs_lists[str(p)].append(fishers_loaded[:,:,pin:pf])
                snrs_lists[str(p)].append(snrs_loaded[pin:pf])
                SNRderivatives_lists[str(p)].append(SNRderivativess_loaded[:,pin:pf])
                nevents_chunk = len(events_lists[str(p)][-1]['dL'])
                assert nevents_chunk == all_batch_sizes[it, p]
                pin = pf
    ncheck=0       
    for evl in  events_lists.values():
        for evs in evl:
            ncheck+=len(evs['dL'])
        
    assert ncheck==nevents_total
    
    FLAGS.all_n_it_pools = all_n_it_pools
    FLAGS.events_lists = events_lists
    FLAGS.idxs_lists = idxs_lists
    FLAGS.fishers_lists = FIMs_lists
    FLAGS.snrs_lists = snrs_lists
    FLAGS.SNRderivatives_lists = SNRderivatives_lists
    
    ############################################################################
    # Run processes in parallel
    ############################################################################
    
    if FLAGS.npools>1:
        print('Parallelizing on %s CPUs ' %FLAGS.npools)    
        print('Total available CPUs: %s' %str(multiprocessing.cpu_count()) )
        pool =  get_pool(mpi=FLAGS.mpi, threads=FLAGS.npools+1)  
        if FLAGS.mpi:
            pool.map( mainMPI, [ i for i in range(1, FLAGS.npools+1)] ) 
        else:
            pool.starmap( main, [ ( i, FLAGS ) for i in range(1, FLAGS.npools+1)] )
        pool.close()
    else:
        (main(1, FLAGS), )
    
    ############################################################################
    # Concatenate results and clean
    ############################################################################
    
    if FLAGS.concatenate:
        
        print('\nSaving final version to file...')
        if FLAGS.idx_f is None:
                idxf = str(nevents_total)
        else: idxf = FLAGS.idx_f
            
        suffstr = '_'+str(FLAGS.idx_in)+'_to_'+str(idxf)      
            
        pin=FLAGS.idx_in
        temrs_initialised = False
        for it in range(all_batch_sizes.shape[0]): # iterations
            for p in range(all_batch_sizes.shape[-1]): # pools
                pf =  pin+all_batch_sizes[it, p]
                if pf>pin:
                    print('Concatenating files from %s to %s' %(pin, pf))
                    suff_batch = '_'+str(pin)+'_to_'+str(pf)
                    termI_der_, termI_hess_, termII_, termIII_, termIV_, termV_ = from_file( FLAGS.fout, suff=suff_batch)
                    if not temrs_initialised:
                        termI_der, termI_hess, termII, termIII, termIV, termV = termI_der_, termI_hess_, termII_, termIII_, termIV_, termV_
                        temrs_initialised=True
                    else:
                        termI_der   = onp.concatenate([termI_der, termI_der_], axis=-1)
                        termI_hess  = onp.concatenate([termI_hess, termI_hess_], axis=-1)
                        termII      = onp.concatenate([termII, termII_], axis=-1)
                        termIII     = onp.concatenate([termIII, termIII_], axis=-1)
                        termIV      = onp.concatenate([termIV, termIV_], axis=-1)
                        termV       = onp.concatenate([termV, termV_], axis=-1)
                
                    pin = pf
    
        to_file(termI_der, termI_hess, termII, termIII, termIV, termV, FLAGS.fout, suff=suffstr)
                
        if (FLAGS.npools>1) or (FLAGS.npools==1 and all_n_it_pools[0]>1): 
            print('Cleaning...')
            pin=FLAGS.idx_in
            res = []
            for it in range(all_batch_sizes.shape[0]): # iterations
                for p in range(all_batch_sizes.shape[-1]): # pools
                    pf =  pin+all_batch_sizes[it, p]
                    if pf>pin:
                        delete_files(pin, pf, FLAGS.fout)
                        
                        pin = pf
    te=time.time()
    print('------ Done for all. Total execution time: %s sec.\n\n' %(str((te-ti))))