import sys,os

SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd())))#gwfast_dev/gwfast/population
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../..')) #gwfast_dev
sys.path.append(PARENT_DIR)

####### gwfast.population
import gwfast.population.POPmodels as POPmodels 
from gwfast.population.popdistributions.massdistribution import *
from gwfast.population.popdistributions.spindistribution import *
from gwfast.population.popdistributions.ratedistribution import *
from gwfast.population.POPmodels import MassSpinRedshiftIndependent_PopulationModel,MassRedshiftIndependent_PopulationModel,MassOnly_PopulationModel

######## gwfast
import gwfast.gwfastGlobals as glob
from gwfast.waveforms import TaylorF2_RestrictedPN, IMRPhenomD, IMRPhenomHM, IMRPhenomD_NRTidalv2, IMRPhenomNSBH
from gwfast.waveforms import LAL_WF






########################################################################################
# Shortcuts for distribution functions names; nothing to edit here
########################################################################################

mass_models_dict = {
                    'TruncatedPowerLaw': TruncatedPowerLaw_modsmooth_MassDistribution,
                    'PowerLawPlusPeak': PowerLawPlusPeak_modsmooth_MassDistribution,
                    }

rate_models_dict = {'PowerLaw': PowerLaw_RateDistribution,
                    'MadauDickinson': MadauDickinson_RateDistribution,
                    'MadauDickinsonPLTimeDelta': MadauDickinsonPLTimeDelta_RateDistribution,
                    }
                    
spin_models_dict = {'Default': DefaultPrecessing_SpinDistribution,
                    'SameFlatNonPrecessing': SameFlatNonPrecessing_SpinDistribution,
                    'FlatNonPrecessing': FlatNonPrecessing_SpinDistribution,
                    'GaussNonPrecessing': GaussNonPrecessing_SpinDistribution,
                    }
POPmodel_dict = {'MassSpinRedshiftIndependent': MassSpinRedshiftIndependent_PopulationModel,
                'MassRedshiftIndependent': MassRedshiftIndependent_PopulationModel,
                'MassOnly': MassOnly_PopulationModel,
                    }


#############################################################################################
########### Section you need to edit: generate pdraw, run gwfast, run population #############
#############################################################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Edit here
# 1: Choose network and waveform, nCPUs, N_chunk and N_goal,snr_th
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

config_network='ETS' # edit here
config_waveform='precessing' #edit here
CPUs=10
N_chunk=20 
N_goal=10 # Number of events you want to reach 
snr_th=2. # Threshold value for the SNR to use in the Pdet function. We generate Ngoal events that have snr>snr_th
snr_th_FIM=0 # snr over which we will compute the Fishers. We compute all the fishers and then cut in the population analysis with pdet(theta). This corresponds to snr_th in gwfast



# nothing to edit here: shortcuts to configure waveform and detector network

########### Waveform model
if config_waveform=='aligned':
    wf_model='IMRPhenomHM'
    waveform=IMRPhenomHM()
elif config_waveform=='precessing':
    wf_model='LAL-IMRPhenomXPHM'
    lalargs=['HM','precessing']
    waveform=LAL_WF('IMRPhenomXPHM', is_HigherModes=True, is_Precessing=True, is_prec_ang=True)


####### DETECTOR NETWORK

if config_network=='O4': ###### LVK O5
    net=['L1', 'H1', 'Virgo','KAGRA']
    psds=[glob.detPath +'/observing_scenarios_paper/AplusDesign.txt',
        glob.detPath +'/observing_scenarios_paper/AplusDesign.txt',
        glob.detPath +'/observing_scenarios_paper/avirgo_O5low_NEW.txt',
        glob.detPath +'/observing_scenarios_paper/kagra_80Mpc.txt']
    fmin=10.
elif config_network=='O4': #LIGO +VIRGO O4
    net=['L1', 'H1', 'Virgo']
    psds=[glob.detPath +'/observing_scenarios_paper/aligo_O4high.txt',
        glob.detPath +'/observing_scenarios_paper/aligo_O4high.txt',
        glob.detPath +'/observing_scenarios_paper/avirgo_O4high_NEW.txt']
    fmin=10.
elif config_network=='ETS': #ET Sardinia
    net=['ETS']
    #psds=[glob.detPath+'/ET-0000A-18.txt']
    psds=[glob.detPath+'/ET_designs_comparison_paper/HFLF_cryo/ETLength10km.txt']
    fmin=2.
elif config_network=='ET+CE': #ET+CE
    net=['ETS','CE1Id']
    psds=[glob.detPath+'/ET_designs_comparison_paper/HFLF_cryo/ETLength10km.txt',
         glob.detPath+'/ce_strain/cosmic_explorer.txt']
    fmin=2.
elif config_network=='ET+2CE': #ET+CE
    net=['ETS','CE1Id','CE2NM'] # 'CE2NSW', 'CE1IdCoBAal'
    psds=[glob.detPath+'/ET_designs_comparison_paper/HFLF_cryo/ETLength10km.txt',
         glob.detPath+'/ce_strain/cosmic_explorer.txt',
         glob.detPath+'/ce_strain/cosmic_explorer_20km.txt']
    fmin=2.
    
# cosmic_explorer_strain.txt is the baseline 40 km detector
# cosmic_explorer_20km_strain.txt is the baseline 20 km detector ("compact binary tuned")
# cosmic_explorer_20km_pm_strain.txt is the 20 km detector tuned for post-merger signals
# cosmic_explorer_40km_lf_strain.txt is the 40 km detector tuned for low-freqency signals




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3: Edit here: choose population from which you want to generate the events, i.e. pdraw
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


alpha=0.5
beta=0.5
sigmah=20.
mass_model_inj='PowerLawPlusPeak'
mass_model_params_names_inj=['alpha_m','beta_q','sigma_h']
mass_model_params_values_inj=[alpha, beta,sigmah]
rate_model_inj='MadauDickinson'
rate_model_params_names_inj=[]
rate_model_params_values_inj=[]
spin_model_inj='Default'
spin_model_params_names_inj=[]
spin_model_params_values_inj=[]
pop_model_inj='MassSpinRedshiftIndependent'

# nothing to edit here
MASS_model_inj = mass_models_dict[mass_model_inj]()
if mass_model_params_names_inj:
    for i,par in enumerate(mass_model_params_names_inj):
        MASS_model_inj.update_hyperparameters({par: mass_model_params_values_inj[i]})

RATE_model_inj = rate_models_dict[rate_model_inj]()
if rate_model_params_names_inj:
    for i,par in enumerate(rate_model_params_names_inj):
        RATE_model_inj.update_hyperparameters({par: rate_model_params_values_inj[i]})

SPIN_model_inj = spin_models_dict[spin_model_inj]()
if spin_model_params_names_inj:
    for i,par in enumerate(spin_model_params_names_inj):
        SPIN_model_inj.update_hyperparameters({par: spin_model_params_values_inj[i]})
     
p_draw = POPmodel_dict[pop_model_inj](mass_function=MASS_model_inj, 
                                rate_function=RATE_model_inj, 
                                spin_function=SPIN_model_inj,
                                verbose=True)





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4. Edit here : The inputs of this section will be used to run popfisher
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FIM_rotated=0 
Pdet_sigma=1 
idxin_popfisher=0 #Index of the event in the catalog from which to start the calculation.
idxf_popfisher=None #Index of the event in the catalog at which to end the calculation. If set to None, the Fisher will be computed using the entire catalog.

concatenate_popfisher=1
mpi_popfisher=0 
batch_size_popfisher=10
npools_popfisher=50
resume_run_popfisher=0 #default=0, Int specifying whether to resume previous run.

# choose population model for the analysis
mass_model='PowerLawPlusPeak'
mass_model_params_names=['sigma_h']
mass_model_params_values=[20.]
rate_model='MadauDickinson'
rate_model_params_names=[]
rate_model_params_values=[]
spin_model='Default'
spin_model_params_names=[]
spin_model_params_values=[]
pop_model_rec='MassSpinRedshiftIndependent'

prior_limits_params_names=['chi1','chi2','phiJL','phi12','tilt1','tilt2'] 
prior_limits_params_values=[1.,1.,0.025330295910584444, 0.025330295910584444, 0.10132118364233778 ,0.10132118364233778]




# nothing to edit here
MASS_model = mass_models_dict[mass_model]()
if mass_model_params_names:
    for i,par in enumerate(mass_model_params_names):
        MASS_model.update_hyperparameters({par: mass_model_params_values[i]})

RATE_model = rate_models_dict[rate_model]()
if rate_model_params_names:
    for i,par in enumerate(rate_model_params_names):
        RATE_model.update_hyperparameters({par: rate_model_params_values[i]})

SPIN_model = spin_models_dict[spin_model]()
if spin_model_params_names:
    for i,par in enumerate(spin_model_params_names):
        SPIN_model.update_hyperparameters({par: spin_model_params_values[i]})
        
p_pop = POPmodel_dict[pop_model_rec](mass_function=MASS_model, 
                                rate_function=RATE_model, 
                                spin_function=SPIN_model,
                                verbose=True)






#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 5. Edit here: the inputs of this section will be used to run popfisher
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fname_evSNRs_gwfast='' #Path to output folder containing SNRs (it has not necessary to exist before the script is launched since gwfast can compute the snrs of the events)
batch_size_gwfast=1
npools_gwfast=50

fmin=2.
fmax=None
idxin_gwfast=0
idxf_gwfast=None #If set to None, the Fishers will be computed using the entire catalog.

return_snr_derivatives=1 
return_derivatives=0  #0=no
return_all=1

concatenate_gwfast = 1
mpi=0 
resume_run = 0

netfile=None

compute_fisher=1 
compute_imrsplit=0 #0=np

rot=1 
duty_factor = 1.


#seeds=[42, 40, 53] 
#params_fix=['z', 'chi1', 'chi2']
seeds=[] #List of seeds to set for the duty factors in individual detectors, to help reproducibility
params_fix=[] #List of parameters to fix to the fiducial values

use_cogwheel_params=0
jit_fisher=0
fix_fgrid=0
fix_fgrid_df=0.125
























         



