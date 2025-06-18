import sys,os,copy,time,argparse,importlib,shutil
import tqdm_pathos



SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd())))#gwfast_dev/gwfast/population
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../..')) #gwfast_dev
sys.path.append(PARENT_DIR)
import gwfast
import gwfast.gwfastGlobals as glob
from gwfast.signal import GWSignal
from gwfast.network import DetNet
from gwfast.waveforms import *
from gwfast.gwfastUtils import save_detectors,save_data

import gwfast.population.POPutils as utils
import numpy as np

##########################################################################
# classes to generate the desired population
##########################################################################

class Observations(object):
    """
    Class to generate the desired population of events.
    """
    def __init__(self,
                 population_model_inj, 
                 N_goal, 
                 out_dir,
                 snr_th ,
                 psds,
                 names,
                 CPUs,
                 N_chunk,
                 waveform,useEarthMotion=False,fmin=None,fmax=None,
                 add_noise=False,
                ):
        self.snr_th=snr_th
        self.N_goal=N_goal
        self.population_model_inj=population_model_inj
        self.names=names
        self.psds=psds
        self.waveform=waveform
        self.out_dir=out_dir
        self.fmin=fmin
        #self.fmax=fmax
        self.useEarthMotion=useEarthMotion
        self.CPUs=CPUs
        self.N_chunk=N_chunk

   
        

    def generate_network(self):
        '''
        Generate the network of detectors with the desired PSDs and parameters.
        '''
        #fmin        = 10. if self.fmin is None else fmin
        #fmax        = 1024. if self.fmax is None else fmax
        alldetectors = copy.deepcopy(gwfast.gwfastGlobals.detectors)
        Net = {det:alldetectors[det] for det in self.names}
        for i in range(len(self.names)): 
            Net[self.names[i]]['psd_path'] = os.path.join(glob.detPath, self.psds[i]) 
        fname_det_new = os.path.join(self.out_dir, 'detectors.json')
        save_detectors(fname_det_new, Net)

        myLVSignals = {}
        for d in Net.keys():
            myLVSignals[d] = GWSignal(self.waveform, 
                        psd_path=Net[d]['psd_path'],
                        detector_shape = Net[d]['shape'],
                        det_lat= Net[d]['lat'],
                        det_long=Net[d]['long'],
                        det_xax=Net[d]['xax'], 
                        verbose=True,
                        useEarthMotion = self.useEarthMotion,
                        fmin=self.fmin,
                        #fmax=self.fmax,
                        IntTablePath=None) 
        MyNet = DetNet(myLVSignals)
        return MyNet





    def generate_events_parallel_snr_th(self, network):
        '''
        Generate catalog with N_goal events having SNR>snr_th
        :param network: Network of detectors.   

        :return: Samples extracted from the desired population model.
        :rtype: dictionary
        '''
        total_events = 0
        events_over_th = 0
        save_theta_samples = []
        
        while events_over_th < self.N_goal:  # Continue until N_goal events over the threshold are reached or exceeded
            np.random.seed()
            theta_samples = self.population_model_inj.sample_population(size=self.N_chunk)
            data_new = []
    
            # Create the list of dictionaries data_new
            for i in range(len(theta_samples['theta'])):
                new_entry = {}
                for key, value in theta_samples.items():
                    new_entry[key] = value[i]
                data_new.append(new_entry)
    
            in_time = time.time()

            snr = tqdm_pathos.map(network.SNR_netFast, data_new, n_cpus=self.CPUs)
            print('\nDone. Total execution time: %.2fs' % (time.time() - in_time))
            snr = np.array(snr)
            theta_samples['snr']=snr
            # Save all theta_samples
            save_theta_samples.append(theta_samples)
            
            chunk_events_over_th = len(snr[snr > self.snr_th])
            total_events += self.N_chunk
            events_over_th += chunk_events_over_th
            
            print(f"Total number of generated events: {total_events}")
            print(f"Events with SNR> {self.snr_th}: {events_over_th}")
        result=save_theta_samples
        arrays_dict = {}
        for key in result[0].keys():
            arrays_dict[key] = np.concatenate([d[key] for d in result])
        return arrays_dict





    def generate_events_parallel(self, network):
        '''
        Generate catalog with N_goal events
        :param network: Network of detectors.
        
        :return: Samples extracted from the desired population model.
        :rtype: dictionary
        '''
        total_events = 0
        save_theta_samples = []
    
        while total_events < self.N_goal:  # Stop when you have N_goal events
            np.random.seed()
            chunk_size = min(self.N_chunk, self.N_goal - total_events)  # Ensure we don't generate more than N_goal events
            theta_samples = self.population_model_inj.sample_population(size=chunk_size)
            data_new = []
    
            for i in range(len(theta_samples['theta'])):
                new_entry = {}
                for key, value in theta_samples.items():
                    new_entry[key] = value[i]
                data_new.append(new_entry)
    
            in_time = time.time()
            print('\nDone. Total execution time: %.2fs' % (time.time() - in_time))

    
            save_theta_samples.append(theta_samples)
            total_events += chunk_size
    
            print(f"Total number of generated events: {total_events}")
    
        result = save_theta_samples
        arrays_dict = {}
        for key in result[0].keys():
            arrays_dict[key] = np.concatenate([d[key] for d in result])
        return arrays_dict




##########################################################################
##########################################################################


def main():
    """
    Main function to generate the population of events.
    Use the following command to run it:
    python generate_pdraw.py --config <config_file> --fout <output_folder>
    """
    
    in_time=time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='', type=str, required=True) # config file
    parser.add_argument("--fout", default='', type=str, required=True) # output folder 
    FLAGS = parser.parse_args()

    
    config = importlib.import_module(FLAGS.config, package=None)
    out_path=os.path.join(FLAGS.fout) # folder in which you want to save your pdraw

    try:
        print('Creating directory %s' %out_path)
        os.makedirs(out_path)

    except FileExistsError:
        print('Using directory %s for output' %out_path)

    #copy config file inside the fout directory
    config_file_path = FLAGS.config + '.py'
    shutil.copy(config_file_path, out_path)
    
    logname='logfile_inj.txt'

    logfile = os.path.join(out_path, logname) #out_path+'logfile.txt'
    myLog = utils.Logger(logfile)
    sys.stdout = myLog
    sys.stderr = myLog 
    
    ########################################################################################
    ########### Define model to generate population
    ########################################################################################
        
    POP_inj = config.p_draw
    POP_rec = config.p_pop
    


    #########################################################################
    ## Instantiate observations object and network
    #########################################################################

    myObs = Observations( population_model_inj=POP_inj, 
                         N_goal=config.N_goal,
                         out_dir=out_path,
                         snr_th=config.snr_th,
                         psds=config.psds,
                         names=config.net,
                         waveform=config.waveform,
                         fmin=config.fmin,
                         #fmax=config.fmax,
                         N_chunk=config.N_chunk,
                         CPUs=config.CPUs
         )
    network=myObs.generate_network()
    



    #########################################################################
    ## Generate pdraw and save in pdraw.h5
    #########################################################################

 
    
    #result=myObs.generate_events_parallel_snr_th(network)
    result=myObs.generate_events_parallel(network)
    save_data(os.path.join(out_path,'pdraw.h5'), result)

      
    


    #########################################################################
    ### Generate file to run gwfast and popfisher
    #########################################################################

    ### gwfast
    if config.idxf_gwfast is not None:
        idxf_gwfast=config.idxf_gwfast
    else:
        idxf_gwfast=len(result['Mc'])
    wf_model=repr(config.wf_model)
    lalargs=config.lalargs
    psds=config.psds
    net=config.net
    psds = ' '.join(f"'{item}'" for item in psds)
    net = ' '.join(f"'{item}'" for item in net)
    lalargs = ' '.join(f"'{item}'" for item in lalargs)
    snr_th=config.snr_th
    snr_th_FIM=config.snr_th_FIM

    #Directory for single event fisher matrices
    fout_gwfast=out_path+'SingleEventFishers'
    fname_obs=out_path+'pdraw.h5'

    try:
        print('Creating directory %s' %fout_gwfast)
        os.makedirs(fout_gwfast)

    except FileExistsError:
        print('Using directory %s for output' %fout_gwfast)


    # with open(out_path+'/scripts.txt', 'w') as file:
    #     file.write(f'Script to run gwfast:\n')
    # with open(out_path+'/scripts.txt', 'a') as file:
    #     script_line = f'python3 calculate_forecasts_from_catalog.py --fname_obs={fname_obs} --fout={fout_gwfast} --wf_model={wf_model} --lalargs {lalargs} --snr_th={snr_th_FIM} --batch_size={config.batch_size_gwfast} --npools={config.npools_gwfast} --fmin={config.fmin} --idx_in={config.idxin_gwfast} --idx_f={idxf_gwfast} --compute_fisher={config.compute_fisher} --return_snr_derivatives={config.return_snr_derivatives} --return_derivatives={config.return_derivatives} --return_all={config.return_all} --net {net} --psds {psds} --mpi={config.mpi} --duty_factor={config.duty_factor} --concatenate={config.concatenate_gwfast} --rot={config.rot} --resume_run={config.resume_run}'# --use_cogwheel_params={config.use_cogwheel_params} --jit_fisher={config.jit_fisher} --fix_fgrid={config.fix_fgrid} --fix_fgrid_df={config.fix_fgrid_df} --compute_imrsplit={config.compute_imrsplit} #--fname_evSNRs={config.fname_evSNRs_gwfast}
    #     if config.fmax is not None:
    #         script_line += f' --fmax={config.fmax}'
            
    #     if config.netfile is not None:
    #         script_line += f' --netfile={config.netfile}'

    #     if config.params_fix:
    #         params_fix=' '.join(config.params_fix)
    #         script_line += f' --params_fix {params_fix}'
            
    #     if config.seeds:
    #         seeds=' '.join(map(str, config.seeds))
    #         script_line += f' --seeds {seeds}'

    
    #     script_line += '\n'
    #     file.write(script_line)


    sh_path = os.path.join(out_path, 'run_gwfast.sh')
    with open(sh_path, 'w') as file:
        file.write('#!/bin/bash\n\n')
        file.write('# Auto-generated script to run gwfast\n\n')
    
        # === Variabili di configurazione ===
        file.write(f'fname_obs="{fname_obs}"\n')
        file.write(f'fout="{fout_gwfast}"\n')
        file.write(f'wf_model="{wf_model}"\n')
        file.write(f"lalargs={lalargs}\n")
        file.write(f"net={net}\n")
        file.write(f"psds={psds}\n")
        file.write(f"snr_th={snr_th}\n")
        file.write(f"snr_th_FIM={snr_th_FIM}\n")
        file.write(f"batch_size={config.batch_size_gwfast}\n")
        file.write(f"npools={config.npools_gwfast}\n")
        file.write(f"fmin={config.fmin}\n")
        file.write(f"idx_in={config.idxin_gwfast}\n")
        file.write(f"idx_f={idxf_gwfast}\n")
        file.write(f"compute_fisher={config.compute_fisher}\n")
        file.write(f"return_snr_derivatives={config.return_snr_derivatives}\n")
        file.write(f"return_derivatives={config.return_derivatives}\n")
        file.write(f"return_all={config.return_all}\n")
        file.write(f"mpi={config.mpi}\n")
        file.write(f"duty_factor={config.duty_factor}\n")
        file.write(f"concatenate={config.concatenate_gwfast}\n")
        file.write(f"rot={config.rot}\n")
        #file.write(f"use_cogwheel_params={config.use_cogwheel_params}\n")
        #file.write(f"jit_fisher={config.jit_fisher}\n")
        #file.write(f"fix_fgrid={config.fix_fgrid}\n")
        #file.write(f"fix_fgrid_df={config.fix_fgrid_df}\n")
        file.write(f"resume_run={config.resume_run}\n")
        #file.write(f"compute_imrsplit={config.compute_imrsplit}\n")
    
        if config.fmax is not None:
            file.write(f"fmax={config.fmax}\n")
        if config.netfile is not None:
            file.write(f"netfile={config.netfile}\n")
        if config.params_fix:
            params_fix = ' '.join(config.params_fix)
            file.write(f"params_fix='{params_fix}'\n")
        if config.seeds:
            seeds = ' '.join(map(str, config.seeds))
            file.write(f"seeds='{seeds}'\n")
    
        # === Exe commands ===
        file.write(f'\n\ncd {PARENT_DIR}/run\n\n')
        file.write('\npython3 calculate_forecasts_from_catalog.py \\\n')
        file.write(f'  --fname_obs={fname_obs} \\\n')
        file.write(f'  --fout={fout_gwfast} \\\n')
        file.write(f'  --wf_model={wf_model} \\\n')
        file.write(f'  --lalargs {lalargs} \\\n')
        file.write(f'  --snr_th={snr_th_FIM} \\\n')
        file.write(f'  --batch_size={config.batch_size_gwfast} \\\n')
        file.write(f'  --npools={config.npools_gwfast} \\\n')
        file.write(f'  --fmin={config.fmin} \\\n')
        file.write(f'  --idx_in={config.idxin_gwfast} \\\n')
        file.write(f'  --idx_f={idxf_gwfast} \\\n')
        file.write(f'  --compute_fisher={config.compute_fisher} \\\n')
        file.write(f'  --return_snr_derivatives={config.return_snr_derivatives} \\\n')
        file.write(f'  --return_derivatives={config.return_derivatives} \\\n')
        file.write(f'  --return_all={config.return_all} \\\n')
        file.write(f'  --net {net} \\\n')
        file.write(f'  --psds {psds} \\\n')
        file.write(f'  --mpi={config.mpi} \\\n')
        file.write(f'  --duty_factor={config.duty_factor} \\\n')
        file.write(f'  --concatenate={config.concatenate_gwfast} \\\n')
        file.write(f'  --rot={config.rot} \\\n')
        #file.write(f'  --use_cogwheel_params={config.use_cogwheel_params} \\\n')
        #file.write(f'  --jit_fisher={config.jit_fisher} \\\n')
        #file.write(f'  --fix_fgrid={config.fix_fgrid} \\\n')
        #file.write(f'  --fix_fgrid_df={config.fix_fgrid_df} \\\n')
        file.write(f'  --resume_run={config.resume_run} \\\n')
        #file.write(f'  --compute_imrsplit={config.compute_imrsplit} \\\n')
        
        if config.fmax is not None:
            file.write(f'  --fmax={config.fmax} \\\n')
        if config.netfile is not None:
            file.write(f'  --netfile={config.netfile} \\\n')
        if config.params_fix:
            file.write(f'  --params_fix {params_fix} \\\n')
        if config.seeds:
            file.write(f'  --seeds {seeds} \\\n')
        

        file.seek(file.tell() - 2)
        file.write('\n')

    os.chmod(sh_path, 0o755)



    #### gwfast.population

    if config.idxf_popfisher is not None:
        idxf_popfisher=config.idxf_popfisher
    else:
        idxf_popfisher=len(result['Mc'])
    
    mass_model=repr(config.mass_model)
    rate_model=repr(config.rate_model)
    spin_model=repr(config.spin_model)
    POPmodel=repr(config.pop_model_rec)
    fout_popfisher=out_path+'PopFisher'
    fname_evSNRs=fout_gwfast+f'/snrs_{config.idxin_popfisher}_to_{idxf_popfisher}.txt' 
    fname_evFIMs=fout_gwfast+f'/fishers_{config.idxin_popfisher}_to_{idxf_popfisher}.npy'
    fname_evSNRders=fout_gwfast+f'/all_derivatives_SNR_{config.idxin_popfisher}_to_{idxf_popfisher}.hdf5'
    try:
        print('Creating directory %s' %fout_popfisher)
        os.makedirs(fout_popfisher)
    except FileExistsError:
        print('Using directory %s for output' %fout_popfisher)
    


    # with open(out_path+'/scripts.txt', 'a') as file:
    #     file.write(f'\n\n\n\n Script to run popfisher: \n\n')
    #     script_line = f'python3 calculate_hyperpar_derivatives_from_catalog.py --fname_obs={fname_obs} --fname_evSNRs={fname_evSNRs} --fname_evFIMs={fname_evFIMs} --fname_evSNRders={fname_evSNRders} --fout={fout_popfisher} --mass_model={mass_model} --POPmodel={POPmodel}'
    
    #     if config.mass_model_params_names:
    #         mass_model_params_names = ' '.join(config.mass_model_params_names)
    #         script_line += f' --mass_model_params_names {mass_model_params_names}'
    
    #     if config.mass_model_params_values:
    #         mass_model_params_values = ' '.join(map(str, config.mass_model_params_values))
    #         script_line += f' --mass_model_params_values {mass_model_params_values}'
    
    #     script_line += f' --rate_model={rate_model}'
    
    #     if config.rate_model_params_names:
    #         rate_model_params_names = ' '.join(config.rate_model_params_names)
    #         script_line += f' --rate_model_params_names {rate_model_params_names}'
    
    #     if config.rate_model_params_values:
    #         rate_model_params_values = ' '.join(map(str, config.rate_model_params_values))
    #         script_line += f' --rate_model_params_values {rate_model_params_values}'
    
    #     script_line += f' --spin_model={spin_model}'
    
    #     if config.spin_model_params_names:
    #         spin_model_params_names = ' '.join(config.spin_model_params_names)
    #         script_line += f' --spin_model_params_names {spin_model_params_names}'
    
    #     if config.spin_model_params_values:
    #         spin_model_params_values = ' '.join(map(str, config.spin_model_params_values))
    #         script_line += f' --spin_model_params_values {spin_model_params_values}'
    
    #     script_line += f' --POPmodel={POPmodel}'
    
    #     if config.prior_limits_params_names:
    #         prior_limits_params_names = ' '.join(config.prior_limits_params_names)
    #         script_line += f' --prior_limits_params_names {prior_limits_params_names}'
    
    #     if config.prior_limits_params_values:
    #         prior_limits_params_values = ' '.join(map(str, config.prior_limits_params_values))
    #         script_line += f' --prior_limits_params_values {prior_limits_params_values}'
    
    #     script_line += f' --batch_size={config.batch_size_popfisher} --npools={config.npools_popfisher} --snr_th={snr_th} --snr_th_FIM={snr_th_FIM} --idx_in={config.idxin_popfisher} --idx_f={idxf_popfisher} --mpi={config.mpi_popfisher} --concatenate={config.concatenate_popfisher} --FIM_rotated={config.FIM_rotated} --Pdet_sigma={config.Pdet_sigma} --resume_run={config.resume_run_popfisher}\n'
    
    #     file.write(script_line)  



    # Create or overwrite the shell script
    sh_path_pop=os.path.join(out_path, 'run_popfisher.sh')
    with open(sh_path_pop, 'w') as file:
        file.write("#!/bin/bash\n\n")
        file.write("# Auto-generated script to run popfisher with defined parameters\n\n")
    
        # Define variables
        file.write(f"fname_obs={fname_obs} \\\n")
        file.write(f"fname_evSNRs={fname_evSNRs}\\\n")
        file.write(f"fname_evFIMs={fname_evFIMs}\\\n")
        file.write(f"fname_evSNRders={fname_evSNRders}\\\n")
        file.write(f"fout={fout_popfisher}\\\n")
        file.write(f"mass_model={mass_model}\n")
        file.write(f"POPmodel={POPmodel}\n")
        file.write(f"rate_model={rate_model}\n")
        file.write(f"spin_model={spin_model}\n")
    
        if config.mass_model_params_names:
            mass_model_params_names = ' '.join(config.mass_model_params_names)
            file.write(f"mass_model_params_names=\"{mass_model_params_names}\"\n")
    
        if config.mass_model_params_values:
            mass_model_params_values = ' '.join(map(str, config.mass_model_params_values))
            file.write(f"mass_model_params_values=\"{mass_model_params_values}\"\n")
    
        if config.rate_model_params_names:
            rate_model_params_names = ' '.join(config.rate_model_params_names)
            file.write(f"rate_model_params_names=\"{rate_model_params_names}\"\n")
    
        if config.rate_model_params_values:
            rate_model_params_values = ' '.join(map(str, config.rate_model_params_values))
            file.write(f"rate_model_params_values=\"{rate_model_params_values}\"\n")
    
        if config.spin_model_params_names:
            spin_model_params_names = ' '.join(config.spin_model_params_names)
            file.write(f"spin_model_params_names=\"{spin_model_params_names}\"\n")
    
        if config.spin_model_params_values:
            spin_model_params_values = ' '.join(map(str, config.spin_model_params_values))
            file.write(f"spin_model_params_values=\"{spin_model_params_values}\"\n")
    
        if config.prior_limits_params_names:
            prior_limits_params_names = ' '.join(config.prior_limits_params_names)
            file.write(f"prior_limits_params_names=\"{prior_limits_params_names}\"\n")
    
        if config.prior_limits_params_values:
            prior_limits_params_values = ' '.join(map(str, config.prior_limits_params_values))
            file.write(f"prior_limits_params_values=\"{prior_limits_params_values}\"\n")
    
        # Write remaining fixed parameters
        file.write(f"batch_size={config.batch_size_popfisher}\n")
        file.write(f"npools={config.npools_popfisher}\n")
        file.write(f"snr_th={snr_th}\n")
        file.write(f"snr_th_FIM={snr_th_FIM}\n")
        file.write(f"idx_in={config.idxin_popfisher}\n")
        file.write(f"idx_f={idxf_popfisher}\n")
        file.write(f"mpi={config.mpi_popfisher}\n")
        file.write(f"concatenate={config.concatenate_popfisher}\n")
        file.write(f"FIM_rotated={config.FIM_rotated}\n")
        file.write(f"Pdet_sigma={config.Pdet_sigma}\n")
        file.write(f"resume_run={config.resume_run_popfisher}\n")
    
        # Final execution command
 
        file.write(f'\n\ncd {SCRIPT_DIR}/population/run\n\n')
        file.write("python3 calculate_hyperpar_derivatives_from_catalog.py \\\n")
        file.write(f"  --fname_obs={fname_obs} \\\n")
        file.write(f"  --fname_evSNRs={fname_evSNRs} \\\n")
        file.write(f"  --fname_evFIMs={fname_evFIMs} \\\n")
        file.write(f"  --fname_evSNRders={fname_evSNRders} \\\n")
        file.write(f"  --fout={fout_popfisher} \\\n")
        file.write(f"  --mass_model={mass_model} \\\n")
        file.write(f"  --POPmodel={POPmodel} \\\n")
        
        if config.mass_model_params_names:
            file.write(f"  --mass_model_params_names {' '.join(config.mass_model_params_names)} \\\n")
        if config.mass_model_params_values:
            file.write(f"  --mass_model_params_values {' '.join(map(str, config.mass_model_params_values))} \\\n")
        
        file.write(f"  --rate_model={rate_model} \\\n")
        
        if config.rate_model_params_names:
            file.write(f"  --rate_model_params_names {' '.join(config.rate_model_params_names)} \\\n")
        if config.rate_model_params_values:
            file.write(f"  --rate_model_params_values {' '.join(map(str, config.rate_model_params_values))} \\\n")
        
        file.write(f"  --spin_model={spin_model} \\\n")
        
        if config.spin_model_params_names:
            file.write(f"  --spin_model_params_names {' '.join(config.spin_model_params_names)} \\\n")
        if config.spin_model_params_values:
            file.write(f"  --spin_model_params_values {' '.join(map(str, config.spin_model_params_values))} \\\n")
        
        if config.prior_limits_params_names:
            file.write(f"  --prior_limits_params_names {' '.join(config.prior_limits_params_names)} \\\n")
        if config.prior_limits_params_values:
            file.write(f"  --prior_limits_params_values {' '.join(map(str, config.prior_limits_params_values))} \\\n")
        
        file.write(f"  --batch_size={config.batch_size_popfisher} \\\n")
        file.write(f"  --npools={config.npools_popfisher} \\\n")
        file.write(f"  --snr_th={snr_th} \\\n")
        file.write(f"  --snr_th_FIM={snr_th_FIM} \\\n")
        file.write(f"  --idx_in={config.idxin_popfisher} \\\n")
        file.write(f"  --idx_f={idxf_popfisher} \\\n")
        file.write(f"  --mpi={config.mpi_popfisher} \\\n")
        file.write(f"  --concatenate={config.concatenate_popfisher} \\\n")
        file.write(f"  --FIM_rotated={config.FIM_rotated} \\\n")
        file.write(f"  --Pdet_sigma={config.Pdet_sigma} \\\n")
        file.write(f"  --resume_run={config.resume_run_popfisher}\n")

        os.chmod(sh_path_pop, 0o755)



    



    ##########################################################################
    ##########################################################################
    
    print('\nDone. Total execution time: %.2fs' % (time.time() - in_time))
    print(f'\nScript to run gwfast in {sh_path}')
    print(f'\nScript to run gwfast.population in {sh_path_pop}')
    print('PARENT_DIR,',PARENT_DIR)
    print('SCRIPT_DIR,',SCRIPT_DIR)
     

    
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    myLog.close()

if __name__ == '__main__':
    main()





