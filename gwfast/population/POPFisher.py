import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt


from POPmodels import *
from POPutils import open_h5py,open_catalog,load_SNRders,print_diagonal_elements




# Module to compute the population Fisher Gamma_Lambda 
# (see Eq. (21) of [J. Gair et al., MNRAS, Volume 519, Issue 2, February 2023, Pages 2736–2753]).




class FisherMatrixCalculator(object):
    r"""
    Class to compute the population Fisher matrix Gamma_Lambda.
    """
    def __init__(self, 
                 population_model_injection,
                 population_model_recovery,
                 snr_th,
                 path_pdraw,
                 path_single_FIMs,
                 path_pop_terms,
                 idx_i,idx_f):
        r"""
        Parameters:
        :param population_model_injection: injection distribution p_draw(\theta).
        :param population_model_recovery: true population model p_pop(\theta|\lambda).
        :param float snr_th: threshold value for the SNR to consider the event detectable.
        :param str path_pdraw: path to the file containing the samples for p_draw(\theta).
        :param str path_single_FIMs: path to the file containing the single Fisher matrices.
        :param str path_pop_terms: path to the file containing the integrands provided by the module gwfast.population.calculate_hyperpar_derivatives_from_catalog
        :param int idx_i: index of the first sample.
        :param int idx_f: index of the last sample.
        """

        self.POP_rec=population_model_recovery
        self.POP_inj=population_model_injection 
        self.parameters = self.POP_rec.par_list #source parameter list
        self.hyperparameters = list(self.POP_rec.hyperpar_dict.keys()) #hyperparameters list
        self.N_par=len(self.parameters) #number of source parameters 
        self.N_hyperpar = len(self.hyperparameters) #number of hyperparameters
        self.hyperpar_dict=self.POP_rec.hyperpar_dict #dictionary of hyperparameters with the corresponding injected values

        self.path_pdraw=path_pdraw
        self.path_single_FIMs=path_single_FIMs
        self.path_pop_terms=path_pop_terms
        self.idx_i=idx_i
        self.idx_f=idx_f

        #load pdraw samples, snrs, derivatives of snr, single event fisher matrices, and the integrands provided by the module gwfast.population.calculate_hyperpar_derivatives_from_catalog
        N_samp,theta_samples,snr,der_snr,FIMs,termI_der,termI_hess,termII,termIII,termIV,termV=open_catalog(path_pdraw,path_single_FIMs,idx_i,idx_f,popterms=True,path_pop_terms=path_pop_terms)
        self.pop_inj=self.POP_inj.pop_function(theta_samples,uselog=False)
        self.pop_rec=self.POP_rec.pop_function(theta_samples,uselog=False)
        self.N_samp=N_samp #number of MC samples
        self.theta_samples=theta_samples #MC samples from pdraw
        self.snr=snr
        self.snr_th=snr_th
        self.der_snr=der_snr
        self.FIMs=FIMs
        self.termI_der=termI_der
        self.termI_hess=termI_hess
        self.termII=termII
        self.termIII=termIII
        self.termIV=termIV
        self.termV=termV
        
    

    def _pdet_theta(self,sigma=1):
        r"""
        Function to calculate the detection probability:

            .. math::
            p_\mathrm{det}(\vec\theta) =
            =\frac{1}{2}\text{erfc}\Bigg(\frac{\rho_\mathrm{th}-\rho_\mathrm{opt}(\theta)}{\sqrt{2}}\Bigg)
            
        Parameters:
        :return array of shape (len(N_samp),)
        """
        return 0.5* erfc((self.snr_th-self.snr)/(np.sqrt(2)*sigma))
    


    def _pdet_lambda(self, pdet_theta): 
        r"""
        Function to calculate the selection function of the hyperparameters.

            ..math::
            P_\mathrm{det}(\vec\lambda) =\int P_{det}(\vec\theta)p(\vec\theta|\vec\lambda)d\vec\theta=
            =\frac{1}{N_{samp}}\sum_{k=0}^{N_{samp}} \frac{1}{2}\text{erfc}\Bigg(\frac{\rho_\mathrm{th}-\rho_\mathrm{opt}(\theta_k)}{\sqrt{2}}\Bigg), 
            where \theta_{k}\sim p(\vec\theta|\vec\lambda)
        
        Parameters:
        :param array pdet_theta: detection probability.
        :return float: selection effects.
        """

        return (1 / self.N_samp)*np.sum(pdet_theta*self.pop_rec/self.pop_inj)




    def _A_integrand(self,pdet_theta,pdet_lambda):
        r"""
        Function to compute the integrand A(\theta)=
        ..math::
        A(\bar\theta)=-\dfrac{\partial^2 \ln [p_{\rm pop}(\bar\theta  |  \lambda)p_{\rm det}^{-1}(\lambda)]}{\partial\lambda_i \partial\lambda_j}\Bigg|_{\lambda=\bar\lambda}=
        \dfrac{1}{p_\mathrm{det}(\bar\lambda)}\dfrac{\partial^2 p_\mathrm{det}(\lambda)}{\partial \lambda^i\partial \lambda^j }\Bigg|_{\lambda=\bar\lambda}
        -\dfrac{1}{p_\mathrm{det}^2(\bar\lambda)}\dfrac{\partial p_\mathrm{det}(\lambda)}{\partial \lambda^i}\dfrac{\partial p_\mathrm{det}(\lambda)}{\partial \lambda^j}\Bigg|_{\lambda=\bar\lambda}-\dfrac{\partial^2 \ln
        p_\mathrm{pop}(\bar\theta|\lambda)}{\partial \lambda^i\partial \lambda^j }\Bigg|_{\lambda=\bar\lambda}

        Parameters:

        :param array pdet_theta: detection probability.
        :param float pdet_lambda: selection effects.
        :return ndarray: matrix of shape (N_hyper,N_hyper,N_samp)
        """
        div=self.pop_rec/self.pop_inj
        
        # compute the first derivative of Pdet(lambda) w.r.t. lambda
        arg_dPdet_lambda=pdet_theta * self.termI_der *div
        dPdet_dlambda_i =(1 / self.N_samp)*np.sum(arg_dPdet_lambda,axis=-1) 
        dPdet_dlambda_i =dPdet_dlambda_i[:,np.newaxis]
        
        # compute the second derivative of Pdet(lambda) w.r.t. lambda
        arg_d2Pdet_lambda=pdet_theta*div*(self.termI_hess+np.einsum('ik, jk -> ijk', self.termI_der,self.termI_der))
        d2Pdet_dlambda_i_dPdet_dlambda_j=(1 / self.N_samp)*np.sum(arg_d2Pdet_lambda,axis=-1)
        
        #A_det
        Adet=(pdet_lambda*d2Pdet_dlambda_i_dPdet_dlambda_j-dPdet_dlambda_i@dPdet_dlambda_i.T)/(pdet_lambda)**2
        Adet=Adet[:, :, np.newaxis]
        
        #Apop
        Apop=-self.termI_hess
        A=Apop+Adet
        return A



    def _der_pdet(self):
        r"""
        Function to compute the derivative of the error function in p_{det}(\theta) w.r.t. the SNR"
        """
        A=np.sqrt(2*np.pi)**-1*np.exp(-0.5*(self.snr_th-self.rho)**2)
        der_pdet=A*self.der_snr
        return der_pdet



    def regularize_MC_integrals(self,quantile_range, arg,check_convergence=False,MC=False,**kwargs):
        r"""    
        Function to regularize Monte Carlo integrals by removing outliers based on a given quantile range.

        Parameters:
        :param float quantile_range: Quantile range to identify and remove outliers.
        :param ndarray arg: Array of shape (N_hyper, N_hyper, N_samp) containing the integrands.
        :param bool check_convergence: If True, plot convergence of integrals.
        :param bool MC: If True, plot effective number of samples.
        
        Returns:
        :return ndarray: Array of shape (N_hyper, N_hyper) containing the regularized integrals.
        """
        
        print('Chosen quantile range:', quantile_range)
        low, up = np.zeros(self.N_hyperpar), np.zeros(self.N_hyperpar)
        mask = []
        index = []
        for i in range(self.N_hyperpar):
            low[i], up[i] = np.quantile(arg[i][i], quantile_range)
            mask.append(np.logical_and(arg[i][i] > low[i], arg[i][i] < up[i]))
            index.append(np.where(mask[i] == False)[0])  
        unique_index = np.array(list(set(np.hstack(index)))) 
        print('Removed elements:', len(unique_index))
        global_mask = np.ones(arg.shape[-1], dtype=bool)
        for m in mask:
            global_mask &= m
            arg_filtered = arg[:, :, global_mask]
        N_samp_filtered = global_mask.sum()
        print('\n MC integral after regularization')
        I = (1 / N_samp_filtered) * np.sum(arg_filtered, axis=-1)
        print_diagonal_elements(I,self.POP_rec)


        
        if check_convergence==True:
            r"""
            Plots the convergence of the Monte Carlo integrals. 
            The first column displays the non-regularized integrals, while the second column shows the regularized integrals, 
            where outliers have been removed based on the specified quantile range.
            """
            Nsamp=np.arange(0,self.N_samp)
            Gamma_notconv=[]
            for i in range(self.N_hyperpar):
                Gamma_notconv.append(np.cumsum(arg[i][i])/Nsamp)
        
            Gamma_vs_Nsamp=(1/Nsamp[global_mask])*np.cumsum(arg_filtered,axis=-1)
            
            Nrows=self.N_hyperpar
            Ncols=2
            ylabels=list(self.POP_rec.hyperpar_dict.keys())
            
            Gamma_vs_Nsamp=(1/Nsamp[global_mask])*np.cumsum(arg_filtered,axis=-1)
            
            colors=['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17']
            titles=['Non regularized','Regularized']
            fig, axs = plt.subplots(Nrows,Ncols, figsize=(10, 25),**kwargs)
            plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)
                        
                        
            for i in range(Nrows):
                axs[i,1].plot(Nsamp,Gamma_notconv[i],lw=1.5,color=colors[i],label='Non regularized',ls='dashed')
                for j in range(Ncols):
                    axs[i,j].minorticks_on()
                    axs[i,j].grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
                    axs[i,j].yaxis.set_ticks_position('both')
                    axs[i,j].xaxis.set_ticks_position('both')
                    axs[i,j].set_xlabel('$N_\mathrm{samples}$')
                    axs[0,j].set_title(titles[j])
                    axs[i,j].set_ylabel(ylabels[i])
                    axs[i,0].plot(Nsamp,Gamma_notconv[i],lw=1.5,color=colors[i],ls='dashed')
                    axs[i,1].plot(Nsamp[global_mask],Gamma_vs_Nsamp[i][i],color=colors[i], label=None)

            axs[0,1].legend()
            plt.show()
    
        if MC==True:
            r"""
            Plots the effective number of samples for the regularized integrals.
            """
            mu=1/N_samp_filtered*np.cumsum(arg[:,:,global_mask],axis=-1)
            sigma_squared=1/N_samp_filtered**2*np.cumsum(arg[:,:,global_mask]**2,axis=-1)-1/N_samp_filtered*(mu)**2
                        
            Neff=mu**2/sigma_squared   
            ylabel=list(self.POP_rec.hyperpar_dict.keys())
            colors=['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17']

            for i in range(self.N_hyperpar):
                #if dim=='3d':
                if i<10:
                    plt.plot(Neff[i][i],label=ylabel[i],color=colors[i])
                else:
                    plt.plot(Neff[i][i],linestyle='--',label=ylabel[i])
            plt.legend(loc=[1.05,0.2])
            plt.xlabel('N')
            plt.yscale('log')
            plt.ylabel('$\mu^2/\sigma^2$')
            #plt.title('N effective')
            plt.show()
            plt.close()
        return I
        


    def compute_POPFisher(self,save_MC_args=True):
        r"""
        Function to compute the population Fisher matrix Gamma_Lambda.
        Parameters:
        :param bool save_MC_args: if True, return the integrands.
        :return ndarray: array of shape (N_hyper,N_hyper) containing the population Fisher matrix.
        """

        pdet_theta=self._pdet_theta(sigma=1)
        pdet_lambda=self._pdet_lambda(pdet_theta)
        div=self.pop_rec/self.pop_inj
        


        ###############################################################
        # Compute terms of the population Fisher
        ###############################################################
        print('\n \n ...Before MC integral regularization...\n \n')

        A=self._A_integrand(pdet_theta,pdet_lambda)
        arg1=A*pdet_theta/pdet_lambda*div
        Gamma_I=self.N_samp**-1*np.sum(arg1,axis=-1)
        #Gamma_I=self.N_samp**-1*A/pdet_lambda
        print('...First term computed...')
        
        B=0.5*self.termII
        arg2=B*pdet_theta/pdet_lambda*div
        Gamma_II=self.N_samp**-1*np.sum(arg2,axis=-1)
        print('...Second  term computed...')
        
        C=-0.5*self.termIII
        arg3=C/pdet_lambda*div
        Gamma_III=self.N_samp**-1*np.sum(arg3,axis=-1)
        print('...Third term computed...')
        
        D=-self.termIV
        arg4=D/pdet_lambda*div
        Gamma_IV=self.N_samp**-1*np.sum(arg4,axis=-1)
        print('...Fourth term computed ...')
        
        E=-0.5*self.termV
        arg5=E*pdet_theta/pdet_lambda*div
        Gamma_V=self.N_samp**-1*np.sum(arg5,axis=-1)
        print('...Fifth term computed...')
        if save_MC_args==False:
            return Gamma_I,Gamma_II,Gamma_III,Gamma_IV,Gamma_V
        elif save_MC_args==True:
            return Gamma_I,Gamma_II,Gamma_III,Gamma_IV,Gamma_V,arg1,arg2,arg3,arg4,arg5










