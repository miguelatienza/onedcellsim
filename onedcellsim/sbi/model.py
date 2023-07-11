"""
 Module for training an SBI posterior for a simulator. The simulator is defined in the Simulator class. This class allows for the user to specify which parameters are variable and which are fixed, and to specify the prior ranges for the variable parameters. 
 """

import pandas as pd
import numpy as np
import os
from sbi import utils as utils
#from sbi import analysis as analysis
from sbi.inference import SNPE
import torch
from ..simulators.multistability.simulator import Simulator 
from ..compression import compress
from tqdm import tqdm

SIMDIR = os.path.join(os.path.dirname(__file__), '../simulations')
VARNAMES = ['zetaf', 'zetab', 'zetac', 'kf', 'kb', 'Lf', 'Lb', 'vrf', 'vrb', 'xf', 'xb', 'xc', 'vf', 'vb']


class Model:
    """
    Class for training an SBI posterior for a simulator. The simulator is defined in the Simulator class. This class allows for the user to specify which parameters are variable and which are fixed, and to specify the prior ranges for the variable parameters. 
    """
    def __init__(self, simulator, parameter_dict=None, variable_parameter_names=None, prior_min=None, prior_max=None, compressor=None, working_dir='./', device='cpu', default_parameter_values=None, t_step=2*3600, t_max=7*3600):
        """
        Constructor method 

        Args:
            
            simulator (Simulator): Simulator class for the model. it must contain a method called simulate that takes a parameter set and returns a simulation output. It should contain an attribute called var_names. It should also contain a set of default parameters if the user does not specify any.
            
            variable_parameter_names (_type_): Names of the parameters that are variable. Must be a subset of the parameters in the simulator.
            
            prior_min (_type_): The minimum value of the prior for the variable parameters.
            
            prior_max (_type_): The maximum value of the prior for the variable parameters.
            
            compressor (_type_, optional): The compressor to use for training the SBI posterior. The simulations are compressed down to a lower dimensional feature space by the compressor. The lower-dimensional output is used for training the posterior. If not None, it should contain the method compress, and should be compatible with the output of the simulator. Defaults to None.

            working_dir (_type_, optional): The directory to save the SBI posterior. Defaults to './'.

            device (_type_, optional): The device to use for training the SBI posterior. Defaults to 'cpu'.

            default_parameter_values (_type_, optional): The default parameter values to use if the user does not specify any. Defaults to None.

        Raises:
            ValueError: _description_
        """
        self.prior_min = prior_min
        self.prior_max = prior_max
        self.working_dir = working_dir
        self.data_dir = os.path.join(self.working_dir, 'data')
        self.device=device
        self.compressor = compressor
        self.simulator = simulator
        self.variable_parameter_names = variable_parameter_names
        self.parameter_names = simulator.parameter_names
        self.full_default_parameter_values = np.array(simulator.default_parameter_values, dtype='float32')
        self.t_step = t_step
        self.t_max = t_max
        self.parameter_dict = parameter_dict
        
        
        # Check that the variable parameter are valid
        self.check_parameters()
        
        self.build_prior()

    
    def check_parameters_old(self):
        #warn that this is now deprecated
        print('This method is deprecated, use parameter_dict instead')

        if not set(self.variable_parameter_names).issubset(set(self.simulator.parameter_names)):
            raise ValueError('The variable parameter names must be a subset of the simulator parameter names')

        #now find the indices corresponding to the variable parameters
        self.variable_parameter_indices = np.array(
                [self.parameter_names.index(var_name) for var_name in self.variable_parameter_names])

        # If default parameters are not specified, use the simulator default parameters
        if self.default_parameter_values is None:
            self.default_parameter_values = np.array(self.simulator.default_parameter_values)[self.variable_parameter_indices]

            #Make sure that the default parameter values are between the prior min and max
            if not all([self.prior_min[i]<self.default_parameter_values[i]<self.prior_max[i] for i in range(len(self.prior_min))]): 
                raise ValueError('The default parameter values must be between the prior min and max')
        # If default parameters are specified, check that they are valid
        else:
            self.default_parameter_values = np.array(self.default_parameter_values)

        # If the compressor is not specified, use the default compressor
        if self.compressor is None:
            self.compressor = lambda x: torch.tensor(x)
        # If the compressor is specified, check that it is valid

    
    def check_parameters(self):
        
        if self.parameter_dict is None:
            self.check_parameters_old()
            return
        
        parameter_names = list(self.parameter_dict.keys())
        
        if not set(parameter_names).issubset(set(self.simulator.parameter_names)):
            
            raise ValueError('The variable parameter names must be a subset of the simulator parameter names')
       
        self.variable_parameter_names = [var_name for var_name in parameter_names if self.parameter_dict[var_name]['type']=='variable']
        self.latent_parameter_names = [var_name for var_name in parameter_names if self.parameter_dict[var_name]['type']=='latent']
        self.fixed_parameter_names = [var_name for var_name in parameter_names if self.parameter_dict[var_name]['type']=='fixed']

        self.variable_parameter_indices = np.array(
                [self.parameter_names.index(var_name) for var_name in parameter_names if self.parameter_dict[var_name]['type']=='variable'])
        self.latent_parameter_indices = np.array(
                [self.parameter_names.index(var_name) for var_name in parameter_names if self.parameter_dict[var_name]['type']=='latent'])
        self.fixed_parameter_indices = np.array(
                [self.parameter_names.index(var_name) for var_name in parameter_names if self.parameter_dict[var_name]['type']=='fixed'])
        
        self.prior_min_full = np.array([self.parameter_dict[var_name]['prior'][1] if self.parameter_dict[var_name]['type']=='fixed' else self.parameter_dict[var_name]['prior'][0] for var_name in parameter_names])

        self.prior_max_full = np.array([self.parameter_dict[var_name]['prior'][1] if self.parameter_dict[var_name]['type']=='fixed' else self.parameter_dict[var_name]['prior'][2] for var_name in parameter_names])

        self.prior_min = np.array([self.parameter_dict[var_name]['prior'][0] for var_name in parameter_names if self.parameter_dict[var_name]['type']=='variable'])

        self.prior_max = np.array([self.parameter_dict[var_name]['prior'][2] for var_name in parameter_names if self.parameter_dict[var_name]['type']=='variable'])

        self.full_default_parameter_values = np.array([self.parameter_dict[var_name]['prior'][1] for var_name in parameter_names])

    def build_prior(self):
        
        # Check that the prior min and max are valid
        if not all([self.prior_min[i]<=self.prior_max[i] for i in range(len(self.prior_min))]):
            raise ValueError('The prior min must be less than or equal to the prior max')

        self.prior_full = utils.torchutils.BoxUniform(
        low=torch.as_tensor(self.prior_min_full), high=torch.as_tensor(self.prior_max_full))
        
        self.prior = utils.torchutils.BoxUniform(
        low=torch.as_tensor(self.prior_min), high=torch.as_tensor(self.prior_max), device=self.device)
   
        return
    
    def build_full_parameter_set(self, variable_parameter_set):

        if variable_parameter_set.ndim==1:
            variable_parameter_set = variable_parameter_set.reshape(1, variable_parameter_set.size)
        
        n_sets = variable_parameter_set.shape[0]

        parameter_set = np.repeat(self.full_default_parameter_values[np.newaxis, :], n_sets, axis=0)
        parameter_set[:, self.variable_parameter_indices] = variable_parameter_set

        if len(self.latent_parameter_indices)>0:
            latent_parameter_set = self.sample(n_sets)
           
            parameter_set[:, self.latent_parameter_indices] = latent_parameter_set[:, self.latent_parameter_indices]

        #parameter_set = torch.tensor(parameter_set, device=device)

        return parameter_set

    def sample(self, n_sets, inference=False):

        if inference:
            prior = self.prior_inference
            if 'restricted_prior' in dir(self):
                prior=self.restricted_prior
        else:
            prior = self.prior_full
        

        parameter_set = np.array(prior.sample((n_sets,)).cpu(), dtype='float32')

        return parameter_set
    
    def simulation_wrapper(self, parameter_set=None, n_sims=1, progress_bar=False):
       
        if parameter_set is None:
            parameter_set=self.sample(n_sims)
        
        parameter_set = np.array(parameter_set)
        
        if parameter_set.ndim==0:
            parameter_set = np.array([parameter_set,])
        
        elif (parameter_set.ndim==1) and n_sims==1:
            pass

        elif (parameter_set.ndim==2) & (parameter_set.shape[0]>1) & (n_sims==1):
            #Infer the number of parameters from the shape of the parameter set
            n_sims = parameter_set.shape[0]

        elif (parameter_set.ndim==1) and n_sims>1:
            #create a parameter set which is repeated n_sims times
            parameter_set = np.repeat(parameter_set.reshape(1, parameter_set.size), n_sims, axis=0)
        
        else:
            raise Exception('The parameter shape is not usable')
        
        ##Make it complete by adding the variable parameters
        parameter_set = self.build_full_parameter_set(parameter_set)
        
        simulations = self.simulator.simulate(parameters = parameter_set, nsims=n_sims, verbose=progress_bar, t_max = self.t_max, t_step = self.t_step)
        
        compressed_simulation_0 = self.compressor(simulations)

        shape = [n_sims] + list(compressed_simulation_0.size())
        
        compressed_simulations = torch.zeros(shape, dtype=torch.float32)
        compressed_simulations[0] = compressed_simulation_0
        
        for i in range(1, n_sims):

            compressed_simulations[i] = self.compress(simulations[i], convstats=self.convstats)

        return compressed_simulations.flatten(start_dim=1)
    
    def simulation_wrapper_for_sbi(self, n_sims, data_dir='data', save=True, progress_bar=False, max_batch_size=500):
        
        if max_batch_size<n_sims:
            return self.batched_simulation_wrapper_for_sbi(n_sims, data_dir='data', save=save, progress_bar=progress_bar, max_batch_size=max_batch_size)
        """This must be readapted to new version!!"""
        self.data_dir = os.path.join(self.working_dir, data_dir)
        if save:
            
            if not os.path.isdir(self.data_dir):
                os.mkdir(self.data_dir)
            else:
                raise IsADirectoryError("Data directory already exits, please copy existing data such as simulations and parameter sets to a different folder to avoid conflicts")

        parameter_set = self.sample(n_sims)
   
        simulator = Simulator()

        simulations = simulator.simulate(parameters = parameter_set, nsims=n_sims, verbose=progress_bar, t_max = self.t_max, t_step = self.t_step)
        
        #simulations = simulator.simulate(parameters = parameter_set, nsims=batch_size, verbose=progress_bar,    t_max = self.t_max, t_step = self.t_step)

        compressed_simulation_0 = compress.compressor(simulations[0])

        shape = [n_sims] + list(compressed_simulation_0.size())
        compressed_simulations = torch.zeros(shape, dtype=torch.float32)
        compressed_simulations[0] = compressed_simulation_0
        
        for i in tqdm(range(n_sims)):


            compressed_simulations[i] = compress.compressor(simulations[i], convstats=self.convstats)

        if save:
            
            np.save(os.path.join(self.data_dir, 'theta.npy'), parameter_set)
            
            np.save(os.path.join(self.data_dir, 'simulations.npy'), simulations)

            torch.save(compressed_simulations, os.path.join(self.data_dir, 'X.pt'))
          
        return compressed_simulations
    
    def batched_simulation_wrapper_for_sbi(self, n_sims, data_dir='data', save=True, progress_bar=False, max_batch_size=500):

        self.data_dir = os.path.join(self.working_dir, data_dir)
        
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
        else:
            raise IsADirectoryError("Data directory already exits, please copy existing data such as simulations and parameter sets to a different folder to avoid conflicts")

        n_batches = int(np.ceil(n_sims/max_batch_size))
   
        simulator = Simulator()
        X_list = []
        theta_list = []
        sims_shape = np.array(simulator.simulate(nsims=2, verbose=False, t_max = self.t_max, t_step=self.t_step).shape, dtype=int)
        sims_shape[sims_shape==2]=n_sims
        sims_shape=tuple(sims_shape)
        full_simulations = np.memmap(os.path.join(self.data_dir, 'simulations.dat'), dtype='float64', mode='w+', shape=sims_shape)
        computed_sims=0
        start_index=0
        for batch in tqdm(range(n_batches)):

            batch_size = min(max_batch_size, n_sims-computed_sims)

            parameter_set = self.sample(batch_size)
            PARAMETER_NAMES = ["E", "L0", "Ve_0", "k_minus", "c1", "c2", "c3", "k_max", "Kk", "nk", "k0", "zeta_max", "Kzeta", "nzeta", "b", "zeta0", "alpha", "aoverN", "epsilon", "B"]
            #print({PARAMETER_NAMES[i]:parameter_set[0, i] for i in range(len(parameter_set[0]))})
            simulations = simulator.simulate(parameters = parameter_set, nsims=batch_size, verbose=progress_bar,    t_max = self.t_max, t_step = self.t_step)

            assert np.mean(np.isnan(simulations))==0, f'Simulations contain NaNs, {np.mean(np.isnan(simulations))}'

            X = self.compressor(simulations)

            theta_list.append(parameter_set)
            X_list.append(X)
            full_simulations[start_index:start_index+batch_size]=simulations

            start_index+=batch_size

            computed_sims+=batch_size
            del simulations
        
        X = torch.cat(X_list, dim=0)
        theta = np.concatenate(theta_list, axis=0)

        if save:
            np.save(os.path.join(self.data_dir, 'theta.npy'), theta)

            full_simulations.flush()
            np.save(os.path.join(self.data_dir, 'simulations.npy'), full_simulations)
            torch.save(X, os.path.join(self.data_dir, 'X.pt'))

    
    def train(self, nsims, batch_size=50, density_estimator='maf', hidden_features=50, n_mades=5, n_mafs=5, n_mcmc=100, n_epochs=100, training_batch_size=100, show_train_summary=True, convstats=True, conv_net=False):

        if self.conv_net:
            return self.train_conv_net(batch_size=batch_size)
        
        theta = torch.tensor(np.load(
            os.path.join(self.data_dir, 'theta.npy')))[:, self.variable_parameter_indices]
        
        X = torch.load(
            os.path.join(self.data_dir, 'X.pt'))
        
        X = X.flatten(1,-1)


        theta = theta.to(device=self.device)
        X = X.to(device=self.device)

        inference = SNPE(self.prior, device=self.device)
        

        density_estimator = inference.append_simulations(theta, X).train(training_batch_size=batch_size, show_train_summary=True)#plot_loss=False)
        posterior = inference.build_posterior(density_estimator, sample_width='mcmc')
        
        self.posterior = posterior
        
        return posterior
    
    def train_conv_net(self, batch_size=50):

        theta = torch.tensor(np.load(
            os.path.join(self.data_dir, 'theta.npy')))[:, self.variable_parameter_indices]
        
        X = torch.load(
            os.path.join(self.data_dir, 'X.pt'))

        X = X.flatten(1,-1)

        theta = theta.to(device=self.device)
        X = X.to(device=self.device)

        embedding_net = compress.SummaryNet()
        neural_posterior=utils.posterior_nn(model='maf', 
            embedding_net=embedding_net,
            )

        self.prior = utils.torchutils.BoxUniform(
        low=torch.as_tensor(self.prior_min, device=self.device), high=torch.as_tensor(self.prior_max, device=self.device),
        device=self.device
        )

        self.inference=SNPE(prior=self.prior, density_estimator=neural_posterior, device=self.device)
    
        self.density_estimator = self.inference.append_simulations(theta, X).train(training_batch_size=batch_size, show_train_summary=True)#plot_loss=False)
        posterior = self.inference.build_posterior(self.density_estimator, sample_with='mcmc')
        
        self.posterior = posterior
        
        return posterior

    
    def restrict_prior(self, nsims=500):
        
        restriction_estimator = utils.RestrictionEstimator(prior=self.prior)      
        
        ##theta containing only varied parameters
        theta = self.prior.sample((nsims,))

        ##Simulations
        x = self.simulation_wrapper(theta, progress_bar=False)

        restriction_estimator.append_simulations(theta, x)
        restriction_estimator.train()
        self.restricted_prior = restriction_estimator.restrict_prior()

        
        return self.restricted_prior


    def save(self, model_name='model.sbi'):

        import pickle
        self.model_path = os.path.join(self.working_dir, model_name)
        with open(self.model_path, "wb") as handle:
            pickle.dump(self, handle)
        
        print(f'Model saved at {self.model_path}')

        return


        
    
        


        

