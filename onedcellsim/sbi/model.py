## class model for SBI

import pandas as pd
import numpy as np
import os
from sbi import utils as utils
#from sbi import analysis as analysis
from sbi.inference import SNPE
import torch
from ..simulations.simulator import Simulator 
from ..compression import compress
from tqdm import tqdm

SIMDIR = os.path.join(os.path.dirname(__file__), '../simulations')
VARNAMES = ['zetaf', 'zetab', 'zetac', 'kf', 'kb', 'Lf', 'Lb', 'vrf', 'vrb', 'xf', 'xb', 'xc', 'vf', 'vb']


class Model:

    def __init__(self, variable_parameter_names, prior_min, prior_max, working_dir='./', default_parameter_values=None, convstats=True, conv_net=False, device='cpu'):
        """_summary_

        Args:
            variable_parameter_names (_type_): _description_
            prior_min (_type_): _description_
            prior_max (_type_): _description_

        Raises:
            ValueError: _description_
        """
        self.prior_min = prior_min
        self.prior_max = prior_max
        self.working_dir = working_dir
        self.convstats=convstats
        self.conv_net=conv_net
        self.device=device
        if self.convstats==True and conv_net==True:
            self.convstats=False
        

        self.prior = utils.torchutils.BoxUniform(
        low=torch.as_tensor(self.prior_min), high=torch.as_tensor(self.prior_max))
        
        self.default_parameters = pd.read_csv(os.path.join(SIMDIR, "default_parameters.csv"))

        self.parameter_names = list(self.default_parameters.columns)
        
        if default_parameter_values is None:
            self.default_parameter_values = self.default_parameters.loc[0, :].values.astype('float32')
        else:
            self.default_parameter_values = np.array(default_parameter_values, dtype='float32')
            assert(len(self.parameter_names)==len(self.default_parameter_values))

        try:
            self.variable_parameter_indices = np.array(
                [self.parameter_names.index(var_name) for var_name in variable_parameter_names])  
        except ValueError:
            raise ValueError(f'One or more of the given parameter names is not in the list of parameters: {self.parameter_names}')
            
    def build_full_parameter_set(self, variable_parameter_set):
        
        variable_parameter_set = np.array(variable_parameter_set)

        if variable_parameter_set.ndim==1:
            variable_parameter_set = variable_parameter_set.reshape(1, variable_parameter_set.size)
        n_sets = variable_parameter_set.shape[0]

        parameter_set = np.repeat(self.default_parameter_values[np.newaxis, :], n_sets, axis=0)
        parameter_set[:, self.variable_parameter_indices] = variable_parameter_set

        #parameter_set = torch.tensor(parameter_set, device=device)

        return parameter_set

    def sample(self, n_sets):

        if 'restricted_prior' in dir(self):
            prior=self.restricted_prior
        else:
            prior = self.prior

        samples = np.array(prior.sample((n_sets,)).cpu(), dtype='float32')

        parameter_set = self.build_full_parameter_set(samples)
        #parameter_set = np.repeat(self.default_parameter_values[np.newaxis, :], n_sets, axis=0)
        #parameter_set[:, self.variable_parameter_indices] = samples

        #parameter_set = torch.tensor(parameter_set, device=device)

        return parameter_set
    
    def simulation_wrapper(self, parameter_set=None, n_sims=1, progress_bar=True):

        simulator = Simulator()

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
        
        simulations = simulator.simulate(parameters = parameter_set, nsims=n_sims, verbose=progress_bar)
        
        compressed_simulation_0 = compress.compressor(simulations[0], convstats=self.convstats, conv_net=self.conv_net)

        shape = [n_sims] + list(compressed_simulation_0.size())
        
        compressed_simulations = torch.zeros(shape, dtype=torch.float32)
        compressed_simulations[0] = compressed_simulation_0
        
        for i in range(1, n_sims):

            compressed_simulations[i] = compress.compressor(simulations[i], convstats=self.convstats)

        return compressed_simulations
    
    def simulation_wrapper_for_sbi(self, n_sims, data_dir='data', save=True, progress_bar=True, max_batch_size=500, convstats=True):
        
        if max_batch_size<n_sims:
            return self.batched_simulation_wrapper_for_sbi(n_sims, data_dir='data', save=True, progress_bar=True, max_batch_size=max_batch_size)

        self.data_dir = os.path.join(self.working_dir, data_dir)
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
        else:
            raise IsADirectoryError("Data directory already exits, please copy existing data such as simulations and parameter sets to a different folder to avoid conflicts")

        parameter_set = self.sample(n_sims)
   
        simulator = Simulator()

        simulations = simulator.simulate(parameters = parameter_set, nsims=n_sims, verbose=progress_bar)

        compressed_simulation_0 = compress.compressor(simulations[0], convstats=self.convstats, conv_net=self.conv_net)

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
    
    def batched_simulation_wrapper_for_sbi(self, n_sims, data_dir='data', save=True, progress_bar=True, max_batch_size=500):

        self.data_dir = os.path.join(self.working_dir, data_dir)
        
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
        else:
            raise IsADirectoryError("Data directory already exits, please copy existing data such as simulations and parameter sets to a different folder to avoid conflicts")

        n_batches = int(np.ceil(n_sims/max_batch_size))
   
        simulator = Simulator()
        X_list = []
        theta_list = []
        sims_shape = np.array(simulator.simulate(nsims=2, verbose=False).shape, dtype=int)
        sims_shape[sims_shape==2]=n_sims
        sims_shape=tuple(sims_shape)
        full_simulations = np.memmap(os.path.join(self.data_dir, 'simulations.dat'), dtype='float64', mode='w+', shape=sims_shape)
        computed_sims=0
        start_index=0
        for batch in tqdm(range(n_batches)):

            batch_size = min(max_batch_size, n_sims-computed_sims)

            parameter_set = self.sample(batch_size)
            simulations = simulator.simulate(parameters = parameter_set, nsims=batch_size, verbose=False)

            compressed_simulation_0 = compress.compressor(simulations[0], convstats=self.convstats, conv_net=self.conv_net)

            shape = [batch_size] + list(compressed_simulation_0.size())
            
            compressed_simulations = torch.zeros(shape, dtype=torch.float32)
            
            compressed_simulations[0] = compressed_simulation_0
            
            for i in range(batch_size):
                
                compressed_simulations[i] = compress.compressor(simulations[i], convstats=self.convstats, conv_net=self.conv_net)
                

            theta_list.append(parameter_set)
            X_list.append(compressed_simulations)
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

    
    def train(self, batch_size=50):

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


        
    
        


        

