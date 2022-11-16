from onedcellsim.sbi.model import Model
import numpy as np
import pytest
import os
#import torch

def test_model():

    variable_parameter_names = ["E", "Ve_0"]
    prior_min = [1e-3, 1e-2]
    prior_max = [8e-3, 8e-2]
    model = Model(variable_parameter_names, prior_min, prior_max)
    
    assert(len(model.default_parameter_values.shape)==1)
    assert(isinstance(model.default_parameter_values, np.ndarray))
    
def test_existing_data_dir():

    try: 
        model_dir = 'pytest_tmp'
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
            os.mkdir(os.path.join(model_dir, 'data'))
        else:
            raise IsADirectoryError('Tmp directory already exists, avoiding running this test to avoid conflicts')
        
        variable_parameter_names = ["E", "Ve_0"]
        prior_min = [1e-3, 1e-2]
        prior_max = [8e-3, 8e-2]
        
        with pytest.raises(IsADirectoryError):
            model = Model(variable_parameter_names, prior_min, prior_max, working_dir=model_dir)
            model.simulation_wrapper_for_sbi(n_sims=10)
        
        os.rmdir(os.path.join(model_dir, 'data'))
        os.rmdir(model_dir)
    
    except Exception as E:
        if os.path.isdir(os.path.join(model_dir, 'data')):
            os.rmdir(os.path.join(model_dir, 'data'))
        if os.path.isdir(model_dir):
            os.rmdir(model_dir)
        
        raise E

def test_wrong_parameter_names():

    fake_names = ["fd", "E"]
    prior_min = [1e-3, 1e-2]
    prior_max = [8e-3, 8e-2]

    with pytest.raises(ValueError):
        model = Model(fake_names, prior_min, prior_max)

def test_sampler():

    variable_parameter_names = ["E", "Ve_0"]
    prior_min = [1e-3, 1e-2]
    prior_max = [8e-3, 8e-2]
    model = Model(variable_parameter_names, prior_min, prior_max)
    
    n_samples = 1000
    samples = model.sample(n_samples)

    assert(samples.shape[0]==n_samples)
    assert(samples.shape[1]==model.default_parameter_values.size)

def test_simulation_wrapper():

    variable_parameter_names = ["E", "Ve_0"]
    prior_min = [1e-3, 1e-2]
    prior_max = [8e-3, 8e-2]

    mean_params = 0.5*(np.array(prior_min)+ np.array(prior_max))
    model = Model(variable_parameter_names, prior_min, prior_max)

    output = model.simulation_wrapper(mean_params, n_sims=10)

    assert(output.size()[0]==10)

def test_simulation_wrapper_single_simulation():

    variable_parameter_names = ["E"]
    prior_min = [1e-3]
    prior_max = [8e-3]    

    mean_params = 0.5*(np.array(prior_min)+ np.array(prior_max))
    model = Model(variable_parameter_names, prior_min, prior_max)

    output = model.simulation_wrapper(mean_params,n_sims=1)
    output_multisims = model.simulation_wrapper(mean_params, n_sims=10)
    print((output).size())
    assert(output.ndim==3)
    assert(output_multisims.size()[0]==10)


def test_simulation_wrapper_for_sbi():

    variable_parameter_names = ["E", "Ve_0"]
    prior_min = [1e-3, 1e-2]
    prior_max = [8e-3, 8e-2]
    model_dir = 'pytest_tmp'
    os.mkdir(model_dir)
    
    model = Model(variable_parameter_names, prior_min, prior_max, working_dir=model_dir)
    
    n_sims = 10
    model.simulation_wrapper_for_sbi(n_sims)
    
    parameters_file = os.path.join(model_dir, 'data/theta.npy')
    simulations_file = os.path.join(model_dir, 'data/simulations.npy')
    compressed_simulations_file = os.path.join(model_dir, 'data/X.pt')

    assert(os.path.isfile(simulations_file))
    assert(os.path.isfile(parameters_file))
    assert(os.path.isfile(compressed_simulations_file))

    os.system(f'rm -r {model_dir}')

def test_training():
    
    n_sims=100
    variable_parameter_names = ["Ve_0"]
    prior_min = [8e-3]
    prior_max = [2e-1]
    model_dir = 'pytest_tmp'
    os.mkdir(model_dir)
    
    default_params = [3e-3, 10, 3e-2, 5e-3, 1.5e-4, 7.5e-5, 7.8e-3, 35, 35, 3, 1e-2, 1.4, 50, 4, 3, 1e-1, 4e-2, 1, 1*0.5,60]
    model = Model(variable_parameter_names, prior_min, prior_max, working_dir=model_dir, default_parameter_values=default_params)
    
    model.simulation_wrapper_for_sbi(n_sims=n_sims)
    model.train()
    
    mean_params = 0.5*(np.array(prior_min)+ np.array(prior_max))
    output = model.simulation_wrapper(mean_params).flatten()

    n_samples=10_000
    inferred_parameters = model.posterior.sample((n_samples, ), x=output)

    print(f'Total prior range: {prior_min, prior_max}')
    print('\n')
    print('Evaluation with mean parameters:')

    intervals = [inferred_parameters.mean(axis=0)- inferred_parameters.std(axis=0), inferred_parameters.mean(axis=0) + inferred_parameters.std(axis=0)]

    print(f'1 std confidence range of inferred parameters:')
    print(f'first parameter: {intervals[0][0], intervals[1][0]}')
    print(f'True value: {mean_params[0]}')
    # print(f'second parameter: {intervals[0][1], intervals[1][1]}')
    # print(f'True value: {mean_params[1]}')
    
    ##Check performance for lower parameter values
    params_25 = 0.75*np.array(prior_min)+ 0.25*np.array(prior_max)
    output = model.simulation_wrapper(params_25).flatten()

    n_samples=10_000
    inferred_parameters = model.posterior.sample((n_samples, ), x=output)

    print('\n')
    print('Evaluation with 25th percentile lower parameters:')

    intervals = [inferred_parameters.mean(axis=0)- inferred_parameters.std(axis=0), inferred_parameters.mean(axis=0) + inferred_parameters.std(axis=0)]

    print(f'1 std confidence range of inferred parameters:')
    print(f'first parameter: {intervals[0][0], intervals[1][0]}')
    print(f'True value: {params_25[0]}')
    # print(f'second parameter: {intervals[0][1], intervals[1][1]}')
    # print(f'True value: {params_25[1]}')
    
    ##Check performance for higher parameter values
    params_75 = 0.25*np.array(prior_min)+ 0.75*np.array(prior_max)
    output = model.simulation_wrapper(params_75).flatten()

    n_samples=10_000
    inferred_parameters = model.posterior.sample((n_samples, ), x=output)

    print('\n')
    print('Evaluation with 75th percentile lower parameters:')

    intervals = [inferred_parameters.mean(axis=0)- inferred_parameters.std(axis=0), inferred_parameters.mean(axis=0) + inferred_parameters.std(axis=0)]

    print(f'1 std confidence range of inferred parameters:')
    print(f'first parameter: {intervals[0][0], intervals[1][0]}')
    print(f'True value: {params_75[0]}')
    # print(f'second parameter: {intervals[0][1], intervals[1][1]}')
    # print(f'True value: {params_75[1]}')
    



    




