"""
Python Simulator class for simulating 1D cell tracks. The core functions are written in Julia for significant speedup. 
"""
import os
from julia.api import Julia
from . import environ
import numpy as np


PARAMETERS={
    'E': [2.5e-3, 3e-3, 3.1e-3],
    'L0': [10, 11, 20],
    'Ve_0': [1.71e-2, 2.5e-2, 3.5e-2],
    'k_minus': [0, 0, 0],
    'c_1': [1e-4, 1.5e-4, 5e-4],
    'c_2': [0.3, 0.5, 0.6],
    'c_3': [3e-3, 7.8e-3, 1.5e-2],
    'kappa_max': [35, 35, 35],
    'K_kappa': [1, 35, 40],
    'n_kappa': [1, 3, 7.6],
    'kappa_0': [0, 1e-3, 1e-2],
    'zeta_max': [1, 1.4, 40],
    'K_zeta': [10, 50, 60],
    'n_zeta': [1, 4, 5.8],
    'b': [2, 2, 3],
    'zeta_0': [0, 1e-3, 1e-1],
    'alpha': [4e-2, 4e-2, 4e-2],
    'aoverN': [0, 0, 0],
    'epsilon': [0, 1, 3],
    'B': [5, 30, 100],
    'epsilon_l' : [0, 0.2, 0.3],
    'gamma': [0, 0.5, 1],
    }


#PARAMETER_NAMES = ["E", "L0", "Ve_0", "k_minus", "c1", "c2", "c3", "k_max", "Kk", "nk", "k0", "zeta_max", "Kzeta", "nzeta", "b", "zeta0", "alpha", "aoverN", "epsilon", "B"]
PARAMETER_NAMES = ['E', 'L0', 'Ve_0', 'k_minus', 'c_1', 'c_2', 'c_3', 'kappa_max', 'K_kappa', 'n_kappa', 'kappa_0', 'zeta_max', 'K_zeta', 'n_zeta', 'b', 'zeta_0', 'alpha', 'aoverN', 'epsilon', 'B', 'epsilon_l', 'gamma']
VAR_NAMES = ["Lf", "Lb", "kf", "kb"]

##Defaults to no force dependence on retrograde flow (aoverN=0)
#DEFAULT_PARAMETER_VALUES = [3e-3, 10, 3e-2, 0, 1.5e-4, 0.5, 7.8e-3, 35, 35, 3, 1e-2, 1.4, 50, 4, 3, 1e-1, 4e-2, 0, 0, 30]
DEFAULT_PARAMETER_VALUES = [3e-3, 11, 2.5e-2, 0, 1.5e-4, 0.5, 7.8e-3, 35, 35, 3, 1e-3, 1.4, 50, 4, 2, 1e-3, 4e-2, 0, 0, 30, 0, 0.5]

class Simulator:
    """
    Simulator class for simulating 1D cell tracks. The core functions are written in Julia for significant speedup.

    Attributes:
    sime

        
    """
    def __init__(self, julia_path=None, p=1, t_step=120, t_step_compute=0.5, t_max=15*3600):
        """
        Initialize the simulator class.
        """
        self.parameter_names = PARAMETER_NAMES
        self.var_names = VAR_NAMES
        self.default_parameter_values = DEFAULT_PARAMETER_VALUES
        self.p = p
        self.t_step = t_step
        self.t_step_compute = t_step_compute
        self.t_max = t_max
        #jpath = "/usr/b"
        if p==1:
            self.jl = Julia(runtime=environ.JULIA_PATH, compiled_modules=False)
            
            julia_simulate_file = os.path.join(
                os.path.dirname(__file__), "simulate.jl"
            )
            self.simulate_jl = self.jl.eval(f"""
            include("{julia_simulate_file}")""")
        else:
            self.jl = Julia(runtime=julia_path, compiled_modules=False)
            
            julia_simulate_file = os.path.join(
                os.path.dirname(__file__), "simulate.jl"
            )
            self.simulate_jl = self.jl.eval(f"""
            include("{julia_simulate_file}")""")
    
    def simulate(self, parameters=None, t_max=None, t_step=None, t_step_compute=None, delta=0, nsims=1, verbose=False, mode="array", max_batch_size=500):

        if t_max is None:
            t_max = self.t_max
        if t_step is None:
            t_step = self.t_step
        if t_step_compute is None:
            t_step_compute = self.t_step_compute
            
        if self.p>1:
            return self.parallel_simulate(parameters=None, t_max=15*3600, t_step=30, t_step_compute=0.5, delta=0, nsims=1, verbose=False, mode="array")

        if parameters is None:
            parameters=DEFAULT_PARAMETER_VALUES
        
        parameters = np.array(parameters)
        if parameters.ndim==1:
            parameters = np.repeat(parameters[np.newaxis, :], nsims, axis=0)

        L0, k0, kmax = parameters[:,1], parameters[:,10], parameters[:,7]
        
        # sample initial conditions from uniform distribution
        vars_0 = np.array([L0, L0, k0, k0]).T
        vars_delta = np.array([4*L0, 4*L0, 0.6*kmax, 0.6*kmax]).T
        #print(vars_delta)
        init_vars = vars_0 + np.random.uniform(low=0, high=1, size=(nsims, 4)) * vars_delta
        # toprint = {prameter_name: parameter_value for prameter_name, parameter_value in zip(PARAMETER_NAMES, parameters[0, :])}
        # print(toprint)

        return self.simulate_jl(parameters=parameters, t_max=t_max, t_step=t_step, t_step_compute=t_step_compute, delta=delta, nsims=nsims, verbose=verbose, mode=mode, init_vars=init_vars)
        

    def parallel_simulate(self, parameters=None, t_max=15*3600, t_step=120, t_step_compute=0.5, delta=0, nsims=1, verbose=False, mode="array"):

        return self.simulate_jl(parameters=parameters, t_max=t_max, t_step=t_step, t_step_compute=t_step_compute, delta=delta, nsims=nsims, verbose=verbose, mode=mode)



        
