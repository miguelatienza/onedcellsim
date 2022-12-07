"""Python Simulator class for simulating 1D cell tracks with Julia"""
import os
from julia.api import Julia

class Simulator():

    def __init__(self, julia_path=None, p=1):

        self.p=p

        "Set default arguments"
        if julia_path is None:
            julia_path=os.path.join(__file__, '../')
            julia_path = '/project/ag-moonraedler/MAtienza/cellsbi/envs/sbi/julia-1.6.7/bin/julia'

        #jpath = "/usr/b"
        if p==1:
            self.jl = Julia(runtime=julia_path, compiled_modules=False)
            
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
    
    def simulate(self, parameters=None, t_max=15*3600, t_step=30, t_step_compute=0.5, delta=0, nsims=1, verbose=False, mode="array", max_batch_size=500):

        if self.p>1:
            return self.parallel_simulate(parameters=None, t_max=15*3600, t_step=30, t_step_compute=0.5, delta=0, nsims=1, verbose=False, mode="array")

        if parameters is None:
            parameters=parameters=[3e-3, 10, 3e-2, 5e-3, 1.5e-4, 7.5e-5, 7.8e-3, 35, 35, 3, 1e-2, 1.4, 50, 4, 3, 1e-1, 4e-2, 1, 1, 45]

        
        return self.simulate_jl(parameters=parameters, t_max=t_max, t_step=t_step, t_step_compute=t_step_compute, delta=delta, nsims=nsims, verbose=verbose, mode=mode)
        


    def parallel_simulate(self, parameters=None, t_max=15*3600, t_step=30, t_step_compute=0.5, delta=0, nsims=1, verbose=False, mode="array"):

        return self.simulate_jl(parameters=parameters, t_max=t_max, t_step=t_step, t_step_compute=t_step_compute, delta=delta, nsims=nsims, verbose=verbose, mode=mode)



        
