from onedcellsim.simulators.multistability import simulator
import pytest
import numpy as np
import time
import pandas as pd


single_core_simulator = simulator.Simulator()
#multicore_simulator = simulator.Simulator(p=10)

class Test_single_core:

    def test_simulate_single_sim(self):
        #self.my_simulator=simulator.Simulator()
        looping = True
        df = single_core_simulator.simulate()
        # while looping:
        #     t, df = single_core_simulator.simulate()
        #     plt.plot(t, df[:, 9])
        #     plt.plot(t, df[:, 10])
        #     plt.plot(t, df[:, 11])
        #     plt.show()
        print(df.shape)
        #assert(t.ndims=)
        assert(df.ndim==3)
        assert(df.shape[0]==1)
        assert((df[0, :, 0]==1).all())
        #     looping = input()!=1

    def test_simulate_multiple_sims(self):
        
        nsims=100
        
        t_0 = time.time()
        df = single_core_simulator.simulate(nsims=nsims)
        print(f"Time per simulation: {round((time.time()-t_0))*1000/nsims}ms")

        t_0 = time.time()
        df = single_core_simulator.simulate(nsims=nsims, mode='dict', verbose=True)
        print(f"Time per simulation: {round((time.time()-t_0))*1000/nsims}ms")

        t_0 = time.time()
        df = single_core_simulator.simulate(nsims=nsims)
        print(f"Time per simulation: {round((time.time()-t_0))*1000/nsims}ms")
        "correct shape"
        assert(df.shape[0]==nsims)
        
        "No nans returned"
        try:
            assert(np.isnan(df).sum()==0)
        except AssertionError:
            print(np.where(np.sum(np.isnan(df), axis=(1))>0))
            assert(np.isnan(df).sum()==0)

        "No array has been left to zeros"
        for dfp in df:
            assert(not 0 in np.mean(dfp, axis=0)[1:])
        
        "Front is larger than nuc and nuc is larger than rear"
        assert(
            np.all(df[:, :, 11]>df[:, :, 13]) & 
            np.all(df[:, :, 12]<df[:, :, 13]))    
    
    def test_input_parameters(self):
        
        parameters = np.array([3e-3, 10, 3e-2, 5e-3, 1.5e-4, 7.5e-5, 7.8e-3, 35, 35, 3, 1e-2, 1.4, 50, 4, 3, 1e-1, 4e-2, 1, 1,45])

        df = single_core_simulator.simulate(parameters=parameters, mode='array')

        return
    
    def test_simulate_multiple_param_sets(self):

        nsims=10

        params = np.array([[3e-3, 10, 3e-2, 5e-3, 1.5e-4, 7.5e-5, 7.8e-3, 35, 35, 3, 1e-2, 1.4, 50, 4, 3, 1e-1, 4e-2, 1, 1,45]])
        parameter_set = np.repeat(params, nsims, axis=0)

        df = single_core_simulator.simulate(parameters=parameter_set, nsims=nsims, mode='array')

        assert(isinstance(df, np.ndarray))
        assert(df.shape[0]==parameter_set.shape[0])
        return