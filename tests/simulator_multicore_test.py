from onedcellsim.simulations.simulator import Simulator
import pytest
import numpy as np
import time
import pandas as pd

simulator=Simulator(p=3)

def test_multi_simulations():

    nsims=100

    params = np.array([[3e-3, 10, 3e-2, 5e-3, 1.5e-4, 7.5e-5, 7.8e-3, 35, 35, 3, 1e-2, 1.4, 50, 4, 3, 1e-1, 4e-2, 1, 1,45]])
    parameter_set = np.repeat(params, nsims, axis=0)

    df = simulator.simulate(parameters=parameter_set, nsims=nsims, mode='array')

    assert(isinstance(df, np.ndarray))
    assert(df.shape[0]==parameter_set.shape[0])
    return