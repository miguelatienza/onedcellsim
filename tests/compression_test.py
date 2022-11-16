from onedcellsim.compression import compress
from onedcellsim.simulations import simulator

sim = simulator.Simulator()

def test_compressor_single_sim():

    """Test the compressor with all defaults and single simulation"""

    t, simulations = sim.simulate(nsims=1)
    
    compressed_results = compress.compressor(simulations)



    