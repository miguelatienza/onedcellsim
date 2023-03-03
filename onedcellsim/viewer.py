"""Class to view simulations interactively on a notebook"""

import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython import display
from .simulations.simulator import Simulator

class Viewer:

    def __init__(self, parameters):
        
        ##Initialize the simulator
        self.simulator = Simulator()

        pass

    def build_canvas(self):

        plt.ion()
        output=widgets.Output()

        with plt.ioff():

            with output:
                fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
                display(fig.canvas)




