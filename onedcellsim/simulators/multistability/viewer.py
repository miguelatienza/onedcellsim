import sys
import os
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from julia.api import Julia
## import pyqtSignal
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QSlider, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QSizePolicy, QSpacerItem, QFileDialog, QLabel)
from PyQt5 import QtWidgets, QtCore

#get the julia path from the command line, if provided
if len(sys.argv) > 1:
    jpath = sys.argv[1]
else:
    #get the path to the current script
    path_to_script = os.path.dirname(os.path.realpath(__file__))
    jpath = f"{path_to_script}/../../../venv/julia-1.6.7/bin/julia"

jl = Julia(runtime=jpath, compiled_modules=False)
julia_simulate_file = os.path.join(
                os.path.dirname(__file__), "simulate.jl"
            )

simulate = jl.eval(f"""
            include("{julia_simulate_file}")""")

class FloatSlider(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(float)

    def __init__(self, minimum, maximum, num_steps, value, label_text):
        super().__init__()
        self.spin_box = QtWidgets.QDoubleSpinBox()
        ##Set the number of decimals according to the minimum value
        self.spin_box.setDecimals(5)
        self.spin_box.setMinimum(minimum)
        self.spin_box.setMaximum(maximum)
        self.spin_box.setSingleStep((maximum - minimum) / num_steps)
        self.spin_box.setValue(value)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(num_steps)
        self.slider.setValue(int((value - minimum) / self.spin_box.singleStep()))

        self.label = QtWidgets.QLabel(label_text)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.spin_box)
        self.setLayout(layout)

        self.slider.valueChanged.connect(self.update_value)
        self.spin_box.valueChanged.connect(self.update_slider)

    def update_value(self, slider_value):
        value = self.spin_box.minimum() + (slider_value * self.spin_box.singleStep())
        self.spin_box.setValue(value)
        self.valueChanged.emit(value)

    def update_slider(self, value):
        slider_value = int((value - self.spin_box.minimum()) / self.spin_box.singleStep())
        self.slider.setValue(slider_value)
        self.valueChanged.emit(value)
    
    def value(self):
        return self.spin_box.value()




# PARAMETER_NAMES = ["E", "L0", "Ve_0", "k_minus", "c_1", "c_2", "c_3", "kappa_max", "K_kappa", "n_kappa", "kappa_0", "zeta_max", "K_zeta", "n_zeta", "b", "zeta_0", "alpha", "aoverN", "epsilon", "B", "epsilon_l", "gamma"]

VAR_NAMES = ["Lf", "Lb", "kf", "kb"]

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
    'epsilon_l' : [0, 2, 3],
    'gamma': [0, 0.5, 1],
    }

DEFAULT_PARAMETER_VALUES = [PARAMETER[1] for PARAMETER in PARAMETERS.values()]
PARAMETER_NAMES = list(PARAMETERS.keys())

variable_parameters = {
    #"E": [2e-3, 3e-3, 4e-3],
    "L0": [1, 11, 40],
    "Ve_0": [1e-2, 2.5e-2, 3.5e-2],
    "c_1": [1e-5, 1.5e-4, 3e-4],
    "c_2": [0.2, 0.5, 0.9],
    "c_3": [7e-3, 1e-2, 2e-2],
    "kappa_max": [1, 35, 80],
    "zeta_max": [5, 6, 20],
    "b": [1, 2, 10],
    #"B": [10, 25, 40],
    "epsilon": [0,0.5,5],
    'epsilon_l' : [0, 2, 3],
    'gamma': [0, 0.5, 1],
}

variable_parameter_indices = [PARAMETER_NAMES.index(key) for key in variable_parameters.keys()]


init_var_names = {
    "Lf": [10, 20, 50],
    "Lb": [10, 20, 50],
    "kf": [0.1, 0.1*variable_parameters["kappa_max"][2], 0.6*variable_parameters["kappa_max"][2]],
    "kb": [0.1, 0.1*variable_parameters["kappa_max"][2], 0.6*variable_parameters["kappa_max"][2]]
}

#simulator = Simulator()
DURATION=5
def run_simulation(full_params, init_vars=None):
    

    DURATION=5
    t_max, t_step = 7*60*60,120
    t_step_compute=0.5
    L0, k0, kmax = full_params[1], full_params[10], full_params[7]
    # sample initial conditions from uniform distribution
    ivs = init_vars.copy()
 
    if init_vars is None:
        vars_0 = np.array([L0, L0, k0, k0]).T
        vars_delta = np.array([4*L0, 4*L0, 0.6*kmax, 0.6*kmax]).T
        #print(vars_delta)
        init_vars = vars_0 + np.random.uniform(low=0, high=1, size=(1, 4)) * vars_delta

    #print(full_params.shape)
    variables = simulate(parameters=full_params, t_max=t_max, t_step=t_step, t_step_compute=t_step_compute, delta=0, init_vars=init_vars, nsims=1)
    #print(variables.shape)
    variables = variables[0, :, :]
    #print(variables.shape)
    #print(variables)
    df = pd.DataFrame(columns=['t', 'xc', 'xb', 'xf', 'kf', 'kb', 'vrf', 'vrb'])
    #ids = np.ones(length, dtype='int')*particle_id
    data = pd.DataFrame({'t': variables[:, 1],
    'xf':variables[:, 11], 
    'xb':variables[:,12],
    'xc':variables[:,13],
    'kf':variables[:,5],
    'kb':variables[:,6],
    'vrf':variables[:,9],
    'vrb':variables[:,10],
    'vf':variables[:,14],
    'vb':variables[:,15]
    })

    df=pd.concat([df, data], ignore_index=True)
    return df

class SimulationApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.setWindowState(QtCore.Qt.WindowMaximized)

        self.fig, self.ax = plt.subplots(figsize=(7*1.6,7), nrows=4, ncols=1, dpi=100)
        self.fig.tight_layout()
        self.fig.subplots_adjust(hspace=0.01, left=0.1, right=0.99)
        self.ax[0].set_ylim([-400, 400])
        self.fig.canvas.header_visible=False
        
        #Plot the track
        self.line_1, = self.ax[0].plot([], [], color='red')
        self.line_2, = self.ax[0].plot([], [], color='green')
        self.line_3, = self.ax[0].plot([], [], color='blue')
        #self.ax[0].set_xlabel('Time in hours', fontsize=20)
        self.ax[0].set_xticks([])
        # self.ax[0].set_ylabel(r'Position in $\mathrm{\mu} m$', fontsize=20)
        self.ax[0].set_ylabel(r'x ($\mathrm{\mu} m$)', fontsize=20)

        
        ##Plot the kappa dynamics
        self.line_4, = self.ax[1].plot([], [], color='red', label='front')
        self.line_5, = self.ax[1].plot([], [], color='blue', label='rear')
        self.line_4_1, = self.ax[1].plot([], [], color='red', alpha=0)
        self.line_5_1, = self.ax[1].plot([], [], color='blue', alpha=0)
        
          
        #self.ax[1].set_xlabel('Time in hours', fontsize=20)
        self.ax[1].set_xticks([])
        self.ax[1].set_ylabel(r'$\mathrm{\kappa (nNs \mu m^{-2})}$', fontsize=20)
        self.ax[1].legend()

        ##Plot the retrograde flow
        self.line_6, = self.ax[2].plot([], [], color='red', label='front')
        self.line_7, = self.ax[2].plot([], [], color='blue', label='rear')
        
        self.ax[2].set_xticks([])
        #self.ax[2].set_xlabel('Time in hours', fontsize=20)
        self.ax[2].set_ylabel(r'$\mathrm{v_r (\mu m/s)}$', fontsize=20)
        self.ax[2].plot(np.linspace(0,DURATION,100), np.zeros(100), color='black', alpha=0.5)
        self.Ve_0_line, = self.ax[2].plot(np.linspace(0,DURATION,100), np.zeros(100), color='green', alpha=1, label='$\mathrm{V_e^0}$')
        self.ax[2].legend()
        
        ##Plot the retrograde force
        self.line_8, = self.ax[3].plot([], [], color='red', label='front')
        self.line_9, = self.ax[3].plot([], [], color='blue', label='rear')
        self.ax[3].plot(np.linspace(0,DURATION,100), np.zeros(100), color='black', alpha=0.5)

        self.ax[3].set_xlabel('Time in hours', fontsize=20)
        #self.ax[3].set_ylabel(r'$\mathrm{Protrusion force (nN \mu m^{-1})}$', fontsize=20)
        self.ax[3].set_ylabel(r'$\mathrm{F_p (nN \mu m^{-1})}$', fontsize=20)
        self.ax[3].legend()
        
        self.create_widgets()
        self.initUI()
        self.update_simulation()


    def create_widgets(self):
        self.parameters_sliders = []
        for i, key in enumerate(variable_parameters.keys()):

            #create a floatslider
            p = FloatSlider(variable_parameters[key][0], variable_parameters[key][2], 100, variable_parameters[key][1], key)
            # # add label
            p.slider.setToolTip(key)
            self.parameters_sliders.append(p)

        self.init_conditions_sliders = []
        # create sliders for the initial conditions
        for i, key in enumerate(init_var_names.keys()):
            
            iv = FloatSlider(init_var_names[key][0], init_var_names[key][2], 100, init_var_names[key][1], key)

            # iv.setMinimum(init_var_names[key][0])
            # iv.setMaximum(init_var_names[key][2])
            # iv.setSingleStep((init_var_names[key][2] - init_var_names[key][0]) // 1000)
            # iv.setValue(init_var_names[key][1])
            # iv.setTickInterval(1)  # Optional: Set tick interval if desired
            # # add label
            iv.slider.setToolTip(key)
            self.init_conditions_sliders.append(iv)

        # Update the simulation when the sliders are moved
        for p in self.parameters_sliders:
            p.valueChanged.connect(self.update_simulation)

        for iv in self.init_conditions_sliders:
            iv.valueChanged.connect(self.update_simulation)


    def initUI(self):
        self.setWindowTitle("Simulation Viewer")
        self.setGeometry(100, 100, 800, 600)
        
        # Create main layout
        main_layout = QtWidgets.QHBoxLayout()
        
        # Create a group box for sliders
        self.full_sliders_box = QtWidgets.QVBoxLayout()
        self.sliders_group_box = QtWidgets.QGroupBox("Parameters")
        sliders_layout = QtWidgets.QVBoxLayout()  # Use QVBoxLayout for vertical layout
        self.init_conditions_sliders_group_box = QtWidgets.QGroupBox("Initial conditions")
        init_conditions_sliders_layout = QtWidgets.QVBoxLayout()  # Use QVBoxLayout for vertical layout

        # Add sliders to the layout
        for p in self.parameters_sliders:
            sliders_layout.addWidget(p)
        
        # Add initial conditions sliders to the layout
        for iv in self.init_conditions_sliders:
            init_conditions_sliders_layout.addWidget(iv)

        # Set the layout for the group boxes
        self.sliders_group_box.setLayout(sliders_layout)
        self.init_conditions_sliders_group_box.setLayout(init_conditions_sliders_layout)
        
        self.full_sliders_box.addWidget(self.sliders_group_box)
        self.full_sliders_box.addWidget(self.init_conditions_sliders_group_box)

        # Add the group box to the main layout
        main_layout.addLayout(self.full_sliders_box)
        
        self.canvas = FigureCanvas(self.fig)
        self.plot_layout = QtWidgets.QVBoxLayout()
        self.plot_layout.addWidget(self.fig.canvas)
        main_layout.addLayout(self.plot_layout)

        main_layout.setStretchFactor(self.full_sliders_box, 20)  # Adjust the stretch factor
        main_layout.setStretchFactor(self.plot_layout, 80)
        self.central_widget.setLayout(main_layout)

    def update_simulation(self):
        
        params = [p.value() for p in self.parameters_sliders]

        full_params = np.array(DEFAULT_PARAMETER_VALUES)
        #full_params = np.array([PARAMETER[1] for PARAMETER in PARAMETERS.values()])
        full_params[variable_parameter_indices] = params

        ivs = np.array([iv.value() for iv in self.init_conditions_sliders]).reshape(1,4)

        E, L0, Ve_0, k_minus, c1, c2, c3, k_max, Kk, nk, k0, zeta_max, Kzeta, nzeta, b, zeta0, alpha, aoverN, epsilon, B, epsilon_l, gamma = full_params
        obs = run_simulation(full_params, init_vars=ivs)
        t, front, rear, nucleus, kf, kb, vrf, vrb, vf, vb = obs.t.values, obs.xf.values, obs.xb.values, obs.xc.values, obs.kf.values, obs.kb.values, obs.vrf.values, obs.vrb.values, obs.vf, obs.vb


        Ff = vrf*kf 
        Fb = vrb*kb 

        # update the track
        self.line_1.set_data(t/3600, front)
        self.line_2.set_data(t/3600, nucleus)
        self.line_3.set_data(t/3600, rear)
       
        k_lim = k0 + (k_max * B**nk/(Kk**nk + B**nk))
        kfeq = c1*k_lim/(c1 + c2*np.exp(np.abs(vrf)/c3))
        kbeq = c1*k_lim/(c1 + c2*np.exp(np.abs(vrb)/c3))
        
        self.line_4.set_data(t/3600, kf)
        self.line_5.set_data(t/3600, kb)

        self.line_4_1.set_data(t/3600, kfeq)
        self.line_5_1.set_data(t/3600, kbeq)
    

        self.line_6.set_data(t/3600, vrf)
        self.line_7.set_data(t/3600, vrb)
        
        checkf = vf + vrf - Ve_0*np.exp(-aoverN*Ff) + k_minus

        checkb = vrb -vb - Ve_0*np.exp(-aoverN*Fb) + k_minus
  
        self.line_8.set_data(t/3600, Ff)
        self.line_9.set_data(t/3600, Fb)
        # self.line_8.set_data(t/3600, checkf)
        # self.line_9.set_data(t/3600, checkb)
        DURATION=5

        self.ax[0].set_xlim(-DURATION*0.01, DURATION*1.01)
        vmax=72
        self.ax[0].set_ylim(-vmax*DURATION, vmax*DURATION)

        self.ax[1].set_xlim(-DURATION*0.01, DURATION*1.01)
        self.ax[1].set_ylim(-0.01, max(kb.max(), kf.max())*1.05)

        self.ax[2].set_xlim(-DURATION*0.01, DURATION*1.01)
        self.ax[2].set_ylim(min(vrb.min(), vrf.min()), 
            max(vrb.max(), vrf.max())*1.05)  

        self.ax[3].set_xlim(-DURATION*0.01, DURATION*1.01)
        self.ax[3].set_ylim(min(Fb.min(), Ff.min()), 
            max(Fb.max(), Ff.max())*1.05) 
        # self.ax[3].set_ylim(min(checkb.min(), checkf.min()), 
        #     max(checkb.max(), checkf.max()))   
        # self.ax[3].set_ylim(-0.01, 0.01)   

        self.Ve_0_line.set_data(t/3600, np.ones(t.size)*Ve_0)

        self.fig.canvas.draw()
        DURATION=5

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = SimulationApp()
    main_window.show()
    sys.exit(app.exec_())

