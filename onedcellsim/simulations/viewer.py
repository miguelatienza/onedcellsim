import sys
import os
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from julia.api import Julia
from PyQt5.QtWidgets import (QApplication, QMainWindow, QSlider, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QSizePolicy, QSpacerItem, QFileDialog, QLabel)

# jpath = "/home/miguel/onedcellsim/venv/julia-1.6.7/bin/julia"
# jl = Julia(runtime=jpath, compiled_modules=False)
# path_to_julia_scripts = "/project/ag-moonraedler/MAtienza/cellsbi/simulations/"
# path_to_julia_scripts = "./"
# julia_simulate_file = os.path.join(
#                 os.path.dirname(__file__), "simulate.jl"
#             )

jpath = "/project/ag-moonraedler/MAtienza/cellsbi/envs/sbi/julia-1.6.7/bin/julia"
jl = Julia(runtime=jpath, compiled_modules=False)
path_to_julia_scripts = "/project/ag-moonraedler/MAtienza/cellsbi/simulations/"
path_to_julia_scripts = "./"
julia_simulate_file = os.path.join(
                os.path.dirname(__file__), "simulate.jl"
            )

simulate = jl.eval(f"""
            include("{julia_simulate_file}")""")
# simulate = jl.eval(f"""
# include("{path_to_julia_scripts}simulate.jl")
# """)

#from simulator import Simulator

#simulator = Simulator()
DURATION=5
def run_simulation(params):
    
    full_params = np.array(params)
    #print(full_params)
    DURATION=5
    t_max, t_step = DURATION*60*60,30
    t_step_compute=0.5
    #params, particle_id, verbose, t_max, t_step, t_step_compute = args
    
    #np.random.seed(int(time.time())
    variables = simulate(parameters=full_params, t_max=t_max, t_step=t_step, t_step_compute=t_step_compute, delta=0, kf0=10)[0]
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
    #     self.parameters={
    #     'E': [7e-5, 5.5e-4, 3e-3],
    #     'L_0': [10, 30, 50],
    #     'V_e^0': [1e-2, 1.9e-2, 1.1e-1],
    #     'zeta_0': [0, 5e-2, 1e-1],
    #     'zeta^{max}': [1, 19, 40],
    #     'b': [3, 3, 3],
    #     'K_zeta': [10, 14.3, 40],
    #     'n_zeta': [1, 1.01, 5.8],
    #     'B': [5, 37.5, 70],
    #     'kappa_0': [1e-2, 1e-2, 1e-2],
    #     'kappa^{max}': [35, 50, 70],
    #     'K_kappa': [1, 8.9, 40],
    #     'n_kappa': [1, 1.3, 7.6],
    #     'c_1': [1.5e-4, 1.5e-4, 1.5e-4],
    #     'c_2': [7.5e-5, 7.5e-5, 7.5e-5],
    #     'c_3': [7.8e-3, 7.8e-3, 7.8e-3],
    #     'epsilon': [0, 1, 3]
    # }

    #     self.parameters={
    #     'E': [7e-5, 3e-3, 5e-3],
    #     'L_0': [10, 30, 50],
    #     'V_e^0': [1e-2, 1.45e-2, 1.1e-1],
    #     'zeta_0': [0, 5e-2, 1e-1],
    #     'zeta^{max}': [1, 1.5, 40],
    #     'b': [3, 3, 3],
    #     'K_zeta': [10, 32.7, 40],
    #     'n_zeta': [1, 1.01, 5.8],
    #     'B': [5, 37.5, 70],
    #     'kappa_0': [1e-2, 1e-2, 1e-2],
    #     'kappa^{max}': [35, 50, 70],
    #     'K_kappa': [1, 1.5, 40],
    #     'n_kappa': [1, 1.3, 7.6],
    #     'c_1': [1e-4, 1.5e-4, 5e-4],
    #     'c_2': [3e-5, 7.5e-5, 9.9e-5],
    #     'c_3': [3e-3, 9.9e-3, 1.5e-2],
    #     'epsilon': [0, 1, 3]
    # }
        

        self.parameters={
        'E': [3e-3, 3e-3, 5e-3],
        'L_0': [10, 10, 50],
        'V_e^0': [1e-2, 2.5e-2, 1.1e-1],
        'zeta_0': [0, 5e-2, 1e-1],
        'zeta^{max}': [1, 1.4, 40],
        'b': [3, 2, 3],
        'K_zeta': [10, 50, 60],
        'n_zeta': [1, 4, 5.8],
        'B': [5, 37.5, 100],
        'kappa_0': [1e-2, 1e-2, 1e-2],
        'kappa^{max}': [35, 50, 70],
        'K_kappa': [1, 35, 40],
        'n_kappa': [1, 3, 7.6],
        'c_1': [1e-4, 1.5e-4, 5e-4],
        'c_2': [3e-5, 7.5e-5, 9.9e-5],
        'c_3': [3e-3, 7.8e-3, 1.5e-2],
        'epsilon': [0, 1, 3]
    }
        # Create sliders for the parameters
        self.E = QtWidgets.QDoubleSpinBox()
        self.L0 = QtWidgets.QDoubleSpinBox()
        self.Ve_0 = QtWidgets.QDoubleSpinBox()
        self.zeta0 = QtWidgets.QDoubleSpinBox()
        self.zeta_max = QtWidgets.QDoubleSpinBox()
        self.b = QtWidgets.QDoubleSpinBox()
        self.K_zeta = QtWidgets.QDoubleSpinBox()
        self.n_zeta = QtWidgets.QDoubleSpinBox()
        self.B = QtWidgets.QDoubleSpinBox()
        self.k0 = QtWidgets.QDoubleSpinBox()
        self.kappa_max = QtWidgets.QDoubleSpinBox()
        self.K_kappa = QtWidgets.QDoubleSpinBox()
        self.n_kappa = QtWidgets.QDoubleSpinBox()
        self.c1 = QtWidgets.QDoubleSpinBox()
        self.c2 = QtWidgets.QDoubleSpinBox()
        self.c3 = QtWidgets.QDoubleSpinBox()
        self.epsilon = QtWidgets.QDoubleSpinBox()

        self.parameters_sliders = [self.E, self.L0, self.Ve_0, self.zeta0, self.zeta_max, self.b, self.K_zeta, self.n_zeta, self.B, self.k0, self.kappa_max, self.K_kappa, self.n_kappa, self.c1, self.c2, self.c3, self.epsilon]

        for i, key in enumerate(self.parameters.keys()):
            self.parameters_sliders[i].setMinimum(self.parameters[key][0])
            self.parameters_sliders[i].setMaximum(self.parameters[key][2])
            self.parameters_sliders[i].setSingleStep((self.parameters[key][2]-self.parameters[key][0])/1000)
            self.parameters_sliders[i].setValue(self.parameters[key][1])
            n_decimals = int(np.log10(1/self.parameters[key][1]))
            self.parameters_sliders[i].setDecimals(n_decimals+3)

        #Update the simulation when the sliders are moved
        for i, key in enumerate(self.parameters.keys()):
            self.parameters_sliders[i].valueChanged.connect(self.update_simulation)


        ##Add labels to the widgets
        for i, key in enumerate(self.parameters.keys()):
            self.parameters_sliders[i].setPrefix(key + ": ")
            self.parameters_sliders[i].setToolTip(key)


    def initUI(self):
        self.setWindowTitle("Simulation Viewer")
        self.setGeometry(100, 100, 800, 600)
        
        # Create main layout
        main_layout = QtWidgets.QHBoxLayout()
        
        # Create a group box for sliders
        self.sliders_group_box = QtWidgets.QGroupBox("Parameters")
        sliders_layout = QtWidgets.QGridLayout()
        #self.screen_size = QtWidgets.QDesktopWidget().availableGeometry().size()

        #sliders_layout.setSpacing(0.2*self.screen_size.width())

        
        # Add the sliders to the layout
        for i, key in enumerate(self.parameters.keys()):
            sliders_layout.addWidget(self.parameters_sliders[i], i, 0)

        
        # Set the layout for the group box
        self.sliders_group_box.setLayout(sliders_layout)
        
        # Add the group box to the main layout
        main_layout.addWidget(self.sliders_group_box)
        
        self.canvas = FigureCanvas(self.fig)
        self.plot_layout = QtWidgets.QVBoxLayout()
        self.plot_layout.addWidget(self.fig.canvas)
        main_layout.addLayout(self.plot_layout)

        main_layout.setStretchFactor(sliders_layout, 20)
        main_layout.setStretchFactor(self.plot_layout, 80)
        self.central_widget.setLayout(main_layout)
  

    def update_simulation(self):
        E = self.E.value()
        L0 = self.L0.value()
        Ve_0 = self.Ve_0.value()
        k_minus = 0# self.k_minus.value()
        c1 = self.c1.value()
        c2 = self.c2.value()
        c3 = self.c3.value()
        k_max = self.kappa_max.value()
        Kk = self.K_kappa.value()
        nk = self.n_kappa.value()
        k0 = self.k0.value()
        zeta_max = self.zeta_max.value()
        Kzeta = self.K_zeta.value()
        nzeta = self.n_zeta.value()
        b = self.b.value()
        zeta0 = self.zeta0.value()
        aoverN = 1
        epsilon = self.epsilon.value()
        B = self.B.value()


        params = [E, L0, Ve_0, k_minus, c1, c2, c3, k_max, Kk, nk, k0, zeta_max, Kzeta, nzeta, b, zeta0, 4e-2, aoverN, epsilon, B]
        obs = run_simulation(params)
        t, front, rear, nucleus, kf, kb, vrf, vrb, vf, vb = obs.t.values, obs.xf.values, obs.xb.values, obs.xc.values, obs.kf.values, obs.kb.values, obs.vrf.values, obs.vrb.values, obs.vf, obs.vb

        # locator = t>=(2*3600)
        # t=t[locator]-(2*3600)
        # front = front[locator]
        # rear = rear[locator]
        # nucleus=nucleus[locator]
        # vrf=vrf[locator]
        # vrb=vrb[locator]
        # kf=kf[locator]
        # kb=kb[locator]
        # vf = vf[locator]
        # vb=vb[locator]

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
        self.ax[0].set_ylim(rear.min()-10, front.max()+10)

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

        self.Ve_0_line.set_data(t/3600, np.ones(t.size)*self.Ve_0.value())

        self.fig.canvas.draw()
        DURATION=5

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = SimulationApp()
    main_window.show()
    sys.exit(app.exec_())
