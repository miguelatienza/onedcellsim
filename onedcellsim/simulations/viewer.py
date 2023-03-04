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

jpath = "/home/miguel/onedcellsim/venv/julia-1.6.7/bin/julia"
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

def run_simulation(params):
    
    full_params = np.array(params)
    #print(full_params)
    t_max, t_step = 15*60*60,30
    t_step_compute=0.5
    #params, particle_id, verbose, t_max, t_step, t_step_compute = args
    
    #np.random.seed(int(time.time())
    variables = simulate(parameters=full_params, t_max=t_max, t_step=t_step, t_step_compute=t_step_compute, delta=0)[0]
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
        self.line_4_1, = self.ax[1].plot([], [], color='red', alpha=0.5)
        self.line_5_1, = self.ax[1].plot([], [], color='blue', alpha=0.5)
        
          
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
        self.ax[2].plot(np.linspace(0,15,100), np.zeros(100), color='black', alpha=0.5)
        self.Ve_0_line, = self.ax[2].plot(np.linspace(0,15,100), np.zeros(100), color='green', alpha=0.5, label='$\mathrm{V_e^0}$')
        self.ax[2].legend()
        
        ##Plot the retrograde force
        self.line_8, = self.ax[3].plot([], [], color='red', label='front')
        self.line_9, = self.ax[3].plot([], [], color='blue', label='rear')
        self.ax[3].plot(np.linspace(0,15,100), np.zeros(100), color='black', alpha=0.5)

        self.ax[3].set_xlabel('Time in hours', fontsize=20)
        #self.ax[3].set_ylabel(r'$\mathrm{Protrusion force (nN \mu m^{-1})}$', fontsize=20)
        self.ax[3].set_ylabel(r'$\mathrm{F_p (nN \mu m^{-1})}$', fontsize=20)
        self.ax[3].legend()
        
        self.create_widgets()
        self.initUI()
        self.update_simulation()


    def create_widgets(self):

        # Create sliders for the parameters
        self.E = QtWidgets.QDoubleSpinBox()
        self.E.setMinimum(0.0000)
        self.E.setMaximum(0.3)
        self.E.setSingleStep(0.000030)
        self.E.setDecimals(6)
        self.E.setValue(3e-3)

        self.L0 = QtWidgets.QDoubleSpinBox()
        self.L0.setMinimum(5)
        self.L0.setMaximum(100.0)
        self.L0.setSingleStep(0.1)
        self.L0.setValue(10)

        self.Ve_0 = QtWidgets.QDoubleSpinBox()
        self.Ve_0.setMinimum(0)
        self.Ve_0.setMaximum(5e-2)
        self.Ve_0.setSingleStep(0.0003)
        self.Ve_0.setDecimals(6)
        self.Ve_0.setValue(3e-2)
        
        
        self.k_minus = QtWidgets.QDoubleSpinBox()
        self.k_minus.setMinimum(0)
        self.k_minus.setMaximum(3e-2)
        self.k_minus.setSingleStep(5e-05)
        self.k_minus.setDecimals(6)
        self.k_minus.setValue(0.005)
        
        self.c1 = QtWidgets.QDoubleSpinBox()
        self.c1.setMinimum(0)
        self.c1.setMaximum(0.015)
        self.c1.setSingleStep(1.4999999999999998e-06)
        self.c1.setDecimals(6)
        self.c1.setValue(0.00015)
        
        self.c2 = QtWidgets.QDoubleSpinBox()
        self.c2.setMinimum(0)
        self.c2.setMaximum(0.0075)
        self.c2.setSingleStep(7.499999999999999e-07)
        self.c2.setDecimals(6)
        self.c2.setValue(7.5e-05)
        
        self.c3 = QtWidgets.QDoubleSpinBox()
        self.c3.setMinimum(7.8e-05)
        self.c3.setMaximum(0.7799999999999999)
        self.c3.setSingleStep(7.8e-05)
        self.c3.setDecimals(6)
        self.c3.setValue(0.0078)
        
        self.k_max = QtWidgets.QDoubleSpinBox()
        self.k_max.setMinimum(0)
        self.k_max.setMaximum(3500.0)
        self.k_max.setSingleStep(0.35000000000000003)
        self.k_max.setDecimals(6)
        self.k_max.setValue(35)
        
        self.Kk = QtWidgets.QDoubleSpinBox()
        self.Kk.setMinimum(0)
        self.Kk.setMaximum(3500)
        self.Kk.setValue(35)

        self.nk = QtWidgets.QDoubleSpinBox()
        self.nk.setMinimum(0)
        self.nk.setMaximum(6)
        self.nk.setSingleStep(0.1)
        self.nk.setValue(3)
        
        self.k0 = QtWidgets.QDoubleSpinBox()
        self.k0.setMinimum(0.000)
        self.k0.setMaximum(1.0)
        self.k0.setSingleStep(0.0001)
        self.k0.setDecimals(6)       
        self.k0.setValue(0.01)
               
        self.zeta_max = QtWidgets.QDoubleSpinBox()
        self.zeta_max.setMinimum(0.0)
        self.zeta_max.setMaximum(140.0)
        self.zeta_max.setSingleStep(0.013999999999999999)
        self.zeta_max.setDecimals(6)
        self.zeta_max.setValue(1.4)

        self.Kzeta = QtWidgets.QDoubleSpinBox()
        self.Kzeta.setMinimum(0)
        self.Kzeta.setMaximum(5000.0)
        self.Kzeta.setSingleStep(0.5)
        self.Kzeta.setDecimals(6)
        self.Kzeta.setValue(50)
        
        self.nzeta = QtWidgets.QDoubleSpinBox()
        self.nzeta.setMinimum(0.0)
        self.nzeta.setMaximum(10)
        self.nzeta.setSingleStep(0.1)
        self.nzeta.setDecimals(6)
        self.nzeta.setValue(4)
                
        self.b = QtWidgets.QDoubleSpinBox()
        self.b.setMinimum(0.0)
        self.b.setMaximum(300.0)
        self.b.setSingleStep(0.03)
        self.b.setDecimals(6)
        self.b.setValue(3)

        self.zeta0 = QtWidgets.QDoubleSpinBox()
        self.zeta0.setMinimum(0.00)
        self.zeta0.setMaximum(10.0)
        self.zeta0.setSingleStep(0.001)
        self.zeta0.setDecimals(6)
        self.zeta0.setValue(0.1)
           
        self.aoverN = QtWidgets.QDoubleSpinBox()
        self.aoverN.setMinimum(0.0)
        self.aoverN.setMaximum(100.0)
        self.aoverN.setSingleStep(0.01)
        self.aoverN.setDecimals(6)
        self.aoverN.setValue(1)

        self.epsilon = QtWidgets.QDoubleSpinBox()
        self.epsilon.setMinimum(0.0)
        self.epsilon.setMaximum(100.0)
        self.epsilon.setSingleStep(0.01)
        self.epsilon.setDecimals(6)
        self.epsilon.setValue(1)
        
        self.B = QtWidgets.QDoubleSpinBox()
        self.B.setMinimum(0)
        self.B.setMaximum(120)
        self.B.setSingleStep(1)
        self.B.setValue(60)
        self.B.setDecimals(6)

        #Update plot when the widgets are modfified
        self.E.valueChanged.connect(self.update_simulation)
        self.L0.valueChanged.connect(self.update_simulation)
        self.Ve_0.valueChanged.connect(self.update_simulation)
        self.k_minus.valueChanged.connect(self.update_simulation)
        self.c1.valueChanged.connect(self.update_simulation)
        self.c2.valueChanged.connect(self.update_simulation)
        self.c3.valueChanged.connect(self.update_simulation)
        self.k_max.valueChanged.connect(self.update_simulation)
        self.Kk.valueChanged.connect(self.update_simulation)
        self.nk.valueChanged.connect(self.update_simulation)
        self.k0.valueChanged.connect(self.update_simulation)
        self.zeta_max.valueChanged.connect(self.update_simulation)
        self.Kzeta.valueChanged.connect(self.update_simulation)
        self.nzeta.valueChanged.connect(self.update_simulation)
        self.b.valueChanged.connect(self.update_simulation)
        self.zeta0.valueChanged.connect(self.update_simulation)
        self.aoverN.valueChanged.connect(self.update_simulation)
        self.epsilon.valueChanged.connect(self.update_simulation)
        self.B.valueChanged.connect(self.update_simulation)

        ##Add labels to the widgets
        self.E.setPrefix(" E")
        self.E.setToolTip("E")
        self.L0.setPrefix("L0: ")
        self.L0.setToolTip("L0")
        self.Ve_0.setPrefix("Ve_0: ")
        self.Ve_0.setToolTip("Ve_0")
        self.k_minus.setPrefix("k_minus: ")
        self.k_minus.setToolTip("k_minus")
        self.c1.setPrefix("c1: ")
        self.c1.setToolTip("c1: ")
        self.c2.setPrefix("c2: ")
        self.c2.setToolTip("c2")
        self.c3.setPrefix("c3: ")
        self.c3.setToolTip("c3")
        self.k_max.setPrefix("k_max: ")
        self.k_max.setToolTip("k_max")
        self.Kk.setPrefix("Kk: ")
        self.Kk.setToolTip("Kk")
        self.nk.setPrefix("nk: ")
        self.nk.setToolTip("nk")
        self.k0.setPrefix("k0: ")
        self.k0.setToolTip("k0")
        self.zeta_max.setPrefix("zeta_max: ")
        self.zeta_max.setToolTip("zeta_max")
        self.Kzeta.setPrefix("Kzeta: ")
        self.Kzeta.setToolTip("Kzeta")
        self.nzeta.setPrefix("nzeta: ")
        self.nzeta.setToolTip("nzeta")
        self.b.setPrefix("b: ")
        self.b.setToolTip("b")
        self.zeta0.setToolTip("zeta0")
        self.zeta0.setPrefix("zeta0: ")
        self.aoverN.setPrefix("aoverN: ")
        self.aoverN.setToolTip("aoverN")
        self.epsilon.setPrefix("epsilon: ")
        self.epsilon.setToolTip("epsilon")
        self.B.setPrefix("B: ")
        self.B.setToolTip("B: ")


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
        sliders_layout.addWidget(self.E, 0, 0)
        sliders_layout.addWidget(self.L0, 1, 0)
        sliders_layout.addWidget(self.Ve_0, 2, 0)
        sliders_layout.addWidget(self.k_minus, 3, 0)
        sliders_layout.addWidget(self.c1, 4, 0)
        sliders_layout.addWidget(self.c2, 5, 0)
        sliders_layout.addWidget(self.c3, 6, 0)
        sliders_layout.addWidget(self.k_max, 7, 0)
        sliders_layout.addWidget(self.Kk, 8, 0)
        sliders_layout.addWidget(self.nk, 9, 0)
        sliders_layout.addWidget(self.k0, 10, 0)
        sliders_layout.addWidget(self.zeta_max, 11, 0)
        sliders_layout.addWidget(self.Kzeta, 12, 0)
        sliders_layout.addWidget(self.nzeta, 13, 0)
        sliders_layout.addWidget(self.b, 14, 0)
        sliders_layout.addWidget(self.zeta0, 15, 0)
        #sliders_layout.addWidget(self.alpha, 16, 0)
        sliders_layout.addWidget(self.aoverN, 17, 0)
        sliders_layout.addWidget(self.epsilon, 18, 0)
        sliders_layout.addWidget(self.B, 19, 0)
        
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
        k_minus = self.k_minus.value()
        c1 = self.c1.value()
        c2 = self.c2.value()
        c3 = self.c3.value()
        k_max = self.k_max.value()
        Kk = self.Kk.value()
        nk = self.nk.value()
        k0 = self.k0.value()
        zeta_max = self.zeta_max.value()
        Kzeta = self.Kzeta.value()
        nzeta = self.nzeta.value()
        b = self.b.value()
        zeta0 = self.zeta0.value()
        #alpha = self.alpha.value()
        aoverN = self.aoverN.value()
        epsilon = self.epsilon.value()
        B = self.B.value()
        params = [E, L0, Ve_0, k_minus, c1, c2, c3, k_max, Kk, nk, k0, zeta_max, Kzeta, nzeta, b, zeta0, 4e-2, aoverN, epsilon, B]
        obs = run_simulation(params)
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


        self.ax[0].set_xlim(-0.1, 15.1)
        self.ax[0].set_ylim(rear.min()-10, front.max()+10)

        self.ax[1].set_xlim(-0.1, 15.1)
        self.ax[1].set_ylim(-0.1, self.k_max.value()+1)

        self.ax[2].set_xlim(-0.1, 15.1)
        self.ax[2].set_ylim(min(vrb.min(), vrf.min()), 
            max(vrb.max(), vrf.max()))  

        self.ax[3].set_xlim(-0.1, 15.1)
        self.ax[3].set_ylim(min(Fb.min(), Ff.min()), 
            max(Fb.max(), Ff.max())) 
        # self.ax[3].set_ylim(min(checkb.min(), checkf.min()), 
        #     max(checkb.max(), checkf.max()))   
        # self.ax[3].set_ylim(-0.01, 0.01)   

        # self.Ve_0_line.set_data(t/3600, np.ones(t.size)*self.Ve_0.value())

        self.fig.canvas.draw()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = SimulationApp()
    main_window.show()
    sys.exit(app.exec_())
