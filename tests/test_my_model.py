# from onedcellsim.simulations import new_model
# import numpy as np

# def test_new_model():

#     parameters = {'L0': 25, 'gamma_f': 1e-3, 'gamma_b': 1e-3, 'mf': 1, 'mb': 1, 'mc': 0.1}
#     model = new_model.Model(parameters)
#     t_span = [0, 3600*5]
#     yinit = [30, 30, 0, 0.002, 0.002, 0]
#     sol = model.solve(t_span, yinit)
#     Lf, Lb, xc, vf, vb, vc = sol.y
#     assert sol.y.shape[0] == 6
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     ax.plot(sol.t/3600, Lf+xc, label='Lf+xc')
#     ax.plot(sol.t/3600, xc-Lb, label='xc-Lb')
#     ax.plot(sol.t/3600, xc, label='xc')
#     ax.set_ylim([-200, 200])
#     #ax.set_xlim(0, 30)
#     ax.legend()
#     plt.show()

from onedcellsim.simulations import new_model
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# Define the parameter values and ranges
L0_min, L0_max, L0_step = 20, 30, 0.1
gamma_f_min, gamma_f_max, gamma_f_step = 1e-4, 1e-2, 1e-4
gamma_b_min, gamma_b_max, gamma_b_step = 1e-4, 1e-2, 1e-4
mf_min, mf_max, mf_step = 0.1, 10, 0.1
mb_min, mb_max, mb_step = 0.1, 10, 0.1
mc_min, mc_max, mc_step = 0.01, 1, 0.01

# Define the slider widgets for each parameter
L0_slider = widgets.FloatSlider(value=25, min=L0_min, max=L0_max, step=L0_step, description='L0')
gamma_f_slider = widgets.FloatSlider(value=1e-3, min=gamma_f_min, max=gamma_f_max, step=gamma_f_step, description='gamma_f')
gamma_b_slider = widgets.FloatSlider(value=1e-3, min=gamma_b_min, max=gamma_b_max, step=gamma_b_step, description='gamma_b')
mf_slider = widgets.FloatSlider(value=1, min=mf_min, max=mf_max, step=mf_step, description='mf')
mb_slider = widgets.FloatSlider(value=1, min=mb_min, max=mb_max, step=mb_step, description='mb')
mc_slider = widgets.FloatSlider(value=0.1, min=mc_min, max=mc_max, step=mc_step, description='mc')

# Define the plotting function
def plot_simulation(L0, gamma_f, gamma_b, mf, mb, mc):
    parameters = {'L0': L0, 'gamma_f': gamma_f, 'gamma_b': gamma_b, 'mf': mf, 'mb': mb, 'mc': mc}
    model = new_model.Model(parameters)
    t_span = [0, 3600*5]
    yinit = [30, 30, 0, 0.002, 0.002, 0]
    sol = model.solve(t_span, yinit)
    Lf, Lb, xc, vf, vb, vc = sol.y
    assert sol.y.shape[0] == 6
    fig, ax = plt.subplots()
    ax.plot(sol.t/3600, Lf+xc, label='Lf+xc')
    ax.plot(sol.t/3600, xc-Lb, label='xc-Lb')
    ax.plot(sol.t/3600, xc, label='xc')
    ax.set_ylim([-200, 200])
    ax.legend()
    plt.show()

# Combine the sliders and plotting function into an interactive widget
interactive_plot = widgets.interactive(plot_simulation, L0=L0_slider, gamma_f=gamma_f_slider, gamma_b=gamma_b_slider, mf=mf_slider, mb=mb_slider, mc=mc_slider)
plt.show()
display(interactive_plot)