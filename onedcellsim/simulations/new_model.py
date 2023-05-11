import numpy as np
from scipy.integrate import solve_ivp

class Model():

    def __init__(self, parameters):

        self.parameters = parameters

        self.L0 = parameters['L0']
        self.gamma_f = parameters['gamma_f']
        self.gamma_b = parameters['gamma_b']
        self.mf = parameters['mf']
        self.mb = parameters['mb']
        self.mc = parameters['mc']
        self.Fc = parameters['Fc']
        self.wc = parameters['wc']
        self.zeta = parameters['zeta']
        self.phi = parameters['phi']

        self.var_names = ['Lf', 'Lb', 'xc', 'vf', 'vb', 'vc']

    def dydt(self, t, y):

        Lf, Lb, xc, vf, vb, vc = y

        Lfdot = vf
        Lbdot = vb
        xcdot = vc
        vfdot = (-self.gamma_f * (Lf-self.L0) /self.mf) - self.zeta*vf + (self.Fc * np.cos(2*np.pi*self.wc*t) / self.mf)
        vbdot = (-self.gamma_b * (Lb-self.L0)/self.mb) - self.zeta*vb + (self.Fc * np.cos(np.pi*2*self.wc*t - self.phi/np.pi) / self.mb)

        vcdot = self.gamma_f * ((Lf-self.L0) - (Lb-self.L0))/self.mc

        return [Lfdot, Lbdot, xcdot, vfdot, vbdot, vcdot]
    
    def solve(self, t_span, yinit, t_eval=None):

        sol = solve_ivp(self.dydt, t_span, yinit, t_eval=t_eval)
        return sol
    
class GovModel():

    def __init__(self, parameters):

        self.parameters = parameters

        #self.L0 = parameters['L0']
        self.E = parameters['E']
        self.Ve0 = parameters['Ve0']
        #self.zeta = parameters['zeta']
        #self.N = parameters['N']
        #self.kon =  parameters['kon']
        #self.koff = parameters['koff']
        self.r = parameters['r']
        self.fs = parameters['fs']
        self.k = parameters['k']
        #self.alpha = parameters['alpha']

        self.var_names = ['Lf', 'n']

        #self.t_step = 5/100
        print(parameters)
    def dydt(self, t, y):

        Lf, n = y

        #Lf = np.clip(Lf, 0, 8)
        exp = self.E*(Lf-1)/(n*self.fs)
        if exp>10:
            pass#exp=10
        Lf_dot = self.Ve0 - ((self.E*(Lf-1))
                             *(1 - (np.exp(exp)/(n*self.k))))
        # if Lf>=8:
        #     Lf_dot=-1
        #Lf_dot = np.clip(Lf_dot, -10, 10)
        #Lf_dot = np.clip(Lf_dot, -10, 10)
        
        n_dot = (self.r*(1-n)) - (n*np.exp(exp))
        if n<=0.3:
            n_dot =-n_dot
            return [Lf_dot, n_dot]
        #n_dot = np.clip(n_dot, (0.01-n)/self.t_step, np.inf)

        #n_dot = np.clip(n_dot, -10, 10)

        return [Lf_dot, n_dot]
    
    def solve(self, t_span, yinit, t_eval=None, t_shift=0.2, method='RK45', atol=1e50):
        t_last = t_span[0]
        y = []
        t= []
        
        # y = np.zeros([2, len(t_eval)])
        # t_step = (t_eval[1]-t_eval[0])/len(t_eval)
        # print(t_step)
        # for step in range(len(t_eval)):
        #     if step==0:
        #         y[:, step] = yinit
        #     else:
        #         dydt = np.array(self.dydt(t_eval[step], y[:, step-1]))
        #         y[:, step] = y[:, step-1] + t_step*dydt

        # return t_eval, y



        sol = solve_ivp(self.dydt, t_span, yinit, t_eval=t_eval, method=method)
        print(sol.message)
        return sol.t, sol.y
        while t_last<=t_span[-1]:
            t_eval_seg = t_eval[t_eval>=t_last+t_shift] if t_last>0 else t_eval
            yinit = y[-1][:, -1] if len(y)>0 else yinit
            print(yinit)
            sol = solve_ivp(self.dydt, t_span, y[-1][:, -1], t_eval=t_eval_seg)
            print(sol.t)
            t_last = sol.t[-1]
            t.append(sol.t)
            y.append(sol.y)
       
        y = np.concatenate(y, axis=-1)
        t = np.concatenate(t, axis=-1)
        return t, y
    