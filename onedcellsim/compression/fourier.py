"""Compression modeule for sbi.model"""

import torch
import numpy as np

class Compressor:
    """Compressor class for sbi.model"""
    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, x, type='simulation'):
        """Compresses the input data x using the model"""
        if type=='experiment':
            x=np.array(x)
            x = x.reshape(1, -1, 3)
            x = x - x[0, 0, 2]
            if x.shape[1]%2==0:
                x = x[:, :-1, :]
            print(x.shape)
            #return self.compress_experiment(x)
        else:
            x = np.array(x)
            x = x[:, :, 11:14]
        Lf = x[:, :, 0]-x[:, :, 2]
        Lb = -x[:, :, 1]+x[:, :, 2]
        n_sims = x.shape[0]
        n_points = x.shape[1]
    
        freqs = np.fft.rfftfreq(n_points, d=2)
        print(1/freqs)
        indices = (freqs>=1/120) & (freqs<=1/10)
        freqs = freqs[indices]
        print(1/freqs)
        coarse_frames = np.round(np.linspace(0, n_points-1, len(freqs))).astype(int)

        t = np.arange(n_points)*2
        t = t[coarse_frames]

        X = np.zeros([n_sims, len(freqs), 3])
        
        for sim in range(n_sims):
            Lf_spectrum = np.fft.rfft(Lf[sim])[indices]
            Lb_spectrum = np.fft.rfft(Lb[sim])[indices]
            v = np.gradient(x[sim, coarse_frames, 2], t)
            X[sim, :, 0] = np.abs(Lf_spectrum)
            X[sim, :, 1] = np.abs(Lb_spectrum)
            X[sim, :, 0] = v
        X = torch.from_numpy(X).float().flatten(start_dim=1).to(self.device)
        #x = x.flatten(start_dim=1)
        return X

    def compress_experiment(self, x):

        x = np.array(x)
        x = x.reshape(1, -1, 3)
        Lf = x[:, 0]-x[:, 2]
        Lb = -x[:, 1]+x[:, 2]
        n_sims = 1
        n_points = x.shape[1]
    
        freqs = np.fft.rfftfreq(n_points, d=2)
        indices = (freqs>=1/120) & (freqs<=1/10)
        freqs = freqs[indices]
        
        coarse_frames = np.round(np.linspace(0, n_points-1, len(freqs))).astype(int)

        t = np.arange(n_points)*2
        t = t[coarse_frames]

        X = np.zeros([n_sims, len(freqs), 3])
        
        for sim in range(n_sims):
            Lf_spectrum = np.fft.rfft(Lf[sim])[indices]
            Lb_spectrum = np.fft.rfft(Lb[sim])[indices]
            v = np.gradient(x[sim, coarse_frames, 2], t)
            X[sim, :, 0] = np.abs(Lf_spectrum)
            X[sim, :, 1] = np.abs(Lb_spectrum)
            X[sim, :, 2] = v
        x = torch.from_numpy(x).float().flatten(start_dim=1).to(self.device)
        #x = x.flatten(start_dim=1)
        return x
    
    def __repr__(self):
        return f'Compressor(device={self.device})'
    
    def get_spectrum(self, X, sim_number, n_points, dt=2, T_smooth=None, T_detrend=None, max_freq=0.1):
        """Get the spectrum of a simulation"""
        if T_smooth is None:
            T_smooth = 1
        if T_detrend is None:
            T_detrend = int(60/dt)
        
        #Get the front and rear positions
        front = X[sim_number, :,0]
        rear = X[sim_number, :,1]
        
        #make sure the front and rear are long enough
        assert front.size>=1/(max_freq*dt), "The size of the array must be larger than or equal to the smoothening filter"
        #Smooth the front and rear positions
        front_smooth = self.smooth_linesegs(front, T_smooth)
        rear_smooth = self.smooth_linesegs(rear, T_smooth)

        #Detrend the front and rear positions
        front_detrend = front_smooth - self.smooth_linesegs(front, T_detrend)
        rear_detrend = rear_smooth - self.smooth_linesegs(rear, T_detrend)

        #Get the signal
        front_spectrum = np.fft.rfft(front_detrend)[1:]
        rear_spectrum = np.fft.rfft(rear_detrend)[1:]
        freqs = np.fft.rfftfreq(front_detrend.size, d=dt)[1:]
        #print(front_spectrum.shape, rear_spectrum.shape, freqs.shape)
        return freqs, front_spectrum, rear_spectrum
    
    def smooth_linesegs(self, x, sm):
        """Smoothen x by making straight lines between every sm points.

        Args:
            x (_type_): Array to be smoothened
            sm (_type_): the width of the smoothening filter
        """
        assert x.size>=sm, "The size of the array must be larger than or equal to the smoothening filter"

        #Create a copy of x
        x_out = x.copy()
        n_valid_points = np.round(x.size/sm).astype(int)+1
        
        valid_points = np.linspace(0, x.size-1, n_valid_points).astype(int)

        x_out[valid_points[1:-1]] = self.smooth(x, int(sm/2))[valid_points[1:-1]]

        points_to_interpolate = np.arange(0, x.size)
        points_to_interpolate = np.delete(points_to_interpolate, valid_points)
        x_out[points_to_interpolate] = np.interp(points_to_interpolate, valid_points, x_out[valid_points])
        return x_out

    def smooth(self, a,ws):
        # a: NumPy 1-D array containing the data to be smoothed
        # WSZ: smoothing window size needs, which must be odd number,
        # as in the original MATLAB implementation
        ws += not (ws%2)
        out0 = np.convolve(a,np.ones(ws,dtype=int),'valid')/ws   
        r = np.arange(1,ws-1,2)
        start = np.cumsum(a[:ws-1])[::2]/r
        stop = (np.cumsum(a[:-ws:-1])[::2]/r)[::-1]
        return np.concatenate((start , out0, stop)) 