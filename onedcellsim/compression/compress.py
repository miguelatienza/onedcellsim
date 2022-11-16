import numpy as np
import torch


def gkernel(lsig=2, sig=30):
    
    l = int(np.ceil(lsig*sig +1))
    ax = np.linspace(-(l-1)/2, (l-1) / 2, l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = gauss
    
    return kernel/ np.sum(kernel)

def gsmooth(x, sig, lsig=2, mode='valid'):
    
    kernel = gkernel(sig=sig, lsig=lsig)
    xs = np.convolve(x, kernel, mode=mode)
    
    return xs


def stickslip_kernel(t_1, t_2, t_kernel, delta_t, h=1, normalise=True, normalise_height=1):
    """
    t_1: time in protuding phase
    t_2: time in relaxation phase
    t_kernel: length of kernel in same units as t_1 and t_2 
    delt_t: time between frames
    """
    t_tot = t_1+t_2
    
    if t_tot>t_kernel:
        print(t1, t2, t_kernel)
        raise ValueError('t_kernel must be at least as large as t_1+t_2')
    
    kernel_size = np.ceil(t_kernel/delta_t).astype(int)
    kernel_size+= (kernel_size%2+1)%2 ##force odd kernel size
    #print(kernel_size)
    
    t1_size = np.round(t_1/delta_t).astype(int)
    t2_size = np.round(t_2/delta_t).astype(int)
    
    x1 = np.linspace(0, h, t1_size)
    x2 = np.linspace(h, 0, t2_size)
    
    x = np.concatenate((x1, x2))
    x-=x.mean()
    #print(tot_size, x.size)
    leftzeros = np.zeros(np.round((kernel_size-x.size)/2).astype(int))
    right_zeros = np.zeros(kernel_size-x.size-leftzeros.size)
    
   
    x = np.concatenate((leftzeros, x, right_zeros))
    
    
    if normalise:
        x/=(x*x*normalise_height).sum()
    
    return x

def get_kernel_set(
    lengths=np.arange(10/60,2+0.01, 10/60),
    t1_fracs=np.round(np.arange(0.2, 1, 0.1), 3),
    max_length=2, 
    delta_t=30/3600):
    
    kernel_size = np.ceil(max_length/delta_t).astype(int)
    kernel_size+= (kernel_size%2+1)%2
    
    kernels=np.zeros([lengths.size*t1_fracs.size, kernel_size])
    
    i=0
    for t_kernel in lengths:
        for t1_frac in t1_fracs:
            t1 = t1_frac*t_kernel
            t2 = (1-t1_frac)*t_kernel
            #print(t1, t2, t_kernel)
            kernel = stickslip_kernel(t1, t2, max_length, delta_t)
            kernels[i]=kernel
            i+=1
    
    return torch.tensor(kernels, dtype=torch.float32)


def compressor(df, shape_kernels=None, v_kernel=torch.tensor([-0.5, 0, 0.5], dtype=torch.float32), sm=100, shape_th=0, v_th=0):
    
    if shape_kernels is None:
        shape_kernels = get_kernel_set()

    if isinstance(df, np.ndarray):
        X = torch.tensor(
        np.stack([df[:, 11], df[:, 12], df[:, 13]]),
             dtype=torch.float32)
        Xs = torch.tensor(
        np.stack([gsmooth(df[:,11], sm), 
              gsmooth(df[:, 12], sm), 
              gsmooth(df[:, 13], sm)]),dtype=torch.float32)

    else:
        X = torch.tensor(
        np.stack([df.xf.values, df.xc.values, df.xb.values]),
                dtype=torch.float32)

        Xs = torch.tensor(
        np.stack([gsmooth(df.xf.values, sm), 
                gsmooth(df.xc.values, sm), 
                gsmooth(df.xb.values, sm)]),dtype=torch.float32)

    X = X[:, sm:-sm]
    Xf = X-Xs
    
    n_channels = X.shape[0]
    n_shape_kernels, kernel_size = shape_kernels.shape
    v_kernel_size = v_kernel.shape[0]
    x_len = X.shape[1]
    n_out = n_shape_kernels*n_channels
    crop_v = kernel_size-v_kernel_size
    
    #if crop_v%2 != 0:
    #    raise ValueError('shape kernel size should be odd')
    crop_v = int(crop_v/2)
    kernels_in = shape_kernels.view(n_shape_kernels, 1, kernel_size).repeat(n_channels, 1, 1)
    
    #shape_bias = torch.tensor(-shape_th, dtype=torch.float32).repeat(n_shape_kernels*n_channels)
    shape_layer = torch.nn.functional.conv1d(Xf, kernels_in, groups=n_channels)#, bias=False)
    shape_layer = shape_layer.view(n_channels, n_shape_kernels, x_len-kernel_size+1)
    
    shape_layer_plus = torch.nn.functional.relu(shape_layer-shape_th, inplace=False)+shape_th 
    shape_layer_minus = torch.nn.functional.relu(-shape_layer-shape_th, inplace=False)+shape_th
    shape_layer = torch.cat((shape_layer_plus, shape_layer_minus), axis=1)
    #shape_layer = torch.clone(shape_layer_plus)
    ##Now apply max pooling
    shape_layer = torch.nn.functional.max_pool1d(shape_layer, kernel_size)
    
    #v_bias = torch.tensor(dtype=torch.float32).repeat(n_channels)
    v_kernel = v_kernel.view(1, 1, v_kernel.size()[0]).repeat(n_channels,1,1)
    velocity_layer = torch.nn.functional.conv1d(Xs, v_kernel, groups=n_channels)[:, crop_v:-crop_v]
    #print(shape_layer.shape, velocity_layer.shape)
    n_velocity_layer = -torch.clone(velocity_layer)
    
    velocity_layer = torch.stack([velocity_layer, n_velocity_layer])-v_th
    
    velocity_layer = torch.nn.functional.relu(velocity_layer)+v_th
    
    velocity_layer = velocity_layer[0]-velocity_layer[1]
    ##Apply average pooling in this case
    velocity_layer = torch.nn.functional.avg_pool1d(velocity_layer, kernel_size)
    
    
    #print(shape_layer.shape, velocity_layer.shape)
    ##Finally combine the shape and velocity layers
    first_layer = torch.cat((shape_layer, velocity_layer[:, None, :]), axis=1)
    #first_layer = torch.stack()
    return first_layer

def stat_to_image(stat, lengths, t1_fracs, delta_t):

    stat = np.array(stat)
    nkernels = lengths.size*t1_fracs.size
    stat = stat.mean(axis=-1).T[:-1, 0]
    stat = (stat[:nkernels])#+stat[nkernels:])
    
    stat = stat.reshape((lengths.size, t1_fracs.size))
    return stat