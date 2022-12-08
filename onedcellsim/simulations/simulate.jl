include("functions.jl")
using ProgressBars

function single_time_step(stepsize, compute_step, n_sub_steps, dtot, params, variables)#), vf, vb, vc)
    
    #local aoverN, k0, W0, Ve_0, k_minus, E, L0, c1, c2, c3, k_lim, alpha
    #local zetaf, zetab, zetac, kf, kb, Lf, Lb, vrf, vrb, xf, xb, xc, vf, vb
    aoverN, k0, Ve_0, k_minus, E, L0, c1, c2, c3, k_lim, alpha, epsilon = copy(params)
    zetaf, zetab, zetac, kf, kb, Lf, Lb, vrf, vrb, xf, xb, xc, vf, vb = copy(variables)

   
    #local dt=compute_step
    for i in 1:n_sub_steps
        
    #Compute Velocities
        vf, vb, vrf, vrb = velocities(aoverN, zetaf, zetab, Ve_0, kf, kb, k_minus, E, Lf, Lb, L0)
        vc=get_vc(E, Lf, Lb, zetac)
   
        dkfdt = dk_dt(c1, c2, c3, k_lim, kf, k0, vrf, alpha, epsilon)
        
        dkbdt = dk_dt(c1, c2, c3, k_lim, kb, k0, vrb, alpha, epsilon)
   
        dt = min(compute_step, stepsize-dtot)
   
        kf = clamp(kf+dkfdt*dt, 0.01, k_lim)
        kb = clamp(kb+dkbdt*dt, 0.01, k_lim)

        #Compute positions
        xf, xb, xc = solve_x(dt, xf, xb, xc, vf, vb, vc)
        Lf = xf-xc
        Lb = xc-xb

    end
  
    variables = [zetaf zetab zetac kf kb Lf Lb vrf vrb xf xb xc vf vb]

    return variables

end


function simulate(params, t_max, t_step, t_step_compute=0.005, delta=0.02)

    local variables, parameters, i

    E,L0,Ve_0,k_minus,c1,c2,c3,k_max,Kk,nk,k0,zeta_max,Kzeta,nzeta,b,zeta0,alpha,aoverN,epsilon,B = params
    
    n_sub_steps = Int64(ceil(t_step/t_step_compute))
    t_step_compute = t_step/n_sub_steps
    n_steps = Int(t_max/t_step)
    ts = range(0, t_max, step=t_step)
    zeros_arr = zeros(size(ts,1))

    xb=copy(zeros_arr); xb[1]=-L0 #rear in um
    xc=copy(zeros_arr); xc[1]=0
    xf=copy(zeros_arr); xf[1]=L0#front in um
    L=copy(zeros_arr); L[1]=2*L0 # Cell length in um

    vb=copy(zeros_arr) #rear velpocity in um/s
    vc=copy(zeros_arr) #nucleus velocity in um/s
    vf=copy(zeros_arr) #front velocity in um/s
    
    vrf=copy(zeros_arr)#; vrf[1]=Ve_0
    vrb=copy(zeros_arr)#; vrb[1]=Ve_0
    
    kf=copy(zeros_arr); kf[1]=k0
    kb=copy(zeros_arr); kb[1]=k0

    zetaf=copy(zeros_arr)
    zetaf.=zeta_edge(zeta0, zeta_max, B, Kzeta, nzeta)
    zetab=copy(zetaf)
    zetac=copy(zeros_arr)
    zetac.=zeta_nuc(b, zeta0, zeta_max, B, Kzeta, nzeta)
    k_lim = k_lim_func(k0, k_max, B, Kk, nk)
    
    Lf = xf-xc
    Lb=xc-xb

    parameters = [aoverN, k0, Ve_0, k_minus, E, L0, c1, c2, c3, k_lim, alpha, epsilon]
    variables = [zetaf zetab zetac kf kb Lf Lb vrf vrb xf xb xc vf vb]
    
    dtot=0
    #single_time_step(t_step, t_step_compute, dtot, params, variables)

    i=1
    #print(variables[i,:])
    #print(variables[i,:])
    for i in 1:n_steps
        #print(i)
        variables[i+1,:]= single_time_step(t_step, t_step_compute, n_sub_steps, dtot, parameters, variables[i,:])
        
    end

    return Array(ts), variables

end

function array_to_dict(df)

    sim_dict = Dict{Int, Dict{String, Array{Float64, 1}}}(
        i => Dict{String, Array{Float64, 1}}(
        "t"=>df[i, :,1],
        "zetaf"=>df[i, :, 2],
        "zetab"=>df[i, :, 3],
        "zetac"=>df[i, :, 4],
        "kf"=>df[i, :, 5],
        "kb"=>df[i, :, 6],
        "Lf"=>df[i, :, 7],
        "Lb"=>df[i, :, 8],
        "vrf"=>df[i, :, 9],
        "vrb"=>df[i, :, 10],
        "xf"=>df[i, :, 11],
        "xb"=>df[i, :, 12],
        "xc"=>df[i, :, 13],
        "vf"=>df[i, :, 14],
        "vb"=>df[i, :, 15],
    ) for i in size(df)[1])

    if ndims(df)==2
        0
    end
    
    return sim_dict
end


function runsims(;parameters=undef, t_max=15*3600, t_step=30, t_step_compute=0.5, delta=0, nsims=1, verbose=false, mode="array")
   
    local t, df
 
    ##handle input parameters
    if parameters==undef
        parameters=[3e-3, 10, 3e-2, 5e-3, 1.5e-4, 7.5e-5, 7.8e-3, 35, 35, 3, 1e-2, 1.4, 50, 4, 3, 1e-1, 4e-2, 1, 1, 45]
    end
    
    if (ndims(parameters)==1) & (nsims>1)
        parameters = reshape(parameters, (1, size(parameters)[1]))
        parameters = repeat(parameters, outer=nsims)
    elseif (nsims==1) & (ndims(parameters)==1) 
        parameters = reshape(parameters, (1, size(parameters)[1]))
    end

    """zetaf zetab zetac kf kb Lf Lb vrf vrb xf xb xc vf vb"""
    if nsims==1e90
        t, df1 = simulate(parameters, t_max, t_step, t_step_compute, delta)
        id = ones(Int, (n_points))
        df = cat(2, id, t)
        df = cat(2, df, df1)
        return df
    end

    t, df1 = simulate(parameters[1,:], t_max, t_step, t_step_compute, delta)
    
    n_points, n_vars = size(df1)
    df = zeros(Float64, (nsims, n_points, n_vars+2))
    df[1, :, 3:end]=df1
    df[1, :, 1].=1
    df[1, :, 2]=t

    if nsims==1
        return df
    end

    if verbose
        for i in ProgressBar(2:nsims)
            t, df[i, :, 3:end]=simulate(parameters[i, :], t_max, t_step, t_step_compute, delta)
            df[i, :, 1] .= i
            df[i, :, 2]= t
        end
        if mode=="dict"
            return array_to_dict(df)
        end
        return df
    end

    for i in 2:nsims
        t, df[i, :, 3:end]=simulate(parameters[i, :], t_max, t_step, t_step_compute, delta)
        df[i, :, 1] .= i
        df[i, :, 2]= t
    end

    if mode=="dict"
        return array_to_dict(df)
    end

    return df
end