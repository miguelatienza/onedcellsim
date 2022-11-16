using NPZ
include("simulate.jl")
#parameters = npzread("theta.npy")

t_max, t_step, t_step_compute, delta = 15*3600, 30, 0.5, 0

function run_simulations(parameters=undef, n_sims=1, t_max=15*3600, t_step=30, t_step_compute=0.5, delta=0)
    if parameters==undef
        parameters=[0]
    end
    
    df1 = simulate(parameters, t_max, t_step, t_step_compute, delta)

    if n_sims==1
        return df1
    end

    n_points, n_vars = size(df1)

    df = Array{Float64, 3}(undef, (n_sims, n_points, n_vars))

    for i in 2:n_sims
        #parameterset = parameters[i, :]
        try
            df[i, :, :]=simulate(parameters, t_max, t_step, t_step_compute, delta)
        catch y
            print(y, "\n")
            #exit()
        end
    end
    return df
end
