using Distributed
using BenchmarkTools
using Plots
using Dates
@everywhere using NPZ
@everywhere using ProgressBars
@everywhere include("../../simulations/simulate.jl")

@everywhere function runsims(args)
    worker, parameters = args
    #print("worker", worker, "\n")
    local t, df
    npoints = Int(t_max/t_step)+1
    nvars = 16
    nsims=500
    #t = zeros(Int64, (nsims, npoints)), 
    df = zeros(Float64, (nsims, npoints, nvars))
    
    if false #if worker==1
        for i in ProgressBar(1:nsims)
            id = nsims*(worker-1) + i
            #print(id, "\t")
            t, df[i, :, 3:end]=simulate(parameters[i, :], t_max, t_step, t_step_compute, delta)
            df[i, :, 1] .= id
            df[i, :, 2]= t
        end
        return df
    end

    for i in 1:nsims
        id = nsims*(worker-1) + i
        #print(id, "\t")
        try
            t, df[i, :, 3:end]=simulate(parameters[i, :], t_max, t_step, t_step_compute, delta)
            df[i, :, 1] .= id
            df[i, :, 2]= t 
        catch y
            print(y, "\n")
            print(parameters[i, :])
        end
        continue
        t, df[i, :, 3:end]=simulate(parameters, t_max, t_step, t_step_compute, delta)
        df[i, :, 1] .= id
        df[i, :, 2]= t
    end
    return df
end

print("Running simulations in parallel", "\n")

@everywhere nsims=10_000
#@everywhere nsims=10
@everywhere ncpus=20
@everywhere t_max, t_step, t_step_compute, delta = 15*3600, 30, 0.5, 0

@everywhere parameters = npzread("data/theta.npy")
@everywhere nsims = size(parameters)[1]
@everywhere worker_batch_size=500
@everywhere batch_size = worker_batch_size*nworkers()


for i in ProgressBar(1:Int(ceil(nsims/batch_size)))
    print(i, "\n")
    start_point = Int(
        (i-1)*batch_size+1)
    end_point = Int(min(start_point+batch_size-1, nsims))
    parameter_batch = parameters[start_point:end_point, :]
    #print((worker-1)*batch_size + 1, worker*batch_size)
    #continue
    #print((start_point, end_point), "\n")
    npzwrite("data/theta"*string(i)*".npy", parameter_batch)
    continue
    #print([((worker-1)*worker_batch_size + 1, worker*worker_batch_size) for worker in 1:nworkers()])
    args=[(worker, parameter_batch[(worker-1)*worker_batch_size + 1:worker*worker_batch_size, :]) for worker in 1:nworkers()]
    if i>10
        break
    end
    #df = pmap(runsims, args)

    #df = reduce(vcat, df)

    #npzwrite("df"*string(i)*".npy", df)
end
