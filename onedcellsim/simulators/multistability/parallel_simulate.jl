using Distributed
using BenchmarkTools
using Plots
using Dates
@everywhere using NPZ
@everywhere using ProgressBars
@everywhere include("simulate.jl")

@everywhere function runsims(args)
    worker, parameters, nsims, t_max, t_step, t_step_compute, delta = args
    #print("worker", worker, "\n")
    local t, df
    npoints = Int(t_max/t_step)+1
    nvars = 16
    #t = zeros(Int64, (nsims, npoints)), 
    df = zeros(Float64, (nsims, npoints, nvars))

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

@everywhere function run_batches(nsims, parameters, nworkers, out_path; t_max=15*3600, t_step=30, t_step_compute=0.5, delta=0)
    
    batch_size=worker_batch_size*ncpus


    for i in ProgressBar(1:Int(ceil(nsims/batch_size)))
        print(i, "\n")
        start_point = Int(
            (i-1)*batch_size+1)
        end_point = Int(min(start_point+batch_size-1, nsims))
        parameter_batch = parameters[start_point:end_point, :]
        npzwrite("data/theta"*string(i)*".npy", parameter_batch)
        continue
        #print([((worker-1)*worker_batch_size + 1, worker*worker_batch_size) for worker in 1:nworkers()])
        args=[
            (worker, 
            parameter_batch[(worker-1)*worker_batch_size + 1:worker*worker_batch_size, :],
            nsims, t_max, t_step, t_step_compute, delta 
            ) for worker in 1:nworkers()]
    
        df = pmap(runsims, args)

        df = reduce(vcat, df)

        npzwrite("df"*string(i)*".npy", df)
    end
