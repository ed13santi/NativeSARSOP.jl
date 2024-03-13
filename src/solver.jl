Base.@kwdef struct SARSOPSolver{LOW,UP} <: Solver
    epsilon::Float64    = 0.5
    precision::Float64  = 1e-3
    kappa::Float64      = 0.5
    delta::Float64      = 1e-1
    max_time::Float64   = 1.0
    max_steps::Int      = typemax(Int)
    verbose::Bool       = true
    initial_bounds_uncertainty::Float64 = 0.01
    init_lower::LOW     = BlindLowerBound(bel_res = initial_bounds_uncertainty)
    init_upper::UP      = FastInformedBound(bel_res=initial_bounds_uncertainty)
    prunethresh::Float64= 0.10
    path::String
end

function POMDPTools.solve_info(solver::SARSOPSolver, pomdp::POMDP)
    tree = SARSOPTree(solver, pomdp)

    t0 = time()
    iter = 0
    flag = "1"
    while time()-t0 < solver.max_time && root_diff(tree) > solver.precision
        # println("STEPS")
        # println(tree.V_upper[1])
        # println(tree.V_lower[1])
        if solver.verbose
            print(time()-t0)
        end
        sample!(solver, tree)
        backup!(tree)
        prune!(solver, tree)
        iter += 1
    end

    # print final difference
    println(root_diff(tree))
    # if the difference is bigger than the precision (program stopped due to time), write a flag to a file
    if root_diff(tree) <= solver.precision
        flag = "0"
    end
    file = open(solver.path, "w")
    println(file, flag)
    close(file)


    pol = AlphaVectorPolicy(
        pomdp,
        getproperty.(tree.Γ, :alpha),
        ordered_actions(pomdp)[getproperty.(tree.Γ, :action)]
    )
    return pol, (;
        time = time()-t0, 
        tree,
        iter,
    )
end

POMDPs.solve(solver::SARSOPSolver, pomdp::POMDP) = first(solve_info(solver, pomdp))
