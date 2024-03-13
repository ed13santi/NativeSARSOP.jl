Base.@kwdef struct SARSOPSolver{LOW,UP} <: Solver
    epsilon::Float64    = 0.5
    precision::Float64  = 1e-3
    kappa::Float64      = 0.5
    delta::Float64      = 1e-1
    max_time::Float64   = 1.0
    max_steps::Int      = typemax(Int)
    verbose::Bool       = true
    init_lower::LOW     = BlindLowerBound(bel_res = 0.1)
    init_upper::UP      = FastInformedBound(bel_res=0.1)
    prunethresh::Float64= 0.10
end

function POMDPTools.solve_info(solver::SARSOPSolver, pomdp::POMDP)
    tree = SARSOPTree(solver, pomdp)

    t0 = time()
    iter = 0
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


    # println("FINAL")
    println(root_diff(tree))


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
