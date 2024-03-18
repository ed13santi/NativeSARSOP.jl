Base.@kwdef struct FastInformedBound <: Solver
    max_iter::Int               = typemax(Int)
    max_time::Float64           = 1.
    bel_res::Float64            = 1e-3
    init_value::Float64         = 0.
    α_tmp::Vector{Float64}      = Float64[]
    residuals::Vector{Float64}  = Float64[]
    r_max::Float64  
end

function bel_res(α1, α2)
    max_res = 0.
    @inbounds for i ∈ eachindex(α1, α2)
        res = abs(α1[i] - α2[i])
        res > max_res && (max_res = res)
    end
    return max_res
end

function update!(𝒫::ModifiedSparseTabular, M::FastInformedBound, Γ, 𝒮, 𝒜, 𝒪)
    (;R,T,O) = 𝒫
    γ = discount(𝒫)
    residuals = M.residuals

    for a ∈ 𝒜
        α_a = M.α_tmp
        T_a = T[a]
        O_a = O[a]
        nz = nonzeros(T_a)
        rv = rowvals(T_a)

        for s ∈ 𝒮
            rsa = R[s,a]

            if isinf(rsa)
                α_a[s] = -Inf
            elseif isterminal(𝒫,s)
                α_a[s] = 0.
            else
                next = 0
                for s_next ∈ 𝒮
                    Tprob = T_a[s, s_next]
                    Vmax = -Inf
                    for α′ ∈ Γ
                        tmp = α′[s_next]
                        tmp > Vmax && (Vmax = tmp)
                    end
                    next += Tprob*Vmax
                end
                α_a[s] = rsa + γ*next
            end
        end
        res = bel_res(Γ[a], α_a)
        residuals[a] = res
        copyto!(Γ[a], α_a)
    end
end

function POMDPs.solve(sol::FastInformedBound, pomdp::POMDP)
    t0 = time()
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    O = ordered_observations(pomdp)
    γ = discount(pomdp)

    init_value = sol.init_value
    # Γ = if isfinite(sol.init_value)
    #     [fill(sol.init_value, length(S)) for a ∈ A]
    # else
    r_max = sol.r_max
    V̄ = r_max/(1-γ)
    Γ = [fill(V̄, length(S)) for a ∈ A]
    # end
    resize!(sol.α_tmp, length(S))
    residuals = resize!(sol.residuals, length(A))

    iter = 0
    res_criterion = <(sol.bel_res)
    while iter < sol.max_iter && time() - t0 < sol.max_time
        update!(pomdp, sol, Γ, S, A, O)
        iter += 1
        all(res_criterion,residuals) && break
    end

    return AlphaVectorPolicy(pomdp, Γ, A)
end


