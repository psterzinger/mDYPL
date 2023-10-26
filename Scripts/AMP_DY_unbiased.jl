## Philipp Sterzinger 18.10.2023
## Provided as is, although state evolutions have been unit tested, no guarantee for correctness is given 
## gamma = 0 is not implemented

module AMP_DY_unbiased 

export find_params_mufixed_nonlinearsolve, find_params_mufixed_nlsolve

using LinearAlgebra, Statistics, Cuba, NLsolve, SciMLBase, NonlinearSolve, FiniteDiff, Optim 

include("AMP_helpers.jl")

## Approximate state evolution equations and jacobians 
function eq_bin!(res, param, kappa, gamma; mu_target = 1.0, verbose = false,  int_abstol = 1e-10, int_reltol = 1e-10)
    a = param[1]
    b = param[2] 
    sigma = param[3]
    if verbose 
        println("α = ", a, ", b = ", b , ", σ = ", sigma, ", μ = ", mu_target)
    end 
    res .= Cuba.cuhre((x,g) -> eq_bin_integrands!(g,x,gamma,kappa,a,mu_target,sigma,b),2,3; atol = int_abstol, rtol = int_reltol).integral
    if verbose
        println("Max residual: $(maximum(abs.(res)))")
    end  
end 

function eq_bin_integrands!(g,x,gamma,kappa,a,mu,sigma,b; cutoff = 50000) 
    jacobian = bivar_coord_transform!(x) 
    Z = x[1]
    G = x[2]
    tdens = jacobian * dnorm(Z) * dnorm(G)

    Z = gamma * Z
    Z_s = mu * Z + sigma * sqrt(kappa) * G

    ∂_ζ = ∂ζ(Z) 
    a_frac = (1 + a) / 2 
    prox_val = prox_bζ(Z_s + a_frac * b, b)
    ∂²_ζ = ∂²ζ(prox_val)
    ∂moreau = a_frac - ∂ζ(prox_val)
    pre_mult = 2.0 *  ∂_ζ * tdens 

    ## mu 
    g[1] = pre_mult * Z * ∂moreau
    ## b 
    g[2] = pre_mult / (1.0 + b * ∂²_ζ) - 1.0 + kappa 
    ##sigma 
    g[3] = pre_mult * ∂moreau^2 * b^2 / kappa^2 - sigma^2 

    g[abs.(g).>cutoff] .= NaN 
end 

function eq_bin_jac!(J, param, kappa, gamma, vec; int_abstol = 1e-10, int_reltol = 1e-10, mu_target = 1.0)
    a = param[1]
    b = param[2] 
    sigma = param[3]
    vec[4:12] .= Cuba.cuhre((x,g) -> eq_bin_jac_vec!(g,x,gamma,kappa,a,mu_target,b,sigma),2,9; atol = int_abstol, rtol = int_reltol).integral 
    rowfill!(J,vec[4:12]) 
end 

function eq_bin_jac_vec!(vec,x,gamma,kappa,a,mu,b,sigma; cutoff = 50000) 
    jacobian = bivar_coord_transform!(x) 
    Z = x[1]
    G = x[2]

    tdens = jacobian * dnorm(Z) * dnorm(G)

    Z = gamma * Z
    Z_s = mu * Z + sigma * sqrt(kappa) * G

    ∂_ζ = ∂ζ(Z) 
    a_frac = (1 + a) / 2 
    prox_val = prox_bζ(Z_s + a_frac * b, b)
    ∂²_ζ = ∂²ζ(prox_val)
    ζ_frac = ∂²_ζ / (1.0 + b * ∂²_ζ)
    ∂moreau = a_frac - ∂ζ(prox_val)

    ## mu 
    pre_mult = -2.0 * ∂_ζ * Z * ζ_frac * tdens 

    vec[1] = Z * ∂_ζ / (1.0 + b * ∂²_ζ) * tdens  #wrt a
    vec[2] = pre_mult * ∂moreau #wrt b
    vec[3] = pre_mult * sqrt(kappa) * G #wrt sigma

    ## b
    inter_val = 0.5 * b^2 * ∂³ζ(prox_val) / (1.0 + b * ∂²_ζ)
    pre_mult = -2.0 * ∂_ζ / (1.0 + b * ∂²_ζ)^2 * tdens

    vec[4] = inter_val * pre_mult #wrt a
    vec[5] = pre_mult * (∂²_ζ + ∂moreau * inter_val) #wrt b
    vec[6] = pre_mult * sqrt(kappa) * G *inter_val #wrt sigma
    
    ### sigma 
    pre_mult = -4.0 * b^2 / kappa^2 * ∂_ζ * ∂moreau * ζ_frac * tdens
    
    vec[7] = 2 * b^2 / kappa^2 * ∂_ζ * ∂moreau / (1.0 + b * ∂²_ζ) * tdens #wrt mu
    vec[8] = pre_mult * ∂moreau #wrt b
    vec[9] = pre_mult * sqrt(kappa) * G #wrt sigma

    vec[8] += 4 * b / kappa^2 * ∂_ζ * ∂moreau^2 * tdens 
    vec[9] += -2.0 * sigma

   vec[abs.(vec).>cutoff] .= NaN 
   return nothing 
end 

function eq_bin_and_jac!(r, J, param, kappa, gamma, vec; verbose = false, int_abstol = 1e-10, int_reltol = 1e-10, mu_target = 1.0) 
    a = param[1]
    b = param[2] 
    sigma = param[3]
    if verbose 
        println("α = ", a, ", b = ", b , ", σ = ", sigma, ", μ = ", mu_target)
    end 

    vec .= Cuba.cuhre((x,g) -> eq_bin_and_jac_vec!(g,x,gamma,kappa,a,mu_target,b,sigma),2,12; atol = int_abstol, rtol = int_reltol).integral 

    fullfill!(r,J,vec) 
    if verbose
        println("Max residual: $(maximum(abs.(r)))")
    end
end 

function eq_bin_and_jac_vec!(vec,x,gamma,kappa,a,mu,b,sigma; cutoff = 50000)
    jacobian = bivar_coord_transform!(x) 
    Z = x[1]
    G = x[2]

    tdens = jacobian * dnorm(Z) * dnorm(G)

    Z = gamma * Z
    Z_s = mu * Z + sigma * sqrt(kappa) * G

    ∂_ζ = ∂ζ(Z) 
    a_frac = (1 + a) / 2 
    prox_val = prox_bζ(Z_s + a_frac * b, b)
    ∂²_ζ = ∂²ζ(prox_val)
    ζ_frac = ∂²_ζ / (1.0 + b * ∂²_ζ)
    ∂moreau = a_frac - ∂ζ(prox_val)

    pre_mult = 2.0 *  ∂_ζ * tdens 

    ## mu 
    vec[1] = pre_mult * Z * ∂moreau
    ## b 
    vec[2] = pre_mult / (1.0 + b * ∂²_ζ) - 1.0 + kappa 
    ##sigma 
    vec[3] = pre_mult * ∂moreau^2 * b^2 / kappa^2 - sigma^2 

    ## mu deriv
    pre_mult = -2.0 * ∂_ζ * Z * ζ_frac * tdens 

    vec[4] =  Z * ∂_ζ / (1.0 + b * ∂²_ζ) * tdens  #wrt a
    vec[5] = pre_mult * ∂moreau #wrt b
    vec[6] = pre_mult * sqrt(kappa) * G #wrt sigma

    ## b deriv
    inter_val = 0.5 * b^2 * ∂³ζ(prox_val) / (1.0 + b * ∂²_ζ)
    pre_mult = -2.0 * ∂_ζ / (1.0 + b * ∂²_ζ)^2 * tdens

    vec[7] = inter_val * pre_mult #wrt a
    vec[8] = pre_mult * (∂²_ζ + ∂moreau * inter_val) #wrt b
    vec[9] = pre_mult * sqrt(kappa) * G *inter_val #wrt sigma
    
    ### sigma deriv
    pre_mult = -4.0 * b^2 / kappa^2 * ∂_ζ * ∂moreau * ζ_frac * tdens
    
    vec[10] =  2 * b^2 / kappa^2 * ∂_ζ * ∂moreau / (1.0 + b * ∂²_ζ) * tdens #wrt mu
    vec[11] = pre_mult * ∂moreau #wrt b
    vec[12] = pre_mult * sqrt(kappa) * G #wrt sigma

    vec[11] += 4 * b / kappa^2 * ∂_ζ * ∂moreau^2 * tdens 
    vec[12] += -2.0 * sigma

    vec[abs.(vec) .> cutoff] .= NaN 
end 

## Solvers for state evolution 
"""
    find_params_mufixed_nlsolve(kappa,gamma; kwargs...) 

Solve the stationary mDYPL state evolution equations for `alpha, b, sigma` at `mu = mu_target` using `NLsolve` given parameters `kappa`, `gamma` and return `NLsolve` return struct. 

# Arguments 
`kappa::Float64`: Asymptotic ratio of columns/rows of design matrix ∈ (0,1)

`gamma::Float64`: Asymptotic variance of true unobserved linear predictors ≥ 0.0

# Keyword Arguments 
`mu_target::Float64=1.0`: Target value of bias paramater μ

`verbose::Bool=false`: print solver information at each Newton step

`x_init::Union{Missing,Vector{Float64}}=missing`: provide custom starting values of alpha, b, sigma

`constrained_solve::Bool=false`: use constrained solver functionality of `NLsolve`

`reformulation::Symbol=:smooth`: use one of `NLsolve`'s constrained_solve reformulation options `:smooth`, `:minmax`

`method::Symbol=:newton`: provide one of `NLsolve`'s solvers `:newton`, `:trust_region`, `:anderson`

`iterations::Int64=100`:  maximum number of iterations

`linesearch::Any=LineSearches.BackTracking(order=3)`: linesearch algorithm from `LineSearches`

`ftol::Float64=1e-6`: infinite norm of residuals under which convergence is declared

`int_abstol::Float64=1e-10` and `int_reltol::Float64 = 1e-10`: the requested relative (ϵ_rel) and absolute (ϵ_abs​) accuracies of the integrals; see `Cuba` for more info 

`numeric::Bool=false`: use `FiniteDiff`'s numeric Jacobian instead of analytical expressions 

# Examples 
```jldoctest
julia> sol = find_params_mufixed_nlsolve(0.2,sqrt(5));
julia> sol.zero 
3-element Vector{Float64}:
 1.4994127712343261
 3.0270353075047765
 4.7437383902484
```

See also [`find_params_mufixed_nonlinearsolve`](@ref)
"""
function find_params_mufixed_nlsolve(kappa, gamma;
    mu_target = 1.0, 
    verbose = false, 
    x_init = missing, 
    constrained_solve = false, 
    method = :newton, 
    reformulation = :smooth,
    iterations = 100,
    linesearch = LineSearches.BackTracking(order=3),
    ftol = 1e-6, 
    int_abstol = 1e-10, 
    int_reltol = 1e-10, 
    numeric = false 
    )

    if ismissing(x_init) 
        x_init = vcat(1.,1.,1 + sqrt(kappa) * gamma)
    end 

    vec = Vector{Float64}(undef,12) 
    f_init =  Vector{Float64}(undef,3) 
    lower = [0.0,0.0,0.0] .+ eps()
    upper = [Inf,Inf,Inf]

    if !numeric 
        # Setup system of equations & Jacobian 
        f!(r,param) = eq_bin!(r, param, kappa, gamma; mu_target = mu_target, verbose = verbose, int_abstol = int_abstol, int_reltol = int_reltol)
        J!(J,param) = eq_bin_jac!(J, param, kappa, gamma, vec; mu_target = mu_target, int_abstol = int_abstol, int_reltol = int_reltol) 
        fJ!(r,J,param) = eq_bin_and_jac!(r, J, param, kappa, gamma, vec; mu_target = mu_target, verbose = verbose, int_abstol = int_abstol, int_reltol = int_reltol)
        df = OnceDifferentiable(f!, J!, fJ!, x_init, f_init)
        # solve 
        if verbose 
            println("Solve parameters for: κ = $kappa, γ = $gamma, μ = $mu_target \n")
        end 
        if constrained_solve
            sol = mcpsolve(df,lower,upper,x_init, reformulation = reformulation, method = method, iterations = iterations, linesearch = linesearch, ftol = ftol)
        else 
            sol = nlsolve(df, x_init, method = method, iterations = iterations, linesearch = linesearch, ftol = ftol) 
        end 

        if sol.zero[end] < 0 & converged(sol) 
            x_init = sol.zero 
            x_init[end] *= -1 
            #println("Got negative standard deviation, try again...")
            if constrained_solve
                sol = mcpsolve(df,lower,upper,x_init, reformulation = reformulation, method = method, linesearch = linesearch, ftol = ftol)
            else 
                sol = nlsolve(df, x_init, method = method, linesearch = linesearch, ftol = ftol) 
            end 
        end 
        sol
    else 
        find_params_nlsolve_numeric(kappa, gamma;
            mu_target = mu_target,
            verbose = verbose, 
            x_init = x_init, 
            constrained_solve = constrained_solve, 
            method = method, 
            reformulation = reformulation,
            iterations = iterations,
            linesearch = linesearch,
            ftol = ftol, 
            int_abstol = int_abstol, 
            int_reltol = int_reltol
        ) 
    end 
end 

"""
    find_params_mufixed_nonlinearsolve(kappa,gamma; <keyword arguments>) 

    Solve the stationary mDYPL state evolution equations for `alpha, b, sigma` at `mu = mu_target` using `NonlinearSolve` given parameters `kappa`, `gamma` and return `Vector{Float64}`. 

# Arguments 
`kappa::Float64`: Asymptotic ratio of columns/rows of design matrix ∈ (0,1)

`gamma::Float64`: Asymptotic variance of true unobserved linear predictors ≥ 0.0

# Keyword Arguments 
`mu_target::Float64=1.0`: Target value of bias paramater μ

`verbose::Bool=false`: print solver information at each Newton step

`x_init::Union{Missing,Vector{Float64}}=missing`: provide custom starting values of alpha, b, sigma

`method::SciMLBase.AbstractNonlinearAlgorithm=NewtonRaphson(linsolve = LineSearches.HagerZhang())`: provide one of `NonlinearSolve`'s solvers `NewtonRaphson()`, `TrustRegion()`

`iterations::Int64=100`: maximum number of iterations

`abstol::Float64=1e-6`: infinite norm of residuals under which convergence is declared

`reltol::Float64=1e-6`: infinite norm of residuals under which convergence is declared

`specialize::DataType=SciMLBase.FullSpecialize`: control the amount of compilation specialization is performed for the NonlinearProblem; see `SciMLBase`

`int_abstol::Float64=1e-10` and `int_reltol::Float64 = 1e-10`: the requested relative (ϵ_rel) and absolute (ϵ_abs​) accuracies of the integrals; see `Cuba` for more info

`numeric::Bool=false`: use `FiniteDiff`'s numeric Jacobian instead of analytical expressions 

# Examples 
```jldoctest
sol = find_params_mufixed_nonlinearsolve(0.2,sqrt(5))
3-element Vector{Float64}:
 1.499412777504286
 3.027035314829301
 4.743738403143466
```

See also [`find_params_nlsolve`](@ref)
"""
function find_params_mufixed_nonlinearsolve(kappa, gamma;
    mu_target::Float64 = 1.0, 
    verbose::Bool = false,
    x_init::Union{Missing,Vector{Float64}} = missing, 
    method::SciMLBase.AbstractNonlinearAlgorithm = NewtonRaphson(), 
    iterations::Int64 = 100,
    abstol::Float64 = 1e-6,
    reltol::Float64 = 1e-8, 
    specialize::DataType = SciMLBase.FullSpecialize,
    int_abstol::Float64 = 1e-10, 
    int_reltol::Float64 = 1e-10, 
    numeric::Bool = false 
    )::Vector{Float64}

    if ismissing(x_init) 
        x_init = vcat(1.,1., 1. + sqrt(kappa) * gamma)  
    end 

    # Setup system of equations & Jacobian 
    v = Vector{Float64}(undef,12) 
    jp = Matrix{Float64}(undef,3,3) 

    if !numeric 
        f!(r,param,p) = eq_bin!(r, param, kappa, gamma; mu_target = mu_target::Float64, verbose = verbose::Bool, int_abstol = int_abstol::Float64, int_reltol = int_reltol::Float64)

        J!(J,param,p) = eq_bin_jac!(J, param, kappa, gamma, v; mu_target = mu_target::Float64, int_abstol = int_abstol::Float64, int_reltol = int_reltol::Float64) 
    
        function Jv!(Jv,vec,param,p; mu_target = mu_target::Float64, int_abstol = int_abstol::Float64, int_reltol = int_reltol::Float64) 
            eq_bin_jac!(J, param, kappa, gamma, v; mu_target = mu_target, int_abstol = int_abstol, int_reltol = int_reltol) 
            mul!(Jv,J,vec) 
            return nothing 
        end 

        function Jvt!(Jv,vec,param,p; mu_target = mu_target::Float64,  int_abstol = int_abstol::Float64, int_reltol = int_reltol::Float64) 
            eq_bin_jac!(J, param, kappa, gamma, v; mu_target = mu_target, int_abstol = int_abstol, int_reltol = int_reltol) 
            mul!(Jv,J',vec)  
            return nothing 
        end

        # define nonlinear problem
        NF = NonlinearFunction{true, specialize}(f!; 
            jac = J!, 
            jvp = Jv!, 
            vjp = Jvt!, 
            jac_prototype = jp
        )

        # solve 
        if verbose 
            println("\n Solve parameters for: κ = $kappa, γ = $gamma, μ = $mu_target \n")
        end 
        probN = NonlinearProblem(NF, x_init, mu_target)
        sol = solve(probN, method, reltol = reltol, abstol = abstol, maxiters = iterations)

        if sol[end] < 0
            x_init = sol
            x_init[end] *= -1 
            probN = NonlinearProblem(NF, x_init, mu_target)
            sol = solve(probN, method, reltol = reltol, abstol = abstol, maxiters = iterations)
        end 

        sol
    else
        find_params_nonlinearsolve_numeric(kappa, gamma;
            mu_target = mu_target, 
            verbose = verbose, 
            x_init = x_init, 
            method = method, 
            iterations = iterations,
            int_abstol = int_abstol, 
            int_reltol = int_reltol, 
            abstol = abstol, 
            reltol = reltol, 
            specialize = specialize
        )::Vector{Float64}
    end 
end 

function find_params_nlsolve_numeric(kappa, gamma;
    mu_target = 1.0, 
    verbose = false, 
    x_init = missing, 
    constrained_solve = true, 
    method = :newton, 
    reformulation = :smooth,
    iterations = 100,
    linesearch = LineSearches.BackTracking(order=3),
    ftol = 1e-6, 
    int_abstol = 1e-10, 
    int_reltol = 1e-10
    )

    if ismissing(x_init) 
        x_init = vcat(1.,1.,1 + sqrt(kappa) * gamma)
    end 

    vec = Vector{Float64}(undef,12) 
    f_init =  Vector{Float64}(undef,3) 
    lower = [0.0,0.0,0.0] .+ eps()
    upper = [Inf,Inf,Inf] 

    # Setup system of equations & Jacobian 
    f!(r,param) = eq_bin!(r, param, kappa, gamma; mu_target = mu_target, verbose = verbose, int_abstol = int_abstol, int_reltol = int_reltol)
    cache = FiniteDiff.JacobianCache(x_init) 
    J!(J,param) = FiniteDiff.finite_difference_jacobian!(J,f!,param,cache)
    df = OnceDifferentiable(f!, J!, x_init, f_init)

    # solve 
    if verbose 
        println("Solve parameters for: κ = $kappa, γ = $gamma, μ = $mu_target \n")
    end 
    if constrained_solve
        sol = mcpsolve(df,lower,upper,x_init, reformulation = reformulation, method = method, iterations = iterations, linesearch = linesearch, ftol = ftol)
    else 
        sol = nlsolve(df, x_init, method = method, iterations = iterations, linesearch = linesearch, ftol = ftol) 
    end 

    if  converged(sol) && sol.zero[end] < 0
        x_init = sol.zero 
        x_init[end] *= -1 
        
        if constrained_solve
            sol = mcpsolve(df,lower,upper,x_init, reformulation = reformulation, method = method, linesearch = linesearch, ftol = ftol)
        else 
            sol = nlsolve(df, x_init, method = method, linesearch = linesearch, ftol = ftol) 
        end 
    end 
    sol
end 

function find_params_nonlinearsolve_numeric(kappa, gamma;
    mu_target::Float64 = 1.0,
    verbose::Bool = false,
    x_init::Union{Missing,Vector{Float64}} = missing, 
    method::SciMLBase.AbstractNonlinearAlgorithm = NewtonRaphson(), 
    iterations::Int64 = 100,
    abstol::Float64 = 1e-6,
    reltol::Float64 = 1e-8, 
    specialize::DataType = SciMLBase.FullSpecialize,
    int_abstol::Float64 = 1e-10, 
    int_reltol::Float64 = 1e-10, 
    )::Vector{Float64}

    if ismissing(x_init) 
        x_init = vcat(1.,1., 1. + sqrt(kappa) * gamma)  
    end 

    # Setup system of equations & Jacobian 
    jp = Matrix{Float64}(undef,3,3) 

    f!(r,param,p) = eq_bin!(r, param, kappa, gamma; mu_target = mu_target::Float64, verbose = verbose::Bool, int_abstol = int_abstol::Float64, int_reltol = int_reltol::Float64)
    cache = FiniteDiff.JacobianCache(x_init) 
    g!(r,param) = eq_bin!(r, param, kappa, gamma; mu_target = mu_target::Float64, verbose = verbose::Bool, int_abstol = int_abstol::Float64, int_reltol = int_reltol::Float64)
    J!(J,param,p) = FiniteDiff.finite_difference_jacobian!(J,g!,param,cache)

    function Jv!(Jv,vec,param,p) 
        J!(J,param,p)
        mul!(Jv,J,vec) 
        return nothing 
    end 

    function Jvt!(Jv,vec,param,p) 
        J!(J, param,pt) 
        mul!(Jv,J',vec)  
        return nothing 
    end

    # define nonlinear problem
    NF = NonlinearFunction{true, specialize}(f!; 
        jac = J!, 
        jvp = Jv!, 
        vjp = Jvt!, 
        jac_prototype = jp
    )

    # solve 
    if verbose 
        println("\n Solve parameters for: κ = $kappa, γ = $gamma, μ = $mu_target \n")
    end 
    probN = NonlinearProblem(NF, x_init, mu_target)
    sol = solve(probN, method, reltol = reltol, abstol = abstol, maxiters = iterations)

    if sol[end] < 0
        x_init = sol
        x_init[end] *= -1 
        probN = NonlinearProblem(NF, x_init, mu_target)
        sol = solve(probN, method, reltol = reltol, abstol = abstol, maxiters = iterations)
    end 

    sol
end 

end 