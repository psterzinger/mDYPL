## Philipp Sterzinger 03.10.2023
## Provided as is, although state evolutions have been unit tested, no guarantee for correctness is given 
## gamma = 0 is only experimental

module AMP_DY 

export find_params_nonlinearsolve, find_params_nlsolve, logistic_mDYPL

using LinearAlgebra, Statistics, Roots, ProximalOperators, Cuba, NLsolve, Dates, SciMLBase, NonlinearSolve, FiniteDiff, Optim 

## Approximate state evolution equations and jacobians 
function eq_bin!(res,param, kappa, gamma, a; verbose = false,  int_abstol = 1e-10, int_reltol = 1e-10)
    if gamma == 0 
        eq_bin_no_gamma!(res, param, kappa, gamma, a; verbose = verbose, int_abstol = int_abstol, int_reltol = int_abstol)
        return 
    end 
    mu = param[1]
    b = param[2] 
    sigma = param[3]
    if verbose 
        println("μ = ", mu, ", b = ", b, ", σ = ", sigma)
    end 

    res .= Cuba.cuhre((x,g) -> eq_bin_integrands!(g,x,gamma,kappa,a,mu,sigma,b),2,3; atol = int_abstol, rtol = int_reltol).integral
    if verbose
        println("Max residual: $(maximum(abs.(res)))")
    end  
end 

function eq_bin_no_gamma!(res,param, kappa, gamma, a;verbose = false, int_abstol = 1e-10, int_reltol = 1e-10) 
    b = param[1]
    sigma = param[2]
    if verbose 
        println("b = ", b, ", σ = ", sigma)
    end 
    res .= Cuba.cuhre((x,g) -> eq_bin_integrands_no_gamma!(g,x,gamma,kappa,a,sigma,b),2,2; atol = int_abstol, rtol = int_reltol).integral 
end 

function eq_bin_integrands!(g,x,gamma,kappa,a,mu,sigma,b; cutoff = 5000) 
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

function eq_bin_integrands_no_gamma!(g,x,gamma,kappa,a,sigma,b) 
    jacobian = bivar_coord_transform!(x) 
    Z = x[1]
    G = x[2]
    tdens = jacobian * dnorm(Z) * dnorm(G)

    Z = gamma * Z
    Z_s = sigma * sqrt(kappa) * G

    ∂_ζ = ∂ζ(Z) 
    a_frac = (1 + a) / 2 
    prox_val = prox_bζ(Z_s + a_frac * b, b)
    ∂²_ζ = ∂²ζ(prox_val)
    ∂moreau = a_frac - ∂ζ(prox_val)
    pre_mult = 2.0 *  ∂_ζ * tdens 

    ## b 
    g[1] = pre_mult / (1.0 + b * ∂²_ζ) - 1.0 + kappa 
    ##sigma 
    g[2] = pre_mult * ∂moreau^2 * b^2 / kappa^2 - sigma^2 
end 

function eq_bin_jac_vec!(vec,x,gamma,kappa,a,mu,sigma,b; cutoff = 5000) 
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

    vec[1] = pre_mult * Z #wrt mu
    vec[2] = pre_mult * ∂moreau #wrt b
    vec[3] = pre_mult * sqrt(kappa) * G #wrt sigma

    ## b
    inter_val = b * ∂³ζ(prox_val) / (1.0 + b * ∂²_ζ)
    pre_mult = -2.0 * ∂_ζ / (1.0 + b * ∂²_ζ)^2 * tdens

    vec[4] = pre_mult * Z * inter_val #wrt mu
    vec[5] = pre_mult * (∂²_ζ + ∂moreau * inter_val) #wrt b
    vec[6] = pre_mult * sqrt(kappa) * G *inter_val #wrt sigma
    
    ### sigma 
    pre_mult = -4.0 * b^2 / kappa^2 * ∂_ζ * ∂moreau * ζ_frac * tdens
    
    vec[7] = pre_mult * Z #wrt mu
    vec[8] = pre_mult * ∂moreau #wrt b
    vec[9] = pre_mult * sqrt(kappa) * G #wrt sigma

    vec[8] += 4 * b / kappa^2 * ∂_ζ * ∂moreau^2 * tdens 
    vec[9] += -2.0 * sigma

   vec[abs.(vec).>cutoff] .= NaN 
   return nothing 
end 

function eq_bin_jac_vec_no_gamma!(vec,x,gamma,kappa,a,sigma,b) 
    jacobian = bivar_coord_transform!(x) 
    Z = x[1]
    G = x[2]
    tdens = jacobian * dnorm(Z) * dnorm(G)

    Z = gamma * Z
    Z_s = sigma * sqrt(kappa) * G

    ∂_ζ = ∂ζ(Z) 
    a_frac = (1 + a) / 2 
    prox_val = prox_bζ(Z_s + a_frac * b, b)
    ∂²_ζ = ∂²ζ(prox_val)
    ζ_frac = ∂²_ζ / (1.0 + b * ∂²_ζ)
    tdens = jacobian * dnorm(x[1]) * dnorm(G)
    ∂moreau = a_frac - ∂ζ(prox_val)

    ## b
    inter_val = b * ∂³ζ(prox_val) / (1.0 + b * ∂²_ζ)
    pre_mult = -2.0 * ∂_ζ / (1.0 + b * ∂²_ζ)^2 * tdens

    vec[1] = pre_mult * (∂²_ζ + ∂moreau * inter_val) #wrt b
    vec[2] = pre_mult * sqrt(kappa) * G *inter_val #wrt sigma 

    ### sigma 
    pre_mult = -4.0 * b^2 / kappa^2 * ∂_ζ * ∂moreau * ζ_frac * tdens
    
    
    vec[3] = pre_mult * ∂moreau #wrt b
    vec[4] = pre_mult * sqrt(kappa) * G #wrt sigma 

    vec[3] += 4 * b / kappa^2 * ∂_ζ * ∂moreau^2 * tdens 
    vec[4] += -2.0 * sigma
end 

function eq_bin_jac!(J, param, kappa, gamma, a, vec; int_abstol = 1e-10, int_reltol = 1e-10) 
    if gamma == 0
        eq_bin_jac_no_gamma!(J, param, kappa, gamma, a, vec; int_abstol = int_abstol, int_reltol = int_reltol)
        return 
    end 
    mu = param[1]
    b = param[2] 
    sigma = param[3]

    vec[4:12] .= Cuba.cuhre((x,g) -> eq_bin_jac_vec!(g,x,gamma,kappa,a,mu,sigma,b),2,9; atol = int_abstol, rtol = int_reltol).integral 
    rowfill!(J,vec[4:12]) 
end 

function eq_bin_jac_no_gamma!(J, param, kappa, gamma, a, vec; int_abstol = 1e-10, int_reltol = 1e-10) 
    b = param[1] 
    sigma = param[2]

    vec[3:6] .= Cuba.cuhre((x,g) -> eq_bin_jac_vec_no_gamma!(g,x,gamma,kappa,a,sigma,b),2,4; atol = int_abstol, rtol = int_reltol).integral 

    rowfill!(J,vec[3:6]) 
end 

function eq_bin_and_jac!(r, J, param, kappa, gamma, a, vec; verbose = false, int_abstol = 1e-10, int_reltol = 1e-10) 
    if gamma == 0 
        eq_bin_and_jac_no_gamma!(r , J, param, kappa, gamma, a, vec; verbose = verbose, int_abstol = int_abstol, int_reltol = int_reltol) 
        return 
    end 
    mu = param[1]
    b = param[2] 
    sigma = param[3]
    if verbose 
        println("μ = ", mu, ", b = ", b, ", σ = ", sigma)
    end 

    vec .= Cuba.cuhre((x,g) -> eq_bin_and_jac_vec!(g,x,gamma,kappa,a,mu,sigma,b),2,12; atol = int_abstol, rtol = int_reltol).integral 

    fullfill!(r,J,vec) 

    if verbose
        println("Max residual: $(maximum(abs.(r)))")
    end  
end 

function eq_bin_and_jac_no_gamma!(r, J, param, kappa, gamma, a, vec; verbose = false, int_abstol = 1e-10, int_reltol = 1e-10)  
    b = param[1] 
    sigma = param[2]
    if verbose 
        println("b = ", b, ", σ = ", sigma)
    end 

    vec .= Cuba.cuhre((x,g) -> eq_bin_and_jac_vec_no_gamma!(g,x,gamma,kappa,a,sigma,b),2,6; atol = int_abstol, rtol = int_reltol).integral 
    fullfill!(r,J,vec) 
end 

function eq_bin_and_jac_vec!(vec,x,gamma,kappa,a,mu,sigma,b; cutoff = 5000)
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

    vec[4] = pre_mult * Z #wrt mu
    vec[5] = pre_mult * ∂moreau #wrt b
    vec[6] = pre_mult * sqrt(kappa) * G #wrt sigma

    ## b deriv
    inter_val = b * ∂³ζ(prox_val) / (1.0 + b * ∂²_ζ)
    pre_mult = -2.0 * ∂_ζ / (1.0 + b * ∂²_ζ)^2 * tdens

    vec[7] = pre_mult * Z * inter_val #wrt mu
    vec[8] = pre_mult * (∂²_ζ + ∂moreau * inter_val) #wrt b
    vec[9] = pre_mult * sqrt(kappa) * G *inter_val #wrt sigma
    
    ### sigma deriv
    pre_mult = -4.0 * b^2 / kappa^2 * ∂_ζ * ∂moreau * ζ_frac * tdens
    
    vec[10] = pre_mult * Z #wrt mu
    vec[11] = pre_mult * ∂moreau #wrt b
    vec[12] = pre_mult * sqrt(kappa) * G #wrt sigma

    vec[11] += 4 * b / kappa^2 * ∂_ζ * ∂moreau^2 * tdens 
    vec[12] += -2.0 * sigma

    vec[abs.(vec) .> cutoff] .= NaN 
end 

function eq_bin_and_jac_vec_no_gamma!(vec,x,gamma,kappa,a,sigma,b) 
    jacobian = bivar_coord_transform!(x) 
    Z = x[1]
    G = x[2]
    tdens = jacobian * dnorm(Z) * dnorm(G)

    Z = gamma * Z
    Z_s = sigma * sqrt(kappa) * G

    ∂_ζ = ∂ζ(Z) 
    a_frac = (1 + a) / 2 
    prox_val = prox_bζ(Z_s + a_frac * b, b)
    ∂²_ζ = ∂²ζ(prox_val)
    ζ_frac = ∂²_ζ / (1.0 + b * ∂²_ζ)
    ∂moreau = a_frac - ∂ζ(prox_val)

    pre_mult = 2.0 *  ∂_ζ * tdens 

    ## b 
    vec[1] = pre_mult / (1.0 + b * ∂²_ζ) - 1.0 + kappa 
    ##sigma 
    vec[2] = pre_mult * ∂moreau^2 * b^2 / kappa^2 - sigma^2 

    ## b deriv
    inter_val = b * ∂³ζ(prox_val) / (1.0 + b * ∂²_ζ)
    pre_mult = -2.0 * ∂_ζ / (1.0 + b * ∂²_ζ)^2 * tdens

    vec[3] = pre_mult * (∂²_ζ + ∂moreau * inter_val) #wrt b
    vec[4] = pre_mult * sqrt(kappa) * G *inter_val #wrt sigma
    
    ### sigma deriv
    pre_mult = -4.0 * b^2 / kappa^2 * ∂_ζ * ∂moreau * ζ_frac * tdens
    
    vec[5] = pre_mult * ∂moreau #wrt b
    vec[6] = pre_mult * sqrt(kappa) * G #wrt sigma

    vec[5] += 4 * b / kappa^2 * ∂_ζ * ∂moreau^2 * tdens 
    vec[6] += -2.0 * sigma
end 

## proximal operator and logistic link function 
function prox_bζ(x,b)::Float64
    f(z) = z + b / (1.0 + exp(-z)) - x
    r = bisection(f,-20,20)[1]
    if !isnan(r)
      return r 
    else
      f = ProximalOperators.LogisticLoss(-1)
      ProximalOperators.prox(f, [x], b)[1][1]
    end 
end

function bisection(f, a_, b_, atol = 2eps(promote_type(typeof(b_),Float64)(b_)); increasing = sign(f(b_))) ## taken from julia discourse
    a_, b_ = minmax(a_, b_)
    c = middle(a_,b_)
    z = f(c) * increasing
    if z > 0 #
        b = c
        a = typeof(b)(a_)
    else
        a = c
        b = typeof(a)(b_)
    end
    while abs(a - b) > atol
        c = middle(a,b)
        if f(c) * increasing > 0 #
            b = c
        else
            a = c
        end
    end
    a, b
end

function ζ(z) log(1.0 + exp(z)) end 

function ∂ζ(z) 1.0 / (1.0 + exp(-z)) end 

function ∂²ζ(z) 
    mu = 1.0 / (1.0 + exp(-z)) 
    mu * (1.0 - mu)
end 

function ∂³ζ(z) 
    mu = 1.0 / (1.0 + exp(-z)) 
    mu * (1.0 - mu) * (1.0 - 2.0 * mu)
end 

## Helper functions 
function dnorm(x)
    1.0/sqrt(2*pi) * exp(-x^2/2) 
end 

function rowfill!(J,vec) 
    count = 1
    @inbounds @simd for i in axes(J,1)
        for j in axes(J,2)
            J[i,j] = vec[count] 
            count += 1 
        end 
    end 
end 

function fullfill!(r,J,vec) 
    p = size(J,2)
    r .= vec[1:p] 
    rowfill!(J,vec[(p+1):end]) 
end 

function bivar_coord_transform!(x)
    u = x[1] 
    u = u * (1.0 - u) 

    v = x[2] 
    v = v * (1.0 - v) 

    x[1] = (2.0 * x[1] - 1.0) / u 
    x[2] = (2.0 * x[2] - 1.0) / v 
    
    (1.0 - 2.0 * u) * (1.0 - 2.0 * v) / (u * v)^2
end 

## Solvers for state evolution 
"""
    find_params_nlsolve(kappa,gamma,a; kwargs...) 

Solve the stationary mDYPL state evolution equations `mu, b, sigma` using `NLsolve` given parameters `kappa`, `gamma`, `a` and return `NLsolve` return struct. 

# Arguments 
`kappa::Float64`: Asymptotic ratio of columns/rows of design matrix ∈ (0,1)

`gamma::Float64`: Asymptotic variance of true unobserved linear predictors ≥ 0.0

`a::Float64`: Shrinkage paramater of mDYPL estimator ∈ (0,1.0]

# Keyword Arguments 
`verbose::Bool=false`: print solver information at each Newton step

`x_init::Union{Missing,Vector{Float64}}=missing`: provide custom starting values of mu, b, sigma

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
julia> sol = find_params_nlsolve(0.2,sqrt(5),1.0; x_init = [1.5,3.,4.7]);
julia> sol.zero 
3-element Vector{Float64}:
 1.4994127712343261
 3.0270353075047765
 4.7437383902484
```

See also [`find_params_nonlinearsolve`](@ref)
"""
function find_params_nlsolve(kappa, gamma, a;
    verbose = false, 
    x_init = missing, 
    constrained_solve = true, 
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
        if gamma == 0 
            x_init = vcat(1.,1.)
        else
            x_init = vcat(1.,1.,1 + sqrt(kappa) * gamma)
        end 
    end 

    if gamma == 0
        vec = Vector{Float64}(undef,6) 
        f_init =  Vector{Float64}(undef,2)
        lower = [0.0,0.0] .+ eps()
        upper = [Inf,Inf]
    else 
        vec = Vector{Float64}(undef,12) 
        f_init =  Vector{Float64}(undef,3) 
        lower = [0.0,0.0,0.0] .+ eps()
        upper = [Inf,Inf,Inf]
    end 
    if !numeric 
        # Setup system of equations & Jacobian 
        f!(r,param) = eq_bin!(r, param, kappa, gamma, a; verbose = verbose, int_abstol = int_abstol, int_reltol = int_reltol)
        J!(J,param) = eq_bin_jac!(J, param, kappa, gamma, a, vec; int_abstol = int_abstol, int_reltol = int_reltol) 
        fJ!(r,J,param) = eq_bin_and_jac!(r, J, param, kappa, gamma, a, vec; verbose = verbose, int_abstol = int_abstol, int_reltol = int_reltol)
        df = OnceDifferentiable(f!, J!, fJ!, x_init, f_init)
        # solve 
        if verbose 
            println("Solve parameters for: κ = $kappa, γ = $gamma, α = $a \n")
        end 
        if constrained_solve
            sol = mcpsolve(df,lower,upper,x_init, reformulation = reformulation, method = method, iterations = iterations, linesearch = linesearch, ftol = ftol)
        else 
            sol = nlsolve(df, x_init, method = method, iterations = iterations, linesearch = linesearch, ftol = ftol) 
        end 

        if sol.zero[end] < 0 & converged(sol) 
            x_init = sol.zero 
            sol.zero[end] *= -1 
            #println("Got negative standard deviation, try again...")
            if constrained_solve
                sol = mcpsolve(df,lower,upper,x_init, reformulation = reformulation, method = method, linesearch = linesearch, ftol = ftol)
            else 
                sol = nlsolve(df, x_init, method = method, linesearch = linesearch, ftol = ftol) 
            end 
        end 
        sol
    else 
        find_params_nlsolve_numeric(kappa, gamma, a;
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
    find_params_nonlinearsolve(kappa,gamma,a; <keyword arguments>) 

Solve the stationary mDYPL state evolution equations `mu, b, sigma` using `NonlinearSolve` given parameters `kappa`, `gamma`, `a` and return `Vector{Float64}`. 

# Arguments 
`kappa::Float64`: Asymptotic ratio of columns/rows of design matrix ∈ (0,1)

`gamma::Float64`: Asymptotic variance of true unobserved linear predictors ≥ 0.0

`a::Float64`: Shrinkage paramater of mDYPL estimator ∈ (0,1.0]

# Keyword Arguments 
`verbose::Bool=false`: print solver information at each Newton step

`x_init::Union{Missing,Vector{Float64}}=missing`: provide custom starting values of mu, b, sigma

`method::SciMLBase.AbstractNonlinearAlgorithm=NewtonRaphson(linsolve = LineSearches.HagerZhang())`: provide one of `NonlinearSolve`'s solvers `NewtonRaphson()`, `TrustRegion()`

`iterations::Int64=100`: maximum number of iterations

`abstol::Float64=1e-6`: infinite norm of residuals under which convergence is declared

`reltol::Float64=1e-6`: infinite norm of residuals under which convergence is declared

`specialize::DataType=SciMLBase.FullSpecialize`: control the amount of compilation specialization is performed for the NonlinearProblem; see `SciMLBase`

`int_abstol::Float64=1e-10` and `int_reltol::Float64 = 1e-10`: the requested relative (ϵ_rel) and absolute (ϵ_abs​) accuracies of the integrals; see `Cuba` for more info

`numeric::Bool=false`: use `FiniteDiff`'s numeric Jacobian instead of analytical expressions 

# Examples 
```jldoctest
sol = find_params_nonlinearsolve(0.2,sqrt(5),1.0)
3-element Vector{Float64}:
 1.499412777504286
 3.027035314829301
 4.743738403143466
```

See also [`find_params_nlsolve`](@ref)
"""
function find_params_nonlinearsolve(kappa, gamma, a;
    verbose::Bool = false,
    x_init::Union{Missing,Vector{Float64}} = missing, 
    method::SciMLBase.AbstractNonlinearAlgorithm = NewtonRaphson(linsolve = LineSearches.HagerZhang()), 
    iterations::Int64 = 100,
    abstol::Float64 = 1e-6,
    reltol::Float64 = 1e-8, 
    specialize::DataType = SciMLBase.FullSpecialize,
    int_abstol::Float64 = 1e-10, 
    int_reltol::Float64 = 1e-10, 
    numeric::Bool = false 
    )::Vector{Float64}

    if ismissing(x_init) 
        if gamma == 0 
            x_init = vcat(1.,1.)
        else
            x_init = vcat(1.,1., 1. + sqrt(kappa) * gamma)  
        end 
    end 

    # Setup system of equations & Jacobian 
    if gamma == 0
        v = Vector{Float64}(undef,6) 
        f_init =  Vector{Float64}(undef,2)
        jp = Matrix{Float64}(undef,2,2)
    else 
        v = Vector{Float64}(undef,12) 
        f_init =  Vector{Float64}(undef,3) 
        jp = Matrix{Float64}(undef,3,3) 
    end 

    if !numeric 
        f!(r,param,a) = eq_bin!(r, param, kappa, gamma, a; verbose = verbose::Bool, int_abstol = int_abstol::Float64, int_reltol = int_reltol::Float64)

        J!(J,param,a) = eq_bin_jac!(J, param, kappa, gamma, a, v; int_abstol = int_abstol::Float64, int_reltol = int_reltol::Float64) 
    
        function Jv!(Jv,vec,param,a; int_abstol = int_abstol::Float64, int_reltol = int_reltol::Float64) 
            eq_bin_jac!(J, param, kappa, gamma, a, v; int_abstol = int_abstol, int_reltol = int_reltol) 
            mul!(Jv,J,vec) 
            return nothing 
        end 

        function Jvt!(Jv,vec,param,a;  int_abstol = int_abstol::Float64, int_reltol = int_reltol::Float64) 
            eq_bin_jac!(J, param, kappa, gamma, a, v; int_abstol = int_abstol, int_reltol = int_reltol) 
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
            println("\n Solve parameters for: κ = $kappa, γ = $gamma, α = $a \n")
        end 
        probN = NonlinearProblem(NF, x_init, a)
        sol = solve(probN, method, reltol = reltol, abstol = abstol, maxiters = iterations)

        if sol[end] < 0
            x_init = sol.zero 
            sol.zero[end] *= -1 
            probN = NonlinearProblem(NF, x_init, a)
            sol = solve(probN, method, reltol = reltol, abstol = abstol, maxiters = iterations)
        end 

        sol
    else
        find_params_nonlinearsolve_numeric(kappa, gamma, a;
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

function find_params_nlsolve_numeric(kappa, gamma, a;
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
        if gamma == 0 
            x_init = vcat(1.,1.)
        else
            x_init = vcat(1.,1.,1 + sqrt(kappa) * gamma)
        end 
    end 

    if gamma == 0
        vec = Vector{Float64}(undef,6) 
        f_init =  Vector{Float64}(undef,2)
        lower = [0.0,0.0] .+ eps()
        upper = [Inf,Inf]
    else 
        vec = Vector{Float64}(undef,12) 
        f_init =  Vector{Float64}(undef,3) 
        lower = [0.0,0.0,0.0] .+ eps()
        upper = [Inf,Inf,Inf]
    end 

    # Setup system of equations & Jacobian 
    f!(r,param) = eq_bin!(r, param, kappa, gamma, a; verbose = verbose, int_abstol = int_abstol, int_reltol = int_reltol)
    cache = FiniteDiff.JacobianCache(x_init) 
    J!(J,param) = FiniteDiff.finite_difference_jacobian!(J,f!,param,cache)
    df = OnceDifferentiable(f!, J!, x_init, f_init)

    # solve 
    if verbose 
        println("Solve parameters for: κ = $kappa, γ = $gamma, α = $a \n")
    end 
    if constrained_solve
        sol = mcpsolve(df,lower,upper,x_init, reformulation = reformulation, method = method, iterations = iterations, linesearch = linesearch, ftol = ftol)
    else 
        sol = nlsolve(df, x_init, method = method, iterations = iterations, linesearch = linesearch, ftol = ftol) 
    end 

    if sol.zero[end] < 0 & converged(sol) 
        x_init = sol.zero 
        sol.zero[end] *= -1 
        #println("Got negative standard deviation, try again...")
        if constrained_solve
            sol = mcpsolve(df,lower,upper,x_init, reformulation = reformulation, method = method, linesearch = linesearch, ftol = ftol)
        else 
            sol = nlsolve(df, x_init, method = method, linesearch = linesearch, ftol = ftol) 
        end 
    end 
    sol
end 

function find_params_nonlinearsolve_numeric(kappa, gamma, a;
    verbose::Bool = false,
    x_init::Union{Missing,Vector{Float64}} = missing, 
    method::SciMLBase.AbstractNonlinearAlgorithm = NewtonRaphson(linsolve = LineSearches.HagerZhang()), 
    iterations::Int64 = 100,
    abstol::Float64 = 1e-6,
    reltol::Float64 = 1e-8, 
    specialize::DataType = SciMLBase.FullSpecialize,
    int_abstol::Float64 = 1e-10, 
    int_reltol::Float64 = 1e-10, 
    )::Vector{Float64}

    if ismissing(x_init) 
        if gamma == 0 
            x_init = vcat(1.,1.)
        else
            x_init = vcat(1.,1., 1. + sqrt(kappa) * gamma)  
        end 
    end 

    # Setup system of equations & Jacobian 
    if gamma == 0
        v = Vector{Float64}(undef,6) 
        f_init =  Vector{Float64}(undef,2)
        jp = Matrix{Float64}(undef,2,2)
    else 
        v = Vector{Float64}(undef,12) 
        f_init =  Vector{Float64}(undef,3) 
        jp = Matrix{Float64}(undef,3,3) 
    end 

    f!(r,param,a) = eq_bin!(r, param, kappa, gamma, a; verbose = verbose::Bool, int_abstol = int_abstol::Float64, int_reltol = int_reltol::Float64)
    cache = FiniteDiff.JacobianCache(x_init) 
    g!(r,param) = eq_bin!(r, param, kappa, gamma, a; verbose = verbose::Bool, int_abstol = int_abstol::Float64, int_reltol = int_reltol::Float64)
    J!(J,param,a) = FiniteDiff.finite_difference_jacobian!(J,g!,param,cache)

    function Jv!(Jv,vec,param,a) 
        J!(J,param,a)
        mul!(Jv,J,vec) 
        return nothing 
    end 

    function Jvt!(Jv,vec,param,a) 
        J!(J, param,a) 
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
        println("\n Solve parameters for: κ = $kappa, γ = $gamma, α = $a \n")
    end 
    probN = NonlinearProblem(NF, x_init, a)
    sol = solve(probN, method, reltol = reltol, abstol = abstol, maxiters = iterations)

    if sol[end] < 0
        x_init = sol
        sol[end] *= -1 
        probN = NonlinearProblem(NF, x_init, a)
        sol = solve(probN, method, reltol = reltol, abstol = abstol, maxiters = iterations)
    end 

    sol
end 

## mDYPL estimator
function loglikl(beta, y, X, eta, mu_buff)  
    mul!(eta,X,beta) 
    mu_buff .= y .* eta 
    eta .= 1.0 .+ exp.(eta)
    eta .= log.(eta) 
    mu_buff .= eta .- mu_buff

    sum(mu_buff)
end 

function loglikl_grad!(r,beta, y, X, eta, mu_buff)  
    mul!(eta,X,beta) 
    mu_buff .= 1.0 ./ (1.0 .+ exp.(.-eta)) 
    mu_buff .= mu_buff .- y 

    mul!(r,X',mu_buff)
end 

function loglikl_hess!(H, beta, X, eta, mu_buff, X_buff)  
    mul!(eta,X,beta) 
    mu_buff .= 1.0 ./ (1.0 .+ exp.(.-eta)) 
    mu_buff .= mu_buff .* (1.0 .- mu_buff) 
    X_buff .= mu_buff .* X 
    mul!(H,X',X_buff) 
end 

"""
    logistic_mDYPL(y, X, alpha; beta_init = missing, kwargs...) 

Compute the mDYPL estimator for a logistic regression model using data `y`,`X` and shrinkage parameter `alpha` and return a `Optim.optimize` return struct. 

# Arguments 
`y::Vector`: Vector of binary responses

`X::Matrix`: Matrix of covariates 

`alpha::Float64`: Shrinkage paramater of mDYPL estimator ∈ (0,1.0]

# Keyword Arguments 
`beta_init::Union{Missing,Vector{Float64}}=missing`: provide starting values for minimization, if missing `beta_init = zeros(size(X, 2))`

`kwargs...`: keyword arguments to be passed to `Optim.optimize` 

# Examples 
```jldoctest
using Random, Optim
Random.seed!(123)
n = 1000
p = 100 
X = randn(n,p) / sqrt(n) 
beta = vcat(fill(0.0, ceil(Int64, p / 2)), fill(10.0, p-ceil(Int64, p / 2)))
y = rand(n) .< 1.0 ./ (1.0 .+ exp.(.-X * beta))
alpha = 1 / 1.1
mDYPL = logistic_mDYPL(y, X, alpha; beta_init = beta) 
Optim.minimizer(mDYPL)
100-element Vector{Float64}:
  0.1431992418085259
 -0.46534789400516297
  ⋮
  3.2790351663040407
```

"""
function logistic_mDYPL(y, X, alpha; beta_init = missing, kwargs...) 
    y_star = alpha * y .+ (1-alpha) / 2
    n = length(y)
    mu_buff = Vector{Float64}(undef, n)
    eta = similar(mu_buff) 
    X_buff = similar(X) 

    f(beta) = loglikl(beta, y_star, X, eta, mu_buff)  
    g!(g,beta) = loglikl_grad!(g, beta, y_star, X, eta, mu_buff)  
    h!(H,beta) = loglikl_hess!(H, beta, X, eta, mu_buff, X_buff)  

    if ismissing(beta_init)
        beta_init = zeros(size(X, 2)) 
    end 
    Optim.optimize(f, g!, h!, beta_init; kwargs...) 
end 

end 
