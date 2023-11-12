module AMP_Lasso

export find_params_lasso_nonlinearsolve, find_params_lasso_nlsolve

using LinearAlgebra, Statistics, Cuba, NLsolve, SciMLBase, NonlinearSolve, FiniteDiff, Optim, Distributions

include("AMP_helpers.jl")
include("logistic_loglikl.jl")

## Approximate state evolution equations
const normal = Normal() 

function Phi(t) 
    cdf(normal,t) 
end 

function eq_bin!(res, param, kappa, gamma, lambda, s; verbose = false,  int_abstol = 1e-10, int_reltol = 1e-10) 
    mu = param[1]
    b = param[2] 
    sigma = param[3]
    theta = param[4] 
    tau = param[5] 
    rho = param[6] 
    if verbose 
        println("μ = ", mu, ", b = ", b, ", σ = ", sigma, ", θ = ", theta, ", τ = ", tau, ", ρ = ", rho )
    end 

    res[1:3] .= Cuba.cuhre((x,g) -> eq_bin_integrands!(g, x, kappa, gamma, mu, sigma, b, theta, tau, rho), 2, 3; atol = int_abstol, rtol = int_reltol).integral

    gts = (gamma * theta)^2
    t_1 = lambda / sqrt(rho^2 * kappa + gts / s)
    t_2 = lambda / (rho * sqrt(kappa)) 
    q_1 = Phi(-t_1)
    q_2 = Phi(-t_2)
    sigma_tau_frac = 1 / (2 * sigma * tau)
    b_kappa_frac = b / kappa 
    lambda_term = lambda^2 * (s * dnorm(t_1) / t_1 + (1.0 - s) * dnorm(t_2) / t_2)
    if isnan(lambda_term) 
        lambda_term = 0.0 
        #lambda_term = Inf
    end 
    res[1] += theta * b 
    res[2] += b / (sigma * tau) - 1 
    res[3] -= rho^2  
    res[4] = theta * q_1 - mu * sigma_tau_frac
    res[5] = s * q_1 + (1.0 - s) * q_2 - b_kappa_frac * sigma_tau_frac 
    res[6] = b_kappa_frac * sigma_tau_frac * lambda^2 + b * rho^2 * sigma_tau_frac + gts * q_1 - lambda_term - 2 * ((gamma * mu)^2 + sigma^2) * sigma_tau_frac^2 
    if verbose
        println("Max residual: $(maximum(abs.(res)))")
    end  
end 

function eq_bin_integrands!(g, x, kappa, gamma, mu, sigma, b, theta, tau, rho; cutoff = 50000) 
    jacobian = bivar_coord_transform!(x) 
    Z_1 = x[1]
    Z_2 = x[2]
    tdens = jacobian * dnorm(Z_1) * dnorm(Z_2)

    Z = -gamma * Z_1
    Z_s = sigma * Z_2 - mu * Z_1

    ∂_ζ = ∂ζ(Z) 
    prox_val = prox_bζ(Z_s, b)
    pre_mult = 2.0 * ∂_ζ * tdens 

    ## mu 
    g[1] = 2.0 * tdens * ∂²ζ(Z) * prox_val 
    ## b 
    g[2] = pre_mult / (1.0 + b * ∂²ζ(prox_val)) 
    ##sigma 
    g[3] = pre_mult * ∂ζ(prox_val)^2
    g[abs.(g).>cutoff] .= NaN 
end 

function find_params_lasso_nlsolve(kappa, gamma, lambda, s;
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
        x_init = vcat(1.,1.,1 + sqrt(kappa) * gamma, 1, 1, 1)
    end 

    f_init =  Vector{Float64}(undef,6) 
    lower = [0.0,0.0,0.0,0.0,0.0,0.0] .+ eps()
    upper = [Inf,Inf,Inf,Inf,Inf,Inf]

    # Setup system of equations & Jacobian 
    f!(r, param) = eq_bin!(r, param, kappa, gamma, lambda, s; verbose = verbose, int_abstol = int_abstol, int_reltol = int_reltol)
    cache = FiniteDiff.JacobianCache(x_init) 
    J!(J,param) = FiniteDiff.finite_difference_jacobian!(J,f!,param,cache)
    df = OnceDifferentiable(f!, J!, x_init, f_init)

    # solve 
    if verbose 
        println("Solve parameters for: κ = $kappa, γ = $gamma, λ = $lambda, s = $s\n")
    end 
    if constrained_solve
        sol = mcpsolve(df, lower, upper,x_init, reformulation = reformulation, method = method, iterations = iterations, linesearch = linesearch, ftol = ftol)
    else 
        sol = nlsolve(df, x_init, method = method, iterations = iterations, linesearch = linesearch, ftol = ftol) 
    end 

    if sol.zero[3] < 0 & converged(sol) 
        x_init = sol.zero 
        x_init[3] *= -1 
        #println("Got negative standard deviation, try again...")
        if constrained_solve
            sol = mcpsolve(df,lower,upper,x_init, reformulation = reformulation, method = method, linesearch = linesearch, ftol = ftol)
        else 
            sol = nlsolve(df, x_init, method = method, linesearch = linesearch, ftol = ftol) 
        end 
    end 
    sol
end 

function find_params_lasso_nonlinearsolve(kappa, gamma, lambda, s;
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
        x_init = vcat(1.,1., 1. + sqrt(kappa) * gamma, 1, 1, 1)  
    end 

    # Setup system of equations & Jacobian 
    jp = Matrix{Float64}(undef,6,6) 

    f!(r,param,lambda) = eq_bin!(r, param, kappa, gamma, lambda, s; verbose = verbose::Bool, int_abstol = int_abstol::Float64, int_reltol = int_reltol::Float64)
    cache = FiniteDiff.JacobianCache(x_init) 
    g!(r,param) = eq_bin!(r, param, kappa, gamma, lambda, s; verbose = verbose::Bool, int_abstol = int_abstol::Float64, int_reltol = int_reltol::Float64)
    J!(J,param,lambda) = FiniteDiff.finite_difference_jacobian!(J,g!,param,cache)

    function Jv!(Jv,vec,param,lambda) 
        J!(J,param,lambda)
        mul!(Jv,J,vec) 
        return nothing 
    end 

    function Jvt!(Jv,vec,param,lambda) 
        J!(J, param,lambda) 
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
        println("\n Solve parameters for: κ = $kappa, γ = $gamma, λ = $lambda, s = $s \n")
    end 
    probN = NonlinearProblem(NF, x_init, lambda)
    sol = solve(probN, method, reltol = reltol, abstol = abstol, maxiters = iterations)

    if sol[3] < 0
        x_init = sol
        x_init[3] *= -1 
        probN = NonlinearProblem(NF, x_init, lambda)
        sol = solve(probN, method, reltol = reltol, abstol = abstol, maxiters = iterations)
    end 

    sol
end 

end 

