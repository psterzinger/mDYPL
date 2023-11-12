## Philipp Sterzinger 03.10.2023
## Provided as is, although state evolutions have been unit tested, no guarantee for correctness is given 
module AMP_MLE

export find_params_MLE_nonlinearsolve, find_params_MLE_nlsolve, logistic_MLE

using LinearAlgebra, Statistics, Cuba, NLsolve, SciMLBase, NonlinearSolve, FiniteDiff, Optim 

include("AMP_helpers.jl")
include("logistic_loglikl.jl")

## Approximate state evolution equations and jacobians 
function eq_bin!(res, param, kappa, gamma; verbose = false,  int_abstol = 1e-10, int_reltol = 1e-10)
    
    mu = param[1]
    b = param[2] 
    sigma = param[3]
    if verbose 
        println("μ = ", mu, ", b = ", b, ", σ = ", sigma)
    end 

    res .= Cuba.cuhre((x,g) -> eq_bin_integrands!(g,x,gamma,kappa,mu,sigma,b),2,3; atol = int_abstol, rtol = int_reltol).integral
    if verbose
        println("Max residual: $(maximum(abs.(res)))")
    end  
end 

function eq_bin_integrands!(g,x,gamma,kappa,mu,sigma,b; cutoff = 5000) 
    jacobian = bivar_coord_transform!(x) 
    Z = x[1]
    G = x[2]
    tdens = jacobian * dnorm(Z) * dnorm(G)

    Q_1 = gamma * Z
    Q_2 = sigma * sqrt(kappa) * G - mu * Q_1

    ∂_ζ = ∂ζ(Q_1) 
    prox_val = prox_bζ(Q_2, b)
    ∂²_ζ = ∂²ζ(prox_val)
    bζ_prox = b * ∂ζ(prox_val)
    pre_mult = 2.0 *  ∂_ζ * tdens 

    ## mu 
    g[1] = pre_mult * Q_1 * bζ_prox
    ## b 
    g[2] = pre_mult / (1.0 + b * ∂²_ζ)  - 1 + kappa 
    ##sigma 
    g[3] = pre_mult * bζ_prox^2 - (sigma * kappa)^2 

    g[abs.(g).>cutoff] .= NaN 
end 

## Solver
function find_params_MLE_nlsolve(kappa, gamma;
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

    f_init =  Vector{Float64}(undef,3) 
    lower = [0.0,0.0,0.0] .+ eps()
    upper = [Inf,Inf,Inf]

    # Setup system of equations & Jacobian 
    f!(r,param) = eq_bin!(r, param, kappa, gamma; verbose = verbose, int_abstol = int_abstol, int_reltol = int_reltol)
    cache = FiniteDiff.JacobianCache(x_init) 
    J!(J,param) = FiniteDiff.finite_difference_jacobian!(J,f!,param,cache)
    df = OnceDifferentiable(f!, J!, x_init, f_init)

    # solve 
    if verbose 
        println("Solve parameters for: κ = $kappa, γ = $gamma \n")
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
end 

function find_params_MLE_nonlinearsolve(kappa, gamma;
    verbose::Bool = false,
    x_init::Union{Missing,Vector{Float64}} = missing, 
    method::SciMLBase.AbstractNonlinearAlgorithm = NewtonRaphson(), 
    iterations::Int64 = 100,
    abstol::Float64 = 1e-8,
    reltol::Float64 = 1e-8, 
    specialize::DataType = SciMLBase.FullSpecialize,
    int_abstol::Float64 = 1e-10, 
    int_reltol::Float64 = 1e-10, 
    a::Number = 1)::Vector{Float64}

    if ismissing(x_init) 
        x_init = vcat(1.,1., 1. + sqrt(kappa) * gamma)  
    end 

    # Setup system of equations & Jacobian 
    jp = Matrix{Float64}(undef,3,3) 

    f!(r,param,a) = eq_bin!(r, param, kappa, gamma; verbose = verbose::Bool, int_abstol = int_abstol::Float64, int_reltol = int_reltol::Float64)
    cache = FiniteDiff.JacobianCache(x_init) 
    g!(r,param) = eq_bin!(r, param, kappa, gamma; verbose = verbose::Bool, int_abstol = int_abstol::Float64, int_reltol = int_reltol::Float64)
    J!(J,param,a) = FiniteDiff.finite_difference_jacobian!(J,g!,param,cache)

    function Jv!(Jv,vec,param,a) 
        J!(J,param,a)
        mul!(Jv,J,vec) 
        return nothing 
    end 

    function Jvt!(Jv,vec,param,a) 
        J!(J, param) 
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
        println("\n Solve parameters for: κ = $kappa, γ = $gamma\n")
    end 
    probN = NonlinearProblem(NF, x_init,a)
    sol = solve(probN, method, reltol = reltol, abstol = abstol, maxiters = iterations)

    if sol[end] < 0
        x_init = sol
        x_init[end] *= -1 
        probN = NonlinearProblem(NF, x_init)
        sol = solve(probN, method, reltol = reltol, abstol = abstol, maxiters = iterations)
    end 

    sol
end 

function logistic_MLE(y, X; beta_init = missing, kwargs...) 
    n = length(y)
    mu_buff = Vector{Float64}(undef,n)
    eta = similar(mu_buff) 
    X_buff = similar(X) 

    f(beta) = loglikl(beta, y, X, eta, mu_buff)  
    g!(g,beta) = loglikl_grad!(g,beta, y, X, eta, mu_buff)  
    h!(H,beta) = loglikl_hess!(H,beta, X, eta, mu_buff, X_buff)  

    if ismissing(beta_init)
        beta_init = zeros(size(X,2)) 
    end 
    Optim.optimize(f, g!, h!, beta_init; kwargs...) 
end 

end 