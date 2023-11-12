using Distributed, Plots, LaTeXStrings, ColorSchemes, JLD2, LinearAlgebra, Optim, Random, GLMNet, Distributions, LineSearches, NonlinearSolve, SharedArrays

addprocs(20) 

@everywhere using Plots, LaTeXStrings, ColorSchemes, JLD2, LinearAlgebra, Optim, Random, GLMNet, Distributions, LineSearches, NonlinearSolve
#cd("/Users/Philipp/Repositories/hdf/code/mDYPL_supplementary")
@everywhere home_dir = "" # Should be the supplementary material folder
@everywhere include(joinpath(home_dir, "Scripts", "AMP_DY.jl"))
@everywhere using .AMP_DY

@everywhere include(joinpath(home_dir, "Scripts", "AMP_Lasso.jl"))
@everywhere using .AMP_Lasso

@everywhere function round_to_sf(x::Real, sf::Int=2)
    if x == 0
        return 0
    end
    d = ceil(log10(abs(x)))
    factor = 10.0^(sf - d)
    return round(x * factor) / factor
end

@everywhere function to_latex(n::Number, sf::Int=1)
    return "\$$(round_to_sf(n, sf))\$" 
end

@everywhere function from_scaled_mixture_normal(c::Float64, s::Float64, n::Int=1)
    if s < 0 || s > 1
        throw(ArgumentError("s must be between 0 and 1"))
    end

    # Array to hold the samples
    samples = zeros(Float64, n)

    # Decide from which distribution to sample for each
    from_normal = rand(n) .< s

    # Sample for the scaled normal entries
    samples[from_normal] = randn(sum(from_normal)) * c

    return samples
end


@everywhere kappa = 0.2
@everywhere gamma = 1
@everywhere lambda = 0.2
@everywhere s = 0.5 
@everywhere lambdas = 0.1:0.01:1.1 

start = find_params_lasso_nlsolve(kappa, gamma, lambda, s; verbose = true, method = :trust_region, x_init = [0.8,0.5,sqrt(2.5),1,5,.5] ).zero

Lasso_pars = [Vector{Float64}() for i in eachindex(lambdas)] 
res = randn(6) 
for i in eachindex(lambdas) 
    println(i)
    if i > 1 && !any(isnan.(Lasso_pars[i-1])) 
        lasso_init = Lasso_pars[i-1]
    else
        lasso_init = start
    end 

    lasso_pars = find_params_lasso_nonlinearsolve(kappa, gamma, lambdas[i], s; x_init = lasso_init, verbose = true)
    Main.AMP_Lasso.eq_bin!(res, lasso_pars, kappa, gamma, lambdas[i], s)
    if maximum(abs.(res)) < 1e-6 
        Lasso_pars[i] = lasso_pars 
    else 
        lasso_pars = find_params_lasso_nlsolve(kappa, gamma, lambdas[i], s; x_init = lasso_init, verbose = true).zero
        Main.AMP_Lasso.eq_bin!(res, lasso_pars, kappa, gamma, lambdas[i], s)
        if  maximum(abs.(res)) < 1e-6 
            Lasso_pars[i] = lasso_pars 
        else 
            Lasso_pars[i] = fill(NaN,6) 
        end 
    end
end 
@save joinpath(home_dir, "Results", "rlr_lasso_pars.jld2")  Lasso_pars 

@everywhere lambda_grid = 0.1:0.05:1.1 
@everywhere n = 2000
@everywhere p = floor(Int64, kappa * n) 
@everywhere n_simulations = 2000

Lasso_betas = SharedArray{Float64}(length(lambda_grid), n_simulations, p)
true_betas = SharedArray(similar(Lasso_betas))
for i in eachindex(lambda_grid)
    println("i:$i / $(length(lambda_grid))")
    @sync @distributed for j in 1:n_simulations 
        println(j)
        Random.seed!((i-1) * n_simulations + j)
        beta_true = from_scaled_mixture_normal(1/sqrt(s), s, p) 
        X = Matrix{Float64}(undef,n,p)
        y = Vector{Float64}(undef,n) 
        X .= randn(n,p) / sqrt(p) 
        y .= rand(n) .< 1.0 ./ (1.0 .+ exp.(.-X*beta_true))
        true_betas[i,j,:] .= beta_true
        Lasso_betas[i,j,:] .= Matrix(glmnet(X, hcat(1 .- y, y), Binomial(); alpha = 1.0, lambda = [lambda_grid[i]/p], intercept = false, standardize = false).betas)
    end
end 
@save joinpath(home_dir, "Results", "rlr_lasso_betas.jld2")  Lasso_betas

## Estimate mu 
lasso_approx_mus = Vector{Float64}(undef, length(lambda_grid))
for i in eachindex(lambda_grid) 
    betas_lasso = Lasso_betas[i,:,:] 
    betas_true = true_betas[i,:,:] 
    lasso_approx_mus[i] = mean(diag(betas_lasso * betas_true')) / (gamma^2 * size(Lasso_betas,3))
end 

lasso_mus = map(v -> v[1], Lasso_pars) 
lasso_sigmas = map(v -> v[3], Lasso_pars) 

p = plot(lambdas, lasso_mus, legend = :none, label = "Lasso", color = ColorSchemes.viridis[2/10], linewidth = 2) 
scatter!(p, lambda_grid, lasso_approx_mus, color = ColorSchemes.viridis[2/10], markerstrokecolor = ColorSchemes.viridis[2/10], marker=:diamond)

lasso_approx_mse_scaled = similar(lasso_approx_mus)
lambda_indices = [findfirst(isequal(value), lambdas) for value in lambda_grid]
mus_lambda_grid = [Lasso_pars[ind][1] for ind in lambda_indices]
for i in eachindex(lambda_grid)
    betas_lasso = Lasso_betas[i,:,:] 
    betas_true = true_betas[i,:,:] 
    mu = mus_lambda_grid[i] 
    centerd_betas = betas_lasso ./ mu .- betas_true
    lasso_approx_mse_scaled[i] =  mean(centerd_betas.^2)
end

## sigma / mu plot 
p = plot(lambdas, ((lasso_sigmas ./ lasso_mus).^2), legend = :none, label = "Lasso", color = ColorSchemes.viridis[2/10], linewidth = 2) 
scatter!(p, lambda_grid, lasso_approx_mse_scaled, color = ColorSchemes.viridis[2/10], markerstrokecolor = ColorSchemes.viridis[2/10], marker=:diamond)

