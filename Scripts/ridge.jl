using MKL, Plots, LaTeXStrings, ColorSchemes, JLD2, LinearAlgebra, Optim, Random, Statistics, LineSearches, Plots.Measures
home_dir = "" # Should be the supplementary material folder
include(joinpath(home_dir, "Scripts", "AMP_DY.jl"))
using .AMP_DY
include(joinpath(home_dir, "Scripts", "AMP_Ridge.jl"))
using .AMP_Ridge
include(joinpath(home_dir, "Scripts", "AMP_MLE.jl"))
using .AMP_MLE
function round_to_sf(x::Real, sf::Int=2)
    if x == 0
        return 0
    end
    d = ceil(log10(abs(x)))
    factor = 10.0^(sf - d)
    return round(x * factor) / factor
end
function to_latex(n::Number, sf::Int=1)
    return "\$$(round_to_sf(n, sf))\$" 
end
function fill_beta_SC(p)
    pp = ceil(Int64, p / 8) 
    vcat(fill(-10,pp),fill(10,pp),fill(0.0,p-2*pp))
end
kappas = (0.1, 0.2, 0.3, 0.4)
gammas = (1, 5, 10)
n = 2000
n_simulations = 1000
maxl = 4.5
for gamma in gammas 
    for kappa in kappas 
        if kappa == 0.1 && gamma == 5 || gamma == 1 && kappa <= 0.4 
            lambdas = range(0.0, maxl, 200)
            alphas = range(1.0, 0.05, 200)
            alpha_grid = alphas[1:10:length(alphas)]
        else 
            lambdas = range(maxl * (1 - 0.99),maxl,200)
            alphas = range(0.99,0.05,200)
            alpha_grid = alphas[1:10:length(alphas)]
        end 
    
        lambda_grid = similar(alpha_grid) 
        for i in eachindex(alpha_grid)
            lambda_grid[i] = lambdas[argmin(abs.((1 .- lambdas ./ maxl) .- alpha_grid[i]))]
        end 
    
        p = floor(Int64, kappa * n) 
        beta_true = fill_beta_SC(p) * gamma / 5

        DY_pars = [Vector{Float64}() for i in eachindex(alphas)] 
        Ridge_pars = [Vector{Float64}() for i in eachindex(lambdas)] 
        start = find_params_nonlinearsolve(kappa,gamma,alphas[1]; verbose = true, numeric = true)
        alt = find_params_ridge_nonlinearsolve(kappa,gamma,lambdas[1]; verbose = true)
        
        DY_pars[1] = start 
        Ridge_pars[1] = alt 
    
        if length(alphas) < length(lambdas) 
            iterates = eachindex(lambdas)[2:end] 
        else
            iterates = eachindex(alphas)[2:end]
        end 
        res = randn(3) 
        for i in iterates
            println("i:$(i-1)/$(length(iterates))")
            if i <= length(alphas)
                if alphas[i] <= 0.05
                    dy_init = [0.015,0.5,0.1]
                elseif !any(isnan.(DY_pars[i-1])) 
                    dy_init = DY_pars[i-1]
                else
                    dy_init = start 
                end 
    
                dy_pars = find_params_nlsolve(kappa,gamma,alphas[i]; x_init = dy_init, method = :trust_region, linesearch = LineSearches.HagerZhang(), numeric = true).zero
                Main.AMP_DY.eq_bin!(res, dy_pars, kappa, gamma, alphas[i])
                if maximum(abs.(res)) < 1e-6 
                    DY_pars[i] = dy_pars 
                else 
                    dy_pars = find_params_nlsolve(kappa,gamma,alphas[i]; x_init = start, method = :trust_region, linesearch = LineSearches.HagerZhang(), numeric = true).zero
                    Main.AMP_DY.eq_bin!(res, dy_pars, kappa, gamma, alphas[i])
                    if  maximum(abs.(res)) < 1e-6 
                        DY_pars[i] = dy_pars 
                    else 
                        DY_pars[i] = fill(NaN,3) 
                    end 
                end 
            end
            if i <= length(lambdas)
                if !any(isnan.(Ridge_pars[i-1])) 
                    ridge_init = Ridge_pars[i-1]
                else
                    ridge_init = alt
                end
                ridge_pars = find_params_ridge_nlsolve(kappa, gamma, lambdas[i]; x_init = ridge_init).zero
                Main.AMP_Ridge.eq_bin!(res, ridge_pars, kappa, gamma, lambdas[i])
                if maximum(abs.(res)) < 1e-6 
                    Ridge_pars[i] = ridge_pars 
                else 
                    ridge_pars = find_params_ridge_nlsolve(kappa, gamma, lambdas[i]; x_init = alt).zero
                    Main.AMP_Ridge.eq_bin!(res, ridge_pars, kappa, gamma, lambdas[i])
                    if  maximum(abs.(res)) < 1e-6 
                        Ridge_pars[i] = ridge_pars 
                    else 
                        Ridge_pars[i] = fill(NaN,3) 
                    end 
                end 
            end
        end 
        @save joinpath(home_dir, "Results", "rlr", "rlr_dy_pars_k$(kappa)_g$(gamma).jld2")  DY_pars 
        @save joinpath(home_dir, "Results", "rlr", "rlr_ridge_pars_k$(kappa)_g$(gamma).jld2")  Ridge_pars 
 
        #DY_pars = load(joinpath(home_dir, "Results", "rlr", "rlr_dy_pars_k$(kappa)_g$(gamma).jld2"))["DY_pars"]
        #Ridge_pars = load(joinpath(home_dir, "Results", "rlr", "rlr_ridge_pars_k$(kappa)_g$(gamma).jld2"))["Ridge_pars"]
    
        DY_betas = Array{Float64}(undef, length(alpha_grid), n_simulations, p)
        Ridge_betas = Array{Float64}(undef, length(lambda_grid), n_simulations, p)
    
        X = Matrix{Float64}(undef,n,p)
        y = Vector{Float64}(undef,n) 
    
        if length(alpha_grid) < length(lambda_grid) 
            iterates = eachindex(lambda_grid) 
        else
            iterates = eachindex(alpha_grid)
        end 

        for i in iterates
            for j in 1:n_simulations 
                println("i:$i/$(length(iterates)), j:$j/$n_simulations")
                Random.seed!((i-1) * n_simulations + j)
                X .= randn(n,p) / sqrt(p) 
                y .= rand(n) .< 1.0 ./ (1.0 .+ exp.(.-X * beta_true))
    
                if i <= length(alpha_grid)
                    DY_betas[i,j,:] .= Optim.minimizer(logistic_mDYPL(y, X, alpha_grid[i]; beta_init = beta_true))
                end
                if i <= length(lambda_grid)
                    Ridge_betas[i,j,:] .= Optim.minimizer(logistic_ridge(y, X, lambda_grid[i]; beta_init = beta_true))
                end
            end
        end 
        @save joinpath(home_dir, "Results", "rlr", "rlr_dy_betas_k$(kappa)_g$(gamma).jld2")  DY_betas
        @save joinpath(home_dir, "Results", "rlr", "rlr_ridge_betas_k$(kappa)_g$(gamma).jld2")  Ridge_betas 

        #DY_betas = load(joinpath(home_dir, "Results", "rlr", "rlr_dy_betas_k$(kappa)_g$(gamma).jld2"))["DY_betas"]
        #Ridge_betas = load(joinpath(home_dir, "Results", "rlr", "rlr_ridge_betas_k$(kappa)_g$(gamma).jld2"))["Ridge_betas"]
        
        DY_pars = load(joinpath(home_dir, "Results", "rlr", "rlr_dy_pars_k$(kappa)_g$(gamma).jld2"))["DY_pars"]
        Ridge_pars = load(joinpath(home_dir, "Results", "rlr", "rlr_ridge_pars_k$(kappa)_g$(gamma).jld2"))["Ridge_pars"]
    
        ## Estimate mu 
        dy_approx_mus = mapslices(X -> mean(X * beta_true) / (gamma^2 * p), DY_betas, dims = [2,3])[:,1,1] 
        ridge_approx_mus = mapslices(X -> mean(X * beta_true) / (gamma^2 * p), Ridge_betas, dims = [2,3])[:,1,1]
        @save joinpath(home_dir, "Results", "rlr", "rlr_dy_approx_mus_k$(kappa)_g$(gamma).jld2")  dy_approx_mus
        @save joinpath(home_dir, "Results", "rlr", "rlr_ridge_approx_mus_k$(kappa)_g$(gamma).jld2")  ridge_approx_mus

        ## Estimate unscaled mse 
        dy_approx_mse = mapslices(X -> mean(mapslices(beta_dy -> mean((beta_dy .- beta_true).^2), X, dims=2)), DY_betas, dims=[2,3])[:,1,1]
        ridge_approx_mse = mapslices(X -> mean(mapslices(beta_ridge -> mean((beta_ridge .- beta_true).^2), X, dims=2)), Ridge_betas, dims=[2,3])[:,1,1]
        @save joinpath(home_dir, "Results", "rlr", "rlr_dy_approx_mse_k$(kappa)_g$(gamma).jld2")  dy_approx_mse
        @save joinpath(home_dir, "Results", "rlr", "rlr_ridge_approx_mse_k$(kappa)_g$(gamma).jld2")  ridge_approx_mse

        ## Estimate scaled mse 
        alpha_indices = [findfirst(isequal(value), alphas) for value in alpha_grid]
        lambda_indices = [findfirst(isequal(value), lambdas) for value in lambda_grid]

        mus_alpha_grid = [DY_pars[ind][1] for ind in alpha_indices]
        mus_lambda_grid = [Ridge_pars[ind][1] for ind in lambda_indices]

        dy_approx_mse_scaled = Vector{Float64}(undef, size(DY_betas, 1))
        for (idx, slice) in enumerate(eachslice(DY_betas; dims=1))
            dy_approx_mse_scaled[idx] = mean(mapslices(beta_dy -> mean(((beta_dy ./ mus_alpha_grid[idx]) .- beta_true).^2), slice, dims=2))
        end

        ridge_approx_mse_scaled = Vector{Float64}(undef, size(Ridge_betas, 1))
        for (idx, slice) in enumerate(eachslice(Ridge_betas; dims=1))
            ridge_approx_mse_scaled[idx] = mean(mapslices(beta_ridge -> mean(((beta_ridge ./ mus_lambda_grid[idx]) .- beta_true).^2), slice, dims=2))
        end
        @save joinpath(home_dir, "Results", "rlr", "rlr_dy_approx_mse_scaled_k$(kappa)_g$(gamma).jld2")  dy_approx_mse_scaled
        @save joinpath(home_dir, "Results", "rlr", "rlr_ridge_approx_mse_scaled_k$(kappa)_g$(gamma).jld2")  ridge_approx_mse_scaled

        ## Estimate sigma 
        dy_approx_sigmas = Vector{Float64}(undef, size(DY_betas, 1))
        for i in eachindex(dy_approx_sigmas)
            count = 0
            for j in axes(DY_betas,2) 
                count += mean((DY_betas[i,j,:] .- mus_alpha_grid[i] .* beta_true).^2)
            end 
            dy_approx_sigmas[i] = count / size(DY_betas,2)
        end

        ridge_approx_sigmas = Vector{Float64}(undef, size(Ridge_betas, 1))
        for i in eachindex(ridge_approx_sigmas)
            count = 0
            for j in axes(Ridge_betas,2) 
                count += mean((Ridge_betas[i,j,:] .- mus_lambda_grid[i] .* beta_true).^2)
            end 
            ridge_approx_sigmas[i] = count / size(Ridge_betas,2)
        end
        @save joinpath(home_dir, "Results", "rlr", "rlr_dy_approx_sigmas_k$(kappa)_g$(gamma).jld2")  dy_approx_sigmas
        @save joinpath(home_dir, "Results", "rlr", "rlr_ridge_approx_sigmas_k$(kappa)_g$(gamma).jld2")  ridge_approx_sigmas
    end 
end
mu_plots = Vector{Any}(undef,3)
mse_plots = Vector{Any}(undef,3)
mse_scaled_plots = Vector{Any}(undef,3)
sigma_plots = Vector{Any}(undef,3)
pos = 1
legend_plot = plot(legend = false, background_color_inside=:transparent, left_margin = -50mm)
colors = [ColorSchemes.viridis[i] for i in range(0, stop=1, length=4)]
for i in eachindex(kappas)
    scatter!(legend_plot,[], [], color=colors[i], label=LaTeXString("\$ \\kappa = $(round(i * 0.1, digits = 1)) \$  "), legend=:right, marker = :square, markerstrokecolor=colors[i])
end
line_styles = [:solid, :dash, :dashdot]
labels = [L"\mathrm{mDYPL}", L"\mathrm{Ridge}", L"\mathrm{ML}"]
for (i,style) in enumerate(line_styles)
    plot!(legend_plot, [], [], linestyle=style, color=:black, label=labels[i])
end
plot!(legend_plot, showaxis=false, grid=false, framestyle=:none, xlims=(0,0), ylims=(0,0), legend=:right)

for gamma in gammas
    mse_plot = plot(legend = false, title = LaTeXString("\$ \\gamma = $gamma \$"), titlefonszie = 11) 
    mse_scaled_plot = plot(legend = false, title = LaTeXString("\$ \\gamma = $gamma \$"), titlefonszie = 11) 
    mu_plot = plot(legend = false, title = LaTeXString("\$ \\gamma = $gamma \$"), titlefonszie = 11)
    sigma_plot = plot(legend = false, title = LaTeXString("\$ \\gamma = $gamma \$"), titlefonszie = 11) 

    for (kappa, col) in zip(kappas, colors)
        if kappa == 0.1 && gamma == 5 || gamma == 1 && kappa <= 0.4 
            lambdas = range(0.0, maxl, 200)
            alphas = range(1.0, 0.05, 200)
            alpha_grid = alphas[1:10:length(alphas)]
        else 
            lambdas = range(maxl * (1 - 0.99),maxl,200)
            alphas = range(0.99,0.05,200)
            alpha_grid = alphas[1:10:length(alphas)]
        end 
        
        maxl = maximum(lambdas)
        lambda_grid = similar(alpha_grid) 
        for i in eachindex(alpha_grid)
            lambda_grid[i] = lambdas[argmin(abs.((1 .- lambdas ./ maxl) .- alpha_grid[i]))]
        end 
        
        DY_pars = load(joinpath(home_dir, "Results", "rlr", "rlr_dy_pars_k$(kappa)_g$(gamma).jld2"))["DY_pars"]
        Ridge_pars = load(joinpath(home_dir, "Results", "rlr", "rlr_ridge_pars_k$(kappa)_g$(gamma).jld2"))["Ridge_pars"]

        ## Compute theoretical values 
        dy_mus = map(v -> v[1], DY_pars)
        dy_sigmas = map(v -> v[3], DY_pars)

        ridge_mus = map(v -> v[1], Ridge_pars)
        ridge_sigmas = map(v -> v[3], Ridge_pars)

        dy_mse = (1.0 .- dy_mus).^2 * gamma^2 + kappa * dy_sigmas.^2
        ridge_mse = (1.0 .- ridge_mus).^2 * gamma^2 + kappa * ridge_sigmas.^2 

        dy_approx_mus = load(joinpath(home_dir, "Results", "rlr", "rlr_dy_approx_mus_k$(kappa)_g$(gamma).jld2"))["dy_approx_mus"]
        ridge_approx_mus = load(joinpath(home_dir, "Results", "rlr", "rlr_ridge_approx_mus_k$(kappa)_g$(gamma).jld2"))["ridge_approx_mus"]
        dy_approx_mse = load(joinpath(home_dir, "Results", "rlr", "rlr_dy_approx_mse_k$(kappa)_g$(gamma).jld2"))["dy_approx_mse"]
        ridge_approx_mse = load(joinpath(home_dir, "Results", "rlr", "rlr_ridge_approx_mse_k$(kappa)_g$(gamma).jld2"))["ridge_approx_mse"]
        dy_approx_mse_scaled = load(joinpath(home_dir, "Results", "rlr", "rlr_dy_approx_mse_scaled_k$(kappa)_g$(gamma).jld2"))["dy_approx_mse_scaled"]
        ridge_approx_mse_scaled = load(joinpath(home_dir, "Results", "rlr", "rlr_ridge_approx_mse_scaled_k$(kappa)_g$(gamma).jld2"))["ridge_approx_mse_scaled"]
        dy_approx_sigmas = load(joinpath(home_dir, "Results", "rlr", "rlr_dy_approx_sigmas_k$(kappa)_g$(gamma).jld2"))["dy_approx_sigmas"]
        ridge_approx_sigmas = load(joinpath(home_dir, "Results", "rlr", "rlr_ridge_approx_sigmas_k$(kappa)_g$(gamma).jld2"))["ridge_approx_sigmas"]

        if kappa == 0.1 && gamma == 5 || gamma == 1 && kappa <= .4 
            mle_mu = dy_mus[1] 
            mle_sigma = dy_sigmas[1] 
            mle_mse = dy_mse[1] 
            # Exclude because of separation
            dy_approx_mse_scaled[1] = NaN 
            ridge_approx_mse_scaled[1] = NaN
            dy_approx_mse[1] = NaN 
            ridge_approx_mse[1] = NaN  
            dy_approx_mus[1] = NaN 
            ridge_approx_mus[1] = NaN 
            dy_approx_sigmas[1] = NaN 
            ridge_approx_sigmas[1] = NaN 
        end 

        ## MSE plot 
        plot!(mse_plot, alphas, dy_mse, color = col, linewidth = 1, linestyle = :solid, label = "") 
        scatter!(mse_plot, alpha_grid, dy_approx_mse, color = col, markerstrokecolor = col, marker=:diamond, label = "")
        plot!(mse_plot, 1 .- lambdas ./ maxl, ridge_mse, color = col, linewidth = 1, linestyle = :dash, label = "")
        scatter!(mse_plot, 1 .- lambda_grid ./ maxl, ridge_approx_mse, color = col, markerstrokecolor = col, marker=:diamond, label = "")
        xflip!(mse_plot)
        if kappa == 0.1 && gamma == 5 || gamma == 1 && kappa <= 0.4 
            plot!(mse_plot, [0,1], fill(mle_mse, 2), color = col, linewidth = 1, linestyle = :dashdot, label = "") 
        end
        
        ## MSE scaled plot 
        plot!(mse_scaled_plot, alphas, (kappa^(1/2) * dy_sigmas ./ dy_mus).^2, color = col, linewidth = 1, linestyle = :solid, label = "") 
        plot!(mse_scaled_plot, 1 .- lambdas ./ maxl, (kappa^(1/2) * ridge_sigmas ./ ridge_mus).^2, color = col, linewidth = 1, linestyle = :dash, label = "")
        scatter!(mse_scaled_plot, alpha_grid, dy_approx_mse_scaled, color = col, markerstrokecolor = col, marker=:diamond, label = "")
        scatter!(mse_scaled_plot, 1 .- lambda_grid ./ maxl, ridge_approx_mse_scaled, color = col, markerstrokecolor = col, marker=:diamond, label = "")
        xflip!(mse_scaled_plot)
        if kappa == 0.1 && gamma == 5 || gamma == 1 && kappa <= 0.4 
            plot!(mse_scaled_plot, [0,1], fill((kappa^(1/2) * mle_sigma ./ mle_mu)^2 , 2), color = col, linewidth = 1, linestyle = :dashdot, label = "") 
        end
        
        ## mu plot 
        plot!(mu_plot,alphas, dy_mus, color = col, linewidth = 1, linestyle = :solid, label = "") 
        plot!(mu_plot, 1 .- lambdas ./ maxl, ridge_mus, color = col, linewidth = 1, linestyle = :dash, label = "")
        scatter!(mu_plot, alpha_grid, dy_approx_mus, color = col, markerstrokecolor = col, marker=:diamond, label = "")
        scatter!(mu_plot, 1 .- lambda_grid ./ maxl, ridge_approx_mus, color = col, markerstrokecolor = col, marker=:diamond, label = "")
        xflip!(mu_plot)
        if kappa == 0.1 && gamma == 5  || gamma == 1 any(kappas .<= .4)
            plot!(mu_plot, [0,1], fill(mle_mu , 2), color = col, linewidth = 1, linestyle = :dashdot, label = "") 
        end

        ## Sigma plot 
        plot!(sigma_plot,alphas, sqrt.(kappa * dy_sigmas.^2), color = col, linewidth = 1, linestyle = :solid, label = "") 
        plot!(sigma_plot, 1 .- lambdas ./ maxl, sqrt.(kappa * ridge_sigmas.^2), color = col, linewidth = 1, linestyle = :dash, label = "")
        scatter!(sigma_plot, alpha_grid, sqrt.(dy_approx_sigmas), color = col, markerstrokecolor = col, marker=:diamond, label = "")
        scatter!(sigma_plot, 1 .- lambda_grid ./ maxl, sqrt.(ridge_approx_sigmas), color = col, markerstrokecolor = col, marker=:diamond, label = "")
        xflip!(sigma_plot)
        if kappa == 0.1 && gamma == 5  || gamma == 1 any(kappas .<= .4)
            plot!(sigma_plot, alphas, fill(sqrt.(kappa * mle_sigma^2) , length(alphas)), color = col, linewidth = 1, linestyle = :dashdot, label = "") 
        end
    end 
    xticks_vals = xticks(mu_plot)[1][1]
    yticks_vals = yticks(mu_plot)[1][1] 
    xticks_labels = to_latex.(xticks_vals, 2)
    yticks_labels = to_latex.(yticks_vals, 3)
    xticks!(mu_plot, (xticks_vals, xticks_labels))
    yticks!(mu_plot, (yticks_vals, yticks_labels))

    xticks_vals = xticks(mse_plot)[1][1]
    yticks_vals = yticks(mse_plot)[1][1] 
    xticks_labels = to_latex.(xticks_vals, 2)
    yticks_labels = to_latex.(yticks_vals, 3)
    xticks!(mse_plot, (xticks_vals, xticks_labels))
    yticks!(mse_plot, (yticks_vals, yticks_labels))

    xticks_vals = xticks(mse_scaled_plot)[1][1]
    yticks_vals = yticks(mse_scaled_plot)[1][1] 
    xticks_labels = to_latex.(xticks_vals, 2)
    yticks_labels = to_latex.(yticks_vals, 3)
    xticks!(mse_scaled_plot, (xticks_vals, xticks_labels))
    yticks!(mse_scaled_plot, (yticks_vals, yticks_labels))

    xticks_vals = xticks(sigma_plot)[1][1]
    yticks_vals = yticks(sigma_plot)[1][1] 
    xticks_labels = to_latex.(xticks_vals, 2)
    yticks_labels = to_latex.(yticks_vals, 3)
    xticks!(sigma_plot, (xticks_vals, xticks_labels))
    yticks!(sigma_plot, (yticks_vals, yticks_labels))
    mu_plots[pos] = mu_plot
    mse_plots[pos] = mse_plot
    mse_scaled_plots[pos] = mse_scaled_plot
    sigma_plots[pos] = sigma_plot
    pos += 1
end
mu_plots_final = plot(mu_plots...,legend_plot, layout = (1, 4), size = (1000, 500))
sigma_plots_final = plot(sigma_plots...,legend_plot, layout = (1, 4), size = (1000, 500))
mse_plots_final = plot(mse_plots..., legend_plot,layout = (1, 4), size = (1000, 500))
mse_scaled_plots_final = plot(mse_scaled_plots...,legend_plot, layout = (1, 4), size = (1000, 500))

savefig(mu_plots_final,joinpath("Figures", "rlr_mu_plots.pdf"))
savefig(mse_plots_final,joinpath("Figures", "rlr_mse_plots.pdf"))
savefig(mse_scaled_plots_final,joinpath("Figures", "rlr_mse_scaled_plots.pdf"))
savefig(sigma_plots_final,joinpath("Figures", "rlr_sigmas_plots.pdf"))