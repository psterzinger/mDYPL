using Random, Plots, LaTeXStrings, JLD2, ColorSchemes, DataFrames, Optim, Distributions, LaTeXStrings, JLD2, Plots.PlotMeasures

home_dir = "" # Should be the supplementary material folder
include(joinpath(home_dir, "Scripts", "AMP_DY.jl"))
using .AMP_DY

function fill_beta_SC_zeros_first(p)
    pp = ceil(Int64, p / 2) 
    vcat(fill(0.0,pp),fill(10,p-pp))
end 

n_simulations = 10000
n = 8000
p = 800
kappa = p/n 
test_statistics = Vector{Float64}(undef, n_simulations)
alphas = [1.0, 1 / (1 + kappa), 0.75, 0.5, 0.25] 
leave_out = [1,10] 
gamma = sqrt(5)

beta_SC = fill_beta_SC_zeros_first(p) #* gamma / sqrt(50 * kappa) 
plots = [] 
count = 0

for a in alphas 
    for leave in leave_out 
        #=
        for i in 1:n_simulations
            Random.seed!(count * n_simulations + i)
            betas = beta_SC[(leave+1):end] 
            X =  randn(n, p) / sqrt(n)
            
            y = rand(n) .< 1 ./ (1 .+ exp.(-X * beta_SC))
            y_star = a * y .+ (1 - a) / 2

            beta_full = Optim.minimizer(logistic_mDYPL(y, X, a; beta_init = beta_SC))
            ll_full = loglikl(beta_full, y_star, X, Vector{Float64}(undef,n), Vector{Float64}(undef,n))

            X_constrained = X[:, (leave+1):end]

            beta_constrained = Optim.minimizer(logistic_mDYPL(y, X_constrained, a; beta_init = betas))
            ll_constrained = loglikl(beta_constrained, y_star, X_constrained, Vector{Float64}(undef,n), Vector{Float64}(undef,n))

            test_statistics[i] = 2 * (ll_constrained - ll_full)
        end
        @save joinpath(home_dir, "Results", "llr", "llr_test_l$(leave)_a$(a).jld2") test_statistics
        =#
        test_statistics = load(joinpath(home_dir, "Results", "llr", "llr_test_l$(leave)_a$(a).jld2"))["test_statistics"]
        if a < 0.75 
            x_init = [0.1,0.5,1.0]
        else
            x_init = missing 
        end 
        pars = find_params_nonlinearsolve(kappa, sqrt(5), a; verbose = true, x_init = x_init)
        b = pars[2]
        sigma = pars[3] 
        t_stats = sort(test_statistics) * b  / (kappa * sigma^2) 
        chi_qs = quantile(Chisq(leave), (1:n_simulations) ./ (n_simulations + 1)) 
        if a == 1
            ps = Plots.scatter(chi_qs, t_stats, 
                label = :none, 
                markersize = 4, 
                color = ColorSchemes.viridis[2/10],
                markerstrokecolor = ColorSchemes.viridis[2/10], 
                alpha = .5, 
                left_margin = 5mm
            )
        else 
            ps = Plots.scatter(chi_qs, t_stats, 
                label = :none, 
                markersize = 4, 
                color = ColorSchemes.viridis[2/10],
                markerstrokecolor = ColorSchemes.viridis[2/10], 
                alpha = .5
            )
        end 

        xlim = extrema(t_stats)
        ylim = extrema(chi_qs)
        limits =  min(xlim[2], ylim[2])
        plot!(ps, [0.0,limits], [0.0,limits], linestyle = :dash, color = :black, label = :none, linewidth = 2, linecolor = ColorSchemes.viridis[floor(Int64,256 * 5 / 6)])
        if a == 1.0 && leave == 1
            title!(ps, L"\alpha = 1", titlefontsize = 11)
            ylabel!(ps, L"\beta_{0,1} = 0")
        elseif a == 1 
            ylabel!(ps,L"\beta_{0,1} = \ldots = \beta_{0,10} = 0")
        elseif a == 1/(1 + kappa) && leave == 1 
            title!(ps, L"\alpha = 1 / (1 + \kappa)", titlefontsize = 11) 
        elseif a == 0.75 && leave == 1
            title!(ps, L"\alpha = 3 / 4", titlefontsize = 11) 
        elseif a == 0.5 && leave == 1
            title!(ps, L"\alpha = 1 / 2", titlefontsize = 11) 
        elseif a == 0.25 && leave == 1
            title!(ps, L"\alpha = 1 / 4", titlefontsize = 11) 
        end 
        push!(plots,ps)
        count += 1
    end
end

inds = [1,3,5,7,9,2,4,6,8,10]
plot_order = plots[inds]

function round_to_sf(x::Real, sf::Int=2)
    if x == 0
        return 0
    end
    d = ceil(log10(abs(x)))
    factor = 10.0^(sf - d)
    return round(x * factor) / factor
end

function to_latex(n::Number)
    return "\$$(round_to_sf(n, 2))\$" 
end

for p in plot_order
    xticks_vals = xticks(p)[1][1]
    yticks_vals = yticks(p)[1][1] 

    xticks_labels = to_latex.(xticks_vals)
    yticks_labels = to_latex.(yticks_vals)

    xticks!(p, (xticks_vals, xticks_labels))
    yticks!(p, (yticks_vals, yticks_labels))
end

final_plot = plot(plot_order...,layout = (2, 5), size = (1000, 400))

Plots.savefig(final_plot,joinpath(home_dir, "Figures", "qq_plot_all.pdf"))