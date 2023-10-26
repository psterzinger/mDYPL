using Random, Plots, LaTeXStrings, JLD2, ColorSchemes, DataFrames

home_dir = "" # Should be the supplementary material folder
include(joinpath(home_dir, "Scripts", "AMP_DY.jl"))
using .AMP_DY

function fill_beta_SC(p)
    pp = ceil(Int64, p / 8) 
    vcat(fill(-10,pp),fill(10,pp),fill(0.0,p-2*pp))
end 


kappas = (0.1, 0.2, 0.4, 0.6, 0.8, 0.9) 
gammas = (2, 6, 10, 14, 18)
alphas = (0.9, 0.95, 0.9, 0.999) # For Figure 3 set alphas = [0.0] 
ns = vcat(200 : 200 : 8000)
reps = 50 

full_grid = [(0.0, 0.0) for i in  Iterators.product(gammas, kappas)]

for i in eachindex(gammas)
gamma = gammas[i]
for j in eachindex(kappas)
    kappa = kappas[j]
    full_grid[i,j] = (kappa, gamma) 
end 
end 

grid_inds = vcat(1,11,21,7,17,3,13,23,9,19,5,15,25)

grid_vals = [(x,y) = full_grid[ind] for ind in grid_inds] 

betas = [Float64[] for i in eachindex(grid_vals)] 
beta_true = [Float64[] for i in eachindex(grid_vals)] 
beta_length = fill(NaN,length(ns)) 
norms = SharedVector{Float64}(reps) 
params = [Float64[] for i in eachindex(grid_vals)] 

#=
count = 0
for a in alphas 
    for i in eachindex(grid_vals)
        (kappa, gamma) = grid_vals[i]
        for k in 1:length(ns) 
            n = ns[k] 
            p = floor(Int64, n * kappa)
            # a = 1 / (1 + kappa) # For Figure 3
            beta = fill_beta_SC(p) * gamma / sqrt(kappa) * 0.2 
            beta_true[i] = beta
            for l in 1:reps 
                Random.seed!(count * reps + l)
                X = randn(n,p) / sqrt(n) 
                mu = 1.0 ./ (1.0 .+ exp.(.-X*beta)) 
                y = rand(n) .< mu 
                beta_DY = logistic_DY_MAPE(y,X,a; beta_init = beta) 
                norms[l] = sum(beta_DY.^2)
            end
            beta_length[k] = mean(norms)  
        end 
        betas[i] = copy(beta_length)
    end 
    @save joinpath(home_dir,"Results", "beta_norm", beta_l2_length_a$a.jld2") betas 
end 
=# 

out = load(joinpath(home_dir,"Results", "phase_trans.jld2"))["out"]
gamma_grid = vcat(0.0,0.001,0.01,collect(0.0:20/100:22.5)[2:end])

plots = [] 
scalefontsizes(1.5)
for a in alphas 
    #betas = load(home_dir,"Results/beta_l2_length.jld2")["betas"] For Figure 3, a = [1 / (1 + kappa)], load this file 
    betas = load(home_dir,"Results", "beta_norm", "beta_l2_length_a$a.jld2")["betas"]
    
    p = plot(out[:,1],out[:,2], 
    label = :none, 
    xlabel = L"$\kappa$", 
    ylabel = L"$\gamma$", 
    linewidth = 0.0, 
    grid=false,
    widen = false, 
    xlims = (0,1), 
    tick_direction = :out, 
    foreground_color_axis = nothing
    )

    plot!(p,out[:,1],out[:,2], 
    fillrange = zero(out[:,1]),
    fc = :white,  
    lc = :white, 
    label = :none, 
    xlabel = L"$\kappa$", 
    ylabel = L"$\gamma$", 
    )

    plot!(p,out[:,1],out[:,2], 
    fillrange = fill(maximum(gamma_grid),length(out[:,1])),
    fillalpha = 0.35,
    fc = :gray,  
    label = :none, 
    xlabel = L"$\kappa$", 
    ylabel = L"$\gamma$", 
    linewidth = 0.0
    )

    xticks!(p,0:.1:1,vcat(L"0", "",L"0.2","",L"0.4","",L"0.6","",L"0.8","", L"1.0"), z = 1)
    yticks!(p,0:5:20,vcat(L"0",L"5",L"10",L"15",L"20"))

    scatter!(p,full_grid[grid_inds], color = :gray, label = :none, fillstyle = :none, markersize = 1,)
    xlim = Plots.xlims(p) 
    ylim = Plots.ylims(p) 

    xticklabels = [L"$0$",L"$4\mathrm{k}$",L"$8\mathrm{k}$"]

    for i in eachindex(grid_inds) 
        (x,y) = full_grid[grid_inds[i]] 
        x_norm = (x  - xlim[1]) / (xlim[2] - xlim[1]) + .005
        y_norm = (y  - ylim[1]) / (ylim[2] - ylim[1]) + .005
        
        y_values = sqrt.(betas[i] ./ collect(200:200:8000))
        
        y_values = sqrt.(betas[i] ./ collect(200:200:8000))
        y_min = minimum(y_values)
        y_max = maximum(y_values)
        y_quarter = y_min + 0.25 * (y_max - y_min)
        y_three_quarter = y_min + 0.75 * (y_max - y_min)

        y_quarter_label = LaTeXString("\$$(round(y_quarter, digits=2))\$")
        y_three_quarter_label = LaTeXString("\$$(round(y_three_quarter, digits=2))\$")

        Plots.plot!(p,
            200:200:8000,
            sqrt.(betas[i] ./ collect(200:200:8000)),
            color = :black,
            fillalpha = .25, 
            alpha = 1, 
            label = :none, 
            markersize = 2, 
            inset = (1, bbox(x_norm, y_norm, 0.15, 0.15, :bottom)),
            subplot = i+1, 
            bg_inside = nothing, 
            #yticks = nothing, 
            xtickfont = font(6), 
            ytickfont = font(6), 
            yticks = ((y_quarter, y_three_quarter), [y_quarter_label, y_three_quarter_label]),
            xticks = ((0,4000,8000), xticklabels),
            #xrotation = 45
            #ticks = nothing
        )
    end
    push!(plots,p)
end 
if length(plots) > 1 
    plots = reverse(plots) 
    i = 1
    for a in alphas 
        p = plots[i]
        title!(p, LaTeXString("\$ \alpha = $(a) \$"), titlefontsize = 11) 
        i += 1 
    end 
    final_plot = plot(plots...,layout=(2,2))
else
    final_plot = plots[1] 
end 

Plots.savefig(final_plot,joinopath("Figures", "beta_norm_alphas.pdf"))
