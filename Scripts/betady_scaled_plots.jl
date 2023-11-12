using Random, Plots, LaTeXStrings, JLD2, ColorSchemes, DataFrames, Optim

home_dir = "" 
include(joinpath(home_dir, "Scripts", "AMP_DY.jl"))
using .AMP_DY


function  fill_beta(p,b)
    beta = Vector{Float64}(undef,p) 
    pp = length(b) 
    for i in eachindex(beta) 
        beta[i] = b[(i-1)%pp+1]
    end 
    sort(beta)
end 

function fill_beta_SC(p)
    pp = ceil(Int64, p / 8) 
    vcat(fill(-10,pp),fill(10,pp),fill(0.0,p-2*pp))
end 

#n = 1000 # for Figure 1 
n = 2000 # for Figure 2
kappas = (0.1,0.2,0.4,0.6,0.8,0.9) 
gammas = (2,6,10,14,18)
b = [-3,-3/2,0,3/2,3]
reps = 10


full_grid = [(0.0,0.0) for i in  Iterators.product(gammas,kappas)]

for i in eachindex(gammas)
    gamma = gammas[i]
    for j in eachindex(kappas)
        kappa = kappas[j]
        full_grid[i,j] = (kappa,gamma) 
    end 
end 

grid_inds = vcat(1,11,21,7,17,3,13,23,9,19,5,15,25)

kg_pairs = vcat((0.2,sqrt(0.9)), (sqrt(.9)/22.4, 0.2*22.4), full_grid[grid_inds][2:end])

betas = [Float64[] for i in eachindex(kg_pairs)] 
beta_true = [Float64[] for i in  eachindex(kg_pairs)] 
params = [Float64[] for i in  eachindex(kg_pairs)] 
counter = 1
for (kappa,gamma) in kg_pairs 
    println("κ:$kappa, ", "γ:$gamma")
    p = floor(Int64, n * kappa)
    beta = fill_beta_SC(p) * gamma / sqrt(kappa) * 0.2 # Figure 2
    #beta = fill_beta(p,b) / sqrt(9/2 * kappa) * gamma # Figure 1
    beta_true[counter] = beta
    a = 1 / (1 + kappa)
    for k in 1:reps 
        Random.seed!((counter - 1) * length(kg_pairs) + k) 
        X = randn(n,p) / sqrt(n) 
        mu = 1.0 ./ (1.0 .+ exp.(.-X*beta)) 
        y = rand(n) .< mu 
        beta_DY = Optim.minimizer(logistic_mDYPL(y, X, a; beta_init = beta, method = Optim.NewtonTrustRegion()))
        if k == 1
            betas[counter] = beta_DY
        else 
            betas[counter] .= betas[counter] .+ beta_DY 
        end 
    end 
    counter += 1
end 
betas = betas ./ reps 
#@save joinpath(home_dir, "Results", "beta_dy_sc.jdl2") betas
#@save joinpath(home_dir, "Results", "beta_dy_ra.jdl2") betas

betas = load(joinpath(home_dir, "Results", "beta_dy_sc.jdl2"))["betas"]
#betas = load(joinpath(home_dir, "Results", "beta_dy_ra.jdl2"))["betas"]

ks = .05:.025:.95 
gs = .5:0.5:20

pars = load(joinpath(home_dir, "Results", "DY_RA_params.jld2"))["params"]

mus = Vector{Float64}(undef,length(kg_pairs))
count = 1 
for (kappa, gamma) in kg_pairs
    if count > 2
        row = findfirst(x -> isequal(x, gamma), gs) 
        col = findfirst(x -> isequal(x, kappa), ks) 
        mus[count] = pars[row,col][1] 
    else 
        mus[count] = find_params_nlsolve(kappa, gamma, 1 / (1 + kappa)).zero[1]
    end 
    count += 1 
end  

gamma_grid = vcat(0.0,0.001,0.01,collect(0.0:20/100:22.5)[2:end])
#=
Random.seed!(123) 
ns = 2000000
out = Main.AMP_DY.h_mle(
    beta0 = 0, 
    XZU = DataFrame([randn(ns),randn(ns),rand(ns)],vcat("X","Z","U")),
    gamma_grid = gamma_grid
)
out = vcat([1 0],out, [0 maximum(gamma_grid)])
=# 
out = load(joinpath(home_dir, "Results", "phase_trans.jld2"))["out"]
scalefontsizes(1.5)

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
    fillalpha = 0.1,
    fc = :gray,  
    label = :none, 
    xlabel = L"$\kappa$", 
    ylabel = L"$\gamma$", 
    linewidth = 0.0
)

xticks!(p,0:.1:1,vcat(L"0", "",L"0.2","",L"0.4","",L"0.6","",L"0.8","", L"1.0"), z = 1)
yticks!(p,0:5:20,vcat(L"0",L"5",L"10",L"15",L"20"))
           
scatter!(p, kg_pairs, color = :gray, label = :none, fillstyle = :none, markersize = 1) 
xlim = Plots.xlims(p) 
ylim = Plots.ylims(p) 

for i in eachindex(kg_pairs) 
    uv = unique(beta_true[i]) 
    mult = vcat(0,cumsum(map( v -> sum(isequal.(v,beta_true[i])), unique(beta_true[i]))))
    seqs = [(1+mult[j-1]):mult[j] for j in 2:length(mult)] 
    (x,y) = kg_pairs[i]
    x_norm = (x  - xlim[1]) / (xlim[2] - xlim[1]) + .005
    y_norm = (y  - ylim[1]) / (ylim[2] - ylim[1]) + .005
    Plots.scatter!(p,
        betas[i],# ./ mus[i], #Uncomment for unscaled
        color = ColorSchemes.viridis[2/10],
        markerstrokecolor = ColorSchemes.viridis[2/10],
        fillalpha = .5, 
        alpha = .5, 
        label = :none, 
        markersize = 2, 
        inset = (1, bbox(x_norm, y_norm, 0.15, 0.15, :bottom)),
        subplot = i+1, 
        bg_inside = nothing, 
        ticks = nothing
    )

    for j in eachindex(seqs)
        Plots.plot!(p[i+1],seqs[j],fill(uv[j],length(seqs[j])), 
            linecolor = ColorSchemes.viridis[floor(Int64,256 * 5 / 6)], 
            linewidth = 2, 
            label = :none#, 
            #framestyle = :box
        )  
    end
end 

Plots.savefig(p, joinpath(home_dir, "Figures", "AMP_DY_RA_phase_trans_plot_unscaled.pdf"))