using Random, Plots, LaTeXStrings, JLD2, ColorSchemes, DataFrames

home_dir = "" 
include(joinpath(home_dir, "AMP_DY.jl"))
include("AMP_DY.jl") 
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

n = 2000
kappas = (0.1,0.2,0.4,0.6,0.8,0.9) 
gammas = (2,6,10,14,18)
b = [-3,-3/2,0,3/2,3]

betas = [Float64[] for i in  Iterators.product(gammas,kappas)] 
beta_true = [Float64[] for i in  Iterators.product(gammas,kappas)] 
params = [Float64[] for i in  Iterators.product(gammas,kappas)] 
alphas = Matrix{Float64}(undef,size(betas)) 

Random.seed!(123)
for i in eachindex(gammas)
    gamma = gammas[i]
    println("γ:$gamma")
    for j in eachindex(kappas) 
        kappa = kappas[j]
        println("κ:$kappa")
        a = 1 / (1 + kappa) 
        p = floor(Int64, n * kappa)
        #beta = fill_beta_SC(p) * gamma / sqrt(kappa) * 0.2 
        beta = fill_beta(p,b) / sqrt(9/2 * kappa) * gamma 
        beta_true[i,j] = beta
        X = randn(n,p) / sqrt(n) 
        mu = 1.0 ./ (1.0 .+ exp.(.-X*beta)) 
        y = rand(n) .< mu 
        beta_DY = logistic_mDYPL(y, X, a; beta_init = beta, method = Optim.NewtonTrustRegion()) 
        betas[i,j] = beta_DY 
    end 
end 

#=
start = find_params_nlsolve(0.1, 1, 1 / (1 + 0.2);
    verbose = false, 
    x_init = missing, 
    method = :newton, 
).zero

for i in eachindex(gammas)
    gamma = gammas[i] 
    for j in eachindex(kappas) 
        kappa = kappas[j] 
        conv = false 
        ccount = 0 
        if i == 1
            if j == 1 
                x_init = start 
            elseif !any(isnan.(params[i,j-1])) 
                x_init = params[i,j-1] 
            else 
                x_init = start 
            end
        else
            if j == 1 
                if !any(isnan.(params[i-1,j])) 
                    x_init = params[i-1,j]
                else 
                    x_init = start
                end 
            elseif !any(isnan.(params[i,j-1]))
                x_init = params[i,j-1]
            else
                x_init = start 
            end 
        end 
        while !conv && ccount < 4
            for constr in (false, true)  
                for method in (:trust_region, :newton) 
                    ccount += 1
                    try
                        pars = find_params_nlsolve(kappa, gamma, 1 / (1 + kappa);
                                            verbose = true, 
                                            x_init = x_init, 
                                            constrained_solve = constr, 
                                            method = method, 
                                            reformulation = :smooth
                        )
                        conv = converged(pars) 
                        if conv
                            params[i,j] = pars.zero
                            break 
                        end 
                    catch
                    end
                    if conv 
                        break 
                    end 
                end
                if conv 
                    break 
                end 
            end 
        end 

        if !conv 
            for k in j:length(kappas) 
                params[i,k] = fill(NaN,3) 
            end 
            break 
        end 
    end 
end 

out = Matrix{Float64,length(gammas), length(kappas)}
res = randn(3)
for i in eachindex(gammas) 
    for j in eachindex(kappas)
        Main.AMP_DY.eq_bin!(res,params[i,j],kappas[j],gammas[i], 1 / (1 + kappas[j])) 
        out[i,j] = maximum(abs.(res)) 
    end 
end 
cutoff = 1e-8 
sum(out .> cutoff) 
=# 

ks = .05:.025:.95 
k_inds = findall(x -> x in kappas, ks)
gs = .5:0.5:20
g_inds = findall(x -> x in gammas, gs)

params = load(joinpath(home_dir, "Results/DY_RA_params.jld2"))["params"]
params = [params[g, k, i] for g in g_inds, k in k_inds, i in axes(params, 3)]

alphas = map(v -> v[1], params)

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
out = load(joinpath(home_dir, "Results/phase_trans.jld2"))["out"]

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
    fillalpha = 0.35,
    fc = :gray,  
    label = :none, 
    xlabel = L"$\kappa$", 
    ylabel = L"$\gamma$", 
    linewidth = 0.0
)

xticks!(p,0:.1:1,vcat(L"0", "",L"0.2","",L"0.4","",L"0.6","",L"0.8","", L"1.0"), z = 1)
yticks!(p,0:5:20,vcat(L"0",L"5",L"10",L"15",L"20"))


full_grid = [(0.0,0.0) for i in  Iterators.product(gammas,kappas)]

for i in eachindex(gammas)
    gamma = gammas[i]
    for j in eachindex(kappas)
        kappa = kappas[j]
        full_grid[i,j] = (kappa,gamma) 
    end 
end 

grid_inds = vcat(1,11,21,7,17,3,13,23,9,19,5,15,25)
            
scatter!(p,full_grid[grid_inds], color = :gray, label = :none, fillstyle = :none, markersize = 1) 
xlim = Plots.xlims(p) 
ylim = Plots.ylims(p) 

for i in eachindex(grid_inds) 
    uv = unique(beta_true[grid_inds[i]]) 
    mult = vcat(0,cumsum(map( v -> sum(isequal.(v,beta_true[grid_inds[i]])), unique(beta_true[grid_inds[i]]))))
    seqs = [(1+mult[j-1]):mult[j] for j in 2:length(mult)] 
    (x,y) = full_grid[grid_inds[i]] 
    x_norm = (x  - xlim[1]) / (xlim[2] - xlim[1]) + .005
    y_norm = (y  - ylim[1]) / (ylim[2] - ylim[1]) + .005
    Plots.scatter!(p,
        betas[grid_inds[i]],# ./ alphas[grid_inds[i]],
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

