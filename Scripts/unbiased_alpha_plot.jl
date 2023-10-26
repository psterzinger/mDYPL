using MKL, LinearAlgebra, JLD2, Plots, LaTeXStrings, ColorSchemes

home_dir = "" 
include(joinpath(home_dir, "Scripts", "AMP_DY.jl"))
using .AMP_DY

kappas = 0.1:0.00625:0.975 
gammas = .5:0.0625:10 

params = load(joinpath(home_dir, "Results", "alpha_unbiased.jld2"))["params"]
alphas = params[:,:,1]
roots = load(joinpath(home_dir, "Results", "alpha_unbiased_res.jld2"))["roots"]
#=
roots = similar(params)
rs = randn(3)
for i in eachindex(gammas) 
    for j in eachindex(kappas)
        if(!any(isnan.(params[i,j,:])))
            pars = params[i,j,:]
            alpha = pars[1] 
            pars[1] = 1.0 
            Main.AMP_DY.eq_bin!(rs, pars, kappas[j], gammas[i], alpha) 
            roots[i,j,:] = rs
        end 
    end 
end 
@save joinpath(home_dir, "Results", "alpha_unbiased_res.jld2") roots
=#
cutoff = 1e-6
res = mapslices(v -> maximum(abs.(v)), roots; dims = 3)[:,:,1]
alphas[ res .> cutoff] .= NaN 


gr() 

ztick = .8:.1:1 
zlabels = [L"0.8", L"0.9", L"1.0"]
ytick = [0.1,0.2,0.3,0.4,0.5] 
ylabels = [L"0.1", L"0.2", L"0.3", L"0.4", L"0.5"]
xtick = [2,4,6,8,10] 
xlabels = [L"2", L"4", L"6", L"8", L"10"]

grange = 1:2:length(gammas) 
krange = 1:2:length(kappas) 
p = Plots.wireframe(gammas[grange], kappas[krange], alphas[grange,krange]', 
        zticks =(ztick, zlabels), 
        yticks = (ytick, ylabels), 
        xticks = (xtick, xlabels),  
        ylabel = L"\kappa", 
        xlabel = L"\gamma",
        yflip = true, 
        xflip = true,
        camera = (45, 30), 
        size = (400,400),
        ylims = (0.1,0.5)
    ) 
Plots.savefig(p, joinpath(home_dir, "Figures", "alpha_unbiased.pdf")) 
