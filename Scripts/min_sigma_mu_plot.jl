using JLD2, Plots, LaTeXStrings
home_dir = ""

include(joinpath(home_dir, "Scripts", "AMP_DY.jl"))
using .AMP_DY

kappas = .05:.0125:0.9875
gammas = .125:0.125:10
alphas = 0.0125:0.0125:1.0

pars_full = load(joinpath(home_dir, "Results", "DY_pars_full.jld2"))["pars_full"]
res = load(joinpath(home_dir, "Results", "DY_pars_full_res.jld2"))["res"]
#=
using Distributed
addprocs(20)
@everywhere home_dir = ""
@everywhere include(joinpath(home_dir, "Scripts", "AMP_DY.jl"))
@everywhere using .AMP_DY, SharedArrays

res = similar(pars_full)
res = SharedArray(res) 
pars_full = SharedArray(pars_full)

for k in eachindex(alphas) 
    for i in eachindex(gammas) 
        @sync @distributed for j in eachindex(kappas)
            if(!any(isnan.(pars_full[i,j,k,:])))
                rs = randn(3)
                Main.AMP_DY.eq_bin!(rs, pars_full[i,j,k,:], kappas[j], gammas[i], alphas[k]) 
                res[i,j,k,:] = rs
            end 
        end 
    end 
end 
@save joinpath(home_dir, "Results", "DY_pars_full_res.jld2") res =#

min_alpha = Matrix{Float64}(undef,(length(gammas),length(kappas))) 
minv = similar(min_alpha)
cutoff = 1e-8
for i in eachindex(gammas)
    for j in eachindex(kappas) 
        vals = pars_full[i,j,:,3] ./ pars_full[i,j,:,1] 
        #vals = (pars_full[i,j,:,1] .- 1.0).^2 .* kappas[j] .* gammas[i].^2 .+ pars_full[i,j,:,3].^2 ## for unscaled MSE 
        max_res = mapslices(v -> maximum(abs.(v)), res[i,j,:,:]; dims = 2)
        inds = isnan.(vals) .|| isnan.(max_res) .|| max_res .> cutoff .|| pars_full[i,j,:,3] .< 1e-5
        vals[inds[:,1]] .= Inf 
        min_alpha[i,j] = alphas[argmin(vals)] 
        minv[i,j] = minimum(vals)
    end 
end 


gr() 
#scalefontsizes(1.5)
grange = 1:length(gammas) 
krange = 1:length(kappas) 

p = Plots.wireframe(gammas[grange],kappas[krange],min_alpha[grange,krange]',
    xflip = true, 
    yflip = true,
    camera  = (30,45),
    xlabel = L"$\gamma$",
    ylabel = L"$\kappa$", 
    size = (400,400), 
    xticks = (vcat(1,2:2:10),[L"1", L"$2$",L"$4$",L"$6$",L"$8$",L"$10$"]), 
    yticks = ([0.2,0.4,0.6,0.8],[L"$0.2$",L"$0.4$",L"$0.6$",L"$0.8$"]), 
    zticks = ([0.2,0.4,0.6,0.8,1.0],[L"$0.2$",L"$0.4$",L"$0.6$",L"$0.8$",L"$1$"]),
) 

Plots.savefig(p, joinpath(home_dir, "Figures", "min_sigma_mu_surface.pdf"))