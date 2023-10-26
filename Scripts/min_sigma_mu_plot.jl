using JLD2, Plots, LaTeXStrings
home_dir = ""

include(joinpath(home_dir, "Scripts", "AMP_DY.jl"))
using .AMP_DY

kappas = .05:.0125:0.9875
gammas = .125:0.125:10
alphas = 0.0125:0.0125:1.0

pars_full = load(joinpath(home_dir, "Results", "DY_pars_full_long_run8.jld2"))["pars_full"]
res = load(joinpath(home_dir, "Results", "DY_pars_full_res_run8.jld2"))["res"]

#=
res = similar(pars_full)
rs = randn(3)
for k in eachindex(alphas) 
    for i in eachindex(gammas) 
        for j in eachindex(kappas)
            if(!any(isnan.(pars_full[i,j,k,:])))
                Main.AMP_DY.eq_bin!(rs, pars_full[i,j,k,:], kappas[j], gammas[i], alphas[k]) 
                roots[i,j,k,:] = rs
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
p = Plots.wireframe(gammas[4:48],kappas[1:69],min_alpha[4:48,1:69]',
    xflip = true, 
    #yflip = true,
    camera  = (30,45),
    xlabel = L"$\gamma$",
    ylabel = L"$\kappa$", 
    size = (400,400), 
    xticks = ([1,2,3,4,5,6],[L"$1$",L"$2$",L"$3$",L"$4$",L"$5$",L"$6$"]), 
    yticks = ([0.2,0.4,0.6,0.8],[L"$0.2$",L"$0.4$",L"$0.6$",L"$0.8$"]), 
    zticks = ([0.2,0.4,0.6,0.8,1.0],[L"$0.2$",L"$0.4$",L"$0.6$",L"$0.8$",L"$1$"]),
) 

Plots.savefig(p, joinpath(home_dir, "Figures", "min_sigma_mu_surface.pdf"))