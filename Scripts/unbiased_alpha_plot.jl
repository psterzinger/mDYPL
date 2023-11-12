using MKL, LinearAlgebra, JLD2, Plots, LaTeXStrings, ColorSchemes, Makie, CairoMakie

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

CairoMakie.activate!()
s = Figure()

grange = 1:2:length(gammas) 
krange = 1:2:length(kappas) 
mina = minimum(alphas[grange, krange][.!isnan.(alphas[grange, krange])])
shift = 0.0

ztick = (0.8:.1:1) .- mina .+ shift
zlabels = [L"0.8", L"0.9", L"1"]
xtick = [0.1,0.2,0.3,0.4,0.5] 
xlabels = [L"0.1", L"0.2", L"0.3", L"0.4", L"0.5"]
ytick = [2,4,6,8,10] 
ylabels = [L"2", L"4", L"6", L"8", L"10"]

ax = Axis3(
    s[1, 1],
    ylabel = L"\gamma",
    xlabel = L"\kappa",
    zlabel = L"\alpha",
    xticklabelsize = 24,
    yticklabelsize = 24,
    zticklabelsize = 24,
    xlabelsize = 30, 
    ylabelsize = 30,
    zlabelsize = 30,
    xreversed = true,
    zticks = (ztick, zlabels), 
    yticks = (ytick, ylabels), 
    xticks = (xtick, xlabels), 
    aspect = :equal, 
    elevation = 0.15 * pi, 
    azimuth = 1.175 * pi, 
    zspinecolor_2 = :lightgray, 
    zspinecolor_3 = :white, 
    xspinecolor_2 = :lightgray, 
    xspinecolor_3 = :white, 
    yspinecolor_2 = :lightgray, 
    yspinecolor_3 = :white, 
    protrusions = (35,5,20,5)
)

Makie.xlims!(ax, 0.5, 0.1)
Makie.zlims!(ax, 0.0, 1-mina+shift)
Makie.ylims!(ax, .5, 10)

Makie.wireframe!(ax, kappas[krange], gammas[grange], alphas[grange, krange]' .- mina .+ shift,
    color = :black, 
    overdraw = true
)

#= Add a contour plot on kappa gamma plane if desired 
Makie.contour!(ax, kappas[krange], gammas[grange], alphas[grange, krange]' .- mina .+ shift,
    color = :black,
    levels = 20
)
=# 
Makie.save(joinpath(home_dir, "Figures", "alpha_unbiased.pdf"), s, resolution = (600, 600))

s = Figure(fonts = (; math = "Latin Modern Math")) 
ax = Axis(
    s[1, 1],
    ylabel = L"\gamma",
    xlabel = L"\kappa",
    xticklabelsize = 27,
    yticklabelsize = 27,
    xlabelsize = 34, 
    ylabelsize = 34,
    xreversed = true,
    yticks = (ytick, ylabels), 
    xticks = (xtick, xlabels)
)
Makie.hidespines!(ax, :t, :r) 
Makie.xlims!(ax, 0.5, 0.1)
Makie.ylims!(ax, .5, 10)

krange = 1:65
grange = 1:length(gammas) 

maxa = maximum(alphas[grange, krange][.!isnan.(alphas[grange, krange])])
levels = log.(range(exp(mina),exp(maxa),20))

Makie.contour!(ax, kappas[krange], gammas[grange], alphas[grange, krange]';
    labels=true, 
    levels, 
    color = :black,
    labelsize = 20, 
    labelfont = :math
)
Makie.save(joinpath(home_dir, "Figures", "alpha_unbiased_contour_only.pdf"), s, resolution = (600, 600))



#=
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
        zlabel = L"\alpha", 
        yflip = true, 
        xflip = true,
        camera = (45, 30), 
        size = (400,400),
        ylims = (0.1,0.5)
    ) 
Plots.savefig(p, joinpath(home_dir, "Figures", "alpha_unbiased.pdf")) 
=#