using Plots, LaTeXStrings, JLD2, LineSearches, NonlinearSolve, NLsolve

home_dir = "" 
include(joinpath(home_dir, "Scripts", "AMP_DY.jl"))
using .AMP_DY

kappas = .05:.025:.95 
gammas = .5:0.5:20

#=
start = find_params_nonlinearsolve(0.05, 1.0, 1 / (1 + 0.05);
    verbose = true, 
    x_init = missing
)
params =[Float64[] for i in eachindex(gammas), j in eachindex(kappas)]
params = load(joinpath(home_dir, "Results", "DY_RA_params.jld2"))["params"]
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
        if kappa == 0.05 && gamma == 20 
            x_init = [0.27038678503812746, 0.8524915525697666, 2.633460787732285]
        end 
        try
            pars = find_params_nlsolve(kappa, gamma, 1 / (1 + kappa);
                            verbose = true, 
                            x_init = x_init, 
                            method = :newton 
                        )
            root = rand(3) 
            Main.AMP_DY.eq_bin!(root,pars.zero, kappa, gamma, 1 / (1 + kappa))
            conv = converged(pars) && maximum(abs.(root)) < 1e-6 
            pars = pars.zero  
            #=
            pars = find_params_nonlinearsolve(kappa, gamma, 1 / (1 + kappa);
                                verbose = true, 
                                x_init = x_init,
                                #numeric = true
                            )
            root = rand(3) 
            Main.AMP_DY.eq_bin!(root,pars, kappa, gamma, 1 / (1 + kappa))
            conv = maximum(abs.(root)) < 1e-6 
            =#
            if conv
                params[i,j] = pars
                #break 
            end 
        catch e 
            println(e)
        end
        if !conv 
            for k in j:length(kappas) 
                params[i,k] = fill(NaN,3) 
            end 
            break 
        end 
    end 
end
@save joinpath(home_dir,"Results", "DY_RA_params.jld2") params 

out = Matrix{Float64}(undef, length(gammas), length(kappas))
res = randn(3)
for i in eachindex(gammas) 
    for j in eachindex(kappas)
        Main.AMP_DY.eq_bin!(res, params[i,j], kappas[j], gammas[i], 1 / (1 + kappas[j])) 
        out[i,j] = maximum(abs.(res)) 
    end 
end 
@save joinpath(home_dir, "Results", "DY_RA_params_roots.jld2") out
=#
params = load(joinpath(home_dir, "Results", "DY_RA_params.jld2"))["params"]
out = load(joinpath(home_dir, "Results", "DY_RA_params_roots.jld2"))["out"]
cutoff = 1e-5 
sum(out .> cutoff) 
mus = map(v -> v[1], params)
mus[out .> cutoff] .= NaN 

RA_pars =  find_params_nlsolve(0.2, sqrt(0.9), 1 / (1 + 0.2);
    verbose = false, 
    x_init = missing, 
    method = :newton, 
).zero 

x,y,z = (sqrt(0.9),0.2,RA_pars[1])

CairoMakie.activate!()
s = Figure()

grange = 1:length(gammas) 
krange = 1:length(kappas) 
mina = 0.0
shift = 0.0

ztick = [0.2,0.4,0.6,0.8,1.0] .- mina .+ shift
zlabels = [L"$0.2$",L"$0.4$",L"$0.6$",L"$0.8$",L"$1$"]
xtick = [0.1,0.3,0.5,0.7,0.9]
xlabels = [L"$0.1$",L"$0.3$",L"$0.5$",L"$0.7$",L"$0.9$"]
ytick =[1,5,10,15,20]
ylabels = [L"$1$",L"$5$",L"$10$",L"$15$",L"$20$"]

ax = Axis3(
    s[1, 1],
    ylabel = L"\gamma",
    xlabel = L"\kappa",
    zlabel = L"\mu_{*}",
    xticklabelsize = 24,
    yticklabelsize = 24,
    zticklabelsize = 24,
    xlabelsize = 30, 
    ylabelsize = 30,
    zlabelsize = 30,
    yreversed = true, 
    xreversed = true, 
    zticks = (ztick, zlabels), 
    yticks = (ytick, ylabels), 
    xticks = (xtick, xlabels), 
    aspect = :equal, 
    elevation = 0.15 * pi, 
    #azimuth = 1.175 * pi, 
    zspinecolor_2 = :lightgray, 
    zspinecolor_3 = :white, 
    xspinecolor_2 = :lightgray, 
    xspinecolor_3 = :white, 
    yspinecolor_2 = :lightgray, 
    yspinecolor_3 = :white, 
    protrusions = (20,0,30,30)
)


Makie.wireframe!(ax, kappas[krange], gammas[grange], mus[grange, krange]' .- mina .+ shift,
    color = :black, 
    overdraw = true
)

x,y,z = (sqrt(0.9),0.2,RA_pars[1])

Makie.scatter!(ax, [y],[x],[z - mina], 
    marker= :diamond, 
    markerstrokewidth = 3, 
    color = :black)
Makie.save(joinpath(home_dir, "Figures", "DY_RA_bias.pdf"), s, resolution = (600, 600))


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
    yreversed = true,
    yticks = (ytick, ylabels), 
    xticks = (xtick, xlabels)
)
Makie.hidespines!(ax, :t, :r) 

krange = 1:length(kappas)
grange = 1:length(gammas) 

levels = vcat(exp.(range(log(minimum(mus)),log(maximum(mus)),15)),z)

Makie.contour!(ax, kappas[krange], gammas[grange], mus[grange, krange]';
    labels=true, 
    levels, 
    color = :black,
    labelsize = 20, 
    labelfont = :math
)
Makie.scatter!(ax, [y],[x], 
    marker= :diamond, 
    markerstrokewidth = 3, 
    color = :black
)
Makie.save(joinpath(home_dir, "Figures", "DY_RA_bias_contour_only.pdf"), s, resolution = (600, 600))

#=
gr()
p = Plots.wireframe(gammas,kappas,mus', 
    camera = (45,30),  
    ylabel =L"$\kappa$", 
    xlabel=L"$\gamma$",
    zlabel=L"$\mu_{*}$", 
    yflip = true, 
    xticks = ([1,5,10,15,20],[L"$1$",L"$5$",L"$10$",L"$15$",L"$20$"]), 
    yticks = ([0.1,0.3,0.5,0.7,0.9],[L"$0.1$",L"$0.3$",L"$0.5$",L"$0.7$",L"$0.9$"]), 
    zticks = ([0.2,0.4,0.6,0.8,1.0],[L"$0.2$",L"$0.4$",L"$0.6$",L"$0.8$",L"$1$"]),
    size = (400,400)
)
scatter!([x],[y],[z], markershape=:diamond, markerstrokewidth = 3, legend = :none, color = "black",)

Plots.savefig(p,joinpath(home_dir, "Figures", "DY_RA_bias.pdf"))
=#

vars = map(v -> v[3], params)

s = Figure()

grange = 1:length(gammas) 
krange = 1:length(kappas) 
mina = 0.0
shift = 0.0

ztick = [2,3,4,5] .- mina .+ shift
zlabels = [L"$2$",L"$3$",L"$4$",L"$5$"]
xtick = [0.1,0.3,0.5,0.7,0.9]
xlabels = [L"$0.1$",L"$0.3$",L"$0.5$",L"$0.7$",L"$0.9$"]
ytick =[1,5,10,15,20]
ylabels = [L"$1$",L"$5$",L"$10$",L"$15$",L"$20$"]

ax = Axis3(
    s[1, 1],
    ylabel = L"\gamma",
    xlabel = L"\kappa",
    zlabel = L"\sigma_{*}",
    xticklabelsize = 24,
    yticklabelsize = 24,
    zticklabelsize = 24,
    xlabelsize = 30, 
    ylabelsize = 30,
    zlabelsize = 30,
    yreversed = true, 
    xreversed = true, 
    zticks = (ztick, zlabels), 
    yticks = (ytick, ylabels), 
    xticks = (xtick, xlabels), 
    aspect = :equal, 
    elevation = 0.15 * pi, 
    #azimuth = 1.175 * pi, 
    zspinecolor_2 = :lightgray, 
    zspinecolor_3 = :white, 
    xspinecolor_2 = :lightgray, 
    xspinecolor_3 = :white, 
    yspinecolor_2 = :lightgray, 
    yspinecolor_3 = :white, 
    protrusions = (20,0,30,30)
)

Makie.wireframe!(ax, kappas[krange], gammas[grange], vars[grange, krange]' .- mina .+ shift,
    color = :black, 
    overdraw = true
)

x,y,z = (sqrt(0.9),0.2,RA_pars[3])

Makie.scatter!(ax, [y],[x],[z - mina], 
    marker= :diamond, 
    markerstrokewidth = 3, 
    color = :black)
Makie.save(joinpath(home_dir, "Figures", "DY_RA_var.pdf"), s, resolution = (600, 600))


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
    yreversed = true,
    yticks = (ytick, ylabels), 
    xticks = (xtick, xlabels)
)
Makie.hidespines!(ax, :t, :r) 

krange = 1:length(kappas)
grange = 1:length(gammas) 

levels = vcat(exp.(range(log(minimum(vars)),log(maximum(vars)),15)),z)

Makie.contour!(ax, kappas[krange], gammas[grange], vars[grange, krange]';
    labels=true, 
    levels, 
    color = :black,
    labelsize = 20, 
    labelfont = :math
)
Makie.scatter!(ax, [y],[x], 
    marker= :diamond, 
    markerstrokewidth = 3, 
    color = :black
)
Makie.save(joinpath(home_dir, "Figures", "DY_RA_var_contour_only.pdf"), s, resolution = (600, 600))


#=
p = Plots.wireframe(gammas,kappas,vars', 
    camera = (45,30),  
    ylabel =L"$\kappa$", 
    xlabel=L"$\gamma$",
    zlabel=L"$\sigma_{*}$", 
    yflip = true, 
    xticks = ([1,5,10,15,20],[L"$1$",L"$5$",L"$10$",L"$15$",L"$20$"]), 
    yticks = ([0.1,0.3,0.5,0.7,0.9],[L"$0.1$",L"$0.3$",L"$0.5$",L"$0.7$",L"$0.9$"]), 
    zticks = ([2,3,4,5],[L"$2$",L"$3$",L"$4$",L"$5$"])
)
x,y,z = (sqrt(0.9),0.2,RA_pars[3])

scatter!([x],[y],[z], markershape=:diamond, markerstrokewidth = 3, legend = :none, color = "black")

#Plots.savefig(p,joinpath(home_dir, "Figures", DY_RA_var.pdf"))
=#

mse = similar(vars) 

for i in eachindex(gammas) 
    for j in eachindex(kappas) 
        mse[i,j] = (mus[i,j]-1)^2 * gammas[i]^2 / kappas[j] + vars[i,j]^2 
    end 
end 

s = Figure()

grange = 1:length(gammas) 
krange = 1:length(kappas) 
mina = 0.0
shift = 0.0

ztick = [1000,2000,3000,4000] .- mina .+ shift
zlabels = [L"$1$",L"$2$",L"$3$",L"$4$"]
xtick = [0.1,0.3,0.5,0.7,0.9]
xlabels = [L"$0.1$",L"$0.3$",L"$0.5$",L"$0.7$",L"$0.9$"]
ytick =[1,5,10,15,20]
ylabels = [L"$1$",L"$5$",L"$10$",L"$15$",L"$20$"]

ax = Axis3(
    s[1, 1],
    ylabel = L"\gamma",
    xlabel = L"\kappa",
    zlabel = L"\sigma_{*}^2 + (1-\mu_{*})^2\gamma^2\kappa^{-1}",
    xticklabelsize = 24,
    yticklabelsize = 24,
    zticklabelsize = 24,
    xlabelsize = 30, 
    ylabelsize = 30,
    zlabelsize = 30,
    yreversed = true, 
    xreversed = true, 
    zticks = (ztick, zlabels), 
    yticks = (ytick, ylabels), 
    xticks = (xtick, xlabels), 
    aspect = :equal, 
    elevation = 0.15 * pi, 
    #azimuth = 1.175 * pi, 
    zspinecolor_2 = :lightgray, 
    zspinecolor_3 = :white, 
    xspinecolor_2 = :lightgray, 
    xspinecolor_3 = :white, 
    yspinecolor_2 = :lightgray, 
    yspinecolor_3 = :white, 
    protrusions = (35,0,30,30)
)

Makie.wireframe!(ax, kappas[krange], gammas[grange], mse[grange, krange]' .- mina .+ shift,
    color = :black, 
    overdraw = true
)

x,y,z = (sqrt(0.9),0.2,(RA_pars[1]-1)^2 * 0.9 / 0.2 + RA_pars[3]^2)

Makie.scatter!(ax, [y],[x],[z - mina], 
    marker= :diamond, 
    markerstrokewidth = 3, 
    color = :black
)
text!(s.scene, 
    Point3f(0.125, 0.77, 0), 
    text = L"\times 10^{3}", 
    space = :relative, 
    rotation = .1 * pi, 
    fontsize = 24
)
Makie.save(joinpath(home_dir, "Figures", "DY_RA_mse.pdf"), s, resolution = (600, 600))

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
    yreversed = true,
    yticks = (ytick, ylabels), 
    xticks = (xtick, xlabels)
)
Makie.hidespines!(ax, :t, :r) 

krange = 1:length(kappas)
grange = 1:length(gammas) 

levels = vcat(exp.(range(log(minimum(mse)),log(maximum(mse)),15)),z)

Makie.contour!(ax, kappas[krange], gammas[grange], mse[grange, krange]';
    labels=true, 
    levels, 
    color = :black,
    labelsize = 20, 
    labelfont = :math
)
Makie.scatter!(ax, [y],[x], 
    marker= :diamond, 
    markerstrokewidth = 3, 
    color = :black
)
Makie.save(joinpath(home_dir, "Figures", "DY_RA_mse_contour_only.pdf"), s, resolution = (600, 600))

#=
p = Plots.wireframe(gammas,kappas,mse', 
    camera = (45,30),  
    ylabel =L"$\kappa$", 
    xlabel=L"$\gamma$",
    zlabel=L"$\sigma_{*}^2 + (1-\mu_{*})^2\gamma^2 \kappa^{-1}$", 
    #xflip = true,
    yflip = true,  
    xticks = ([1,5,10,15,20],[L"$1$",L"$5$",L"$10$",L"$15$",L"$20$"]), 
    yticks = ([0.1,0.3,0.5,0.7,0.9],[L"$0.1$",L"$0.3$",L"$0.5$",L"$0.7$",L"$0.9$"]), 
    zticks = ([1000,2000,3000,4000],[L"$1000$",L"$2000$",L"$3000$",L"$4000$"]),
    size = (400,400)
)

x,y,z = (sqrt(0.9),0.2,(RA_pars[1]-1)^2 * 0.9 / 0.2 + RA_pars[3]^2 )
scatter!([x],[y],[z], markershape=:diamond, markerstrokewidth = 3, legend = :none, color = "black")
Plots.savefig(p,joinpath(home_dir, "Figures", "DY_RA_mse.pdf"))
=#
