using Plots, LaTeXStrings, JLD2, LineSearches, NonlinearSolve, NLsolve

home_dir = "" 
include(joinpath(home_dir, "Scripts", "AMP_DY.jl"))
using .AMP_DY

kappas = .05:.025:.95 
gammas = .5:0.5:20
 

start = find_params_nonlinearsolve(0.05, 1.0, 1 / (1 + 0.05);
    verbose = true, 
    x_init = missing
)

#=
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

gr()

RA_pars =  find_params_nlsolve(0.2, sqrt(0.9), 1 / (1 + 0.2);
    verbose = false, 
    x_init = missing, 
    method = :newton, 
).zero 

x,y,z = (sqrt(0.9),0.2,RA_pars[1])

p = Plots.wireframe(gammas,kappas,mus', 
    camera = (45,30),  
    ylabel =L"$\kappa$", 
    xlabel=L"$\gamma$",
    #zlabel=L"$\mu_{*}$", 
    yflip = true, 
    xticks = ([1,5,10,15,20],[L"$1$",L"$5$",L"$10$",L"$15$",L"$20$"]), 
    yticks = ([0.1,0.3,0.5,0.7,0.9],[L"$0.1$",L"$0.3$",L"$0.5$",L"$0.7$",L"$0.9$"]), 
    zticks = ([0.2,0.4,0.6,0.8,1.0],[L"$0.2$",L"$0.4$",L"$0.6$",L"$0.8$",L"$1$"]),
    size = (400,400)
)
scatter!([x],[y],[z], markershape=:diamond, markerstrokewidth = 3, legend = :none, color = "black",)

Plots.savefig(p,joinpath(home_dir, "Figures", "DY_RA_bias.pdf"))


vars = map(v -> v[3], params)
p = Plots.wireframe(gammas,kappas,vars', 
    camera = (45,30),  
    ylabel =L"$\kappa$", 
    xlabel=L"$\gamma$",
    #zlabel=L"$\sigma_{*}$", 
    yflip = true, 
    xticks = ([1,5,10,15,20],[L"$1$",L"$5$",L"$10$",L"$15$",L"$20$"]), 
    yticks = ([0.1,0.3,0.5,0.7,0.9],[L"$0.1$",L"$0.3$",L"$0.5$",L"$0.7$",L"$0.9$"]), 
    zticks = ([2,3,4,5],[L"$2$",L"$3$",L"$4$",L"$5$"])
)
x,y,z = (sqrt(0.9),0.2,RA_pars[3])

scatter!([x],[y],[z], markershape=:diamond, markerstrokewidth = 3, legend = :none, color = "black")

#Plots.savefig(p,joinpath(home_dir, "Figures", DY_RA_var.pdf"))


mse = similar(vars) 

for i in eachindex(gammas) 
    for j in eachindex(kappas) 
        mse[i,j] = (mus[i,j]-1)^2 * gammas[i]^2 / kappas[j] + vars[i,j]^2 
    end 
end 

p = Plots.wireframe(gammas,kappas,mse', 
    camera = (45,30),  
    ylabel =L"$\kappa$", 
    xlabel=L"$\gamma$",
    #zlabel=L"$(\mu_{*}-1)^2  \frac{\gamma^2 }{\kappa}+ \sigma_{*}^2$", 
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

