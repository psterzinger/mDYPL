#Helpers for AMP_DY, AMP_Ridge, AMP_Lasso
using Statistics, DataFrames, Optim, Random 

### Proximal Operator 
function prox_bζ(x,b)::Float64
    f(z) = z + b / (1.0 + exp(-z)) - x
    (l,u) = (-10,10)
    fu = f(u) 
    fl = f(l)
    while fu * fl > 0 
        u += 10 
        l -= 10 
        fu = f(u) 
        fl = f(l) 
    end 
    bisection(f,l,u)[1]
end

function bisection(f, a_, b_, atol = 2eps(promote_type(typeof(b_),Float64)(b_)); increasing = sign(f(b_))) ## taken from julia discourse
    a_, b_ = minmax(a_, b_)
    c = middle(a_,b_)
    z = f(c) * increasing
    if z > 0 #
        b = c
        a = typeof(b)(a_)
    else
        a = c
        b = typeof(a)(b_)
    end
    while abs(a - b) > atol
        c = middle(a,b)
        if f(c) * increasing > 0 #
            b = c
        else
            a = c
        end
    end
    a, b
end

### logistic link 
function ζ(z) log(1.0 + exp(z)) end 

function ∂ζ(z) 1.0 / (1.0 + exp(-z)) end 

function ∂²ζ(z) 
    mu = 1.0 / (1.0 + exp(-z)) 
    mu * (1.0 - mu)
end 

function ∂³ζ(z) 
    mu = 1.0 / (1.0 + exp(-z)) 
    mu * (1.0 - mu) * (1.0 - 2.0 * mu)
end 

### misc 
function dnorm(x)
    exp(-x^2/2) / sqrt(2 * pi)
end 

function rowfill!(J,vec) 
    count = 1
    @inbounds @simd for i in axes(J,1)
        for j in axes(J,2)
            J[i,j] = vec[count] 
            count += 1 
        end 
    end 
end 

function fullfill!(r,J,vec) 
    p = size(J,2)
    r .= vec[1:p] 
    rowfill!(J,vec[(p+1):end]) 
end 

function bivar_coord_transform!(x)
    u = x[1] 
    u = u * (1.0 - u) 

    v = x[2] 
    v = v * (1.0 - v) 

    x[1] = (2.0 * x[1] - 1.0) / u 
    x[2] = (2.0 * x[2] - 1.0) / v 
    
    (1.0 - 2.0 * u) * (1.0 - 2.0 * v) / (u * v)^2
end 

function h_mle(;beta0 = 0, 
    gamma_grid = 0:(20/100):20, 
    nsimu = 1e7,
    XZU = nothing
    ) 

    if isnothing(XZU) 
        X = randn(nsimu) 
        U = rand(nsimu) 
        Z = randn(nsimu) 
    else 
        X = XZU.X 
        U = XZU.U 
        Z = XZU.Z 
    end 

    function phase_transition(gamma) 
        gamma0 = sqrt(gamma^2 - beta0^2) 
        Y = -1 .+ 2 .* ( 1 ./ (1 .+ exp.(-beta0 .- gamma0 .* X)) .> U)
        V = X .* Y 

        obj(ts) = mean(max.(ts[1] .* Y .+ ts[2] .* V .- Z, zero(Y)).^2)

        initial_x = [0.0,0.0]
        od = OnceDifferentiable(obj, initial_x; autodiff = :forward)
        Optim.minimum(Optim.optimize(od,initial_x,BFGS()))
    end 

    phase = SharedVector{Float64}(length(gamma_grid)) 

    for i in eachindex(gamma_grid)
        phase[i] = phase_transition(gamma_grid[i]) 
    end 

    return hcat(phase,gamma_grid)
end 