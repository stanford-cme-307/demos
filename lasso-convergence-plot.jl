using Random, LinearAlgebra
using Plots
using Convex, JuMP, GeNIOS, Clarabel
prefix = "lectures/operators/"

function generate_lasso(m=200, n=1000, k = 20)
    k = min(n,k)
    A = randn(m,n)
    A .-= sum(A, dims=1) ./ m
    normalize!.(eachcol(A))
    x_true = zeros(n); x_true[1:k] = randn(k)
    b = A * x_true + 0.05 * randn(m)
    # common heuristic: lambda proportional to ||A^T b||_inf
    λ = 0.1 * maximum(abs.(A'*b))
    return A, b, λ
end

function convex_lasso(A,b,λ)
    n = size(A,2)
    x = Variable(n)
    objective = sumsquares(A*x - b) + λ*norm(x, 1)
    problem = minimize(objective)
    Convex.solve!(problem, Clarabel.Optimizer)
    return problem
end

function jump_lasso(A,b,λ)
    n = size(A,2)
    model = Model(Clarabel.Optimizer)
    @variable(model, x[1:n])
    @variable(model, t[1:n] >= 0)

    @objective(model, Min, sum((A*x - b).^2) + λ * sum(t))
    @constraint(model, t .>= x)
    @constraint(model, t .>= -x)

    optimize!(model)
    return model
end

######## solve lasso problem with proximal gradient (FISTA) #######
function soft_thresholded(y, thresh)
    sign.(y) .* max.(abs.(y) .- thresh, 0.0)
end

function fista_lasso(A, b, λ; max_iters=100000, tol=1e-8, accelerated=false)
    m, n = size(A)
    L = 2 * (opnorm(A))^2           # Lipschitz constant of the gradient
    t = 1.0 / L
    x = zeros(n)
    y = copy(x)
    t_k = 1.0
    obj_prev = sum((A*x - b).^2) + λ*sum(abs.(x))

    for k in 1:max_iters
        if accelerated
            grad = 2 .* (A' * (A*y - b))
            x_new = soft_thresholded(y .- t .* grad, t * λ)
            t_kp1 = (1 + sqrt(1 + 4*t_k^2)) / 2
            y = x_new .+ ((t_k - 1) / t_kp1) .* (x_new .- x)
            x .= x_new
            t_k = t_kp1
        else
            grad = 2 .* (A' * (A*x - b))
            x_new = soft_thresholded(x .- t .* grad, t * λ)
            x .= x_new
        end

        obj = sum((A*x - b).^2) + λ*sum(abs.(x))
        if abs(obj_prev - obj) <= tol * max(1.0, obj_prev)
            return x, obj, k
        end
        obj_prev = obj
    end
    return x, obj_prev, max_iters
end

function genios_lasso(λ, A, b, tol=1e-4)
    solver = GeNIOS.LassoSolver(λ, A, b)
    options = GeNIOS.SolverOptions(use_dual_gap=true, dual_gap_tol=tol, print_iter=500, max_iters=4000)
    return GeNIOS.solve!(solver; options=options)
end

### plot runtime for each method as a function of problem size for n increasing from 100 to 10000

ns = [10, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
ms = Integer.(0.2 .* ns)
k = 20

times_pg = Float64[]
times_convex = Float64[]
times_jump = Float64[]
times_genios = Float64[]

for ii in 1:length(ns)
    m = ms[ii]; n = ns[ii]
    A, b, λ = generate_lasso(m, n, k)

    t = @elapsed fista_lasso(A, b, λ; max_iters=5000, tol=1e-8)
    push!(times_pg, t)
    println("n=$n, PG: $(t)s")

    t = @elapsed convex_lasso(A, b, λ)
    push!(times_convex, t)
    println("n=$n, Convex: $(t)s")

    t = @elapsed jump_lasso(A, b, λ)
    push!(times_jump, t)
    println("n=$n, JuMP: $(t)s")

    t = @elapsed genios_lasso(λ, A, b)
    push!(times_genios, t)
    println("n=$n, GeNIOS: $(t)s")
end

plot(ns[2:end], times_pg[2:end], xscale=:log10, yscale=:log10, marker=:o, label="FISTA", xlabel="n", ylabel="Time (s)")
plot!(ns[2:end], times_convex[2:end], marker=:s, label="Convex.jl")
plot!(ns[2:end], times_jump[2:end], marker=:d, label="JuMP")
plot!(ns[2:end], times_genios[2:end], marker=:x, label="GeNIOS")
title!("LASSO solver runtime comparison")
savefig(prefix*"lasso-convergence.pdf")

### plot runtime for GeNIOS as a function of tol for fixed n

n = 10000
m = 2000
k = 20
A, b, λ = generate_lasso(m, n, k)
tols = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

times_genios_tol = Float64[]

for tol in tols
    t = @elapsed genios_lasso(λ, A, b, tol)
    push!(times_genios_tol, t)
    println("n=$n, GeNIOS: $(t)s")
end

plot(times_genios_tol, tols, xscale=:log10, yscale=:log10, marker=:o, label="GeNIOS", ylabel="tol", xlabel="Time (s)")
title!("Operator splitting gives approximate solutions fast")
savefig(prefix*"lasso-tol-convergence.pdf")
