using Random, LinearAlgebra
using Plots 
using Convex, JuMP, GeNIOS, Clarabel
prefix = "lectures/operators/"

# time to solve a linear system 
# n=1000: ~.015s
# n=10000: ~2.5s
n = 10000
A = randn(n,n); A = A*A'; b = randn(n)
@elapsed A\b

using Plots

ns = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
times = Float64[]

for n in ns
    A = randn(n, n); A = A * A'
    b = randn(n)
    t = @elapsed A \ b
    push!(times, t)
end

plot(ns, times, xscale=:log10, yscale=:log10, marker=:o, xlabel="n", ylabel="Time (s)", label="Solve time")
plot!(ns, times[1] * (ns ./ ns[1]).^3, linestyle=:dash, label="Cubic scaling")
title!("Linear system solve scaling")
savefig(prefix*"lin-sys-solve.pdf")

####### generate data ########
function generate_lasso(m=200, n=1000, k = 20)
    A = randn(m,n)
    A .-= sum(A, dims=1) ./ m
    normalize!.(eachcol(A))
    x_true = zeros(n); x_true[1:k] = randn(k)
    b = A * x_true + 0.05 * randn(m)
    # common heuristic: lambda proportional to ||A^T b||_inf
    λ = 0.1 * maximum(abs.(A'*b))
    return A, b, λ
end 
m, n = 200, 1000
A, b, λ = generate_lasso(m, n) 

########## solve with Convex ###########
x = Variable(n)
objective = sumsquares(A*x - b) + λ*norm(x, 1)
problem = minimize(objective)
@elapsed Convex.solve!(problem, Clarabel.Optimizer)

########## solve lasso problem with JuMP ##########
model = Model(Clarabel.Optimizer)
@variable(model, x[1:n])
@variable(model, t[1:n] >= 0)

@objective(model, Min, sum((A*x - b).^2) + λ * sum(t))
@constraint(model, t .>= x)
@constraint(model, t .>= -x)

@elapsed optimize!(model)

######## solve lasso problem with proximal gradient (FISTA) #######
function soft_thresholded(y, thresh)
    sign.(y) .* max.(abs.(y) .- thresh, 0.0)
end

function fista_lasso(A, b, λ; max_iters=1000, stepsize=nothing)
    m, n = size(A)
    x = zeros(n)
    if stepsize is nothing 
        v = randn(n) 
        # compute Lipschitz constant of the gradient
        for i=1:20 
            v = A'*(A*v); v = v / norm(v)
            # L = 2 * (opnorm(A))^2           
            L = norm(A'*(A*v))
        t = 1.0 / L
    end
    for k in 1:max_iters
            grad = 2 .* (A' * (A*x - b))
            x_new = soft_thresholded(x .- t .* grad, t * λ)
    end
    obj = sum((A*x - b).^2) + λ*sum(abs.(x))
    return x, obj
end

m, n = 2000, 10000
A, b, λ = generate_lasso(m, n) 


t_pg = @elapsed x_pg, obj_pg, iters_pg = fista_lasso(A, b, λ; max_iters=10000, tol=1e-8)
println("Proximal-gradient (FISTA) finished in $(t_pg) s, iters=$(iters_pg), obj=$(obj_pg)")

######## solve lasso problem with preconditioned operator splitting method ##########
solver = GeNIOS.LassoSolver(λ, A, b)
options = GeNIOS.SolverOptions(use_dual_gap=true, dual_gap_tol=1e-6, print_iter=500, max_iters=4000)
@elapsed GeNIOS.solve!(solver; options=options)