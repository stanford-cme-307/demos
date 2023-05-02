# Write a demo in Julia for an advanced optimization class that compares the performance of gradient descent, BFGS, L-BFGS, and Newton's method on the Rosenbrock function.
# Draw a figure with four subfigures showing the first ten iterates of each method, with dashed lines connecting subsequent iterates.
# Also print a table with the error ||x-x^\star|| for each method, and make a line plot of the error as a function of the iteration for each method.

using Plots
using Optim
using LinearAlgebra

# Define the Rosenbrock function
f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

# Initialize the starting point
x0 = [0.0, 0.0]
xstar = [1.0, 1.0]

# Define the optimization methods
iterations = 10
methods = [GradientDescent(), BFGS(), LBFGS(), Newton()]
method_names = ["Gradient Descent", "BFGS", "L-BFGS", "Newton"]
errors = Array{Any}(undef, length(methods))

# Initialize the table
println("Method\t\tError")

# Iterate over the methods
plots = []
for i in 1:length(methods)
    # Optimize the Rosenbrock function
    res = optimize(f, x0, methods[i], 
                   Optim.Options(iterations=iterations, store_trace=true, extended_trace = true))
    x_star = Optim.minimizer(res)
    println(method_names[i], "\t\t", string(error))
    
    # Plot the iterates
    xs = [x[1] for x in Optim.x_trace(res)]
    ys = [x[2] for x in Optim.x_trace(res)]
    errors[i] = [norm(xstar - x) for x in Optim.x_trace(res)]
    fig = plot(xs, ys, label=method_names[i], linestyle=:dash, marker=:circle)
    # add solution and init
    scatter!(xstar[1:1], xstar[2:2], xlabel="x1", ylabel="x2", 
                  marker=:star, color=:yellow, markersize=10, label="solution",
                  legend=:bottomright)
    scatter!(x0[1:1], x0[2:2], marker=:square, color=:red, markersize=5, label="start")
    # plot level sets of the function
    # contour!(f, xlims=(-.5, 1.5), ylims=(-0.5, 1.5), levels=20, color=:black, linewidth=0.5, label="")
    push!(plots, fig)
end

# Show the figure
display(plot(plots..., layout=(2,2)))
# Save as pdf 
savefig(fig, "qn_iterates.pdf")

# Plot the error as a function of the iteration
fig = plot(title="Error vs Iteration", xlabel="Iteration", ylabel="Error")
for i=1:length(methods)
    plot!(errors[i], label=method_names[i])
end
# Show the figure
display(fig)
# Save as pdf 
savefig(fig, "qn_convergence.pdf")