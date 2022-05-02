
using Plots

include("utils.jl")

function plot_points(X, y; J_y=nothing, T=nothing)
    pyplot()
    
    plt = plot(size=(800, 600))
    for c in 1:size(y, 1)
        X_c = [x_i for x_i in eachrow(X) if get_cluster_assignment(x_i, y) == c]
        if size(X_c, 1) > 0
            X_c = reduce(hcat, X_c)'
            scatter!(plt, X_c[:, 1], X_c[:, 2], msw=0, label="Cluster $(c)")
        end
    end
    scatter!(plt, y[:, 1], y[:, 2], color=:black, markershape=:star5, label=nothing)
    xlims!(plt, (-0.1, 1.1))
    ylims!(plt, (-0.1, 1.1))
    
    title_str = ""
    if J_y !== nothing
        title_str *= "Final solution with value J(\$x_{min}\$) = $(round(J_y, digits=3)).\n"
    end
    if T !== nothing
        title_str *= "Temperature: $(T)"
    end
    title!(plt, title_str)

    plt
end

function plot_history(history)
    J_history, temp_drops, best_min_iter, _ = history
    J_curr_history, J_min_history = unzip(map(Tuple, J_history))
    
    pyplot()
    plt = plot(size=(800, 600))
    plot!(plt, 1:size(J_curr_history, 1), J_curr_history, label="J_curr")
    plot!(plt, 1:size(J_min_history, 1), J_min_history, label="J_min")
    vline!(temp_drops, color=:lightgray, linestyle=:dash,
        label="Temperature drops")
    
    ylims!(plt, (0, J_curr_history[20]))
    ylabel!(plt, "Cost function")
    xlabel!(plt, "Iterations")
    title!(plt, "Best solution \$J_{min} = $(round(J_min_history[end], digits=3))\$ 
        found at iteration $(best_min_iter).")
    plt
end