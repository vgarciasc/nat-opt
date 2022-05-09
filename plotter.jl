
using Plots

include("utils.jl")

function plot_points(X, y; base_title="", J_y=nothing, T=nothing, assignments=[])
    pyplot()

    function get_assignment(i, x_i, centroids, assignments)
        if assignments == []
            get_cluster_assignment(x_i, centroids)
        else
            assignments[i]
        end
    end

    plt = plot(size=(800, 600))
    for c in 1:size(y, 1)
        X_c = [x_i for (i, x_i) in enumerate(eachrow(X)) if get_assignment(i, x_i, y, assignments) == c]
        if size(X_c, 1) > 0
            X_c = reduce(hcat, X_c)'
            scatter!(plt, X_c[:, 1], X_c[:, 2], msw=0, label="Cluster $(c)")
        end
    end
    scatter!(plt, y[:, 1], y[:, 2], color=:black, markershape=:star5, label=nothing)
    xlims!(plt, (-0.1, 1.1))
    ylims!(plt, (-0.1, 1.1))
    
    title_str = base_title
    if J_y !== nothing
        title_str *= "Final solution with value J(\$x_{min}\$) = $(round(J_y, digits=3)).\n"
    end
    if T !== nothing
        title_str *= "Temperature: $(round(T, digits=3))"
    end
    title!(plt, title_str)

    plt
end

function plot_history(; base_title="", J_curr_history, best_min_iter=-1,
    J_min_history=[], temp_drops=[])
    
    pyplot()
    plt = plot(size=(800, 600))
    plot!(plt, 1:size(J_curr_history, 1), J_curr_history, label="J_curr")
    
    title_str = base_title
    if J_min_history != []
        title_str *= "Best solution \$J_{min} = $(round(J_min_history[end], digits=3))\$.\n"
        plot!(plt, 1:size(J_min_history, 1), J_min_history, label="J_min")
    end
    if temp_drops != []
        vline!(temp_drops, color=:lightgray, linestyle=:dash,
            label="Temperature drops")
    end
    if best_min_iter != -1
        title_str *= "Best solution found at iteration $(best_min_iter)."
    end
    
    # if size(J_curr_history, 1) > 100
        # ylims!(plt, (0, J_curr_history[20]))
    # end
    ylabel!(plt, "Cost function")
    xlabel!(plt, "Iterations")
    title!(plt, title_str)

    plt
end