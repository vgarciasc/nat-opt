include("utils.jl")
include("plotter.jl")

function run_generalized_lloyd_algorithm(X, C, T; centroids=nothing)
    # Initialization
    if centroids === nothing
        centroids = reduce(hcat, [X[rand(1:size(X, 1)), :] for i in 1:C])'
        centroids = Matrix{Float64}(centroids)
    end

    history = [centroids]

    while size(history, 1) < 2 || distance(history[end], history[end - 1]) > 0.0001
        if size(history, 1) > 100
            break
        end

        # cluster condition
        partition = [exp(- distance(X[i, :], centroids[j, :]) / T) for i=1:size(X, 1), j=1:C]
        partition ./= sum(partition, dims=2)

        # centroid condition
        centroids = (partition' * X) ./ sum(partition, dims=1)'

        push!(history, centroids)
    end

    centroids, history
end

function run_deterministic_annealing(X; C, T0, K_max, cooling=:linear)
    # initialization
    y = nothing
    history = []

    for k in 1:K_max
        T = calc_temp(cooling, T0, k)
        y, y_history = run_generalized_lloyd_algorithm(X, C, T; centroids=y)

        append!(history, [(y_i, T) for y_i in y_history])
    end

    y, history
end

X = load_clustering_dataset("data/clustering/r15.txt")

# y, history = run_generalized_lloyd_algorithm(X, 2, 0.5)
# anim = @animate for y_i in history
#     plot_points(X, y_i)
# end
# gif(anim, "output/gla_1.gif", fps = 5)

if abspath(PROGRAM_FILE) == @__FILE__
    y, history = run_deterministic_annealing(X, C=15, T0=0.5, K_max=200, cooling=:linear)

    y_history, T_history = unzip(map(Tuple, history))
    temp_drops = [i for (i, t) in enumerate(T_history[1:end-1]) if t != T_history[i+1]]
    J_history = [distance_cost(X, y_i) for y_i in y_history]
    J_min_history = [minimum(J_history[1:j]) for j in 1:size(J_history, 1)]
    plot_history(
        J_curr_history=J_history,
        J_min_history=J_min_history,
        temp_drops=temp_drops,
        best_min_iter=argmin(J_history))

    anim = @animate for (y_i, t) in history
        plot_points(X, y_i, J_y=distance_cost(X, y_i), T=t)
    end
    gif(anim, "output/da_1.gif", fps = 5)
end