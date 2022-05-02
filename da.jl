include("utils.jl")
include("plotter.jl")

function run_generalized_lloyd_algorithm(X, C, T; centroids=nothing)
    # Initialization
    if centroids === nothing
        centroids = reduce(hcat, [X[rand(1:size(X, 1)), :] for i in 1:C])'
        centroids = Matrix{Float64}(centroids)
    end

    history = [centroids]

    while size(history, 1) < 2 || history[end] != history[end - 1]
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
    J(y) = distance_cost(X, y)
    
    # initialization
    y, J_y = nothing, nothing
    overall_history = []
    
    for k in 1:K_max
        T = calc_temp(cooling, T0, k)
        y, history = run_generalized_lloyd_algorithm(X, C, T; centroids=y)
        J_y = J(y)
        
        push!(overall_history, (history, J_y, T))
    end

    full_history = [[(y_i, J_y_t, t) for y_i in h] for (h, J_y_t, t) in overall_history]
    full_history = collect(Iterators.flatten(full_history))

    y, J_y, full_history
end

X = load_clustering_dataset("data/clustering/toy1.txt")

y, history = run_generalized_lloyd_algorithm(X, 2, 0.5)
anim = @animate for y_i in history
    plot_points(X, y_i)
end
gif(anim, "output/gla_1.gif", fps = 5)

y, _, history = run_deterministic_annealing(X, C=2, T0=1, K_max=5, cooling=:linear)
anim = @animate for (y_i, J_y_t, t) in history
    plot_points(X, y_i, J_y=J_y_t, T=t)
end
gif(anim, "output/da_1.gif", fps = 5)