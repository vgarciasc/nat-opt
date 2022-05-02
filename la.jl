include("utils.jl")
include("plotter.jl")

function run_lloyd_algorithm(X, C)
    # Initialization
    centroids = reduce(hcat, [X[rand(1:size(X, 1)), :] for i in 1:C])'
    centroids = Matrix{Float64}(centroids)

    partition = get_cluster_assignments(X, centroids)
    history = [centroids]

    while size(history, 1) < 2 || history[end] != history[end - 1]
        # cluster condition
        partition = get_cluster_assignments(X, centroids)

        # centroid condition
        centroids = zeros(size(centroids))
        cluster_pop = zeros(size(centroids, 1))
        for i in 1:size(X, 1)
            centroids[partition[i], :] += X[i, :]
            cluster_pop[partition[i]] += 1
        end
        centroids ./= cluster_pop

        push!(history, centroids)
    end

    centroids, history
end

X = load_clustering_dataset("data/clustering/s1.txt")
y, history = run_lloyd_algorithm(X, 15)

anim = @animate for y_i in history
    plot_points(X, y_i)
end
gif(anim, "output/la_1.gif", fps = 5)