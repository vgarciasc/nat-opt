using Distributions

#https://stackoverflow.com/questions/36367482/unzip-an-array-of-tuples-in-julia
function unzip(a) 
    map(x->getfield.(a, x), fieldnames(eltype(a)))
end

function printv(str, verbose=false)
    if verbose
        println(str)
    end
end

function generate_dataset(; C=2, N=10, ϵ=0.1)
    points = []
    centers = []
    
    for c in 1:C
        # random_center = rand(Uniform(0.2, 0.8), (1, 2))
        random_center = Vector{Float64}([(c % 5) / 5, 0.2 + 4 * floor(c / 5) / 5])'
        push!(centers, random_center)
        
        for n in 1:N
            point = random_center + ϵ * rand(Normal(0.0, 1.0), (1, 2))
            push!(points, point)
        end
    end

    points = reduce(vcat, points)
    centers = reduce(vcat, centers)
    points, centers
end

function load_clustering_dataset(filename)
    output = []
    open(filename) do file
        for line in eachline(file)
            vec = map(x->parse(Float64, x), split(line, " "))
            push!(output, vec)
        end
    end

    # remove header
    output = output[2:end]
    
    # normalize
    X = reduce(hcat, output)'
    X = [(x .- minimum(x)) for x in eachcol(X)]
    X = reduce(hcat, X)
    X = [(x ./ maximum(x)) for x in eachcol(X)]

    reduce(hcat, X)
end

function get_cluster_assignment(x_i, y)
    argmin([distance(x_i, y_c) for y_c in eachrow(y)])
end

function get_cluster_assignments(X, y)
    [get_cluster_assignment(x_i, y) for x_i in eachrow(X)]
end

function get_centroid(X, assignments, c)
    centroid = [x_i for (i, x_i) in enumerate(eachrow(X)) if assignments[i] == c]
    centroid = reduce(hcat, centroid)'
    mean(centroid, dims=1)
end

function get_centroids(X, C, assignments)
    centroids = [get_centroid(X, assignments, c) for c in 1:C]
    reduce(vcat, centroids)
end

function distance(x, y)
    sum((x - y).^2)
end

function distance_cost(X, sol)
    total = 0
    for x in eachrow(X)
        total += minimum([distance(x, c) for c in eachrow(sol)])
    end
    total
end

function calc_temp(cooling, T0, k)
    if cooling == :log
        T0 / log2(1 + k)
    elseif cooling == :linear
        T0 / (1 + k)
    end
end

function perturb_solution(sol, ϵ, perturbation; C=2)
    if perturbation == :gaussian
        sol + ϵ * rand(Normal(0, 1), size(sol))
    elseif perturbation == :cauchy
        sol + ϵ * rand(Cauchy(0, 1), size(sol))
    elseif perturbation == :change_partition
        sol_hat = copy(sol)
        for i in 1:ϵ
            rand_idx = rand(1:size(sol)[1])
            
            while size([c for c in sol if c == sol[rand_idx]])[1] == 1
                rand_idx = rand(1:size(sol)[1])
            end

            sol_hat[rand_idx] = rand(1:C)
        end
        sol_hat
    end
end