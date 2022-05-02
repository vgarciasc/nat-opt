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
    
    for c in 1:C
        random_center = rand(Uniform(0.2, 0.8), (1, 2))
        
        for n in 1:N
            point = random_center + ϵ * rand(Normal(0.0, 1.0), (1, 2))
            push!(points, point)
        end
    end

    reduce(vcat, points)
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

function perturb_solution(sol, ϵ, perturbation)
    if perturbation == :gaussian
        sol + ϵ * rand(Normal(0, 1), size(sol))
    elseif perturbation == :cauchy
        sol + ϵ * rand(Cauchy(0, 1), size(sol))
    end
end