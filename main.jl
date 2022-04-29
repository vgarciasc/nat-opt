using Plots
using Distributions
using Debugger

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

function plot_points(X)
    gr()
    plt = scatter(X[:, 1], X[:, 2], markerstrokewidth=0)
    xlims!(plt, (0, 1))
    ylims!(plt, (0, 1))
    plt
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

function run_sa_clustering(X; D=2, C=2, T0=1, N=10, K_max=10, ϵ=0.1)
    J(y) = distance_cost(X, y)
    
    y_0 = zeros(C, D)
    J_0 = J(y_0)

    y_curr, J_curr = y_0, J_0
    y_min, J_min = y_0, J_0

    for k in 1:K_max
        T = T0 / log2(1 + k)
        println("Temperature: $(T)")

        for n in 1:N
            @bp
            y_hat = y_curr + ϵ * rand(Normal(0, 1), (C, D))
            J_hat = J(y_hat)

            println("\tJ_hat: $(J_hat) \tJ_curr: $(J_curr) \tJ_min: $(J_min)")

            if rand() <= exp((J_curr - J_hat) / T)
                y_curr = y_hat
                J_curr = J_hat

                if J_curr < J_min
                    y_min = y_curr
                    J_min = J_curr
                end
            end
        end
    end
end

X = generate_dataset(C=3, N=100, ϵ=0.05)
plot_points(X)
run_sa_clustering(X)