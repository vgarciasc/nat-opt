using Distributions

include("utils.jl")
include("plotter.jl")

function run_sa_clustering(X; D=2, C=2, N=10, K_max=10, 
    T0=1, ϵ=0.1, cooling=:log, no_empty_cells=false,
    perturbation=:gaussian,
    verbose=false)

    J(y) = distance_cost(X, y)
    
    y_0 = zeros(C, D)
    J_0 = J(y_0)
    
    y_curr, J_curr = y_0, J_0
    y_min, J_min = y_0, J_0

    J_history = []
    temp_drops = []
    iter = 1
    best_min_iter = -1
        
    for k in 1:K_max
        T = calc_temp(cooling, T0, k)

        printv("Temperature: $(T)", verbose)
        push!(temp_drops, iter)

        for n in 1:N
            iter += 1

            y_hat = perturb_solution(y_curr, ϵ, perturbation)
            J_hat = J(y_hat)

            if no_empty_cells
                curr_assignments = get_cluster_assignments(X, y_hat)
                cluster_pops = [sum([1 for y_i in curr_assignments if y_i == c]) for c in 1:C]
                largest_cluster = argmax(cluster_pops)
                empty_cell_found = false

                for c in 1:C
                    if cluster_pops[c] == 0
                        empty_cell_found = true
                        y_hat[c, :] = perturb_solution(y_hat[largest_cluster, :], ϵ, perturbation)
                    end
                end

                if empty_cell_found
                    J_hat = J(y_hat)
                end
            end

            printv("\tJ_hat: $(J_hat) \tJ_curr: $(J_curr) \tJ_min: $(J_min)", verbose)
            push!(J_history, [J_curr, J_min])

            if rand() <= exp((J_curr - J_hat) / T)
                y_curr = y_hat
                J_curr = J_hat

                if J_curr < J_min
                    y_min = y_curr
                    J_min = J_curr

                    best_min_iter = iter
                end
            end
        end
    end

    history = (J_history, temp_drops, best_min_iter)
    y_min, J_min, history
end

if abspath(PROGRAM_FILE) == @__FILE__
    X, y_sol = generate_dataset(C=10, N=100, ϵ=0.05)
    println("J_opt = $(distance_cost(X, y_sol))")
    plot_points(X, y_sol)

    # X = load_clustering_dataset("data/clustering/s1.txt")

    y, J_y, history = @time run_sa_clustering(
        X, C=10, T0=0.5, K_max=100, N=100,
        perturbation=:gaussian, cooling=:log,
        no_empty_cells=true, verbose=false)
    
    println("J_min = $(distance_cost(X, y))")
    plot_points(X, y; J_y)
    plot_points(X, y_sol; J_y)
    
    J_history, temp_drops, best_min_iter, _ = history
    J_curr_history, J_min_history = unzip(map(Tuple, J_history))
    plot_history(
        J_curr_history=J_curr_history,
        best_min_iter=best_min_iter,
        J_min_history=J_min_history,
        temp_drops=temp_drops)
end