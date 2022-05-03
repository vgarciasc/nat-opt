import Dates

include("plotter.jl")
include("sa.jl")
include("da.jl")

function run_and_save_results_sa(samples=1; C, T0, K_max, N, 
    perturbation, cooling, no_empty_cells, should_plot=false)

    println("Running SA (Simulated Annealing).")
    println("\tParameters:")
    println("\t\tC = $(C), T0 = $(T0), K_max = $(K_max)")
    println("\t\t perturbation = $(perturbation), cooling = $(cooling)")
    println("\t\t no_empty_cells = $(no_empty_cells)")
    
    elapsed_times = []
    J_mins = []
    iterations_ran = []
    best_min_iters = []

    println("\tRunning...")
    for _ in 1:samples
        elapsed_time = @elapsed begin
            y_min, J_min, history = run_sa_clustering(
                X, C=C, T0=T0, K_max=K_max, N=N,
                perturbation=perturbation, cooling=cooling, 
                no_empty_cells=no_empty_cells)
        end
        
        J_history, temp_drops, best_min_iter = history
        J_curr_history, J_min_history = unzip(map(Tuple, J_history))

        push!(iterations_ran, size(J_history, 1))
        push!(elapsed_times, elapsed_time)
        push!(J_mins, J_min)
        push!(best_min_iters, best_min_iter)
    end

    println("\tResults:")
    println("\t\tAverage iterations ran: $(round(mean(iterations_ran), digits=3)).")
    println("\t\tAverage time elapsed: $(round(mean(elapsed_times), digits=3)) seconds.")
    println("\t\tBest solution overall: $(round(minimum(J_mins), digits=3))")
    println("\t\tAverage best solution: $(round(mean(J_mins), digits=3))")
    println("\t\tAverage best min iters: $(round(mean(best_min_iters), digits=3))")

    if should_plot
        filecode = replace(string(Dates.now()), r":|[.]"=>"-")
        println("\tSaving files to '$(filecode)'.")

        plt = plot_points(X, y_min; J_y=J_min)
        png(plt, "output/$(filecode)_solution.png")
        plt = plot_history(
            J_curr_history=J_curr_history,
            best_min_iter=best_min_iter,
            J_min_history=J_min_history,
            temp_drops=temp_drops)
        png(plt, "output/$(filecode)_history.png")
    end
end

function run_and_save_results_da(; C, T0, K_max)
    println("Running DA (Deterministic Annealing).")
    println("\tParameters:")
    println("\t\tC = $(C), T0 = $(T0), K_max = $(K_max)")
    
    println("\tRunning...")
    elapsed_time = @elapsed begin
        y_min, history = run_deterministic_annealing(X, C=C, T0=T0, K_max=K_max)
    end
    
    y_history, T_history = unzip(map(Tuple, history))
    temp_drops = [i for (i, t) in enumerate(T_history[1:end-1]) if t != T_history[i+1]]
    J_history = [distance_cost(X, y_i) for y_i in y_history]
    J_min_history = [minimum(J_history[1:j]) for j in 1:size(J_history, 1)]
    J_min = J_min_history[end]
    best_min_iter = argmin(J_history)

    println("\tResults:")
    println("\t\tTime elapsed: $(round(elapsed_time, digits=3)) seconds.")
    println("\t\tIterations ran: $(size(J_history, 1)).")
    println("\t\tBest solution: $(round(J_min, digits=3))",
        " found after $(best_min_iter) iterations.")
    
    filecode = replace(string(Dates.now()), r":|[.]"=>"-")
    println("\tSaving files to '$(filecode)'.")

    plt = plot_points(X, y_min; J_y=J_min)
    png(plt, "output/$(filecode)_solution.png")
    plt = plot_history(
        J_curr_history=J_history,
        J_min_history=J_min_history,
        temp_drops=temp_drops,
        best_min_iter=best_min_iter)
    png(plt, "output/$(filecode)_history.png")
end

X = load_clustering_dataset("data/clustering/toy2.txt")

run_and_save_results_sa(100, C=2, T0=1, K_max=10, N=10, perturbation=:gaussian, cooling=:log, no_empty_cells=false)
run_and_save_results_sa(100, C=2, T0=1, K_max=10, N=100, perturbation=:gaussian, cooling=:log, no_empty_cells=false)
run_and_save_results_sa(100, C=2, T0=1, K_max=10, N=1000, perturbation=:gaussian, cooling=:log, no_empty_cells=false)
run_and_save_results_sa(100, C=2, T0=1, K_max=100, N=10, perturbation=:gaussian, cooling=:log, no_empty_cells=false)
run_and_save_results_sa(100, C=2, T0=1, K_max=100, N=100, perturbation=:gaussian, cooling=:log, no_empty_cells=false)
run_and_save_results_sa(100, C=2, T0=1, K_max=100, N=1000, perturbation=:gaussian, cooling=:log, no_empty_cells=false)

run_and_save_results_sa(100, C=2, T0=1, K_max=10, N=10, perturbation=:cauchy, cooling=:linear, no_empty_cells=false)
run_and_save_results_sa(100, C=2, T0=1, K_max=10, N=100, perturbation=:cauchy, cooling=:linear, no_empty_cells=false)
run_and_save_results_sa(100, C=2, T0=1, K_max=10, N=1000, perturbation=:cauchy, cooling=:linear, no_empty_cells=false)
run_and_save_results_sa(100, C=2, T0=1, K_max=100, N=10, perturbation=:cauchy, cooling=:linear, no_empty_cells=false)
run_and_save_results_sa(100, C=2, T0=1, K_max=100, N=100, perturbation=:cauchy, cooling=:linear, no_empty_cells=false)
run_and_save_results_sa(100, C=2, T0=1, K_max=100, N=1000, perturbation=:cauchy, cooling=:linear, no_empty_cells=false)

# run_and_save_results_da(C=15, T0=1, K_max=2)