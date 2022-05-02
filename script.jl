import Dates

include("plotter.jl")
include("sa.jl")

function run_and_save_results(; C, T0, K_max, N, perturbation, cooling, no_empty_cells)
    println("Running SA (Simulated Annealing).")
    println("\tParameters:")
    println("\t\tC = $(C), T0 = $(T0), K_max = $(K_max)")
    println("\t\t perturbation = $(perturbation), cooling = $(cooling)")
    println("\t\t no_empty_cells = $(no_empty_cells)")
    
    println("\tRunning...")
    history = run_sa_clustering(
        X, C=C, T0=T0, K_max=K_max, N=N,
        perturbation=perturbation, cooling=cooling, 
        no_empty_cells=no_empty_cells)
    
    y_min, J_min, series_history = history
    (J_history, _, best_min_iter, elapsed_time) = series_history
    
    println("\tResults:")
    println("\t\tTime elapsed: $(round(elapsed_time, digits=3)) seconds.")
    println("\t\tIterations ran: $(size(J_history, 1)).")
    println("\t\tBest solution: $(round(J_min, digits=3))",
        " found after $(best_min_iter) iterations.")
    
    filecode = replace(string(Dates.now()), r":|[.]"=>"-")
    println("\tSaving files to '$(filecode)'.")

    plt = plot_points(X, y_min; J_y=J_min)
    png(plt, "output/$(filecode)_solution.png")
    plt = plot_history(series_history)
    png(plt, "output/$(filecode)_history.png")
end

X = load_clustering_dataset("data/clustering/s1.txt")

run_and_save_results(C=15, T0=1, K_max=10, N=10, perturbation=:gaussian, cooling=:log, no_empty_cells=false)