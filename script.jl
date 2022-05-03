import Dates

include("plotter.jl")
include("sa.jl")
include("da.jl")
include("datasets.jl")

function run_and_save_results_sa(df, n_samples=1; C, T0, K_max, N, ϵ,
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
    for sample in 1:n_samples
        elapsed_time = @elapsed begin
            y_min, J_min, history = run_sa_clustering(
                X, C=C, T0=T0, K_max=K_max, N=N, ϵ=ϵ,
                perturbation=perturbation, cooling=cooling, 
                no_empty_cells=no_empty_cells)
        end
        
        J_history, temp_drops, best_min_iter = history
        J_curr_history, J_min_history = unzip(map(Tuple, J_history))

        push!(iterations_ran, size(J_history, 1))
        push!(elapsed_times, elapsed_time)
        push!(J_mins, J_min)
        push!(best_min_iters, best_min_iter)

        if should_plot && sample == n_samples
            filecode = replace(string(Dates.now()), r":|[.]"=>"-")
            println("\tSaving files to '$(filecode)'.")
            base_title = "SA for dataset '$(df["name"])'.\n" *
                "T0: $(T0), K_max: $(K_max), cs: $(cooling), ptb: $(perturbation), NEC: $(no_empty_cells)\n"
    
            plt = plot_points(X, y_min; J_y=J_min, base_title=base_title)
            png(plt, "output/$(filecode)_solution.png")
            plt = plot_history(
                J_curr_history=J_curr_history,
                best_min_iter=best_min_iter,
                J_min_history=J_min_history,
                temp_drops=temp_drops,
                base_title=base_title)
            png(plt, "output/$(filecode)_history.png")
        end
    end

    println("\tResults:")
    println("\t\tAverage iterations ran: $(round(mean(iterations_ran), digits=3)).")
    println("\t\tAverage time elapsed: $(round(mean(elapsed_times), digits=3)) seconds.")
    println("\t\tBest solution overall: $(round(minimum(J_mins), digits=3))")
    println("\t\tAverage best solution: $(round(mean(J_mins), digits=3))")
    println("\t\tAverage best min iters: $(round(mean(best_min_iters), digits=3))")
end

function run_and_save_results_da(df, n_samples=1;
    C, T0, K_max, cooling, should_plot=false)

    println("Running DA (Deterministic Annealing).")
    println("\tParameters:")
    println("\t\tC = $(C), T0 = $(T0), K_max = $(K_max)")
    
    elapsed_times = []
    J_mins = []
    iterations_ran = []
    best_min_iters = []
    
    println("\tRunning...")
    for sample in 1:n_samples
        elapsed_time = @elapsed begin
            y_min, history = run_deterministic_annealing(X, C=C, T0=T0, K_max=K_max, cooling=cooling)
        end
    
        y_history, T_history = unzip(map(Tuple, history))
        temp_drops = [i for (i, t) in enumerate(T_history[1:end-1]) if t != T_history[i+1]]
        J_history = [distance_cost(X, y_i) for y_i in y_history]
        J_min_history = [minimum(J_history[1:j]) for j in 1:size(J_history, 1)]
        J_min = J_min_history[end]
        best_min_iter = argmin(J_history)

        push!(iterations_ran, size(J_history, 1))
        push!(elapsed_times, elapsed_time)
        push!(J_mins, J_min)
        push!(best_min_iters, best_min_iter)
    
        if should_plot && sample == n_samples
            filecode = replace(string(Dates.now()), r":|[.]"=>"-")
            println("\tSaving files to '$(filecode)'.")
            base_title = "DA for dataset '$(df["name"])'.\n"
    
            plt = plot_points(X, y_min; J_y=J_min, base_title=base_title)
            png(plt, "output/$(filecode)_solution.png")
            plt = plot_history(
                J_curr_history=J_history,
                J_min_history=J_min_history,
                best_min_iter=best_min_iter,
                temp_drops=temp_drops,
                base_title=base_title)
            png(plt, "output/$(filecode)_history.png")
        end
    end

    println("\tResults:")
    println("\t\tAverage iterations ran: $(round(mean(iterations_ran), digits=3)).")
    println("\t\tAverage time elapsed: $(round(mean(elapsed_times), digits=3)) seconds.")
    println("\t\tBest solution overall: $(round(minimum(J_mins), digits=3))")
    println("\t\tAverage best solution: $(round(mean(J_mins), digits=3))")
    println("\t\tAverage best min iters: $(round(mean(best_min_iters), digits=3))")
end

function run_and_save_results_gla(df, n_samples=1;
    C, T0, should_plot)

    println("Running GLA (Generalized Lloyd Algorithm).")
    println("\tParameters:")
    println("\t\tC = $(C), T0 = $(T0)")
    
    elapsed_times = []
    J_mins = []
    iterations_ran = []
    best_min_iters = []
    
    println("\tRunning...")
    for sample in 1:n_samples
        elapsed_time = @elapsed begin
            y_min, y_history = run_generalized_lloyd_algorithm(X, C, T0)
        end
    
        J_history = [distance_cost(X, y_i) for y_i in y_history]
        J_min_history = [minimum(J_history[1:j]) for j in 1:size(J_history, 1)]
        J_min = J_min_history[end]
        best_min_iter = argmin(J_history)

        push!(iterations_ran, size(J_history, 1))
        push!(elapsed_times, elapsed_time)
        push!(J_mins, J_min)
        push!(best_min_iters, best_min_iter)
    
        if should_plot && sample == n_samples
            filecode = replace(string(Dates.now()), r":|[.]"=>"-")
            println("\tSaving files to '$(filecode)'.")
            base_title = "GLA for dataset '$(df["name"])'.\n"
    
            plt = plot_points(X, y_min; J_y=J_min, base_title=base_title)
            png(plt, "output/$(filecode)_solution.png")
            plt = plot_history(
                J_curr_history=J_history,
                J_min_history=J_min_history,
                best_min_iter=best_min_iter,
                base_title=base_title)
            png(plt, "output/$(filecode)_history.png")
        end
    end

    println("\tResults:")
    println("\t\tAverage iterations ran: $(round(mean(iterations_ran), digits=3)).")
    println("\t\tAverage time elapsed: $(round(mean(elapsed_times), digits=3)) seconds.")
    println("\t\tBest solution overall: $(round(minimum(J_mins), digits=3))")
    println("\t\tAverage best solution: $(round(mean(J_mins), digits=3))")
    println("\t\tAverage best min iters: $(round(mean(best_min_iters), digits=3))")
end

df = datasets["r15"]

X = load_clustering_dataset(df["path"])

run_and_save_results_sa(df, 10, C=df["k"], T0=1, K_max=10, N=10, ϵ=0.1, perturbation=:gaussian, cooling=:log, no_empty_cells=false)
run_and_save_results_sa(df, 10, C=df["k"], T0=1, K_max=10, N=100, ϵ=0.1, perturbation=:gaussian, cooling=:log, no_empty_cells=false)
run_and_save_results_sa(df, 10, C=df["k"], T0=1, K_max=10, N=1000, ϵ=0.1, perturbation=:gaussian, cooling=:log, no_empty_cells=false)
run_and_save_results_sa(df, 10, C=df["k"], T0=1, K_max=100, N=10, ϵ=0.1, perturbation=:gaussian, cooling=:log, no_empty_cells=false)
run_and_save_results_sa(df, 10, C=df["k"], T0=1, K_max=100, N=100, ϵ=0.1, perturbation=:gaussian, cooling=:log, no_empty_cells=false)
run_and_save_results_sa(df, 10, C=df["k"], T0=1, K_max=100, N=1000, ϵ=0.1, perturbation=:gaussian, cooling=:log, no_empty_cells=false)

run_and_save_results_sa(df, 10, C=df["k"], T0=1, K_max=10, N=10, ϵ=0.1, perturbation=:cauchy, cooling=:linear, no_empty_cells=false)
run_and_save_results_sa(df, 10, C=df["k"], T0=1, K_max=10, N=100, ϵ=0.1, perturbation=:cauchy, cooling=:linear, no_empty_cells=false)
run_and_save_results_sa(df, 10, C=df["k"], T0=1, K_max=10, N=1000, ϵ=0.1, perturbation=:cauchy, cooling=:linear, no_empty_cells=false)
run_and_save_results_sa(df, 10, C=df["k"], T0=1, K_max=100, N=10, ϵ=0.1, perturbation=:cauchy, cooling=:linear, no_empty_cells=false)
run_and_save_results_sa(df, 10, C=df["k"], T0=1, K_max=100, N=100, ϵ=0.1, perturbation=:cauchy, cooling=:linear, no_empty_cells=false)
run_and_save_results_sa(df, 10, C=df["k"], T0=1, K_max=100, N=1000, ϵ=0.1, perturbation=:cauchy, cooling=:linear, no_empty_cells=false)

run_and_save_results_sa(df, 10, C=df["k"], T0=1, K_max=10, N=10, ϵ=0.1, perturbation=:gaussian, cooling=:log, no_empty_cells=true)
run_and_save_results_sa(df, 10, C=df["k"], T0=1, K_max=10, N=100, ϵ=0.1, perturbation=:gaussian, cooling=:log, no_empty_cells=true)
run_and_save_results_sa(df, 1, C=df["k"], T0=1, K_max=10, N=1000, ϵ=0.1, perturbation=:gaussian, cooling=:log, no_empty_cells=true, should_plot=true)
run_and_save_results_sa(df, 10, C=df["k"], T0=1, K_max=100, N=10, ϵ=0.1, perturbation=:gaussian, cooling=:log, no_empty_cells=true)
run_and_save_results_sa(df, 10, C=df["k"], T0=1, K_max=100, N=100, ϵ=0.1, perturbation=:gaussian, cooling=:log, no_empty_cells=true)
run_and_save_results_sa(df, 10, C=df["k"], T0=1, K_max=100, N=1000, ϵ=0.1, perturbation=:gaussian, cooling=:log, no_empty_cells=true)

run_and_save_results_da(df, 10, C=df["k"], T0=1, K_max=10, cooling=:linear, should_plot=true)
run_and_save_results_da(df, 10, C=df["k"], T0=1, K_max=100, cooling=:linear, should_plot=true)
run_and_save_results_da(df, 10, C=df["k"], T0=0.5, K_max=100, cooling=:linear, should_plot=true)
run_and_save_results_da(df, 10, C=df["k"], T0=0.1, K_max=100, cooling=:linear, should_plot=true)
run_and_save_results_da(df, 10, C=df["k"], T0=0.01, K_max=100, cooling=:log, should_plot=true)

run_and_save_results_da(df, 1, C=15, T0=0.5, K_max=200, cooling=:linear)
run_and_save_results_da(df, 1, C=df["k"], T0=0.1, K_max=100, cooling=:linear, should_plot=true)

run_and_save_results_gla(df, 100, C=df["k"], T0=1, should_plot=true)
run_and_save_results_gla(df, 100, C=df["k"], T0=0.5, should_plot=true)
run_and_save_results_gla(df, 100, C=df["k"], T0=0.1, should_plot=true)
run_and_save_results_gla(df, 100, C=df["k"], T0=0.05, should_plot=true)
run_and_save_results_gla(df, 100, C=df["k"], T0=0.01, should_plot=true)
run_and_save_results_gla(df, 100, C=df["k"], T0=0.005, should_plot=true)
run_and_save_results_gla(df, 100, C=df["k"], T0=0.001, should_plot=true)

run_and_save_results_gla(df, 1, C=df["k"], T0=1, should_plot=true)
