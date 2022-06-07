import argparse
import json
from rich import print
import numpy as np
import matplotlib.pyplot as plt
from evo_tree import EvoTreeNode

from evo_tree_aa import AAETNode
from env_configs import get_config
from sga import calc_fitness, tournament_selection
from utils import printv, console
import utils

def run_evolutionary_programming(config, mu, lamb, generations,
    tournament_size, initial_sigma_max, initial_depth, alpha,
    initial_pop, repclass, fit_episodes=10, should_adapt_sigma=True,
    should_plot=False, should_render=False, render_every=None, 
    should_print_individuals=False, verbose=False):

    # Initialization
    population = []
    if initial_pop != []:
        for tree in initial_pop: #assuming initial pop of EvoTreeNodes
            population.append(AAETNode(config=config,
                sigma=np.random.uniform(0, initial_sigma_max, size=config["n_attributes"]),
                tree=tree))

    for _ in range(len(population), lamb):
        population.append(repclass.generate_random_tree(
            config, depth=initial_depth,
            sigma=np.random.uniform(0, initial_sigma_max, size=config["n_attributes"])))

    best = None
    evaluations = 0
    evaluations_to_success = 0
    history = []

    if should_plot:
        fig, (ax1, ax2) = plt.subplots(2)

    # Main loop
    for generation in range(generations):
        for individual in population:
            individual.reward = calc_fitness(individual, episodes=fit_episodes)
            evaluations += 1
        
        avg_reward = np.mean([i.reward for i in population])
        curr_alpha = 0 if avg_reward == config["min_score"] else alpha

        for individual in population:
            individual.fitness = individual.reward - curr_alpha * individual.get_tree_size()
            if best is None or individual.fitness > best.fitness:
                best = individual.copy()
            
            if should_print_individuals:
                console.log(f"Fitness: {individual.reward} - {curr_alpha} * {individual.get_tree_size()}")
                console.log(individual)
        
        if best.reward >= config["max_score"] and evaluations_to_success == 0:
            evaluations_to_success = evaluations

        population.sort(key=lambda x : x.fitness, reverse=True)
        for parent in population[:mu]:
            for _ in range(lamb//mu):
                child = parent.copy()
                child.mutate(should_adapt_sigma)
                population.append(child)

        new_population = []
        for _ in range(lamb):
            selected = tournament_selection(population, tournament_size)
            new_population.append(selected)

        population = new_population

        # Printing and plotting
        min_reward = best.reward
        avg_reward = np.mean([i.reward for i in population])
        std_reward = np.std([i.reward for i in population])
        min_size = best.get_tree_size()
        avg_size = np.mean([i.get_tree_size() for i in population])
        std_size = np.std([i.get_tree_size() for i in population])
        
        if verbose:
            console.rule(f"[bold red]Generation #{generation}")
        printv(f"[underline]Reward[/underline]: {{[green]Best: {'{:.3f}'.format(min_reward)}[/green], [yellow]Avg: {'{:.3f}'.format(avg_reward)}[/yellow]}}", verbose)
        printv(f"[underline]Size  [/underline]: {{[green]Best: {min_size}[/green], [yellow]Avg: {'{:.3f}'.format(avg_size)}[/yellow]}}", verbose)
        printv(f"{' ' * 3} - Best Sigma: {best.sigma})", verbose)
        printv(f"{' ' * 3} - Avg Sigma: {np.mean([i.sigma for i in population], axis=0)}", verbose)

        history.append(((min_reward, avg_reward, std_reward),
                        (min_size, avg_size, std_size)))
        rewards, sizes = zip(*history)
        min_rewards, avg_rewards, std_rewards = zip(*rewards)
        avg_rewards = np.array(avg_rewards)
        std_rewards = np.array(std_rewards)
        min_sizes, avg_sizes, std_sizes = zip(*sizes)
        avg_sizes = np.array(avg_sizes)
        std_sizes = np.array(std_sizes)

        if should_render and generation % render_every == 0:
            population.sort(key=lambda x : x.fitness, reverse=True)
            utils.evaluate_fitness(config, population[0], episodes=1, render=True)

        if should_plot:
            ax1.clear()
            ax1.plot(range(len(history)), min_rewards, color="green", label="Best reward")
            ax1.plot(range(len(history)), avg_rewards, color="blue", label="Average rewards")
            ax1.fill_between(range(len(history)), avg_rewards - std_rewards, avg_rewards + std_rewards, color="blue", alpha=0.2)
            ax2.clear()
            ax2.plot(range(len(history)), min_sizes, color="red", label="Best size")
            ax2.plot(range(len(history)), avg_sizes, color="orange", label="Average sizes")
            ax2.fill_between(range(len(history)), avg_sizes - std_sizes, avg_sizes + std_sizes, color="orange", alpha=0.2)
            
            ax2.set_xlabel("Generations")
            ax1.set_ylabel("Fitness")
            ax2.set_ylabel("Tree size")
            ax1.legend()
            ax2.legend()
            plt.pause(0.0001)
    
    printv(f"[yellow]Best individual w/ reward {best.reward}:", verbose)
    printv(best, verbose)

    if should_plot:
        plt.show()
    
    return best, best.reward, best.get_tree_size(), evaluations_to_success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evolutionary Programming')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-s','--simulations',help="How many simulations?", required=True, type=int)
    parser.add_argument('--mu',help="Value of mu", required=True, type=int)
    parser.add_argument('--lambda',help="Value of lambda", required=True, type=int)
    parser.add_argument('--tournament_size',help="Size of tournament", required=True, type=int)
    parser.add_argument('--generations',help="Number of generations", required=True, type=int)
    parser.add_argument('--initial_sigma_max',help="Initial maximum value of sigma", required=True, type=float)
    parser.add_argument('--initial_depth',help="Randomly initialize the algorithm with trees of what depth?", required=True, type=int)
    parser.add_argument('--alpha',help="How to penalize tree size?", required=True, type=float)
    parser.add_argument('--episodes', help='Number of episodes to run when evaluating model', required=False, default=10, type=int)
    parser.add_argument('--should_plot', help='Should plot performance?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_render', help='Should render at all?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--render_every', help='Should render every N iterations?', required=False, default=1, type=int)
    parser.add_argument('--should_adapt_sigma', help='Should adapt sigma?', required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_print_individuals', help='Should print individuals?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--initial_pop',help="File with initial population", required=False, default='', type=str)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())
    
    history = []
    config = get_config(args["task"])

    initial_pop = []
    if args['initial_pop'] != '':
        with open(args['initial_pop']) as f:
            json_obj = json.load(f)
        initial_pop = [EvoTreeNode.read_from_string(config, json_str) 
            for json_str in json_obj]

    for _ in range(args['simulations']):
        tree, reward, size, evals2suc = run_evolutionary_programming(
            config=config, 
            mu=args['mu'], lamb=args['lambda'],
            initial_pop=initial_pop,
            generations=args['generations'], 
            tournament_size=args['tournament_size'],
            initial_sigma_max=args['initial_sigma_max'],
            initial_depth=args['initial_depth'], 
            alpha=args['alpha'], repclass=AAETNode,
            fit_episodes=args['episodes'],
            should_adapt_sigma=args['should_adapt_sigma'],
            render_every=args['render_every'], should_render=args['should_render'],
            verbose=args['verbose'], should_plot=args['should_plot'])

        # tree, reward, size, evals2suc = run_evolutionary_programming(
        #     config=get_config("cartpole"), mu=15, lamb=30,
        #     generations=100, initial_sigma_max=1, initial_depth=3, 
        #     alpha=5, repclass=AAETNode,
        #     verbose=True, should_plot=False)

        # tree, reward, size, evals2suc = run_evolutionary_programming(
        #     config=get_config("mountain_car"),
        #     mu=15, lamb=30,
        #     generations=100, initial_sigma_max=1, initial_depth=2, 
        #     should_adapt_sigma=True,
        #     alpha=5, repclass=AAETNode,
        #     fit_episodes=50,
        #     verbose=True, should_plot=False)

        # tree, reward, size, evals2suc = run_evolutionary_programming(
        #     config=get_config("lunar_lander"),
        #     mu=30, lamb=60,
        #     generations=100, initial_sigma_max=1, initial_depth=2, 
        #     alpha=0, repclass=AAETNode, 
        #     fit_episodes=5, should_adapt_sigma=False,
        #     render_every=1, should_render=True,
        #     verbose=True, should_plot=False)
        
        history.append((tree, reward, size, evals2suc))
        print(f"Simulations run until now: {len(history)} / {args['simulations']}")
        print(history)
        # utils.evaluate_fitness(tree.config, tree, episodes=10, render=True)

    trees, rewards, sizes, evals2suc = zip(*history)
    trees = np.array(trees)
    successes = [1 if e > 0 else 0 for e in evals2suc]
    evals2suc = [e for e in evals2suc if e > 0]
    
    if args["verbose"]:
        console.rule(f"[bold red]Hall of Fame")
        print(f"[green][bold]5 best trees:[/bold][/green]")
        sorted(trees, key=lambda x: x.reward, reverse=True)
        for i, tree in enumerate(trees[:5]):
            print(f"#{i}: [reward {tree.reward}, size {tree.get_tree_size()}]")
            print(tree)

        console.rule(f"[bold red]RESULTS")
    print(f"[green][bold]Mean Best Reward[/bold]: {np.mean(rewards)}[/green]")
    print(f"[green][bold]Mean Best Size[/bold]: {np.mean(sizes)}[/green]")
    print(f"[green][bold]Average Evaluations to Success[/bold]: {np.mean(evals2suc)}[/green]")
    print(f"[green][bold]Success Rate[/bold]: {np.mean(successes)}[/green]")
