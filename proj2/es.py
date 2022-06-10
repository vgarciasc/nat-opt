import argparse
import json
import pdb
from rich import print
import numpy as np
import matplotlib.pyplot as plt
from evo_tree import EvoTreeNode

from evo_tree_aa import AAETNode
from env_configs import get_config
from sga import tournament_selection, initialize_population
from utils import fill_rewards, printv, console
import utils

def calc_reward(tree, episodes=10, norm_state=False):
    mean, _ = utils.evaluate_fitness(
        tree.config, tree, 
        episodes=episodes,
        should_normalize_state=norm_state)
    return mean

def run_evolutionary_strategy(config, mu, lamb, generations,
    initial_depth, alpha, initial_pop, fit_episodes=10, 
    mutation="A", tournament_size=0, 
    should_plot=False, should_render=False, render_every=None,
    norm_state=False, should_adapt_sigma=False,
    verbose=False):

    # Initialization
    population = initialize_population(config, initial_depth, lamb, initial_pop, norm_state)
    best = population[np.argmax([i.fitness for i in population])]
    evaluations = 0
    evaluations_to_success = 0
    history = []

    if should_plot:
        fig, (ax1, ax2) = plt.subplots(2)

    # Main loop
    for generation in range(generations):        
        avg_reward = np.mean([i.reward for i in population])
        curr_alpha = 0 if avg_reward == config["min_score"] else alpha

        # Creating pool of children
        population.sort(key=lambda x : x.fitness, reverse=True)
        child_population = []
        for parent in population[:mu]:
            for _ in range(lamb // mu):
                child = parent.copy()
                child.mutate(mutation=mutation, use_sigma=should_adapt_sigma)
                child.reward = calc_reward(child, 
                    episodes=fit_episodes, 
                    norm_state=norm_state)
                child.fitness = child.reward - curr_alpha * child.get_tree_size()
                child_population.append(child)
                evaluations += 1                
        
        # Selecting next generation
        if tournament_size == 0:
            population = child_population
        else:
            candidate_population = population + child_population
            population = []

            for _ in range(lamb):
                selected = tournament_selection(candidate_population, tournament_size)
                population.append(selected)

        # Housekeeping history
        rewards = [i.reward for i in population]
        fitnesses = [i.fitness for i in population]
        tree_sizes = [i.get_tree_size() for i in population]
        individual_max_fitness = population[np.argmax(fitnesses)]

        max_reward = individual_max_fitness.reward
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        min_size = np.min(tree_sizes)
        avg_size = np.mean(tree_sizes)
        std_size = np.std(tree_sizes)

        history.append(((max_reward, avg_reward, std_reward),
                        (min_size, avg_size, std_size)))
        
        if individual_max_fitness.fitness > best.fitness:
            fill_rewards(config, [individual_max_fitness], alpha,
                episodes=100, should_normalize_state=norm_state)
            if individual_max_fitness.fitness > best.fitness:
                best = individual_max_fitness.copy()
        
        # Checking for success
        if max_reward >= 490:
            if evaluations_to_success == 0:
                reward_precise = calc_reward(individual_max_fitness, episodes=50, norm_state=norm_state)
                printv(f"Checking for break: (reward: {max_reward}, precise reward: {reward_precise}, tree_size: {individual_max_fitness.get_tree_size()})", verbose)
                if reward_precise >= 490:
                    evaluations_to_success = evaluations
            if individual_max_fitness.get_tree_size() <= 5:
                reward_precise = calc_reward(individual_max_fitness, episodes=50, norm_state=norm_state)
                printv(f"Checking for break: (reward: {max_reward}, precise reward: {reward_precise}, tree_size: {individual_max_fitness.get_tree_size()})", verbose)
                if reward_precise >= 490:
                    break
        
        # Printing
        if verbose:
            console.rule(f"[bold red]Generation #{generation}")
            printv(f"[underline]Reward[/underline]: {{[green]Best: {'{:.3f}'.format(individual_max_fitness.reward)}[/green], [yellow]Avg: {'{:.3f}'.format(avg_reward)}[/yellow]}}", verbose)
            printv(f"[underline]Size  [/underline]: {{[green]Best: {individual_max_fitness.get_tree_size()}[/green], [yellow]Avg: {'{:.3f}'.format(avg_size)}[/yellow]}}", verbose)
            printv(f"{' ' * 3} - Best Sigma: {best.sigma})", verbose and should_adapt_sigma)
            printv(f"{' ' * 3} - Avg Sigma: {np.mean([i.sigma for i in population], axis=0)}", verbose and should_adapt_sigma)

        # Rendering
        if should_render and generation % render_every == 0:
            population.sort(key=lambda x : x.fitness, reverse=True)
            utils.evaluate_fitness(config, population[0], episodes=1, render=True)

        # Plotting
        if should_plot:
            rewards, sizes = zip(*history)
            max_rewards, avg_rewards, std_rewards = zip(*rewards)
            avg_rewards = np.array(avg_rewards)
            std_rewards = np.array(std_rewards)
            min_sizes, avg_sizes, std_sizes = zip(*sizes)
            avg_sizes = np.array(avg_sizes)
            std_sizes = np.array(std_sizes)

            ax1.clear()
            ax1.plot(range(len(history)), max_rewards, color="green", label="Best reward")
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
    parser.add_argument('--generations',help="Number of generations", required=True, type=int)
    parser.add_argument('--tournament_size',help="Size of tournament", required=True, type=int)
    parser.add_argument('--mutation_type',help="Type of mutation", required=True, default="A", type=str)
    parser.add_argument('--norm_state',help="Should normalize state?", required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--initial_depth',help="Randomly initialize the algorithm with trees of what depth?", required=True, type=int)
    parser.add_argument('--initial_pop',help="File with initial population", required=False, default='', type=str)
    parser.add_argument('--alpha',help="How to penalize tree size?", required=True, type=float)
    parser.add_argument('--should_adapt_sigma', help='Should adapt sigma?', required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--episodes', help='Number of episodes to run when evaluating model', required=False, default=10, type=int)
    parser.add_argument('--should_plot', help='Should plot performance?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_render', help='Should render at all?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--render_every', help='Should render every N iterations?', required=False, default=1, type=int)
    parser.add_argument('--should_print_individuals', help='Should print individuals?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
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
        tree, reward, size, evals2suc = run_evolutionary_strategy(
            config=config, 
            mu=args['mu'], lamb=args['lambda'],
            initial_pop=initial_pop,
            generations=args['generations'], 
            initial_depth=args['initial_depth'], 
            tournament_size=args['tournament_size'],
            mutation=args['mutation_type'],
            alpha=args['alpha'],
            fit_episodes=args['episodes'],
            norm_state=args['norm_state'],
            render_every=args['render_every'],
            should_adapt_sigma=args['should_adapt_sigma'],
            should_render=args['should_render'],
            should_plot=args['should_plot'],
            verbose=args['verbose'])
        
        reward, _ = utils.evaluate_fitness(
                tree.config, tree,
                episodes=100,
                should_normalize_state=args['norm_state'])
        history.append((tree, reward, size, evals2suc))
        print(f"Simulations run until now: {len(history)} / {args['simulations']}")
        print(history)
        # utils.evaluate_fitness(tree.config, tree, episodes=10, render=True, norm_state=True)

    trees, rewards, sizes, evals2suc = zip(*history)
    trees = np.array(trees)
    for tree in trees:
        tree.reward, _ = utils.evaluate_fitness(
            tree.config, tree,
            episodes=100,
            should_normalize_state=args['norm_state'])
        tree.fitness = tree.reward - args["alpha"] * tree.get_tree_size()
    
    successes = [1 if e > 0 else 0 for e in evals2suc]
    evals2suc = [e for e in evals2suc if e > 0]
    
    if args["verbose"]:
        console.rule(f"[bold red]Hall of Fame")
        print(f"[green][bold]5 best trees:[/bold][/green]")
        sorted(trees, key=lambda x: x.fitness, reverse=True)
        for i, tree in enumerate(trees[:5]):
            print(f"#{i}: (reward {tree.reward}, size {tree.get_tree_size()})")
            print(tree)

        console.rule(f"[bold red]RESULTS")
    print(f"[green][bold]Mean Best Reward[/bold][/green]: {np.mean(rewards)}")
    print(f"[green][bold]Mean Best Size[/bold][/green]: {np.mean(sizes)}")
    print(f"[green][bold]Average Evaluations to Success[/bold][/green]: {np.mean(evals2suc)}")
    print(f"[green][bold]Success Rate[/bold][/green]: {np.mean(successes)}")
