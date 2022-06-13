import argparse
import copy
import json
import math
import pdb
from rich import print
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import Counter

from evo_tree import EvoTreeNode
from evo_tree_aa import AAETNode
from env_configs import get_config
from tree import TreeNode
from utils import evaluate_fitness, fill_rewards, printv, console
import utils

def initialize_population(config, initial_depth, popsize, initial_pop, norm_state):
    population = []
    if initial_pop != []:
        for tree in initial_pop: #assuming initial pop of EvoTreeNodes
            if norm_state:
                tree.normalize_thresholds()
            individual = AAETNode(config=config,
                sigma=np.random.uniform(0, 1, size=config["n_attributes"]),
                tree=tree)
            population.append(individual)

    for _ in range(len(population), popsize):
        population.append(AAETNode.generate_random_tree(
            config, depth=initial_depth,
            sigma=np.random.uniform(0, 1, size=config["n_attributes"])))
    
    for individual in population:
        individual.reward, _ = evaluate_fitness(config, individual, episodes=100, should_normalize_state=norm_state)
        individual.fitness = individual.reward
    
    return population

def calc_reward(tree, episodes=10, norm_state=False):
    mean, _ = utils.evaluate_fitness(
        tree.config, tree,
        episodes=episodes,
        should_normalize_state=norm_state)
    return mean

def tournament_selection(population, q):
    candidates = np.random.choice(population, size=q, replace=False)
    return max(candidates, key=lambda x : x.fitness)

def run_genetic_algorithm(config, popsize, initial_pop, p_crossover, p_mutation, 
    generations, initial_sigma_max, initial_depth, alpha, tournament_size,
    mutation, norm_state=False, fit_episodes=10, elitism=0, should_adapt_sigma=True,
    should_plot=False, should_render=False, render_every=None, verbose=False):

    # Initialization
    population = initialize_population(config, initial_depth, popsize, initial_pop, norm_state)
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

        # Creating next generation
        new_population = []

        # - Creating new children
        for _ in range(popsize//2):
            parent_a = tournament_selection(population, tournament_size)
            parent_b = tournament_selection(population, tournament_size)

            if np.random.uniform(0, 1) < p_crossover:
                child_a, child_b = AAETNode.crossover(parent_a, parent_b)
            else:
                child_a, child_b = parent_a.copy(), parent_b.copy()

            if np.random.uniform(0, 1) < p_mutation:
                child_a.mutate(mutation=mutation, use_sigma=should_adapt_sigma)
            if np.random.uniform(0, 1) < p_mutation:
                child_b.mutate(mutation=mutation, use_sigma=should_adapt_sigma)
        
            new_population += [child_a, child_b]

        # Evaluating population
        fill_rewards(config, new_population, curr_alpha, episodes=fit_episodes, should_normalize_state=norm_state)

        # - Allowing elitism
        if elitism > 0:
            population.sort(key=lambda x : x.fitness)
            elite = population[:int(popsize * elitism)]
            new_population += elite
        
        population = new_population

        # Housekeeping history
        rewards = [i.reward for i in population]
        fitnesses = [i.fitness for i in population]
        tree_sizes = [i.get_tree_size() for i in population]
        individual_max_fitness = population[np.argmax(fitnesses)]
        
        if individual_max_fitness.fitness > best.fitness:
            fill_rewards(config, [individual_max_fitness], alpha,
                episodes=100, should_normalize_state=norm_state)
            if individual_max_fitness.fitness > best.fitness:
                best = individual_max_fitness.copy()

        max_reward = individual_max_fitness.reward
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        min_size = np.min(tree_sizes)
        avg_size = np.mean(tree_sizes)
        std_size = np.std(tree_sizes)

        history.append(((max_reward, avg_reward, std_reward),
                        (min_size, avg_size, std_size)))
        
        # Printing and plotting
        if verbose:
            console.rule(f"[bold red]Generation #{generation}")
            printv(f"[underline]Reward[/underline]: {{[green]Best: {'{:.3f}'.format(max_reward)}[/green], [yellow]Avg: {'{:.3f}'.format(avg_reward)}[/yellow]}}", verbose)
            printv(f"[underline]Size  [/underline]: {{[green]Best: {individual_max_fitness.get_tree_size()}[/green], [yellow]Avg: {'{:.3f}'.format(avg_size)}[/yellow]}}", verbose)
            printv(f"{' ' * 3} - Best Sigma: {best.sigma})", verbose)
            printv(f"{' ' * 3} - Avg Sigma: {np.mean([i.sigma for i in population], axis=0)}", verbose)

        if should_render and generation % render_every == 0:
            utils.evaluate_fitness(config, best, episodes=1, render=True, should_normalize_state=norm_state)

        if should_plot:
            rewards, sizes = zip(*history)
            min_rewards, avg_rewards, std_rewards = zip(*rewards)
            avg_rewards = np.array(avg_rewards)
            std_rewards = np.array(std_rewards)
            min_sizes, avg_sizes, std_sizes = zip(*sizes)
            avg_sizes = np.array(avg_sizes)
            std_sizes = np.array(std_sizes)

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
    parser.add_argument('--popsize',help="Population size", required=True, type=int)
    parser.add_argument('--p_crossover',help="Probability of crossover", required=True, type=float)
    parser.add_argument('--p_mutation',help="Probability of mutation", required=True, type=float)
    parser.add_argument('--generations',help="Number of generations", required=True, type=int)
    parser.add_argument('--initial_sigma_max',help="Initial maximum value of sigma", required=True, type=float)
    parser.add_argument('--initial_depth',help="Randomly initialize the algorithm with trees of what depth?", required=True, type=int)
    parser.add_argument('--alpha',help="How to penalize tree size?", required=True, type=int)
    parser.add_argument('--mutation_type',help="Type of mutation", required=True, default="A", type=str)
    parser.add_argument('--norm_state',help="Should normalize state?", required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--tournament_size',help="Size of tournament", required=True, type=int)
    parser.add_argument('--elitism',help="Elitism?", required=True, type=float)
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
    
    for _ in range(args["simulations"]):
        tree, reward, size, evals2suc = run_genetic_algorithm(
            config=config,
            popsize=args["popsize"],
            initial_pop=initial_pop,
            p_crossover=args["p_crossover"],
            p_mutation=args["p_mutation"],
            generations=args["generations"],
            initial_sigma_max=args["initial_sigma_max"],
            initial_depth=args["initial_depth"], 
            mutation=args['mutation_type'],
            norm_state=args['norm_state'],
            alpha=args["alpha"],
            tournament_size=args["tournament_size"],
            elitism=args["elitism"],
            verbose=args["verbose"],
            should_plot=args["should_plot"])
        
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
