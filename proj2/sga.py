import copy
import math
import pdb
from rich import print
import numpy as np
import time
import matplotlib.pyplot as plt

from evo_tree import EvoTreeNode
from evo_tree_aa import AAETNode
from env_configs import get_config
from utils import printv, console
import utils

def calc_fitness(tree, episodes=10):
    mean, _ = utils.evaluate_fitness(tree.config, tree, episodes=episodes)
    return mean

def tournament_selection(population, q):
    np.random.shuffle(population)
    return max(population[:q], key=lambda x : x.fitness)

def run_genetic_algorithm(config, popsize, p_crossover, p_mutation, 
    generations, initial_sigma_max, initial_depth, alpha, tournament_size,
    repclass, fit_episodes=10, 
    should_plot=False, should_render=False, render_every=None, verbose=False):

    # Initialization
    population = [repclass.generate_random_tree(
                    config, depth=initial_depth,
                    sigma=np.random.uniform(0, initial_sigma_max, size=config["n_attributes"]))
                  for _ in range(popsize)]
    best = None
    history = []

    if should_plot:
        fig, (ax1, ax2) = plt.subplots(2)

    # Main loop
    for generation in range(generations):
        new_population = []
        for _ in range(popsize//2):
            parent_a = tournament_selection(population, tournament_size)
            parent_b = tournament_selection(population, tournament_size)

            if np.random.uniform(0, 1) < p_crossover:
                child_a, child_b = repclass.crossover(parent_a, parent_b)
            else:
                child_a, child_b = parent_a.copy(), parent_b.copy()
            
            if np.random.uniform(0, 1) < p_mutation:
                child_a.mutate()

            if np.random.uniform(0, 1) < p_mutation:
                child_b.mutate()
        
            new_population += [child_a, child_b]

        population = new_population

        for individual in population:
            individual.reward = calc_fitness(individual, episodes=fit_episodes)
        avg_reward = np.mean([i.reward for i in population])
        curr_alpha = 0 if avg_reward == config["min_score"] else alpha

        for individual in population:
            individual.fitness = individual.reward - curr_alpha * individual.get_tree_size()

            if best is None or individual.fitness > best.fitness:
                best = individual.copy()

        population += [best]

        # if best.reward >= config["max_score"]:
        #     break

        # Printing and plotting
        min_reward = best.reward
        avg_reward = np.mean([i.reward for i in population])
        std_reward = np.std([i.reward for i in population])
        min_size = best.get_tree_size()
        avg_size = np.mean([i.get_tree_size() for i in population])
        std_size = np.std([i.get_tree_size() for i in population])
        
        console.rule(f"[bold red]Generation #{generation}")
        printv(f"[underline]Reward[/underline]: {{[green]Best: {min_reward}[/green], [yellow]Avg: {avg_reward}[/yellow]}}", verbose)
        printv(f"[underline]Size  [/underline]: {{[green]Best: {min_size}[/green], [yellow]Avg: {avg_size}[/yellow]}}", verbose)
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
            utils.evaluate_fitness(config, best, episodes=1, render=True)

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
    
    return best, best.reward, best.get_tree_size()

if __name__ == "__main__":
    history = []

    for _ in range(50):
        # tree, reward, size = run_genetic_algorithm(
        #     config=get_config("cartpole"),
        #     popsize=50, p_crossover=0.8, p_mutation=0.05,
        #     generations=100, initial_sigma_max=1, initial_depth=3, 
        #     alpha=5, tournament_size=20, repclass=AAETNode, 
        #     verbose=True, should_plot=False)
        # tree, reward, size = run_genetic_algorithm(
        #     config=get_config("mountain_car"),
        #     popsize=50, p_crossover=0.8, p_mutation=0.2,
        #     generations=100, initial_sigma_max=1, initial_depth=3, 
        #     alpha=5, tournament_size=20, repclass=AAETNode, 
        #     verbose=True, should_plot=False)
        tree, reward, size = run_genetic_algorithm(
            config=get_config("lunar_lander"),
            popsize=50, p_crossover=0.8, p_mutation=0.2,
            generations=200, initial_sigma_max=1, initial_depth=5, 
            alpha=1, tournament_size=10, repclass=AAETNode, 
            fit_episodes=10, 
            verbose=True, should_plot=False)
        history.append((tree, reward, size))
        print(history)
        utils.evaluate_fitness(tree.config, tree, episodes=10, render=True)

    trees, rewards, sizes = zip(*history)
    print(f"[green][bold]Average reward[/bold]: {np.mean(rewards)}[/green]")
    print(f"[green][bold]Average size[/bold]: {np.mean(rewards)}[/green]")
    print(f"-" * 30)
    print(f"[green][bold]All trees:[/bold][/green]")
    for tree in trees:
        print(tree)
