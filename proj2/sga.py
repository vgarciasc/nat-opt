import copy
import math
import pdb
from rich import print
import numpy as np
import time
import matplotlib.pyplot as plt

from evo_tree import EvoTreeNode
from env_configs import get_config
import utils

def calc_fitness(tree, episodes=10):
    mean, _ = utils.evaluate_fitness(tree.config, tree, episodes=episodes)
    return mean

def tournament_selection(population, q):
    candidates = np.random.choice(population, size=q)
    return max(candidates, key=lambda x : x.fitness)

if __name__ == "__main__":
    config = get_config("cartpole")

    popsize = 50
    p_crossover = 0.8
    p_mutation = 0.05
    generations = 200
    sigma = 1
    alpha = 1
    q = 10

    # Initialization
    population = [EvoTreeNode.generate_random_tree(config, depth=2)
                  for _ in range(popsize)]
    best = None
    history = []
    fig, (ax1, ax2) = plt.subplots(2)

    # Main loop
    for generation in range(generations):
        for individual in population:
            individual.reward = calc_fitness(individual)
            individual.fitness = individual.reward - alpha * individual.get_tree_size()

            if best is None or individual.fitness > best.fitness:
                best = copy.deepcopy(individual)
        
        new_population = []
        for _ in range(popsize//2):
            parent_a = tournament_selection(population, q)
            parent_b = tournament_selection(population, q)

            if np.random.uniform(0, 1) < p_crossover:
                child_a, child_b = EvoTreeNode.crossover(parent_a, parent_b)
            else:
                child_a, child_b = parent_a, parent_b
            
            if np.random.uniform(0, 1) < p_mutation:
                child_a.mutate(sigma)

            if np.random.uniform(0, 1) < p_mutation:
                child_b.mutate(sigma)
        
            new_population += [child_a, child_b]

        population = new_population

        # Printing and plotting
        min_reward = best.reward
        avg_reward = np.mean([i.reward for i in population])
        std_reward = np.std([i.reward for i in population])
        min_size = best.get_tree_size()
        avg_size = np.mean([i.get_tree_size() for i in population])
        std_size = np.std([i.get_tree_size() for i in population])
        print(f"[green]Generation #{generation}: (min: {min_reward}, avg: {avg_reward})")
        print(f"[green]              - best size: {min_size}, avg size: {avg_size})")

        history.append(((min_reward, avg_reward, std_reward),
                        (min_size, avg_size, std_size)))
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
    
    print(f"[yellow]Best individual w/ reward {best.reward}:")
    print(utils.get_treeviz(config, best))

    plt.show()
