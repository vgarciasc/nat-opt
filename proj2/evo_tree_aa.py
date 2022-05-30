import time
import numpy as np
import pdb
from rich import print
from evo_tree import EvoTreeNode

import utils
from utils import printv, console
from tree import TreeNode
from env_configs import get_config

TAU1 = 0.1
TAU2 = 0.1

#Auto-adaptative Evolutionary Tree Node
class AAETNode():
    def __init__(self, config, sigma, tree, **kwargs):
        self.config = config
        self.sigma = sigma
        self.tree = tree

        self.reward = 0
        self.fitness = 0
    
    def act(self, state):
        return self.tree.act(state)

    def generate_random_tree(config, depth, sigma=None):
        if sigma is None:
            sigma = np.ones(config["n_attributes"])
        tree = EvoTreeNode.generate_random_tree(config, depth)
        return AAETNode(config=config, sigma=sigma, tree=tree)
    
    def mutate(self):
        N_1 = np.random.normal(0, 1, size=len(self.sigma))
        N_2 = np.ones(len(self.sigma)) * np.random.normal(0, 1)
        self.sigma *= np.exp(TAU1 * N_1 + TAU2 * N_2)
        
        self.tree.mutate(self.sigma)
    
    def crossover(parent_a, parent_b):
        tree_a, tree_b = EvoTreeNode.crossover(parent_a.tree, parent_b.tree)
        
        sigma_a = utils.crossover_float_intermediary(parent_a.sigma, parent_b.sigma)
        sigma_b = utils.crossover_float_intermediary(parent_b.sigma, parent_a.sigma)

        child_a = AAETNode(config=tree_a.config, sigma=sigma_a, tree=tree_a)
        child_b = AAETNode(config=tree_b.config, sigma=sigma_b, tree=tree_b)

        return child_a, child_b
    
    def get_tree_size(self):
        return self.tree.get_tree_size()
    
    def __str__(self):
        output = f"σ: {self.sigma}\n"
        output += "-" * 10 + "\n"
        output += str(self.tree)

        return output

if __name__ == "__main__":
    config = get_config("cartpole")
    
    print("[yellow]> Generating tree...[/yellow]")
    tree = AAETNode.generate_random_tree(config, depth=2)

    print("[yellow]> Generated tree:[/yellow]")
    printv(tree, verbose=True)

    tree.mutate()

    print("[yellow]> Mutated tree:[/yellow]")
    printv(tree, verbose=True)

    # print("[yellow]> Evaluating fitness:[/yellow]")
    # print(f"Mean reward, std reward: {utils.evaluate_fitness(config, tree, episodes=10)}")

    tree_a = AAETNode.generate_random_tree(config, depth=2,
        sigma=np.random.uniform(0, 5, size=config["n_attributes"]))
    tree_b = AAETNode.generate_random_tree(config, depth=2,
        sigma=np.random.uniform(0, 5, size=config["n_attributes"]))
    
    print(f"[yellow]Doing crossover...[/yellow]\n")
    child_a, child_b = AAETNode.crossover(tree_a, tree_b)

    print(f"[yellow]Parent A:[/yellow]")
    printv(tree_a, verbose=True)
    print(f"[yellow]Parent B:[/yellow]")
    printv(tree_b, verbose=True)

    print(f"[yellow]Child A:[/yellow]")
    printv(child_a, verbose=True)
    print(f"[yellow]Child B:[/yellow]")
    printv(child_b, verbose=True)    