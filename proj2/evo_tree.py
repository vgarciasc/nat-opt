from re import S
import time
import numpy as np
import pdb
import copy
from rich import print

import utils
from utils import printv
from tree import TreeNode
from env_configs import get_config

class EvoTreeNode(TreeNode):
    def __init__(self, fitness, reward, **kwargs):
        self.fitness = fitness
        self.reward = reward

        super(EvoTreeNode, self).__init__(**kwargs)

    def get_random_node(self):
        random_idx = np.random.randint(0, self.get_tree_size())
        stack = [self]
        processed = []

        while len(stack) > 0:
            node = stack.pop()
            processed.append(node)
            if len(processed) == random_idx + 1:
                return node

            if not node.is_leaf:
                stack.append(node.right)
                stack.append(node.left)
        
        printv(f"[red]Couldn't find node with id {random_idx}.[/red]", verbose=True)
        return None

    def generate_random_node(config, is_leaf=None):
        attribute = np.random.randint(config["n_attributes"])
        threshold = np.random.uniform(-1, 1)
        label = np.random.randint(config["n_actions"])
        is_leaf = is_leaf if is_leaf is not None \
            else np.random.choice([True, False])
        
        return EvoTreeNode(fitness=-1, reward=-1,
                           config=config, attribute=attribute, 
                           threshold=threshold, label=label,
                           is_leaf=is_leaf)

    def generate_random_tree(config, depth=2):
        node = EvoTreeNode.generate_random_node(config, is_leaf=(depth==0))
        if depth > 0:
            node.left = EvoTreeNode.generate_random_tree(config, depth - 1)
            node.right = EvoTreeNode.generate_random_tree(config, depth - 1)
            node.left.parent = node
            node.right.parent = node
        return node
    
    def mutate_attribute(self, verbose=False):
        printv("Mutating attribute...", verbose)
        self.attribute = np.random.randint(self.config["n_attributes"])
    
    def mutate_threshold(self, sigma, verbose=False):
        printv("Mutating threshold...", verbose)
        self.threshold += np.random.normal(0, 1) * sigma
    
    def mutate_label(self, verbose=False):
        printv("Mutating label...", verbose)
        self.label = np.random.randint(self.config["n_actions"])

    def mutate_is_leaf(self, verbose=False):
        printv("Mutating is leaf...", verbose)
        self.is_leaf = not self.is_leaf

        if self.is_leaf:
            self.left = None
            self.right = None

    def mutate(self, sigma=1):
        node = self.get_random_node()

        if node.is_leaf:
            node.mutate_label()
        else:
            operation = np.random.choice(["attribute", "threshold", "is_leaf"])
            if operation == "attribute":
                node.mutate_attribute()
            elif operation == "threshold":
                node.mutate_threshold(sigma)
            elif operation == "is_leaf":
                node.mutate_is_leaf()
        
    def crossover(parent_a, parent_b):
        start = time.time()
        parent_a = parent_a.copy()
        parent_b = parent_b.copy()
        end = time.time()
        if end - start > 0.5:
            pdb.set_trace()
        # print(f"Elapsed time for copying: {end - start} seconds")

        start = time.time()
        node_a = parent_a.get_random_node()
        node_b = parent_b.get_random_node()
        end = time.time()
        # print(f"Elapsed time for getting random nodes: {end - start} seconds")
        
        start = time.time()
        parent_a.replace_node(node_a, node_b)
        parent_b.replace_node(node_b, node_a)
        end = time.time()
        # print(f"Elapsed time for replacing nodes: {end - start} seconds")

        return parent_a, parent_b
    
    def copy(self):
        if self.is_leaf:
            return EvoTreeNode(fitness=self.fitness, reward=self.reward,
                               config=self.config, attribute=self.attribute, 
                               threshold=self.threshold, label=self.label,
                               is_leaf=self.is_leaf)
        else:
            new_left = self.left.copy()
            new_right = self.right.copy()

            new_node = EvoTreeNode(fitness=self.fitness, reward=self.reward,
                               config=self.config, attribute=self.attribute, 
                               threshold=self.threshold, label=self.label,
                               is_leaf=self.is_leaf, 
                               left=new_left, right=new_right)
            
            new_left.parent = new_node
            new_right.parent = new_node

            return new_node

if __name__ == "__main__":
    config = get_config("cartpole")
    
    print("[yellow]> Generating tree...[/yellow]")
    tree = EvoTreeNode.generate_random_tree(config, depth=2)

    print("[yellow]> Generated tree:[/yellow]")
    printv(utils.get_treeviz(config, tree), verbose=True)

    tree.mutate()

    print("[yellow]> Mutated tree:[/yellow]")
    printv(utils.get_treeviz(config, tree), verbose=True)

    # print("[yellow]> Evaluating fitness:[/yellow]")
    # print(f"Mean reward, std reward: {utils.evaluate_fitness(config, tree, episodes=10)}")

    tree_a = EvoTreeNode.generate_random_tree(config, depth=2)
    tree_b = EvoTreeNode.generate_random_tree(config, depth=2)
    
    print(f"[yellow]Doing crossover...[/yellow]\n")
    child_a, child_b = EvoTreeNode.crossover(tree_a, tree_b)

    print(f"[yellow]Parent A:[/yellow]")
    printv(utils.get_treeviz(config, tree_a), verbose=True)
    print(f"[yellow]Parent B:[/yellow]")
    printv(utils.get_treeviz(config, tree_b), verbose=True)

    print(f"[yellow]Child A:[/yellow]")
    printv(utils.get_treeviz(config, child_a), verbose=True)
    print(f"[yellow]Child B:[/yellow]")
    printv(utils.get_treeviz(config, child_b), verbose=True)    