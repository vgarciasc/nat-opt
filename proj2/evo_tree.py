import numpy as np
import pdb
from rich import print

import utils
from tree import TreeNode
from env_configs import get_config

class EvoTreeNode(TreeNode):
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
        
        utils.printv(f"[red]Couldn't find node with id {random_idx}.[/red]", verbose=True)
        return None

    def generate_random_node(config, is_leaf=None):
        attribute = np.random.randint(config["n_attributes"])
        threshold = np.random.uniform(-1, 1)
        label = np.random.randint(config["n_actions"])
        is_leaf = is_leaf if is_leaf is not None \
            else np.random.choice([True, False])
        
        return EvoTreeNode(config, attribute, threshold, label, is_leaf)

    def generate_random_tree(config, depth=2):
        root = EvoTreeNode.generate_random_node(config, is_leaf=(depth==0))
        if depth > 0:
            root.left = EvoTreeNode.generate_random_tree(config, depth - 1)
            root.right = EvoTreeNode.generate_random_tree(config, depth - 1)
        return root
    
    def mutate_attribute(self):
        print("Mutating attribute...")
        self.attribute = np.random.randint(self.config["n_attributes"])
    
    def mutate_threshold(self):
        print("Mutating threshold...")
        self.threshold += np.random.normal(0, 1)
    
    def mutate_label(self):
        print("Mutating label...")
        self.label = np.random.randint(self.config["n_actions"])

    def mutate_is_leaf(self):
        print("Mutating is leaf...")
        self.is_leaf = not self.is_leaf

    def mutate(self):
        node = self.get_random_node()

        if node.is_leaf:
            node.mutate_label()
        else:
            operation = np.random.choice(["attribute", "threshold", "is_leaf"])
            if operation == "attribute":
                node.mutate_attribute()
            elif operation == "threshold":
                node.mutate_threshold()
            elif operation == "is_leaf":
                node.mutate_is_leaf()

if __name__ == "__main__":
    config = get_config("cartpole")
    
    print("[yellow]> Generating tree...[/yellow]")
    tree = EvoTreeNode.generate_random_tree(config, depth=2)

    print("[yellow]> Generated tree:[/yellow]")
    utils.printv(utils.get_treeviz(config, tree), verbose=True)

    tree.mutate()
    print("[yellow]> Mutated tree:[/yellow]")
    utils.printv(utils.get_treeviz(config, tree), verbose=True)
    # print("[yellow]> Evaluating fitness:[/yellow]")
    # print(f"Mean reward, std reward: {utils.evaluate_fitness(config, tree, episodes=10)}")