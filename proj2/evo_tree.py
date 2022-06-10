import time
import numpy as np
import pdb
from rich import print

import utils
from utils import printv
from tree import TreeNode
from qtree import QTreeNode
from env_configs import get_config

class EvoTreeNode(TreeNode):
    def __init__(self, fitness, reward, **kwargs):
        super(EvoTreeNode, self).__init__(**kwargs)

        self.fitness = fitness
        self.reward = reward

    def get_random_node(self, get_inners=True, get_leaves=True):
        # node_list = self.get_node_list(get_inners, get_leaves)
        # return np.random.choice(node_list)
        num_nodes = self.get_tree_size()
        
        num_leaves = num_nodes // 2 + 1
        num_inners = num_nodes // 2
        total_num = (num_leaves if get_leaves else 0) + (num_inners if get_inners else 0)
        
        if total_num <= 1:
            return self
        
        random_idx = np.random.randint(1, total_num)

        stack = [self]
        processed = []

        while len(stack) > 0:
            node = stack.pop()
            if (node.is_leaf and get_leaves) or (not node.is_leaf and get_inners):
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
    
    def mutate_attribute(self, force_change=False, verbose=False):
        printv("Mutating attribute...", verbose)
        if force_change:
            new_attribute = self.attribute
            while new_attribute == self.attribute:
                new_attribute = np.random.randint(self.config["n_attributes"])
            self.attribute = new_attribute
            return
        
        self.attribute = np.random.randint(self.config["n_attributes"])
    
    def mutate_threshold(self, sigma, verbose=False):
        if sigma is None:
            sigma = np.ones(self.config["n_attributes"])
        
        printv("Mutating threshold...", verbose)
        self.threshold += np.random.normal(0, 1) * sigma[self.attribute]

        attr_data = self.config["attributes"][self.attribute]
        attr_type = attr_data[1]
        min_val, max_val = attr_data[2]

        if attr_type == "continuous":
            self.threshold = min([max([self.threshold, min_val]), max_val])
        elif attr_type == "binary":
            self.threshold = 0.5
    
    def mutate_label(self, verbose=False):
        printv("Mutating label...", verbose)
        gen_label = lambda : np.random.randint(self.config["n_actions"])

        if self.parent is not None:
            if self == self.parent.left:
                other_label = self.parent.right.label
            elif self == self.parent.right:
                other_label = self.parent.left.label
            else:
                other_label = self.label
            
            new_label = gen_label()
            while new_label == other_label:
                new_label = gen_label()
        else:
            new_label = gen_label()
        
        self.label = new_label

    def mutate_is_leaf(self, verbose=False):
        printv("Mutating is leaf...", verbose)
        self.is_leaf = not self.is_leaf

        if self.is_leaf:
            self.left = None
            self.right = None
        else:
            self.left = EvoTreeNode.generate_random_node(self.config, True)
            self.right = EvoTreeNode.generate_random_node(self.config, True)
            self.left.parent = self
            self.right.parent = self

            labels = np.random.choice(
                range(0, self.config["n_actions"]), 
                size=2, replace=False)
            self.left.label = labels[0]
            self.right.label = labels[1]
    
    def cut_parent(self):
        if self.parent is None or self.parent.parent is None:
            return

        if self.parent.parent.left == self.parent:
            self.parent.parent.left = self
        elif self.parent.parent.right == self.parent:
            self.parent.parent.right = self

    def mutate_A(self, sigma=None):
        node = self.get_random_node()

        if node.is_leaf:
            operation = np.random.choice(["label", "is_leaf"])
            # operation = np.random.choice(["label", "is_leaf", "cut_parent"])
            
            if operation == "label":
                node.mutate_label()
            elif operation == "is_leaf":
                node.mutate_is_leaf()
            elif operation == "cut_parent":
                node.cut_parent()
        else:
            operation = np.random.choice([
                "attribute", "threshold", 
                "is_leaf"])
                # "is_leaf", "cut_parent"])
            
            if operation == "attribute":
                node.mutate_attribute()
                node.threshold = np.random.uniform(-1, 1)
            elif operation == "threshold":
                node.mutate_threshold(sigma)
            elif operation == "is_leaf":
                node.mutate_is_leaf()
            elif operation == "cut_parent":
                node.cut_parent()

    def mutate_B(self, sigma=None):
        operation = np.random.choice(
            ["leaf_label", "inner_attribute",
             "inner_threshold", "is_leaf", 
             "cut_parent"])

        if operation == "leaf_label":
            leaf = self.get_random_node(get_inners=False, get_leaves=True)
            leaf.mutate_label()
            return 0
        elif operation == "inner_attribute" or operation == "inner_threshold":
            inner = self.get_random_node(get_inners=True, get_leaves=False)
            
            if operation == "inner_attribute":
                inner.mutate_attribute(force_change=False)
                inner.threshold = np.random.uniform(-1, 1)
                return 1
            elif operation == "inner_threshold":
                inner.threshold += np.random.normal(0, 1)
                return 2
        elif operation == "is_leaf":
            node = self.get_random_node()
            node.mutate_is_leaf()
            return 3
        elif operation == "cut_parent":
            node = self.get_random_node()
            node.cut_parent()
            return 4
        
    def crossover(parent_a, parent_b):
        parent_a = parent_a.copy()
        parent_b = parent_b.copy()

        node_a = parent_a.get_random_node(get_leaves=False)
        node_b = parent_b.get_random_node(get_leaves=False)
        
        parent_a.replace_node(node_a, node_b)
        parent_b.replace_node(node_b, node_a)

        return parent_a, parent_b
    
    def normalize_thresholds(self):
        (_, _, (xmin, xmax)) = self.config["attributes"][self.attribute]
        self.threshold = (self.threshold - xmin) / (xmax - xmin) * 2 - 1

        if not self.is_leaf:
            self.left.normalize_thresholds()
            self.right.normalize_thresholds()

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

    def read_from_string(config, string):
        actions = [a.lower() for a in config['actions']]
        attributes = [name.lower() for name, _, _ in config['attributes']]

        lines = [line.strip() for line in string.split("\n")]

        parents = [None for _ in lines]
        child_count = [0 for _ in lines]

        for line in lines[1:]:
            depth = line.rindex("- ") + 1

            content = line[depth:].strip()

            parent = parents[depth - 1] if depth > 1 else None
            is_left = (child_count[depth - 1] == 0) if depth > 1 else None
            
            is_leaf = "<=" not in content

            if not is_leaf:
                attribute, threshold = content.split(" <= ")
                
                attribute = attributes.index(attribute.lower())
                threshold = float(threshold)

                node = EvoTreeNode(fitness=-1, reward=-1, config=config,
                    attribute=attribute, threshold=threshold, label=0,
                    is_leaf=False, left=None, right=None, parent=parent)
                
            if is_leaf:
                label = actions.index(content.lower())

                node = EvoTreeNode(fitness=-1, reward=-1, config=config,
                    attribute=0, threshold=0, label=label,
                    is_leaf=True, left=None, right=None, parent=parent)
            
            if parent:
                if is_left:
                    parent.left = node
                else:
                    parent.right = node
            else:
                root = node

            parents[depth] = node
            child_count[depth] = 0
            child_count[depth - 1] += 1
        
        return root

if __name__ == "__main__":
    config = get_config("mountain_car")
    
    # print("[yellow]> Generating tree...[/yellow]")
    # tree = EvoTreeNode.generate_random_tree(config, depth=2)

    # print("[yellow]> Generated tree:[/yellow]")
    # printv(tree, verbose=True)

    # tree.run_episodes(100)
    # printv(tree, verbose=True)

    # tree.mutate()

    # print("[yellow]> Mutated tree:[/yellow]")
    # printv(tree, verbose=True)

    # print("[yellow]> Evaluating fitness:[/yellow]")
    # print(f"Mean reward, std reward: {utils.evaluate_fitness(config, tree, episodes=1000)}")

    # tree_a = EvoTreeNode.generate_random_tree(config, depth=2)
    # tree_b = EvoTreeNode.generate_random_tree(config, depth=2)
    
    # print(f"[yellow]Doing crossover...[/yellow]\n")
    # child_a, child_b = EvoTreeNode.crossover(tree_a, tree_b)

    # print(f"[yellow]Parent A:[/yellow]")
    # printv(tree_a, verbose=True)
    # print(f"[yellow]Parent B:[/yellow]")
    # printv(tree_b, verbose=True)

    # print(f"[yellow]Child A:[/yellow]")
    # printv(child_a, verbose=True)
    # print(f"[yellow]Child B:[/yellow]")
    # printv(child_b, verbose=True)    

    # ---------------------------------------------

    # Custode and Iacca, 2020: CARTPOLE
    # string = "\n- Pole Angular Velocity <= 0.074\n-- Pole Angle <= 0.022\n--- LEFT\n--- RIGHT\n-- RIGHT"
    # norm_state=False
    # Custode and Iacca, 2020: MOUNTAIN CAR
    # string = "\n- Car Velocity <= -0.0001\n-- Car Position <= -0.9\n--- RIGHT\n--- LEFT\n-- Car Position <= -0.3\n--- Car Velocity <= 0.035\n---- Car Position <= -0.45\n----- RIGHT\n----- Car Position <= -0.4\n------ RIGHT\n------ LEFT\n---- RIGHT\n--- RIGHT"
    # norm_state=False
    
    # Our DT obtained via DAgger: MOUNTAIN CAR
    # string = "\n- Car Velocity <= -0.0001\n-- LEFT\n-- Car Velocity <= 0.003\n--- Car Position <= -0.486\n---- RIGHT\n---- LEFT\n--- RIGHT"
    # norm_state=False
    # Our DT obtained via EA: MOUNTAIN CAR
    string = "\n- Car Velocity <= -0.043\n -- LEFT\n -- Car Position <= 0.577\n --- Car Velocity <= 0.043\n ---- Car Position <= -0.210\n ----- RIGHT\n ----- LEFT\n ---- RIGHT\n --- RIGHT"
    norm_state=True

    tree = EvoTreeNode.read_from_string(config, string=string)
    print(tree)

    print("[yellow]> Evaluating fitness:[/yellow]")
    print(f"Mean reward, std reward: {utils.evaluate_fitness(config, tree, episodes=1000, should_normalize_state=norm_state)}")

    # utils.evaluate_fitness(config, tree, 10, should_normalize_state=norm_state, render=True, verbose=True)