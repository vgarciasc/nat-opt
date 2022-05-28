import utils
import pdb
from rich import print

from env_configs import get_config

class TreeNode:
    def __init__(self, config, attribute, threshold, 
        label, is_leaf, left=None, right=None, parent=None):

        self.config = config

        self.attribute = attribute
        self.threshold = threshold
        self.label = label
        self.is_leaf = is_leaf

        self.left = left
        self.right = right
        self.parent = parent
    
    def __str__(self):
        return f"[attrib: {self.attribute}, threshold: {self.threshold}, " + \
            f"label: {self.label}, is_leaf: {self.is_leaf}]"
    
    def act(self, state):
        if self.is_leaf:
            return self.label
        
        if state[self.attribute] <= self.threshold:
            return self.left.act(state)
        else:
            return self.right.act(state)
    
    def get_tree_size(self):
        # if self.is_leaf:
        #     return 1
        # else:
        #     return 1 + self.left.get_tree_size() + self.right.get_tree_size()
        total = 1
        if self.left != None:
            total += self.left.get_tree_size() 
        if self.right != None:
            total += self.right.get_tree_size() 
        return total
    
    def replace_node(self, node_src, node_dst):
        if node_src.parent != None:
            if node_src.parent.left == node_src:
                node_src.parent.left = node_dst
            else:
                node_src.parent.right = node_dst

if __name__ == "__main__":
    config = get_config("cartpole")
    tree = TreeNode(config, 3, 0.44, 1, False, 
        left=TreeNode(config, 2, 0.01, 1, False,
            left=TreeNode(config, 1, 2.1, 0, True),
            right=TreeNode(config, 1, 0.2, 1, True)),
        right=TreeNode(config, 2, -0.41, 0, False, 
            left=TreeNode(config, 1, 2.1, 0, True),
            right=TreeNode(config, 1, 0.2, 1, True)))


    utils.printv(utils.get_treeviz(config, tree), verbose=True)

    print("[yellow]> Evaluating fitness:[/yellow]")
    print(f"Mean reward, std reward: {utils.evaluate_fitness(config, tree, episodes=10)}")