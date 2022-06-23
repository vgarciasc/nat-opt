import pdb
import numpy as np

from env_configs import get_config
from rich import print
from utils import sigmoid
from plotter import plot_decision_surface
from sklearn.datasets import make_blobs

class SoftTree:
    def __init__(self, num_attributes=2, num_classes=2, weights=[], labels=[]):
        self.num_attributes = num_attributes
        self.num_classes = num_classes
        self.weights = weights
        self.labels = labels
        
        self.num_nodes = len(weights)
        self.num_leaves = len(labels)
        self.depth = np.log2(len(weights) + 1)
    
    def randomize(self, depth):
        self.num_nodes = 2 ** depth - 1
        self.num_leaves = 2 ** depth

        self.weights = np.random.uniform(-1, 1, size=(self.num_nodes, self.num_attributes + 1))
        self.labels = np.ones((self.num_leaves, self.num_classes)) / self.num_classes

    def get_left(self, node):
        return node * 2 + 1
    
    def get_right(self, node):
        return node * 2 + 2
    
    def is_leaf(self, node):
        return node >= self.num_nodes

    def get_leaf(self, state):
        state = np.insert(state, 0, 1, axis=0)

        stack = [0]

        while stack != []:
            node = stack.pop(0)

            if self.is_leaf(node):
                return node - self.num_nodes
            else:
                if self.weights[node] @ state <= 0:
                    stack.append(self.get_left(node))
                else:
                    stack.append(self.get_right(node))
    
    def update_leaves_by_dataset(self, X, y):
        count = [np.zeros(self.num_classes) for _ in range(self.num_leaves)]

        for x_i, y_i in zip(X, y):
            leaf = self.get_leaf(x_i)
            count[leaf][y_i] += 1
        
        self.labels = np.zeros((self.num_leaves, self.num_classes))
        for leaf, samples in enumerate(count):
            self.labels[leaf][np.argmax(samples)] = 1

    def predict(self, state):
        return np.argmax(self.labels[self.get_leaf(state)])

    def predict_batch(self, X):
        return np.array([self.predict(x) for x in X])
    
    def turn_univariate(self):
        self.weights = np.array([[(w if i == 0 or i == np.argmax(split) else 0) for i, w in enumerate(split)] for split in self.weights])
    
    def str_univariate(self):
        stack = [(0, 1)]
        output = ""

        while len(stack) > 0:
            node, depth = stack.pop()
            output += "-" * depth + " "

            if self.is_leaf(node):
                output += f"Class {np.argmax(self.labels[node - self.num_nodes])}"
            else:
                bias = self.weights[node][0]
                attribute = np.argmax(np.abs(self.weights[node][1:])) + 1
                weight = self.weights[node][attribute]

                output += f"x{attribute} <= {'{:.3f}'.format(-bias / weight)}"
                
                if (weight < 0):
                    stack.append((self.get_left(node), depth + 1))
                    stack.append((self.get_right(node), depth + 1))
                else:
                    stack.append((self.get_right(node), depth + 1))
                    stack.append((self.get_left(node), depth + 1))
            output += "\n"

        return output

    def __str__(self):
        stack = [(0, 1)]
        output = ""

        while len(stack) > 0:
            node, depth = stack.pop()
            output += "-" * depth + " "

            if self.is_leaf(node):
                output += str(self.labels[node - self.num_nodes])
            else:
                output += '{:.3f}'.format(self.weights[node][0]) + " + " + \
                    " + ".join([f"{'{:.3f}'.format(self.weights[node][i])} x{i}" for i in range(1, self.num_attributes + 1)])
                
                stack.append((self.get_right(node), depth + 1))
                stack.append((self.get_left(node), depth + 1))
            output += "\n"

        return output

class SoftTreeSigmoid(SoftTree):
    def get_leaf(self, state):
        state = np.insert(state, 0, 1, axis=0)

        stack = [(0, 1)]
        output = []

        while stack != []:
            node, membership = stack.pop(0)

            if self.is_leaf(node):
                output.append((node - self.num_nodes, membership))
            else:
                val = sigmoid(self.weights[node] @ state)
                stack.append((self.get_left(node), membership * val))
                stack.append((self.get_right(node), membership * (1 - val)))
        
        max_leaf, max_membership = max(output, key=lambda x:x[1])
        return max_leaf

if __name__ == "__main__":
    config = get_config("cartpole")
    
    # tree = SoftTree(num_attributes=3, num_classes=2)
    # tree.randomize(depth=2)
    # print(tree)
    # state = np.array([1, 2, 3])
    # y = tree.predict(state)

    # tree = SoftTree(num_attributes=2, num_classes=2)
    # tree.randomize(depth=2)
    # tree.labels = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    # print(tree)
    # X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1, cluster_std=3)
    # plot_decision_surface(X, y, tree)

    tree = SoftTree(num_attributes=2, num_classes=2)
    tree.randomize(depth=2)
    X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1, cluster_std=3)
    y_pred = tree.predict_batch(X)
    accuracy = np.mean([(1 if y_pred[i] == y[i] else 0) for i in range(len(X))])
    print(accuracy)