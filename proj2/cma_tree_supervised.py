import pdb
import cma
import numpy as np

from rich import print
from soft_tree import SoftTree, SoftTreeSigmoid
from plotter import plot_decision_surface
from sklearn.datasets import make_blobs

ALPHA = 1

# Evolve the multivariate weights and infer leaf classes based on the training data
if __name__ == "__main__":
    tree = SoftTree(num_attributes=2, num_classes=2)
    tree.randomize(depth=3)
    print(tree)

    # X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1, cluster_std=3)
    X, y = make_blobs(n_samples=1000, centers=[[-1, 1], [1, 1], [1, -1], [-1, -1]], n_features=2, random_state=1, cluster_std=0.5)
    y = np.array([y_i % 2 for y_i in y])

    def get_accuracy(weights):
        tree.weights = weights.reshape((tree.num_nodes, tree.num_attributes + 1))

        tree.update_leaves_by_dataset(X, y)
        y_pred = tree.predict_batch(X)
        accuracy = np.mean([(1 if y_pred[i] == y[i] else 0) for i in range(len(X))])

        return accuracy

    def get_accuracy_with_penalty(weights):
        accuracy = get_accuracy(weights)
        penalty = np.sum([np.sum(row[1:]) - np.max(row[1:]) for row in np.abs(tree.weights)])

        return 1 - accuracy + ALPHA * penalty

    accuracy = get_accuracy(tree.weights)
    print(f"Accuracy: {accuracy}")

    fun = get_accuracy_with_penalty
    x0 = tree.weights.flatten()
    sigma0 = 1
    x, es = cma.fmin2(fun, x0, sigma0, 
        options={'maxfevals': 4000,
                 'bounds': [-1, 1]})
    
    tree.weights = x.reshape((tree.num_nodes, tree.num_attributes + 1))
    tree.update_leaves_by_dataset(X, y)
    print(tree)
    print(f"Accuracy: {get_accuracy(tree.weights)}")

    plot_decision_surface(X, y, tree)

    pdb.set_trace()