import argparse
import copy
from datetime import datetime
import pdb
import time
import cma
import numpy as np

from rich import print
from soft_tree import SoftTree, SoftTreeSigmoid
from plotter import plot_decision_surface
from sklearn.datasets import make_blobs

from utils import printv

def save_history_to_file(history, prefix=""):
    multiv_accuracies, univ_accuracies, elapsed_times, trees, univ_trees = zip(*history)
    trees = np.array(trees)
    univ_trees = np.array(univ_trees)

    string = prefix
    string += f"Average multivariate accuracy: {'{:.3f}'.format(np.mean(multiv_accuracies))} ± {'{:.3f}'.format(np.std(multiv_accuracies))}\n"
    string += f"Average univariate accuracy: {'{:.3f}'.format(np.mean(univ_accuracies))} ± {'{:.3f}'.format(np.std(univ_accuracies))}\n"
    string += f"Average elapsed time: {'{:.3f}'.format(np.mean(elapsed_times))} ± {'{:.3f}'.format(np.std(elapsed_times))}\n"
    
    output_path = "data/cma-log_" + datetime.now().strftime("%Y_%m_%d-%I_%M_%S") + ".txt"

    with open(output_path, "w", encoding="utf-8") as text_file:
        text_file.write(string)

def get_vector_from_tree(tree):
    return np.hstack((tree.weights.flatten(), tree.labels.flatten()))

def get_accuracy(vector, X, y, tree, should_evolve_leaves):
    weights = vector[:tree.num_nodes * (tree.num_attributes + 1)]
    tree.weights = weights.reshape((tree.num_nodes, tree.num_attributes + 1))

    if should_evolve_leaves:
        labels = vector[tree.num_nodes * (tree.num_attributes + 1):]
        tree.labels = labels.reshape((tree.num_leaves, tree.num_classes))
    else:
        tree.update_leaves_by_dataset(X, y)

    y_pred = tree.predict_batch(X)
    accuracy = np.mean([(1 if y_pred[i] == y[i] else 0) for i in range(len(X))])

    return accuracy

def get_accuracy_with_penalty(weights, X, y, tree, should_evolve_leaves, alpha):
    accuracy = get_accuracy(weights, X, y, tree, should_evolve_leaves)
    penalty = np.sum([np.sum(row[1:]) - np.max(row[1:]) for row in np.abs(tree.weights)])

    return 1 - accuracy + alpha * penalty

def run_CMAES(X, y, tree, options={}, should_evolve_leaves=True, sigma0=1, alpha=1):
    if should_evolve_leaves:
        x0 = np.hstack((tree.weights.flatten(), tree.labels.flatten()))
    else:
        x0 = tree.weights.flatten()

    x, _ = cma.fmin2(
        objective_function=get_accuracy_with_penalty, 
        x0=x0, sigma0=sigma0, 
        options=options,
        args=(X, y, tree, should_evolve_leaves, alpha))

    weights = x[:tree.num_nodes * (tree.num_attributes + 1)]
    tree.weights = weights.reshape((tree.num_nodes, tree.num_attributes + 1))

    if should_evolve_leaves:
        labels = x[tree.num_nodes * (tree.num_attributes + 1):]
        tree.labels = labels.reshape((tree.num_leaves, tree.num_classes))

    return tree

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CMA-ES for Tree Annealing')
    parser.add_argument('-s','--simulations',help="How many simulations?", required=True, type=int)
    parser.add_argument('-e','--max_evals',help="How many function evaluations to stop at?", required=True, type=int)
    parser.add_argument('-d','--depth', help="Depth of tree", required=True, type=int)
    parser.add_argument('--should_evolve_leaves', help='Should evolve leaves?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--alpha',help="How to penalize tree multivariateness?", required=True, type=float)
    parser.add_argument('--sigma0',help="How to initialize sigma?", required=False, default=1.0, type=float)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())
    
    X, y = make_blobs(n_samples=1000, centers=[[-1, 1], [1, 1], [1, -1], [-1, -1]], n_features=2, random_state=1, cluster_std=0.5)
    y = np.array([y_i % 2 for y_i in y])

    history = []
    for iteration in range(args['simulations']):
        print(f"Iteration #{iteration}:")

        tree = SoftTree(num_attributes=2, num_classes=2)
        tree.randomize(depth=args['depth'])

        start_time = time.time()
        tree = run_CMAES(X, y, tree,
            options={
                'maxfevals': args['max_evals'],
                'bounds': [-1, 1]},
            should_evolve_leaves=args['should_evolve_leaves'],
            sigma0=args['sigma0'],
            alpha=args['alpha'])
        end_time = time.time()
        elapsed_time = end_time - start_time

        multiv_accuracy = get_accuracy(
            get_vector_from_tree(tree), X, y, tree, 
            should_evolve_leaves=args['should_evolve_leaves'])
        printv(f"Accuracy (multivariate): {multiv_accuracy}", args['verbose'])

        printv(tree, args['verbose'])
        printv(tree.str_univariate(), args['verbose'])

        univ_tree = copy.deepcopy(tree)
        univ_tree.turn_univariate()
        univ_accuracy = get_accuracy(
            get_vector_from_tree(univ_tree), X, y, univ_tree, 
            should_evolve_leaves=args['should_evolve_leaves'])
        printv(f"Accuracy (univariate): {univ_accuracy}", args['verbose'])

        history.append((multiv_accuracy, univ_accuracy, elapsed_time, tree, univ_tree))
    
    command_line = str(args)
    command_line += "\n\npython cma_tree_supervised_2.py " + " ".join([f"--{key} {val}" for (key, val) in args.items()]) + "\n\n---\n\n"
    save_history_to_file(history, command_line)

    multiv_accuracies, univ_accuracies, elapsed_times, trees, univ_trees = zip(*history)

    print(f"Average multivariate accuracy: {'{:.3f}'.format(np.mean(multiv_accuracies))} ± {'{:.3f}'.format(np.std(multiv_accuracies))}")
    print(f"Average univariate accuracy: {'{:.3f}'.format(np.mean(univ_accuracies))} ± {'{:.3f}'.format(np.std(univ_accuracies))}")
    print(f"Average elapsed time: {'{:.3f}'.format(np.mean(elapsed_times))} ± {'{:.3f}'.format(np.std(elapsed_times))}")