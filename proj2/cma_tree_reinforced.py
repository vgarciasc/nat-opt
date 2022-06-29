import pdb
import time
from webbrowser import get
import cma
import numpy as np

from env_configs import get_config
from utils import evaluate_fitness
from rich import print
from soft_tree import SoftTree, SoftTreeSigmoid
from plotter import plot_decision_surface
from sklearn.datasets import make_blobs

ALPHA = 10

def get_vector_from_tree(tree):
    return np.hstack((tree.weights.flatten(), tree.labels.flatten()))

def get_reward(vector, config, tree, episodes):
    weights = vector[:tree.num_nodes * (tree.num_attributes + 1)]
    labels = vector[tree.num_nodes * (tree.num_attributes + 1):]
    tree.weights = weights.reshape((tree.num_nodes, tree.num_attributes + 1))
    tree.labels = labels.reshape((tree.num_leaves, tree.num_classes))

    avg_reward, _  = evaluate_fitness(
        config, tree, episodes,
        should_normalize_state=True)

    return avg_reward

def get_reward_with_penalty(weights, config, tree, episodes):
    avg_reward = get_reward(weights, config, tree, episodes)
    penalty = np.sum([np.sum(row[1:]) - np.max(row[1:]) for row in np.abs(tree.weights)])


    return - avg_reward + ALPHA * penalty

def run_CMAES(config, episodes, tree, options={}):
    x0 = np.hstack((tree.weights.flatten(), tree.labels.flatten()))

    x, _ = cma.fmin2(
        objective_function=get_reward_with_penalty, 
        x0=x0, sigma0=1, 
        options=options,
        args=(config, tree, episodes))

    weights = x[:tree.num_nodes * (tree.num_attributes + 1)]
    tree.weights = weights.reshape((tree.num_nodes, tree.num_attributes + 1))

    if should_evolve_leaves:
        labels = x[tree.num_nodes * (tree.num_attributes + 1):]
        tree.labels = labels.reshape((tree.num_leaves, tree.num_classes))

    return tree

if __name__ == "__main__":
    config = get_config("cartpole")
    should_evolve_leaves = True
    episodes = 10

    history = []
    for iteration in range(10):
        print(f"Iteration #{iteration}:")
        
        tree = SoftTree(
            num_attributes=config["n_attributes"],
            num_classes=config["n_actions"])
        tree.randomize(depth=3)

        start_time = time.time()
        tree = run_CMAES(
            config, episodes, tree,
            options={
                'maxfevals': 10000,
                'bounds': [-1, 1]})
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(tree)

        multiv_accuracy = get_reward(
            get_vector_from_tree(tree), 
            config, tree, episodes)
        print(f"Accuracy (multivariate): {multiv_accuracy}")
        
        pdb.set_trace()

        tree.turn_univariate()

        univ_accuracy = get_reward(
            get_vector_from_tree(tree), 
            config, tree, episodes)
        print(f"Accuracy (univariate): {univ_accuracy}")

        history.append((multiv_accuracy, univ_accuracy, elapsed_time))
    
    multiv_accuracies, univ_accuracies, elapsed_times = zip(*history)

    print(f"Average multivariate accuracy: {'{:.3f}'.format(np.mean(multiv_accuracies))} ± {'{:.3f}'.format(np.std(multiv_accuracies))}")
    print(f"Average univariate accuracy: {'{:.3f}'.format(np.mean(univ_accuracies))} ± {'{:.3f}'.format(np.std(univ_accuracies))}")
    print(f"Average elapsed time: {'{:.3f}'.format(np.mean(elapsed_times))} ± {'{:.3f}'.format(np.std(elapsed_times))}")