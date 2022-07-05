from datetime import datetime
import pdb
import copy
import time
from webbrowser import get
import cma
import numpy as np
import argparse

from env_configs import get_config
from utils import evaluate_fitness
from rich import print
from soft_tree import SoftTree, SoftTreeSigmoid
from plotter import plot_decision_surface
from sklearn.datasets import make_blobs

def save_history_to_file(history, config, prefix=""):
    multiv_rewards, univ_rewards, elapsed_times, trees, univ_trees = zip(*history)
    trees = np.array(trees)
    univ_trees = np.array(univ_trees)

    string = prefix
    string += f"Average multivariate reward: {'{:.3f}'.format(np.mean(multiv_rewards))} ± {'{:.3f}'.format(np.std(multiv_rewards))}\n"
    string += f"Average univariate reward: {'{:.3f}'.format(np.mean(univ_rewards))} ± {'{:.3f}'.format(np.std(univ_rewards))}\n"
    string += f"Average elapsed time: {'{:.3f}'.format(np.mean(elapsed_times))} ± {'{:.3f}'.format(np.std(elapsed_times))}\n"
    string += f"\n\n{'-' * 20}\n\n"

    for i, tree in enumerate(trees):
        string += f"Multivariate Tree #{i} (Reward: {tree.reward})\n"
        string += str(tree)
        string += "\n"
        string += f"Univariate Tree #{i} (Reward: {univ_trees[i].reward})\n"
        string += tree.str_univariate(config)
        string += "\n"

    output_path = "data/cma-RL-log_" + datetime.now().strftime("%Y_%m_%d-%I_%M_%S") + ".txt"

    with open(output_path, "w", encoding="utf-8") as text_file:
        text_file.write(string)

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

def get_reward_with_penalty(weights, config, tree, episodes, alpha):
    avg_reward = get_reward(weights, config, tree, episodes)
    weight_penalties = [np.sum(row[1:]) - np.max(row[1:]) for row in np.abs(tree.weights)]
    
    # penalty = 0
    # same_action_mask = np.zeros(tree.num_nodes)
    
    # for depth in range(int(np.log2(tree.num_leaves)), 0, -1):
    #     for i in range(2 ** (depth - 1), 2 ** depth):
    #         if depth == int(np.log2(tree.num_leaves)):
    #             label_left = tree.get_left(i-1) - tree.num_nodes
    #             label_right = tree.get_right(i-1) - tree.num_nodes
    #             same_action_mask[i-1] = np.argmax(tree.labels[label_left]) if np.argmax(tree.labels[label_left]) == np.argmax(tree.labels[label_right]) else -1
    #         else:
    #             label_left = tree.get_left(i-1)
    #             label_right = tree.get_right(i-1)
    #             same_action_mask[i-1] = same_action_mask[label_left] if same_action_mask[label_left] == same_action_mask[label_right] else -1

    # same_action_mask = np.array([1 if i == -1 else 0 for i in same_action_mask])
    # penalty = np.sum(weight_penalties * same_action_mask)

    penalty = np.sum(weight_penalties)

    return - avg_reward + alpha * penalty

def run_CMAES(config, episodes, tree, options={}, alpha=1, sigma0=1):
    x0 = np.hstack((tree.weights.flatten(), tree.labels.flatten()))

    x, _ = cma.fmin2(
        objective_function=get_reward_with_penalty, 
        x0=x0, sigma0=sigma0, 
        options=options,
        args=(config, tree, episodes, alpha))

    weights = x[:tree.num_nodes * (tree.num_attributes + 1)]
    tree.weights = weights.reshape((tree.num_nodes, tree.num_attributes + 1))
    labels = x[tree.num_nodes * (tree.num_attributes + 1):]
    tree.labels = labels.reshape((tree.num_leaves, tree.num_classes))

    return tree

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CMA-ES for Tree Annealing')
    parser.add_argument('-t','--task', help="Task to train", required=True, type=str)
    parser.add_argument('-s','--simulations',help="How many simulations?", required=True, type=int)
    parser.add_argument('-e','--max_evals',help="How many function evaluations to stop at?", required=True, type=int)
    parser.add_argument('-d','--depth', help="Depth of tree", required=True, type=int)
    parser.add_argument('--episodes',help="How many episodes to evaluate?", required=True, type=int)
    parser.add_argument('--alpha',help="How to penalize tree multivariateness?", required=True, type=float)
    parser.add_argument('--sigma0',help="How to initialize sigma?", required=False, default=1.0, type=float)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    command_line = str(args)
    command_line += "\n\npython cma_tree_reinforced.py " + " ".join([f"--{key} {val}" for (key, val) in args.items()]) + "\n\n---\n\n"
    
    config = get_config(args["task"])
    episodes = args["episodes"]

    history = []
    for iteration in range(args["simulations"]):
        print(f"Iteration #{iteration}:")
        
        tree = SoftTree(
            num_attributes=config["n_attributes"],
            num_classes=config["n_actions"])
        tree.randomize(depth=args["depth"])

        start_time = time.time()
        tree = run_CMAES(
            config, episodes, tree,
            options={
                'maxfevals': args["max_evals"],
                'bounds': [-1, 1]},
            alpha=args["alpha"],
            sigma0=args["sigma0"])
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(tree)

        multiv_reward = get_reward(get_vector_from_tree(tree), config, tree, episodes)
        print(f"Reward (multivariate): {multiv_reward}")

        univ_tree = copy.deepcopy(tree)
        univ_tree.turn_univariate()

        print(univ_tree)

        univ_reward = get_reward(get_vector_from_tree(univ_tree), config, univ_tree, episodes)
        print(f"Reward (univariate): {univ_reward}")

        tree.reward = multiv_reward
        univ_tree.reward = univ_reward

        history.append((multiv_reward, univ_reward, elapsed_time, tree, univ_tree))
        save_history_to_file(history, config, command_line)

    multiv_rewards, univ_rewards, elapsed_times, trees, univ_trees = zip(*history)

    print(f"Average multivariate reward: {'{:.3f}'.format(np.mean(multiv_rewards))} ± {'{:.3f}'.format(np.std(multiv_rewards))}")
    print(f"Average univariate reward: {'{:.3f}'.format(np.mean(univ_rewards))} ± {'{:.3f}'.format(np.std(univ_rewards))}")
    print(f"Average elapsed time: {'{:.3f}'.format(np.mean(elapsed_times))} ± {'{:.3f}'.format(np.std(elapsed_times))}")