import pdb
import numpy as np
import gym
from rich.console import Console

console = Console()
norm_state_mins = None
norm_state_maxs = None

def printv(str, verbose=False):
    if verbose:
        console.log(str)

def save_history_to_file(history, filepath, prefix=""):
    trees, rewards, sizes, evals2suc = zip(*history)
    successes = [1 if e > 0 else 0 for e in evals2suc]
    evals2suc = [e for e in evals2suc if e > 0]
    trees = np.array(trees)

    string = prefix
    string += f"Mean Best Reward: {np.mean(rewards)}\n"
    string += f"Mean Best Size: {np.mean(sizes)}\n"
    string += f"Average Evaluations to Success: {np.mean(evals2suc)}\n"
    string += f"Success Rate: {np.mean(successes)}\n"
    string += "\n-----\n\n"

    for i, tree in enumerate(trees):
        string += f"Tree #{i} (Reward: {tree.reward}, Size: {tree.get_tree_size()})\n"
        string += str(tree)
        string += "\n"
    
    with open(filepath, "w", encoding="utf-8") as text_file:
        text_file.write(string)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def normalize_state(config, state):
    global norm_state_mins
    global norm_state_maxs

    if norm_state_mins is None:
        norm_state_mins = np.array([(xmin if abs(xmin) < 9999 else -1) for (_, _, (xmin, xmax)) in config["attributes"]])
    if norm_state_maxs is None:
        norm_state_maxs = np.array([(xmax if abs(xmax) < 9999 else 3) for (_, _, (xmin, xmax)) in config["attributes"]])
    
    return (state - norm_state_mins) / (norm_state_maxs - norm_state_mins) * 2 - 1

def evaluate_fitness(config, tree, episodes=10, should_normalize_state=False, render=False, verbose=False):
    env = gym.make(config["name"])
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if should_normalize_state:
                state = normalize_state(config, state)
            
            action = tree.act(state)
            if render:
                env.render()
            next_state, reward, done, _ = env.step(action)

            state = next_state
            total_reward += reward

        printv(f"Episode #{episode} finished with total reward {total_reward}", verbose)
        total_rewards.append(total_reward)
    
    env.close()
    return np.mean(total_rewards), np.std(total_rewards)

def calc_reward(tree, episodes=10, norm_state=False, penalize_std=False):
    mean, std = evaluate_fitness(
        tree.config, tree,
        episodes=episodes,
        should_normalize_state=norm_state)
    return mean - std if penalize_std else mean

def fill_rewards(config, trees, alpha, episodes=10, should_normalize_state=False, penalize_std=False):
    env = gym.make(config["name"])
    
    for tree in trees:
        # total_rewards = 0
        rewards = []

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                if should_normalize_state:
                    state = normalize_state(config, state)
                
                action = tree.act(state)
                state, reward, done, _ = env.step(action)
                total_reward += reward

            # total_rewards += total_reward
            rewards.append(total_reward)

        # tree.reward = total_rewards / episodes
        tree.reward = np.mean(rewards) - np.std(rewards) if penalize_std else np.mean(rewards)
        tree.fitness = tree.reward - alpha * tree.get_tree_size()
    
    env.close()

def crossover_float_intermediary(parent_a, parent_b):
    assert len(parent_a) == len(parent_b)

    idx = np.random.randint(0, len(parent_a))

    child = np.copy(parent_a)
    child[idx] = (parent_a[idx] + parent_b[idx]) / 2

    return child