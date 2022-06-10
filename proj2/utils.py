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

def fill_rewards(config, trees, alpha, episodes=10, should_normalize_state=False):
    env = gym.make(config["name"])
    
    for tree in trees:
        total_rewards = 0

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

            total_rewards += total_reward
    
        tree.reward = total_rewards / episodes
        tree.fitness = tree.reward - alpha * tree.get_tree_size()
    
    env.close()

def crossover_float_intermediary(parent_a, parent_b):
    assert len(parent_a) == len(parent_b)

    idx = np.random.randint(0, len(parent_a))

    child = np.copy(parent_a)
    child[idx] = (parent_a[idx] + parent_b[idx]) / 2

    return child