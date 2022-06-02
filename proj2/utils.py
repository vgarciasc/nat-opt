import numpy as np
import gym
from rich.console import Console

console = Console()

def printv(str, verbose=False):
    if verbose:
        console.log(str)

def evaluate_fitness(config, tree, episodes=10, render=False, verbose=False):
    env = gym.make(config["name"])
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
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

def crossover_float_intermediary(parent_a, parent_b):
    assert len(parent_a) == len(parent_b)

    idx = np.random.randint(0, len(parent_a))

    child = np.copy(parent_a)
    child[idx] = (parent_a[idx] + parent_b[idx]) / 2

    return child