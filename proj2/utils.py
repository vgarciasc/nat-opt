import numpy as np
import gym
from rich import print

def printv(str, verbose=False):
    if verbose:
        print(str)

def get_treeviz(config, tree):
    stack = [(tree, 1)]
    output = ""

    while len(stack) > 0:
        node, depth = stack.pop()

        output += "\n"
        output += "-" * depth
        output += " "

        if node.is_leaf:
            output += (config['actions'][node.label]).upper()
        else:
            output += config['attributes'][node.attribute][0]
            output += " <= "
            output += '{:.3f}'.format(node.threshold)
            
            if node.right:
                stack.append((node.right, depth + 1))
            if node.left:
                stack.append((node.left, depth + 1))

    return output

def evaluate_fitness(config, tree, episodes=10, verbose=False):
    env = gym.make(config["name"])
    total_rewards = []

    for episode in range(episodes):
        raw_state = env.reset()
        state = config['conversion_fn'](env, None, raw_state)
        total_reward = 0
        done = False
        
        while not done:
            action = tree.act(state)
            raw_next_state, reward, done, _ = env.step(action)
            next_state = config['conversion_fn'](env, raw_state, raw_next_state)

            state = next_state
            raw_state = raw_next_state
            total_reward += reward

        printv(f"Episode #{episode} finished with total reward {total_reward}", verbose)
        total_rewards.append(total_reward)
    
    env.close()
    return np.mean(total_rewards), np.std(total_rewards)