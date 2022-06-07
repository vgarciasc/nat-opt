import gym
import utils
import pdb
import numpy as np

from tree import TreeNode
from rich import print

from env_configs import get_config

class QTreeNode(TreeNode):
    def __init__(self, **kwargs):
        super(QTreeNode, self).__init__(**kwargs)

        self.q_values = np.random.uniform(-1, 1, size=self.config["n_actions"])
    
    def predict_with_leaf(self, state):
        if self.is_leaf:
            return self, np.argmax(self.q_values)
        
        if state[self.attribute] <= self.threshold:
            return self.left.predict_with_leaf(state)
        else:
            return self.right.predict_with_leaf(state)
    
    def act(self, state):
        if self.is_leaf:
            return np.argmax(self.q_values)
        
        if state[self.attribute] <= self.threshold:
            return self.left.act(state)
        else:
            return self.right.act(state)
    
    def run_episodes(self, n_episodes):
        gym_env = gym.make(self.config['name'])
        alpha = 0.001
        epsilon = 0.5
        gamma = 1

        for _ in range(1, n_episodes):
            state = gym_env.reset()
            leaf, action = self.predict_with_leaf(state)
            reward = 0
            done = False
            
            while not done:
                if np.random.random() < epsilon:
                    action = np.random.randint(0, self.config['n_actions'])
                else:
                    action = self.act(state)
                
                next_state, reward, done, _ = gym_env.step(action)
                if done and self.config['should_force_episode_termination_score']:
                    reward = self.config['episode_termination_score']

                next_leaf, _ = self.predict_with_leaf(next_state)

                leaf.q_values[action] = leaf.q_values[action] + alpha * (reward + gamma * np.max(next_leaf.q_values) - leaf.q_values[action])

                state = next_state

        gym_env.close()

if __name__ == "__main__":
    config = get_config("cartpole")
    tree = QTreeNode(config=config, attribute=3, threshold=0.44, label=1, is_leaf=False, 
        left=QTreeNode(config=config, attribute=2, threshold=0.01, label=0, is_leaf=False,
            left=QTreeNode(config=config, attribute=1, threshold=2.1, label=0, is_leaf=True),
            right=QTreeNode(config=config, attribute=1, threshold=0.2, label=1, is_leaf=True)),
        right=QTreeNode(config=config, attribute=2, threshold=-0.41, label=1, is_leaf=False,
            left=QTreeNode(config=config, attribute=1, threshold=2.1, label=0, is_leaf=True),
            right=QTreeNode(config=config, attribute=1, threshold=0.2, label=1, is_leaf=True)))
    
    # config = get_config("mountain_car")
    # tree = QTreeNode(config=config, attribute=0, threshold=0.158, label=1, is_leaf=False, 
    #     left=QTreeNode(config=config, attribute=1, threshold=0.000, label=1, is_leaf=False,
    #         left=QTreeNode(config=config, attribute=1, threshold=2.1, label=0, is_leaf=True),
    #         right=QTreeNode(config=config, attribute=1, threshold=0.2, label=2, is_leaf=True)),
    #     right=QTreeNode(config=config, attribute=2, threshold=-0.41, label=2, is_leaf=True, 
    #         left=QTreeNode(config=config, attribute=1, threshold=2.1, label=0, is_leaf=True),
    #         right=QTreeNode(config=config, attribute=1, threshold=0.2, label=1, is_leaf=True)))

    utils.printv(tree, verbose=True)
    tree.run_episodes(1000)
    utils.printv(tree, verbose=True)

    print("[yellow]> Evaluating fitness:[/yellow]")
    print(f"Mean reward, std reward: {utils.evaluate_fitness(config, tree, episodes=100)}")