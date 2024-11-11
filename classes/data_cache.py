import utils.enums as enums
import numpy as np

class DataCache:
    def __init__(self):
        self.train_rewards = []
        self.val_rewards = []
        self.test_rewards = []
        
        self.agent_policy = {}
        
    def cache_reward(self, reward, mode):
        if isinstance(reward, np.ndarray):
            reward = reward.ravel()
        if mode == enums.EnvMode.TRAIN:
            self.train_rewards.extend(reward)
        elif mode == enums.EnvMode.VAL:
            self.val_rewards.extend(reward)
        else:
            self.test_rewards.extend(reward)
            
    def cache_policy(self, x, y, action):
        if (x, y) not in self.agent_policy:
            self.agent_policy[(x, y)] = {
                0: 0,
                1: 0,
                2: 0,
                3: 0
            }        
        self.agent_policy[(x, y)][action] += 1