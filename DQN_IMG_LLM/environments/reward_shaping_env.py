# environments/reward_shaping_env.py

import gymnasium as gym

class RewardShapingEnv(gym.RewardWrapper):
    def __init__(self, env):
        super(RewardShapingEnv, self).__init__(env)

    def reward(self, reward):
        # Provide a small negative reward for each step to encourage shorter paths
        return reward - 0.01
