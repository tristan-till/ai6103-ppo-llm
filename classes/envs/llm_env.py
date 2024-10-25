if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import numpy as np

from utils.helpers import load_npy_files_to_dict

class LLMEnv(gym.Env):
    def __init__(self, env_id, run_name='runs', capture_video=False):
        super().__init__()
        self.env_id = env_id
        if capture_video:
            self.env = gym.make(env_id, render_mode="rgb_array", is_slippery=False)
            self.env = gym.wrappers.RecordVideo(self.env, f"videos/{run_name}")
        else:
            self.env = gym.make(env_id, render_mode=None, is_slippery=False)
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        self.states = load_npy_files_to_dict("./states/embedding/")
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(384,), dtype=np.float32)
        self.action_space = self.env.action_space
        
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        llm_state = self.states.get(f"{observation}_1", np.zeros(384))
        return llm_state, info
    
    def step(self, action):
        next_observation, reward, termination, truncations, infos = self.env.step(action)
        llm_state = self.states.get(f"{next_observation}_{action}", np.zeros(384))
        return llm_state, reward, termination, truncations, infos
    
    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()
        
if __name__ == '__main__':
    env = LLMEnv('FrozenLake-v1')
    # print()
    env.reset()
    env.step(1)
    env.step(1)
    env.step(2)
    env.step(1)
    _, reward, terminations, _, infos = env.step(2)
    print(reward, terminations, infos)
    _, reward, terminations, _, infos = env.step(2)
    print(reward, terminations, infos)
    
    # print(env.reset())
    # print(env.step(1))
    env.close()