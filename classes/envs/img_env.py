if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import random
from utils.helpers import load_npy_files_to_dict

class ImgEnv(gym.Env):
    def __init__(self, env_id, idx, run_name='runs', capture_video=False, mode='light', use_pre_computed_states=True, size = 4, isRandom = False, is_slippery=False):
        super().__init__()
        seed = 42
        random.seed(seed)
        self.env_id = env_id
        self.idx = idx
        self.size = size
        self.isRandom = isRandom
        random_map = generate_random_map(size=size, p=0.4, seed = seed)
        self.env = gym.make(env_id, render_mode="rgb_array", is_slippery=is_slippery, desc=random_map)

        if capture_video and idx == 0:
            self.env = gym.wrappers.RecordVideo(self.env, f"videos/{run_name}")

        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        self.states = load_npy_files_to_dict(f"./states/{mode}/flat_arr/")
        self.img_size = size*64 # turns out grid size is n*64 by n*64 pixels
        self.img_resolution = 128
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(3*self.img_size**2,), dtype=np.float32)
        self.action_space = self.env.action_space
        self.mode = mode
        self.use_pre_computed_states = use_pre_computed_states
        
    def reset(self, **kwargs):
        if self.isRandom:
            random_map = generate_random_map(size=self.size, p=0.7, seed = random.randint(0, 1000))
            self.env.unwrapped.__init__(render_mode="rgb_array", is_slippery=False, desc=random_map)
        observation, info = self.env.reset(**kwargs)
        if self.use_pre_computed_states:
            llm_state = self.states.get(f"{observation}_1", np.zeros(3*self.img_size**2))
        else:
            llm_state = np.array(self.env.render(), dtype=np.uint8).flatten()

        return llm_state, info
    
    def step(self, action):
        next_observation, reward, termination, truncations, infos = self.env.step(action)
        if self.use_pre_computed_states:
            img_state = self.states.get(f"{next_observation}_{action}", np.zeros(3*self.img_size**2))
        else:
            img_state = np.array(self.env.render(), dtype=np.uint8).flatten()

        return img_state, reward, termination, truncations, infos
    

    def close(self):
        self.env.close()
        
if __name__ == '__main__':
    env = ImgEnv('FrozenLake-v1')
    print(env.states['0_0'].shape)