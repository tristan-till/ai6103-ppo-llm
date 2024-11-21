import os
import random

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..')))

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map, is_valid
import numpy as np

from utils.helpers import load_npy_files_to_dict
import utils.enums as enums
import utils.render_env as render_utils
import utils.env as env_utils

from typing import List, Tuple

class LLMv2Env(gym.Env):
    def __init__(self, env_id, run_name='runs', mode=enums.EnvMode.TRAIN, 
                 use_pre_computed_states=True, size = 4, is_random = False,
                 is_ma = False, seed=42, data_cache=None, is_slippery=False):
        super().__init__()
        self.env_id = env_id
        self.run_name = run_name
        self.size = size
        self.is_random = is_random
        self.is_ma = is_ma
        self.current_map = ["SFFF", "FHFH", "FFFH", "HFFG"]
        self.current_map_id = "SFFFFHFHFFFHHFFG"
        if self.is_ma:
            self.current_map = ["SFFHHFFS", "FHFHFHFH", "FFFHFFFH", "FHFFHFFF", "FFFHHFFF", "FHFHFHFH", "FFFHFFFH", "PHFFHFFG"]
            self.current_map_id = "SFFHHFFSFHFHFHFHFFFHFFFHFHFFHFFFFFFHHFFFFHFHFHFHFFFHFFFHPHFFHFFG"
        self.mode = mode
        self.episodes = 0
        self.is_slippery = is_slippery

        self.img_size = size*64
        self.img_resolution = 128
        self.embedding_size = 384
        self.render_mode='rgb_array'
        
        self.seed = seed
        random.seed(seed)
        
        self.env = None
        self.env = self.init_env(env_id)
        
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(3*self.img_size**2 + self.embedding_size,), dtype=np.float32)
        self.action_space = self.env.action_space
        
        self.states = {}
        self.current_action_str = "0_1"
        
        self.state_path = f"states/llmv2_env/{env_utils.mode_str_from_enum(self.mode)}"
        if not os._exists(f"{self.state_path}"):
            os.makedirs(f"{self.state_path}", exist_ok=True)
        
        if use_pre_computed_states:
            self.load_precomputed_states()

        self.data_cache = data_cache
    
    def init_env(self, env_id):
        if self.is_random:
            self.randomize_map()
        env = gym.make(env_id, render_mode="rgb_array", is_slippery=self.is_slippery, desc=self.current_map)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    
    def load_precomputed_states(self):
        if not os.path.exists(f"{self.state_path}"):
            print("Folder for precomputed states not found")
            return
        self.states = load_npy_files_to_dict(f"{self.state_path}/")

    def is_valid(board: List[List[str]], max_size: int) -> bool:
        # Find all start ('S') and goal ('G') positions
        starts = []
        goals = []
        for r in range(max_size):
            for c in range(max_size):
                if board[r][c] == 'S':
                    starts.append((r, c))
                elif board[r][c] == 'G' or board[r][c] == 'P':
                    goals.append((r, c))
        
        if len(starts) != 2 or len(goals) != 2:
            return False  # Invalid configuration
        
        # Helper function for BFS to check if a single agent can reach a goal
        def can_reach_goal(start: Tuple[int, int], goals: List[Tuple[int, int]]) -> bool:
            frontier, discovered = [], set()
            frontier.append(start)
            while frontier:
                r, c = frontier.pop()
                if (r, c) in discovered:
                    continue
                discovered.add((r, c))
                if (r, c) in goals:
                    return True
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new, c_new = r + x, c + y
                    if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                        continue
                    if board[r_new][c_new] != "H" and (r_new, c_new) not in discovered:
                        frontier.append((r_new, c_new))
            return False

        # Check if both agents have a path to a goal
        for start in starts:
            if not any(can_reach_goal(start, [goal]) for goal in goals):
                return False
   
        return True


    def randomize_map(self):
        valid = False
        while not valid:
            tmp_map = generate_random_map(size=self.size, p=0.7, seed=random.randint(0, 1000))
            if self.is_ma:
                tmp_map[0][-1] = 'S'
                #tmp_map[-1][0] = 'G'
                flat_board = [char for row in tmp_map for char in row]

                # Step 2: Find all indices of 'F' and 'H'
                indices = [i for i, char in enumerate(flat_board) if char in ['F', 'H']]

                # Step 3: Randomly select one index
                if indices:  # Check if there are any 'F' or 'H' cells
                    random_index = random.choice(indices)
                    flat_board[random_index] = 'P'

                # Step 4: Rebuild the board with the substituted character
                tmp_map = ["".join(flat_board[i:i+len(board[0])]) for i in range(0, len(flat_board), len(board[0]))]
                valid = is_valid(tmp_map, self.size)
            else:
                valid = True
        self.current_map = tmp_map
        self.current_map_id = "".join(self.current_map)
        
        
    def reset(self, **kwargs):
        if self.is_random:
            self.randomize_map()
        observation, info = self.env.reset(**kwargs)
        self.current_action_str = f"{observation}_1"
        state = self.get_state()
        # state = state.flatten()
        self.episodes += 1
        return state, info
    
    def get_state(self):
        key = f"{self.current_map_id}_{self.current_action_str}"
        if key not in self.states:
            img = render_utils.render_img_and_embedding(self.current_map_id, self.current_action_str, self.mode)
            self.states[key] = img
            np.save(f"{self.state_path}/{self.current_map_id}_{self.current_action_str}", img)
        return self.states[key]

    def get_platform_position(self):
        _, _, _, platform, _ = render_utils.parse_grid(self.current_map_id)
        return platform
    
    def step(self, action):
        next_observation, reward, termination, truncations, infos = self.env.step(action)
        self.current_action_str = f"{next_observation}_{action}"
        state = self.get_state()
        if self.data_cache is not None:
            x, y = int(next_observation % self.size), int(next_observation / self.size)
            self.data_cache.cache_policy(x, y, action)
        return state, reward, termination, truncations, infos
    
    def render(self):
        return render_utils.render_arr(self.current_map_id, self.current_action_str, self.mode)

    def close(self):
        self.env.close()
        
if __name__ == '__main__':
    env = LLMEnv('FrozenLake-v1', use_pre_computed_states=True)
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