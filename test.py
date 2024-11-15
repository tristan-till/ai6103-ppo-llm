from utils.helpers import load_npy_files_to_dict
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import gymnasium as gym
from classes.envs.img_env import ImgEnv
import numpy as np
import utils.agent as agent_utils
import torch
import utils.config as config_utils

# mode = 'light'
# states = load_npy_files_to_dict(f"./states/{mode}/flat_arr/")

# im = states['0_0']

# im = im.reshape((256, 256, 3))

# zoom_factors = (60 / 256, 128 / 60, 1)

# # Apply the zoom
# im = zoom(im, zoom_factors, order=1)  # order=1 for bilinear interpolation

# # im = im[::2, ::2, :]

# plt.imshow(im, cmap='gray')
# plt.title('Image 0_0')
# plt.show()

size = 4
# random_map = generate_random_map(size=size, p=0.7, seed = 2)
# env = gym.make('FrozenLake-v1', render_mode="rgb_array", is_slippery=False, desc=random_map)

# env = ImgEnv('FrozenLake-v1', run_name='v2', idx=0, capture_video=True, use_pre_computed_states=False, size=size)
# env.reset()


env = gym.vector.SyncVectorEnv([
        lambda idx=idx: ImgEnv('FrozenLake-v1', run_name='v5', idx=idx, capture_video=True, use_pre_computed_states=False, size=size)
        for idx in range(1)
    ])

env.reset()

config = config_utils.parse_config("hyp.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = agent_utils.get_agent(
    env,
    config['env']['type'], 
    config['optimization']['rpo_alpha'], 
    device
)

model_path = r"C:\Cyril\Work\NTU\ai6103-ppo-llm\runs\FrozenLake-v1__v1__1__1730189413\v1.rl_model"
agent.load_state_dict(torch.load(model_path, map_location=device))

action = [1]
termination = 0
while not termination:
    next_observation, reward, termination, truncations, infos = env.step(action)
    next_observation = next_observation.reshape(1, -1)
    next_observation = torch.as_tensor(next_observation, dtype=torch.float32, device=device)
    action, logprob = agent.sample_action_and_compute_log_prob(
                        next_observation
                    )
    action = action.cpu().numpy()

env.close()


# env.step(2)
# env.step(2)
# env.reset()
# env.step(2)

# plt.imshow(env.render())
# plt.title('FrozenLake Environment')
# plt.show()


# envs = gym.vector.SyncVectorEnv([
#         lambda idx=idx: ImgEnv('FrozenLake-v1', run_name='v2', idx=idx, capture_video=True, use_pre_computed_states=False, size=size)
#         for idx in range(1)
#     ])

# envs.reset()
# envs.step([2])
# envs.step([1])
# envs.step([1])

# envs.close()

# initial_observation, _ = envs.reset(seed=2)