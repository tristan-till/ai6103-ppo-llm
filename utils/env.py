import gymnasium as gym
import numpy as np

from classes.envs.llm_env import LLMEnv
from classes.envs.img_env import ImgEnv
import utils.helpers as helpers
import utils.enums as enums

def make_discrete_env(env_id, idx, capture_video, run_name):
    if capture_video and idx == 0:
        if env_id == 'FrozenLake-v1':
            env = gym.make(env_id, render_mode="rgb_array", is_slippery=False)
        else:
            env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        if env_id == 'FrozenLake-v1':
            env = gym.make(env_id, is_slippery=False)
        else:
            env = gym.make(env_id)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

def make_continuous_env(env_id, idx, capture_video, run_name, gamma):
    if capture_video and idx == 0:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gym.make(env_id)
    env = gym.wrappers.FlattenObservation(
        env
    ) 
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env

def make_continuous_envs(env_id, capture_video, run_name, num_envs, gamma):
    envs = gym.vector.SyncVectorEnv(
        [
            make_continuous_env(env_id, i, capture_video, run_name, gamma)
            for i in range(num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

def make_discrete_envs(env_id, capture_video, run_name, num_envs):
    envs = gym.vector.SyncVectorEnv(
        lambda: make_discrete_env(env_id, i, capture_video, run_name) for i in range(num_envs)
    )
    assert isinstance(
            envs.single_action_space, gym.spaces.Discrete
        ), "only discrete action space is supported"
    return envs

def make_llm_envs(env_id, capture_video, run_name, num_envs):
    envs = gym.vector.SyncVectorEnv([
        lambda: LLMEnv(env_id, run_name=run_name, capture_video=capture_video) for _ in range(num_envs)
    ])
    return envs

def make_img_envs(env_id, capture_video, run_name, num_envs, use_pre_computed_states, size):
    envs = gym.vector.SyncVectorEnv([
        lambda idx=idx: ImgEnv(env_id, run_name=run_name, idx=idx, capture_video=capture_video, use_pre_computed_states=use_pre_computed_states, size=size)
        for idx in range(num_envs)
    ])
    return envs

def create_envs(config):
    _, run_name = helpers.get_run_name(config)
    env_id = config['env']['id']
    capture_video = config['simulation']['capture_video']
    num_envs = config['training']['num_envs']
    env_type = config['env']['type']
    if env_type == enums.EnvType.LLM.value:
        return make_llm_envs(env_id, capture_video, run_name, num_envs)
    elif env_type  == enums.EnvType.DISCRETE.value:
        return make_discrete_envs(env_id, capture_video, run_name, num_envs)
    elif env_type  == enums.EnvType.CONTINUOUS.value:
        return make_continuous_envs(env_id, capture_video, run_name, num_envs, config['optimization']['gamma'])
    elif env_type  == enums.EnvType.IMG.value:
        return make_img_envs(env_id, capture_video, run_name, num_envs, use_pre_computed_states = config['simulation']['use_pre_computed_states'], size = config['env']['size'])