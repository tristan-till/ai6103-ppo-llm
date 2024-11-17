import os
import numpy as np
import time

def main(path):
    names = os.listdir(f'states/llmv2_env/{path}/')
    keys = [file.strip('.npy').split("_") for file in names]
    states = [key[0] for key in keys]
    state_set = set(states)
    print(len(state_set))
    state_pos = [f"{key[0]}_{key[1]}" for key in keys]
    counter = 0
    state_pos_set = set(state_pos)
    print(len(state_pos_set))
    print(len(state_pos_set) / len(state_set))
    return
    for state in state_set:
        for i in range(4):
            if os.path.isfile(f'states/llmv2_env/{path}/{state}_{i}.npy'):
                arr = np.load(f'states/llmv2_env/{path}/{state}_{i}.npy')
                break
        for i in range(4):
            if not os.path.isfile(f'states/llmv2_env/{path}/{state}_{i}.npy'):
                print(f"saved new {path} state {state}_{i}.npy")
                np.save(f'states/llmv2_env/{path}/{state}_{i}.npy', arr)
                counter += 1

    print(f"generated {counter} new states")

import utils.render_env as render_env
import utils.enums as enums
import utils.env as env_utils

def get_custom_state_str(grid, agent):
    agent_tile = render_env.get_agent_tile(int(agent), 4)
    _, _, _, holes = render_env.parse_grid(grid)
    agent_x, agent_y = agent_tile
    embedd_str = "Agent "
    embedd_str += "top " if agent_y < 2 else "bottom "
    embedd_str += "left, " if agent_x < 2 else "right, "

    for hole in holes:
        hole_x, hole_y = hole
        if agent_x == hole_x - 1 and agent_y == hole_y:
            embedd_str += "lake right of agent, "
        elif agent_x == hole_x + 1 and agent_y == hole_y:
            embedd_str += "Lake left of player, "
        elif agent_y == hole_y - 1 and agent_x == hole_x:
            embedd_str += "lake below agent, "
        elif agent_y == hole_x + 1 and agent_x == hole_x:
            embedd_str += "lake above agent, "

    embedd_str += "present bottom right."
    return embedd_str

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

def custom_embeddings():
    modes = [enums.EnvMode.TRAIN, enums.EnvMode.VAL]
    mode_strs = {mode: env_utils.mode_str_from_enum(mode) for mode in modes}
    names = os.listdir(f'states/llmv2_env/training/')
    keys = [file.strip('.npy').split("_") for file in names]
    states = list(set([key[0] for key in keys]))
    total = len(states) * 16 * 4
    i = 0
    
    def process_state(grid, agent, direction):
        agent_d = f"{agent}_{direction}"
        state_str = f"{grid}_{agent_d}"
        embedd_str = get_custom_state_str(grid, agent)
        embedding = render_env.encode_str(embedd_str)

        for mode in modes:
            render = render_env.render_arr(grid, agent_d, mode)
            arr = np.append(render.flatten(), embedding)
            save_path = f"states/demo1/{mode_strs[mode]}/{state_str}.npy"
            np.save(save_path, arr)
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for grid in states:
            for agent in range(16):
                for direction in range(4):
                    futures.append(executor.submit(process_state, grid, agent, direction))
        
        # Wait for all threads to complete
        for future in as_completed(futures):
            future.result()
            i += 1
            print(f"{i}/{total}")

    
if __name__ == '__main__':
    custom_embeddings()
