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

import matplotlib.pyplot as plt

def cosine_distances():
    original_grid = "SFFFFHFHFFFHHFFG"
    embeddings = []
    
    # Generate embeddings for each agent
    for i in range(16):
        agent = i
        state_str = get_custom_state_str(original_grid, agent)
        embedding = render_env.encode_str(state_str)
        embeddings.append(embedding)
    
    print(len(embeddings))
    similarity_matrix = np.zeros((16, 16))
    from PIL import Image
    img = render_env.render_state(original_grid, "0_1", enums.EnvMode.TRAIN)
    img = img.convert("RGB")
    img = img.resize((256, 256), Image.NEAREST)
    bg_image = img.transpose(Image.FLIP_LEFT_RIGHT)
        
    # Convert the flipped image to a format suitable for imshow
    bg_image = np.array(bg_image)
    fig, ax = plt.subplots()

    # Display the background image
    ax.imshow(bg_image, extent=[3.5, -0.5, 3.5, -0.5], aspect='auto', zorder=0)
    
    # Calculate cosine similarity between embeddings
    for i, embedding in enumerate(embeddings):
        for j, other_embedding in enumerate(embeddings):
            cos_sim = np.dot(embedding, other_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(other_embedding)
            )
            similarity_matrix[i, j] = cos_sim
        
        pos_dist = similarity_matrix[i]
        pos_dist = np.reshape(pos_dist, (4, 4))
        
        # Create a heatmap with annotations
        heatmap = ax.imshow(pos_dist, cmap="coolwarm", interpolation="nearest", vmin=0, vmax=1, alpha=0.5)
        
        # Add color bar
        plt.colorbar(heatmap)
        
        # Annotate each cell with the numeric value
        for (j, k), val in np.ndenumerate(pos_dist):
            plt.text(k, j, f"{val:.2f}", ha='center', va='center', color='black')
        
        plt.title("Cosine Similarity Heatmap Between States")
        plt.xlabel("State Index")
        plt.ylabel("State Index")
        plt.xticks(ticks=np.arange(4), labels=np.arange(4))
        plt.yticks(ticks=np.arange(4), labels=np.arange(4))
        
        plt.show()
        return
    

if __name__ == '__main__':
    # custom_embeddings()
    cosine_distances()
