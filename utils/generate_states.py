import gymnasium as gym
import numpy as np
from PIL import Image
import io
import base64
import os
from collections import deque
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.req as req
from classes.sentence_transformer import SentenceTransformer


folder_root="states"
base64_folder = f"{folder_root}/base64"
image_folder = f"{folder_root}/images"
llava_folder = f"{folder_root}/llava"
llama_folder = f"{folder_root}/llama"
embedding_folder = f"{folder_root}/embedding"

if not os.path.exists(base64_folder):
    os.makedirs(base64_folder)
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

def save_state_arr_base64(state, orientation, state_arr):
    base64_filename = f"{state}_{orientation}.txt"
    if base64_filename in os.listdir(base64_folder):
        return
    image = Image.fromarray(state_arr)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_filepath = os.path.join(base64_folder, base64_filename)
    with open(base64_filepath, 'w') as f:
        f.write(img_base64)

def save_state_arr_image(state, orientation, state_arr, ):
    image_filename = f"{state}_{orientation}.png"
    if image_filename in os.listdir(image_folder):
        return
    image = Image.fromarray(state_arr)
    image_filepath = os.path.join(image_folder, image_filename)
    image.save(image_filepath, format="PNG")
    
def random_explore(env):
    i = 0
    _, _ = env.reset()
    while i < 100:
        action = orientation = np.random.choice([0, 1, 2, 3])
        state, reward, done, _, _ = env.step(action)
        state_arr = env.render()
        save_state_arr_image(state, orientation, state_arr)
        save_state_arr_base64(state, orientation, state_arr)
        if reward > 1.0 or done:
            state, _ = env.reset()

def save_llava_from_base64():
    if not os.path.exists(llava_folder):
        os.makedirs(llava_folder)

    for filename in os.listdir(base64_folder):
        if filename.endswith(".txt"):
            base64_file_path = os.path.join(base64_folder, filename)
            with open(base64_file_path, 'r') as f:
                txt = f.read()
            res = req.get_llava(txt)
            llava_file_path = os.path.join(llava_folder, filename)
            with open(llava_file_path, 'w') as llava_f:
                llava_f.write(res)
            print(f"Processed {filename} and saved response to {llava_file_path}")
            
def save_llama_from_llava():
    if not os.path.exists(llama_folder):
        os.makedirs(llama_folder)

    for filename in os.listdir(llava_folder):
        if filename.endswith(".txt"):
            llava_file_path = os.path.join(llava_folder, filename)
            with open(llava_file_path, 'r') as f:
                txt = f.read()
            res = req.get_llama(txt)
            llama_file_path = os.path.join(llama_folder, filename)
            with open(llama_file_path, 'w') as llama_f:
                llama_f.write(res)
            print(f"Processed {filename} and saved response to {llama_file_path}")
            
def save_embedding_from_llama():
    if not os.path.exists(embedding_folder):
        os.makedirs(embedding_folder)
        
    sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")

    for filename in os.listdir(llama_folder):
        if filename.endswith(".txt"):
            llama_file_path = os.path.join(llama_folder, filename)
            with open(llama_file_path, 'r') as f:
                txt = f.read()
            embedding = sentence_transformer.encode(txt)
            embedding_file_path = os.path.join(embedding_folder, filename.split(".")[0])
            np.save(f'{embedding_file_path}.npy', embedding)
            print(f"Processed {filename} and saved response to {embedding_file_path}.npy")

def main():
    # env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="rgb_array")
    # random_explore(env)
    # save_llava_from_base64()
    # save_llama_from_llava()
    save_embedding_from_llama()

if __name__ == '__main__':
    main()