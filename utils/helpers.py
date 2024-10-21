import os
import numpy as np
import time

import random
import torch

def load_npy_files_to_dict(folder_path="./states/embedding/"):
    """
    Load all .npy files from a folder into memory and store them in a dictionary.
    
    Args:
    folder_path (str): Path to the folder containing .npy files.
    
    Returns:
    dict: Dictionary with filename as key and NumPy array as value.
    """
    
    # Initialize an empty dictionary to store file contents
    file_dict = {}
    
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Check if it's a .npy file
        if filename.endswith('.npy'):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            
            try:
                # Load the .npy file into a NumPy array
                array = np.load(file_path)
                key = filename.split('.')[0]
                # Store the NumPy array in the dictionary
                file_dict[key] = array
            except Exception as e:
                print(f"Error loading file {filename}: {str(e)}")
    
    return file_dict

def get_run_name(config):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    run_name = f"{config['env']['id']}__{exp_name}__{config['simulation']['seed']}__{int(time.time())}"
    return exp_name, run_name

def set_seed(seed, torch_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    
if __name__ == '__main__':
    folder_path = './states/embedding/'
    file_dict = load_npy_files_to_dict(folder_path)
    assert len(file_dict.keys()) > 0, "No files found in the folder."
    print(file_dict.keys())
    print(file_dict['0'])
    file_dict.get(f"{0}_{0}", np.zeros(384))