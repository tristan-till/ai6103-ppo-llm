# main.py

from training.train_test import Model_TrainTest
from config import (
    TRAIN_MODE, SAVE_DIR, INPUT_DIM, SEED,
    ENV_NAME, MAP_SIZE, IS_SLIPPERY, MAX_EPISODE_STEPS,
    RENDER_MODE, VIDEO_FOLDER
)
import os

def main():
    # Define the local directory where models and plots will be saved
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = SAVE_DIR  # All models and plots will be saved here

    RL_hyperparams = {
        "train_mode": TRAIN_MODE,
        "RL_load_path": os.path.join(SAVE_DIR, 'final_weights.pth'),
        "save_path": SAVE_DIR,
        "save_interval": 1000,

        "clip_grad_norm": 5,
        "learning_rate": 1e-3,  # Increased learning rate
        "discount_factor": 0.99,  # Increased discount factor
        "batch_size": 128,  # Increased batch size
        "update_frequency": 100,  # Adjusted update frequency
        "max_episodes": 12000 if TRAIN_MODE else 5,  # Increased training episodes
        "max_steps": MAX_EPISODE_STEPS,
        "render": False,  # Rendering is handled by RecordVideo

        "epsilon_max": 1.0 if TRAIN_MODE else -1,  # Start with full exploration
        "epsilon_min": 0.01,
        "epsilon_decay": 0.9999,  # Slower decay for epsilon

        "memory_capacity": 10000 if TRAIN_MODE else 0,  # Increased memory capacity

        "map_size": MAP_SIZE,
        "num_states": MAP_SIZE ** 2,
        "render_fps": 6,
        "input_dim": INPUT_DIM,  # Embedding dimension
    }

    # Initialize and run training or testing
    DRL = Model_TrainTest(RL_hyperparams)
    if TRAIN_MODE:
        DRL.train()
    else:
        DRL.test(max_episodes=RL_hyperparams['max_episodes'])

if __name__ == '__main__':
    main()
