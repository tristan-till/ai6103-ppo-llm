# config.py

import torch

# ===========================
# Device Configuration
# ===========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# Seed Configuration
# ===========================
SEED = 2024

# ===========================
# Ollama API Configuration
# ===========================
OLLAMA_BASE_URL = 'http://localhost:11434/api/generate'  # Replace with your Ollama API endpoint

# ===========================
# Environment Configuration
# ===========================
ENV_NAME = 'FrozenLake-v1'
MAP_SIZE = 4  # 4x4 or 8x8
IS_SLIPPERY = False
MAX_EPISODE_STEPS = 100
RENDER_MODE = "rgb_array"

# ===========================
# Training Hyperparameters
# ===========================
TRAIN_MODE = True
RENDER = False  # Rendering is handled by RecordVideo
SAVE_INTERVAL = 500
LEARNING_RATE = 1e-3
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 128
UPDATE_FREQUENCY = 100
MAX_EPISODES = 10000 if TRAIN_MODE else 5
MAX_STEPS = 100
EPSILON_MAX = 1.0 if TRAIN_MODE else -1
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9999
MEMORY_CAPACITY = 10000 if TRAIN_MODE else 0
INPUT_DIM = 384  # Embedding dimension

# ===========================
# Save Configuration
# ===========================
SAVE_DIR = 'DQN_Models_LLM'
VIDEO_FOLDER = 'videos'

# ===========================
# Plot Configuration
# ===========================
PLOT_DPI = 600
