# utils/helpers.py

import matplotlib.pyplot as plt
import seaborn as sns  # Added seaborn for heatmap
import os
import torch
import numpy as np
from PIL import Image
import io
import base64
from sentence_transformers import SentenceTransformer
from api.ollama_api import get_llava, get_llama
from config import DEVICE

def plot_training(reward_history, loss_history, q_value_history, save_dir, episode, window_size=50, dpi=600):
    """
    Plots the training rewards, losses, and Q-value trends, and saves the plots.
    """
    # Filter positive rewards for plotting
    positive_rewards = [reward if reward > 0 else 0 for reward in reward_history]

    # Calculate the Simple Moving Average (SMA) for positive rewards
    if len(positive_rewards) >= window_size:
        sma = np.convolve(positive_rewards, np.ones(window_size) / window_size, mode='valid')
    else:
        sma = positive_rewards  # Not enough data for SMA

    plt.figure(figsize=(18, 5))

    # Plot Positive Rewards
    plt.subplot(1, 3, 1)
    plt.title("Positive Rewards")
    plt.plot(reward_history, label='Raw Reward (All)', color='gray', alpha=0.5)
    plt.plot(positive_rewards, label='Positive Reward', color='#F6CE3B', alpha=1)
    if len(sma) > 0:
        plt.plot(range(window_size - 1, len(positive_rewards)), sma, label=f'SMA {window_size}', color='#385DAA')
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 3, 2)
    plt.title("Loss")
    plt.plot(loss_history, label='Loss', color='#CB291A', alpha=1)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot Q-value Trends
    plt.subplot(1, 3, 3)
    plt.title("Q-value Trends")
    if q_value_history:
        avg_q = [q[0] for q in q_value_history]
        max_q = [q[1] for q in q_value_history]
        plt.plot(avg_q, label='Average Q-value', color='blue')
        plt.plot(max_q, label='Max Q-value', color='red')
    plt.xlabel("Learning Steps")
    plt.ylabel("Q-values")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save plots to the local save directory
    combined_plot_path = os.path.join(save_dir, f'combined_plots_episode_{episode}.png')
    plt.savefig(combined_plot_path, format='png', dpi=dpi, bbox_inches='tight')

    plt.show()
    plt.clf()
    plt.close()

def extract_q_values(agent, env, sentence_transformer, state_embedding_cache):
    """
    Extracts Q-values for all states and actions in the environment.
    Returns a matrix of shape (num_states, num_actions).
    """
    num_states = env.unwrapped.n  # Assuming discrete states
    num_actions = env.action_space.n
    q_values_matrix = np.zeros((num_states, num_actions))

    for state in range(num_states):
        # Preprocess the state to get its embedding
        state_emb = preprocess_state(state, env, sentence_transformer, state_embedding_cache)
        with torch.no_grad():
            q_values = agent.main_network(state_emb.unsqueeze(0)).cpu().numpy()[0]
        q_values_matrix[state] = q_values

    return q_values_matrix

def plot_q_value_heatmap(q_values_matrix, save_dir, map_size, dpi=600):
    """
    Plots and saves a heatmap of Q-values for all states and actions.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(q_values_matrix, annot=True, fmt=".2f", cmap='viridis',
                xticklabels=[f"Action {i}" for i in range(q_values_matrix.shape[1])],
                yticklabels=[f"State {i}" for i in range(q_values_matrix.shape[0])])
    plt.title("Q-value Heatmap")
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.tight_layout()

    heatmap_path = os.path.join(save_dir, 'q_value_heatmap.png')
    plt.savefig(heatmap_path, format='png', dpi=dpi, bbox_inches='tight')

    plt.show()
    plt.clf()
    plt.close()

def preprocess_state(state, env, sentence_transformer, state_embedding_cache):
    """
    Converts the state to an embedding using LLaVA and LLaMA models.
    """
    # Check if the embedding is already cached
    if state in state_embedding_cache:
        return state_embedding_cache[state]

    # Render the current state as an image
    image = env.render()

    # Convert the image to base64 encoding
    buffered = io.BytesIO()
    image = Image.fromarray(image)
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Get description from LLaVA
    llava_response = get_llava(img_base64)

    # Get summarized description from LLaMA
    llama_response = get_llama(llava_response)

    # Convert text to embedding
    embedding = sentence_transformer.encode(llama_response)

    # Normalize the embedding
    embedding = (embedding - np.mean(embedding)) / (np.std(embedding) + 1e-8)

    # Convert embedding to tensor
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32, device=DEVICE)

    # Cache the embedding
    state_embedding_cache[state] = embedding_tensor

    return embedding_tensor
