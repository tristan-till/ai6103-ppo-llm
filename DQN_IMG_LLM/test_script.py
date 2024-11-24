# Install necessary packages
# Note: Uncomment the following lines if you haven't installed these packages locally.
# You can run these commands in your terminal or command prompt.
# pip install torch torchvision torchaudio
# pip install gymnasium
# pip install pygame
# pip install matplotlib
# pip install sentence-transformers
# pip install requests

# Import necessary libraries
import os
import gc
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import requests
import json
import io
import base64
from PIL import Image
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clear cache
gc.collect()
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Used for debugging; CUDA related errors shown immediately.

# Seed everything for reproducible results
seed = 2024
np.random.seed(seed)
np.random.default_rng(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------- Utility Functions --------------------

# Implement necessary utility functions since utils.render_env and other utils are not available.

def get_agent_tile(agent_id, grid_size):
    """
    Determines the (x, y) position of the agent based on agent_id and grid_size.
    Assumes agent_id ranges from 0 to grid_size^2 - 1.
    """
    x = agent_id % grid_size
    y = agent_id // grid_size
    return (x, y)

def parse_grid(grid):
    """
    Parses the grid to identify holes (lakes).
    Assumes grid is a string representing the environment state.
    """
    # For FrozenLake, grid is a string of characters representing the grid.
    # Example for 4x4: 'SFFF', 'FHFH', 'FFFH', 'HFFG'
    grid_size = int(np.sqrt(len(grid)))
    holes = []
    for i, cell in enumerate(grid):
        if cell == 'H':
            x = i % grid_size
            y = i // grid_size
            holes.append((x, y))
    return holes

def get_custom_state_str(grid, agent_id, grid_size=4):
    """
    Generates a custom descriptive string based on the agent's position and nearby lakes.
    """
    agent_tile = get_agent_tile(agent_id, grid_size)
    holes = parse_grid(grid)
    agent_x, agent_y = agent_tile
    embedd_str = "Agent "
    embedd_str += "top " if agent_y < grid_size / 2 else "bottom "
    embedd_str += "left, " if agent_x < grid_size / 2 else "right, "

    for hole in holes:
        hole_x, hole_y = hole
        if agent_x == hole_x - 1 and agent_y == hole_y:
            embedd_str += "lake right of agent, "
        elif agent_x == hole_x + 1 and agent_y == hole_y:
            embedd_str += "lake left of agent, "
        elif agent_y == hole_y - 1 and agent_x == hole_x:
            embedd_str += "lake below agent, "
        elif agent_y == hole_y + 1 and agent_x == hole_x:
            embedd_str += "lake above agent, "

    embedd_str += "present bottom right."
    return embedd_str

def encode_str(embedd_str, sentence_transformer):
    """
    Encodes the descriptive string into a numerical embedding.
    """
    embedding = sentence_transformer.encode(embedd_str)
    # Normalize the embedding
    embedding = (embedding - np.mean(embedding)) / (np.std(embedding) + 1e-8)
    return embedding

def render_arr(grid, agent_d, mode):
    """
    Renders the environment array based on grid, agent position, and mode.
    Placeholder function: Replace with actual rendering logic if needed.
    """
    # For simplicity, we'll create a dummy array. Replace with actual rendering if necessary.
    arr = np.zeros(100)  # Example: Flattened 10x10 grid
    return arr

# -------------------- DQN Components --------------------

# Define ReplayMemory class
class ReplayMemory:
    def __init__(self, capacity):
        """
        Experience Replay Memory defined by deques to store transitions/agent experiences
        """
        self.capacity = capacity
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)

    def store(self, state, action, next_state, reward, done):
        """
        Append (store) the transitions to their respective deques
        """
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)

    def sample(self, batch_size):
        """
        Randomly sample transitions from memory, then convert sampled transitions
        to tensors and move to device (CPU or GPU).
        """
        indices = np.random.choice(len(self), size=batch_size, replace=False)

        states = torch.stack([self.states[i] for i in indices]).to(device)
        actions = torch.as_tensor([self.actions[i] for i in indices], dtype=torch.long, device=device)
        next_states = torch.stack([self.next_states[i] for i in indices]).to(device)
        rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=device)
        dones = torch.as_tensor([self.dones[i] for i in indices], dtype=torch.bool, device=device)

        return states, actions, next_states, rewards, dones

    def __len__(self):
        """
        To check how many samples are stored in the memory.
        """
        return len(self.dones)

# Define DQN_Network class
class DQN_Network(nn.Module):
    """
    The Deep Q-Network (DQN) model for reinforcement learning.
    """
    def __init__(self, num_actions, input_dim):
        """
        Initialize the DQN network.

        Parameters:
            num_actions (int): The number of possible actions in the environment.
            input_dim (int): The dimensionality of the input state space.
        """
        super(DQN_Network, self).__init__()

        self.FC = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, num_actions)
        )

        # Initialize FC layer weights using He initialization
        for module in self.FC:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        """
        Forward pass of the network to find the Q-values of the actions.

        Parameters:
            x (torch.Tensor): Input tensor representing the state.

        Returns:
            Q (torch.Tensor): Tensor containing Q-values for each action.
        """
        Q = self.FC(x)
        return Q

# Define DQN_Agent class
class DQN_Agent:
    """
    DQN Agent Class. This class defines key elements of the DQN algorithm.
    """
    def __init__(self, env, epsilon_max, epsilon_min, epsilon_decay,
                 clip_grad_norm, learning_rate, discount, memory_capacity, input_dim):
        # To save the history of network loss
        self.loss_history = []
        self.running_loss = 0
        self.learned_counts = 0

        # RL hyperparameters
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount

        self.action_space = env.action_space
        self.action_space.seed(seed)  # Set the seed to get reproducible results when sampling the action space
        self.observation_space = env.observation_space
        self.replay_memory = ReplayMemory(memory_capacity)

        # Initiate the network models
        self.main_network = DQN_Network(num_actions=self.action_space.n, input_dim=input_dim).to(device)
        self.target_network = DQN_Network(num_actions=self.action_space.n, input_dim=input_dim).to(device).eval()
        self.target_network.load_state_dict(self.main_network.state_dict())

        self.clip_grad_norm = clip_grad_norm  # For clipping exploding gradients
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)

    def select_action(self, state):
        """
        Selects an action using epsilon-greedy strategy or based on the Q-values.

        Parameters:
            state (torch.Tensor): Input tensor representing the state.

        Returns:
            action (int): The selected action.
        """
        # Exploration: epsilon-greedy
        if np.random.random() < self.epsilon_max:
            return self.action_space.sample()

        # Exploitation: the action is selected based on the Q-values.
        with torch.no_grad():
            self.main_network.eval()  # Set to evaluation mode
            Q_values = self.main_network(state.unsqueeze(0))
            action = torch.argmax(Q_values).item()
            self.main_network.train()  # Switch back to training mode
            return action

    def learn(self, batch_size, done):
        """
        Train the main network using a batch of experiences sampled from the replay memory.
        """
        # Sample a batch of experiences from the replay memory
        states, actions, next_states, rewards, dones = self.replay_memory.sample(batch_size)

        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        predicted_q = self.main_network(states)
        predicted_q = predicted_q.gather(dim=1, index=actions)

        with torch.no_grad():
            next_target_q_value = self.target_network(next_states).max(dim=1, keepdim=True)[0]

        next_target_q_value[dones] = 0  # Set the Q-value for terminal states to zero
        y_js = rewards + (self.discount * next_target_q_value)
        loss = self.criterion(predicted_q, y_js)

        # Update the running loss and learned counts for logging and plotting
        self.running_loss += loss.item()
        self.learned_counts += 1

        if done:
            episode_loss = self.running_loss / self.learned_counts  # The average loss for the episode
            self.loss_history.append(episode_loss)  # Append the episode loss to the loss history for plotting
            # Reset the running loss and learned counts
            self.running_loss = 0
            self.learned_counts = 0

        self.optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Perform backward pass and update the gradients

        # Clip the gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)

        self.optimizer.step()  # Update the parameters of the main network using the optimizer

    def hard_update(self):
        """
        Update the target network parameters by directly copying from the main network.
        """
        self.target_network.load_state_dict(self.main_network.state_dict())

    def update_epsilon(self):
        """
        Update the value of epsilon for epsilon-greedy exploration.
        """
        self.epsilon_max = max(self.epsilon_min, self.epsilon_max * self.epsilon_decay)

    def save(self, path):
        """
        Save the parameters of the main network to a file with .pth extension.
        """
        torch.save(self.main_network.state_dict(), path)

# Define Model_TrainTest class
class Model_TrainTest:
    def __init__(self, hyperparams):
        # Define RL Hyperparameters
        self.train_mode = hyperparams["train_mode"]
        self.RL_load_path = hyperparams["RL_load_path"]
        self.save_path = hyperparams["save_path"]
        self.save_interval = hyperparams["save_interval"]

        self.clip_grad_norm = hyperparams["clip_grad_norm"]
        self.learning_rate = hyperparams["learning_rate"]
        self.discount_factor = hyperparams["discount_factor"]
        self.batch_size = hyperparams["batch_size"]
        self.update_frequency = hyperparams["update_frequency"]
        self.max_episodes = hyperparams["max_episodes"]
        self.max_steps = hyperparams["max_steps"]
        self.render = hyperparams["render"]

        self.epsilon_max = hyperparams["epsilon_max"]
        self.epsilon_min = hyperparams["epsilon_min"]
        self.epsilon_decay = hyperparams["epsilon_decay"]

        self.memory_capacity = hyperparams["memory_capacity"]

        self.num_states = hyperparams["num_states"]
        self.map_size = hyperparams["map_size"]
        self.render_fps = hyperparams["render_fps"]
        self.input_dim = hyperparams["input_dim"]  # Embedding dimension (384)

        # Define local directories for saving models and plots
        self.local_save_dir = os.path.join(os.getcwd(), 'DQN_Models_LLM')
        self.models_dir = os.path.join(self.local_save_dir, 'models')
        self.plots_dir = os.path.join(self.local_save_dir, 'plots')

        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        # Initialize log files for descriptions and summaries
        self.descriptions_log = os.path.join(self.local_save_dir, 'descriptions.log')
        self.summaries_log = os.path.join(self.local_save_dir, 'summaries.log')

        # Clear existing log files or create new ones
        with open(self.descriptions_log, 'w') as f:
            f.write("Episode, Step, State, Description\n")
        with open(self.summaries_log, 'w') as f:
            f.write("Episode, Step, State, Summary\n")

        # Define Env
        self.env = gym.make('FrozenLake-v1', map_name=f"{self.map_size}x{self.map_size}",
                            is_slippery=False, max_episode_steps=self.max_steps,
                            render_mode="rgb_array")

        # Initialize the sentence transformer model
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

        # Define the agent class
        self.agent = DQN_Agent(env=self.env,
                               epsilon_max=self.epsilon_max,
                               epsilon_min=self.epsilon_min,
                               epsilon_decay=self.epsilon_decay,
                               clip_grad_norm=self.clip_grad_norm,
                               learning_rate=self.learning_rate,
                               discount=self.discount_factor,
                               memory_capacity=self.memory_capacity,
                               input_dim=self.input_dim)

        # Initialize state embedding cache
        self.state_embedding_cache = {}

        # Define the grid map based on map_size
        self.grid_map = self.define_grid_map()

    def define_grid_map(self):
        """
        Defines the grid map based on the map_size.
        Returns a list of strings representing each row of the grid.
        """
        if self.map_size == 4:
            grid_map = [
                'SFFF',
                'FHFH',
                'FFFH',
                'HFFG'
            ]
        elif self.map_size == 8:
            grid_map = [
                'SFFFFFFF',
                'FHHHHFHF',
                'FFFHFFFF',
                'FFFFHFHF',
                'FFFFFHHF',
                'FHHFFFFF',
                'FFFFFHFH',
                'HFFFFFGF'
            ]
        else:
            raise ValueError("Unsupported map size. Please use 4x4 or 8x8.")
        return grid_map

    def decode_state(self, state):
        """
        Maps the state index to the grid string.

        Parameters:
            state (int): The current state in the environment.

        Returns:
            grid_str (str): The grid string representing the environment's state.
        """
        # Reconstruct the grid string based on the state index
        # The grid is static; only the agent's position changes
        grid_str = ''.join(self.grid_map)
        return grid_str

    def parse_summary_for_rewards(self, description):
        """
        Parses the description and assigns additional rewards or penalties.

        Parameters:
            description (str): The descriptive string of the current state.

        Returns:
            additional_reward (float): The additional reward to be added to the environment's reward.
        """
        additional_reward = 0.0
        summary = description.lower()

        # Assign positive reward for being near a present
        if 'present' in summary:
            additional_reward += 0.1  # Positive reward

        # Assign negative reward for being near a lake
        if 'lake' in summary:
            additional_reward -= 0.1  # Negative reward

        return additional_reward

    def state_preprocess(self, state, num_states, episode=None, step=None):
        """
        Convert the state to an embedding using custom state descriptions.

        Parameters:
            state (int): The current state in the environment.
            num_states (int): Total number of states in the environment.
            episode (int): Current episode number (for logging).
            step (int): Current step number within the episode (for logging).

        Returns:
            embedding_tensor (torch.Tensor): The embedding tensor for the current state.
        """
        # Check if the embedding is already cached
        if state in self.state_embedding_cache:
            return self.state_embedding_cache[state]

        # Get the grid string
        grid_str = self.decode_state(state)

        # Generate descriptive string
        description = get_custom_state_str(grid_str, state, grid_size=self.map_size)

        # Encode the descriptive string
        embedding = encode_str(description, self.sentence_transformer)

        # Convert embedding to tensor
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32, device=device)

        # Cache the embedding
        self.state_embedding_cache[state] = embedding_tensor

        # Print and save the state description
        if episode is not None and step is not None:
            print(f"Episode: {episode}, Step: {step}, State: {state}")
            print(f"Description: {description}")

            with open(self.descriptions_log, 'a') as f_desc, open(self.summaries_log, 'a') as f_sum:
                f_desc.write(f"{episode}, {step}, {state}, {description}\n")
                # Since we're using custom descriptions, summaries are the same
                f_sum.write(f"{episode}, {step}, {state}, {description}\n")

        return embedding_tensor

    def train(self):
        """
        Reinforcement learning training loop.
        """
        total_steps = 0
        self.reward_history = []

        # Training loop over episodes
        for episode in range(1, self.max_episodes + 1):
            state, _ = self.env.reset(seed=seed)
            state_emb = self.state_preprocess(state, num_states=self.num_states, episode=episode, step=0)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0

            while not done and not truncation and step_size < self.max_steps:
                action = self.agent.select_action(state_emb)
                next_state, reward, done, truncation, _ = self.env.step(action)
                step_size += 1

                # Preprocess the next state
                next_state_emb = self.state_preprocess(next_state, num_states=self.num_states, episode=episode, step=step_size)

                # Parse the description to assign additional rewards
                # Since we're using custom descriptions, embedd_str is same as description
                # Hence, we can directly use the description for reward shaping
                grid_str = self.decode_state(next_state)
                description = get_custom_state_str(grid_str, next_state, grid_size=self.map_size)
                additional_reward = self.parse_summary_for_rewards(description)
                combined_reward = reward + additional_reward

                # Debugging: Print the rewards
                print(f"Episode: {episode}, Step: {step_size}, Original Reward: {reward}, "
                      f"Additional Reward: {additional_reward}, Combined Reward: {combined_reward}")

                # Log the descriptions and summaries
                with open(self.descriptions_log, 'a') as f_desc, open(self.summaries_log, 'a') as f_sum:
                    f_desc.write(f"{episode}, {step_size}, {next_state}, {description}\n")
                    f_sum.write(f"{episode}, {step_size}, {next_state}, {description}\n")

                self.agent.replay_memory.store(state_emb, action, next_state_emb, combined_reward, done)

                if len(self.agent.replay_memory) > self.batch_size:
                    self.agent.learn(self.batch_size, (done or truncation))

                    # Update target-network weights
                    if total_steps % self.update_frequency == 0:
                        self.agent.hard_update()

                state_emb = next_state_emb
                episode_reward += combined_reward  # Use combined reward for tracking
                total_steps += 1

            # Append for tracking history
            self.reward_history.append(episode_reward)

            # Decay epsilon at the end of each episode
            self.agent.update_epsilon()

            # Save model at intervals
            if episode % self.save_interval == 0:
                # Define save paths locally
                save_path = os.path.join(self.models_dir, f'final_weights_{episode}.pth')
                self.agent.save(save_path)
                if episode != self.max_episodes:
                    self.plot_training(episode)
                print('\n~~~~~~Interval Save: Model saved.\n')

            # Print episode results
            result = (f"Episode: {episode}, "
                      f"Total Steps: {total_steps}, "
                      f"Ep Step: {step_size}, "
                      f"Raw Reward: {episode_reward:.2f}, "
                      f"Epsilon: {self.agent.epsilon_max:.4f}")
            print(result)
        self.plot_training(episode)

    def test(self, max_episodes):
        """
        Reinforcement learning policy evaluation.
        """
        # Load the weights of the test_network
        if not os.path.exists(self.RL_load_path):
            print(f"Model file {self.RL_load_path} does not exist.")
            return

        self.agent.main_network.load_state_dict(torch.load(self.RL_load_path, map_location=device))
        self.agent.main_network.eval()

        # Testing loop over episodes
        for episode in range(1, max_episodes + 1):
            state, _ = self.env.reset(seed=seed)
            state_emb = self.state_preprocess(state, num_states=self.num_states, episode=episode, step=0)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0

            while not done and not truncation and step_size < self.max_steps:
                action = self.agent.select_action(state_emb)
                next_state, reward, done, truncation, _ = self.env.step(action)
                step_size += 1
                next_state_emb = self.state_preprocess(next_state, num_states=self.num_states, episode=episode, step=step_size)

                state_emb = next_state_emb
                episode_reward += reward
                # For testing, we don't perform reward shaping

            # Print log
            result = (f"Test Episode: {episode}, "
                      f"Steps: {step_size}, "
                      f"Reward: {episode_reward:.2f}")
            print(result)

        # Close the environment
        self.env.close()

    def plot_training(self, episode):
        """
        Plots the training rewards and loss history.

        Parameters:
            episode (int): The current episode number.
        """
        # Calculate the Simple Moving Average (SMA) with a window size of 50
        window_size = 50
        if len(self.reward_history) >= window_size:
            sma = np.convolve(self.reward_history, np.ones(window_size) / window_size, mode='valid')
        else:
            sma = self.reward_history  # Not enough data for SMA

        plt.figure(figsize=(12, 5))

        # Plot Rewards
        plt.subplot(1, 2, 1)
        plt.title("Rewards")
        plt.plot(self.reward_history, label='Raw Reward', color='#F6CE3B', alpha=1)
        if len(sma) > 0:
            plt.plot(range(window_size - 1, len(self.reward_history)), sma, label=f'SMA {window_size}', color='#385DAA')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()
        plt.grid(True)

        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.title("Loss")
        plt.plot(self.agent.loss_history, label='Loss', color='#CB291A', alpha=1)
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        # Define plot file paths locally
        reward_plot_path = os.path.join(self.plots_dir, f'reward_plot_episode_{episode}.png')
        loss_plot_path = os.path.join(self.plots_dir, f'loss_plot_episode_{episode}.png')

        # Save plots locally
        plt.savefig(reward_plot_path, format='png', dpi=300, bbox_inches='tight')
        plt.savefig(loss_plot_path, format='png', dpi=300, bbox_inches='tight')

        plt.show()
        plt.clf()
        plt.close()

# -------------------- Main Execution --------------------

if __name__ == '__main__':
    # Parameters:
    train_mode = True  # Set to False to run testing
    render = False  # Disable rendering; set to True if you have a display and want to visualize
    map_size = 4  # 4x4 or 8x8

    # Define the directory where models and plots will be saved locally
    local_save_dir = os.path.join(os.getcwd(), 'DQN_Models_LLM')

    # Define hyperparameters
    RL_hyperparams = {
        "train_mode": train_mode,
        "RL_load_path": os.path.join(local_save_dir, 'models', 'final_weights_10000.pth'),  # Update as needed
        "save_path": os.path.join(local_save_dir, 'models', 'final_weights'),
        "save_interval": 500,

        "clip_grad_norm": 5,
        "learning_rate": 1e-3,  # Increased learning rate
        "discount_factor": 0.99,  # Increased discount factor
        "batch_size": 128,  # Increased batch size
        "update_frequency": 100,  # Adjusted update frequency
        "max_episodes": 10000 if train_mode else 5,  # Increased training episodes
        "max_steps": 100,
        "render": render,

        "epsilon_max": 1.0 if train_mode else -1,  # Start with full exploration
        "epsilon_min": 0.01,
        "epsilon_decay": 0.9999,  # Slower decay for epsilon

        "memory_capacity": 10000 if train_mode else 0,  # Increased memory capacity

        "map_size": map_size,
        "num_states": map_size ** 2,
        "render_fps": 6,
        "input_dim": 384,  # Embedding dimension
    }

    # Create save directories if they don't exist
    os.makedirs(RL_hyperparams["save_path"], exist_ok=True)

    # Initialize the training/testing class
    DRL = Model_TrainTest(RL_hyperparams)

    # Train or Test
    if train_mode:
        DRL.train()
    else:
        DRL.test(max_episodes=RL_hyperparams['max_episodes'])
