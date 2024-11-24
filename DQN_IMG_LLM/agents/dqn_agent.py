# agents/dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.replay_memory import ReplayMemory
from models.dqn_network import DQN_Network
from config import DEVICE, SEED
from utils.logger import logger  # Import the logger

class DQN_Agent:
    """
    DQN Agent Class. This class defines key elements of the DQN algorithm.
    """
    def __init__(self, env, epsilon_max, epsilon_min, epsilon_decay,
                 clip_grad_norm, learning_rate, discount, memory_capacity, input_dim):
        # To save the history of network loss and Q-value trends
        self.loss_history = []
        self.q_value_history = []  # Track average and max Q-values
        self.q_value_history_episode = []  # To track Q-values per episode
        self.running_loss = 0
        self.learned_counts = 0

        # RL hyperparameters
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount

        self.action_space = env.action_space
        self.action_space.seed(SEED)  # Set the seed for reproducibility
        self.observation_space = env.observation_space
        self.replay_memory = ReplayMemory(memory_capacity)

        # Initiate the network models
        self.main_network = DQN_Network(num_actions=self.action_space.n, input_dim=input_dim).to(DEVICE)
        self.target_network = DQN_Network(num_actions=self.action_space.n, input_dim=input_dim).to(DEVICE).eval()
        self.target_network.load_state_dict(self.main_network.state_dict())

        self.clip_grad_norm = clip_grad_norm  # For clipping exploding gradients
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)

        logger.info("DQN Agent initialized successfully.")

    def select_action(self, state):
        """
        Selects an action using epsilon-greedy strategy or based on the Q-values.
        """
        # Exploration: epsilon-greedy
        if np.random.random() < self.epsilon_max:
            action = self.action_space.sample()
            logger.debug(f"Selected random action: {action}")
            return action

        # Exploitation: the action is selected based on the Q-values.
        with torch.no_grad():
            self.main_network.eval()  # Set to evaluation mode
            Q_values = self.main_network(state.unsqueeze(0))
            action = torch.argmax(Q_values).item()
            self.main_network.train()  # Switch back to training mode
            logger.debug(f"Selected action based on Q-values: {action}")
            return action

    def learn(self, batch_size, done):
        """
        Train the main network using a batch of experiences sampled from the replay memory.
        """
        if len(self.replay_memory) < batch_size:
            logger.debug("Not enough samples to learn.")
            return  # Not enough samples to learn

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

        # Track Q-value statistics
        avg_q_value = predicted_q.mean().item()
        max_q_value = predicted_q.max().item()
        self.q_value_history.append((avg_q_value, max_q_value))
        self.q_value_history_episode.append((avg_q_value, max_q_value))

        # Update the running loss and learned counts for logging and plotting
        self.running_loss += loss.item()
        self.learned_counts += 1

        if done:
            # Calculate average and max Q-values for the episode
            if self.q_value_history_episode:
                episode_avg_q = np.mean([q[0] for q in self.q_value_history_episode])
                episode_max_q = np.max([q[1] for q in self.q_value_history_episode])
                self.q_value_history.append((episode_avg_q, episode_max_q))
                logger.info(f"Episode Q-values - Avg: {episode_avg_q:.4f}, Max: {episode_max_q:.4f}")
            self.q_value_history_episode = []  # Reset for next episode

            episode_loss = self.running_loss / self.learned_counts  # The average loss for the episode
            self.loss_history.append(episode_loss)  # Append the episode loss to the loss history for plotting
            # Reset the running loss and learned counts
            self.running_loss = 0
            self.learned_counts = 0
            logger.info(f"Episode loss: {episode_loss:.4f}")

        self.optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Perform backward pass and update the gradients

        # Clip the gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)

        self.optimizer.step()  # Update the parameters of the main network using the optimizer

        logger.debug(f"Training step completed. Loss: {loss.item():.4f}")

    def hard_update(self):
        """
        Update the target network parameters by directly copying from the main network.
        """
        self.target_network.load_state_dict(self.main_network.state_dict())
        logger.info("Target network updated.")

    def update_epsilon(self):
        """
        Update the value of epsilon for epsilon-greedy exploration.
        """
        old_epsilon = self.epsilon_max
        self.epsilon_max = max(self.epsilon_min, self.epsilon_max * self.epsilon_decay)
        logger.info(f"Epsilon updated from {old_epsilon:.4f} to {self.epsilon_max:.4f}")

    def save(self, path):
        """
        Save the parameters of the main network to a file with .pth extension.
        """
        torch.save(self.main_network.state_dict(), path)
        logger.info(f"Model saved at {path}")
