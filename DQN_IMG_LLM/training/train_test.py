# training/train_test.py

import os
from environments.reward_shaping_env import RewardShapingEnv
from agents.dqn_agent import DQN_Agent
from models.dqn_network import DQN_Network
from utils.helpers import plot_training, preprocess_state, extract_q_values, plot_q_value_heatmap
from sentence_transformers import SentenceTransformer
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
from config import (
    DEVICE, SEED, ENV_NAME, MAP_SIZE, IS_SLIPPERY, MAX_EPISODE_STEPS,
    RENDER_MODE, SAVE_DIR, VIDEO_FOLDER, INPUT_DIM
)

class Model_TrainTest:
    def __init__(self, hyperparams):
        # Unpack hyperparameters
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

        # Initialize the sentence transformer model
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

        # Define Env with video recording
        env = gym.make(
            ENV_NAME,
            map_name=f"{self.map_size}x{self.map_size}",
            is_slippery=IS_SLIPPERY,
            max_episode_steps=self.max_steps,
            render_mode=RENDER_MODE
        )
        # Apply Reward Shaping
        env = RewardShapingEnv(env)

        # Wrap the environment for video recording
        self.env = RecordVideo(
            env,
            video_folder=os.path.join(SAVE_DIR, VIDEO_FOLDER),
            episode_trigger=lambda episode_id: True if self.train_mode else False,
            name_prefix="train" if self.train_mode else "test"
        )

        # Define the agent class
        self.agent = DQN_Agent(
            env=self.env,
            epsilon_max=self.epsilon_max,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            clip_grad_norm=self.clip_grad_norm,
            learning_rate=self.learning_rate,
            discount=self.discount_factor,
            memory_capacity=self.memory_capacity,
            input_dim=self.input_dim
        )

        # Initialize state embedding cache
        self.state_embedding_cache = {}

        # Initialize reward history
        self.reward_history = []

    def train(self):
        """
        Reinforcement learning training loop.
        """
        total_steps = 0

        # Training loop over episodes
        for episode in range(1, self.max_episodes + 1):
            state, _ = self.env.reset(seed=SEED)
            state_emb = preprocess_state(
                state,
                self.env,
                self.sentence_transformer,
                self.state_embedding_cache
            )
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0

            while not done and not truncation:
                action = self.agent.select_action(state_emb)
                next_state, reward, done, truncation, _ = self.env.step(action)
                next_state_emb = preprocess_state(
                    next_state,
                    self.env,
                    self.sentence_transformer,
                    self.state_embedding_cache
                )

                self.agent.replay_memory.store(state_emb, action, next_state_emb, reward, done)

                if len(self.agent.replay_memory) > self.batch_size:
                    self.agent.learn(self.batch_size, (done or truncation))

                    # Update target-network weights
                    if total_steps % self.update_frequency == 0:
                        self.agent.hard_update()

                state_emb = next_state_emb
                episode_reward += reward
                step_size += 1
                total_steps += 1

            # Append for tracking history
            self.reward_history.append(episode_reward)

            # Decay epsilon at the end of each episode
            self.agent.update_epsilon()

            # Save model at intervals
            if episode % self.save_interval == 0:
                # Define save paths within the local directory
                save_path = os.path.join(self.save_path, f'final_weights_{episode}.pth')
                os.makedirs(self.save_path, exist_ok=True)
                self.agent.save(save_path)
                if episode != self.max_episodes:
                    plot_training(
                        self.reward_history,
                        self.agent.loss_history,
                        self.agent.q_value_history,  # Pass Q-value history
                        self.save_path,
                        episode
                    )
                print('\n~~~~~~Interval Save: Model saved.\n')

            # Print episode results
            result = (
                f"Episode: {episode}, "
                f"Total Steps: {total_steps}, "
                f"Ep Step: {step_size}, "
                f"Raw Reward: {episode_reward:.2f}, "
                f"Epsilon: {self.agent.epsilon_max:.4f}"
            )
            print(result)

        # Final plot after training
        plot_training(
            self.reward_history,
            self.agent.loss_history,
            self.agent.q_value_history,
            self.save_path,
            episode
        )

        # Extract and plot Q-value heatmap
        q_values_matrix = extract_q_values(
            agent=self.agent,
            env=self.env,
            sentence_transformer=self.sentence_transformer,
            state_embedding_cache=self.state_embedding_cache
        )
        plot_q_value_heatmap(
            q_values_matrix=q_values_matrix,
            save_dir=self.save_path,
            map_size=self.map_size
        )

    def test(self, max_episodes):
        """
        Reinforcement learning policy evaluation.
        """
        # Load the weights of the test_network
        self.agent.main_network.load_state_dict(torch.load(self.RL_load_path, map_location=DEVICE))
        self.agent.main_network.eval()

        # Testing loop over episodes
        for episode in range(1, max_episodes + 1):
            state, _ = self.env.reset(seed=SEED)
            state_emb = preprocess_state(
                state,
                self.env,
                self.sentence_transformer,
                self.state_embedding_cache
            )
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0

            while not done and not truncation:
                action = self.agent.select_action(state_emb)
                next_state, reward, done, truncation, _ = self.env.step(action)
                next_state_emb = preprocess_state(
                    next_state,
                    self.env,
                    self.sentence_transformer,
                    self.state_embedding_cache
                )

                state_emb = next_state_emb
                episode_reward += reward
                step_size += 1

            # Print log
            result = (
                f"Episode: {episode}, "
                f"Steps: {step_size}, "
                f"Reward: {episode_reward:.2f}"
            )
            print(result)

        # Close the environment
        self.env.close()
