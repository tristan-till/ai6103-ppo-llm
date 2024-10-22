import os
import time



import numpy as np
import torch
import torch.optim as optim

from ppo import PPO, PPOLogger

import utils.helpers as helpers
import utils.env as env_utils
import utils.agent as agent_utils
import utils.eval as eval_utils
import utils.config as config_utils

def setup():
    config = config_utils.parse_config("hyp.yaml")
    exp_name, run_name = helpers.get_run_name(config)
    if not os.path.exists(f"runs/{run_name}/{exp_name}"):
        os.makedirs(f"runs/{run_name}/{exp_name}")
    helpers.set_seed(config['simulation']['seed'], config['simulation']['torch_deterministic'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return config, exp_name, run_name, device

def main():
    """
    Main function to run the PPO (Proximal Policy Optimization) algorithm.

    This function sets up the environment, creates the PPO agent, and runs the training process.
    It handles both discrete and continuous action spaces, and includes options for
    various PPO algorithm parameters and training configurations.

    Args:
        # Environment parameters
        env_id (str): Identifier for the Gymnasium environment to use.
        env_is_discrete (bool): Whether the environment has a discrete action space.
        num_envs (int): Number of parallel environments to run.

        # Core training parameters
        total_timesteps (int): Total number of timesteps to run the training for. This is the number of interactions with the environment
        num_rollout_steps (int): Number of steps to run in each environment per rollout.
        update_epochs (int): Number of epochs to update the policy for each rollout.
        num_minibatches (int): Number of minibatches for each update.

        # Core PPO algorithm parameters
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        gae_lambda (float): Lambda parameter for Generalized Advantage Estimation.
        surrogate_clip_threshold (float): Clipping parameter for the surrogate objective.

        # Loss function coefficients
        entropy_loss_coefficient (float): Coefficient for the entropy term in the loss.
        value_function_loss_coefficient (float): Coefficient for the value function loss.

        # Advanced PPO parameters
        normalize_advantages (bool): Whether to normalize advantages.
        clip_value_function_loss (bool): Whether to clip the value function loss.
        max_grad_norm (float): Maximum norm for gradient clipping.
        target_kl (float): Target KL divergence for early stopping, if not None.

        # Learning rate schedule
        anneal_lr (bool): Whether to use learning rate annealing.

        # Continuous action space specific
        rpo_alpha (float): Alpha parameter for Regularized Policy Optimization (continuous action spaces only).

        # Reproducibility and logging
        seed (int): Random seed for reproducibility.
        torch_deterministic (bool): Whether to use deterministic algorithms in PyTorch.
        capture_video (bool): Whether to capture and save videos of the agent's performance.
        use_tensorboard (bool): Whether to use TensorBoard for logging.
        save_model (bool): Whether to save the trained model to disk and validate this by running a simple evaluation.
    """
    config, exp_name, run_name, device = setup()
    envs = env_utils.create_envs(config)
    
    agent = agent_utils.get_agent(
        envs,
        config['env']['type'], 
        config['optimization']['rpo_alpha'], 
        device
    )

    optimizer = optim.Adam(agent.parameters(), lr=config['optimization']['learning_rate'], eps=1e-5)

    ppo = PPO(
        agent=agent,
        optimizer=optimizer,
        envs=envs,
        config=config,
    )

    # Train the agent
    trained_agent = ppo.learn()

    if config['simulation']['save_model']:
        model_path = f"runs/{run_name}/{exp_name}.rl_model"
        torch.save(trained_agent.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    #     eval_utils.load_and_evaluate_model(
    #         run_name,
    #         env_id,
    #         env_is_discrete,
    #         envs,
    #         num_envs,
    #         agent_class,
    #         device,
    #         model_path,
    #         gamma,
    #         capture_video,
    #     )

    # Close environments
    envs.close()


if __name__ == "__main__": 
    main()