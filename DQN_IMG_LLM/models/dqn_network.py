# models/dqn_network.py

import torch.nn as nn
import torch

class DQN_Network(nn.Module):
    """
    The Deep Q-Network (DQN) model for reinforcement learning.
    """
    def __init__(self, num_actions, input_dim):
        """
        Initialize the DQN network.
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
        """
        Q = self.FC(x)
        return Q
