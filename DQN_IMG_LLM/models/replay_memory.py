# models/replay_memory.py

import torch
import numpy as np
from collections import deque
from config import DEVICE

class ReplayMemory:
    def __init__(self, capacity):
        """
        Experience Replay Memory using deques to store transitions/agent experiences.
        """
        self.capacity = capacity
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)

    def store(self, state, action, next_state, reward, done):
        """
        Append transitions to their respective deques.
        """
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)

    def sample(self, batch_size):
        """
        Randomly sample transitions from memory and convert them to tensors.
        """
        indices = np.random.choice(len(self), size=batch_size, replace=False)

        states = torch.stack([self.states[i] for i in indices]).to(DEVICE)
        actions = torch.as_tensor([self.actions[i] for i in indices], dtype=torch.long, device=DEVICE)
        next_states = torch.stack([self.next_states[i] for i in indices]).to(DEVICE)
        rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=DEVICE)
        dones = torch.as_tensor([self.dones[i] for i in indices], dtype=torch.bool, device=DEVICE)

        return states, actions, next_states, rewards, dones

    def __len__(self):
        """
        Check how many samples are stored in the memory.
        """
        return len(self.dones)
