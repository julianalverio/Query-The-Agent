import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, device):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((capacity, state_dim))
        self.action = np.zeros((capacity, action_dim))
        self.next_state = np.zeros((capacity, state_dim))
        self.reward = np.zeros((capacity, 1))
        self.not_done = np.zeros((capacity, 1))
        
        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[idxs]).to(self.device),
            torch.FloatTensor(self.action[idxs]).to(self.device),
            torch.FloatTensor(self.next_state[idxs]).to(self.device),
            torch.FloatTensor(self.reward[idxs]).to(self.device),
            torch.FloatTensor(self.not_done[idxs]).to(self.device)
        )
