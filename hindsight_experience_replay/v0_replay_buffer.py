import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, args):
        self.args = args
        
        self.states = np.empty((self.args.buffer_size, self.args.obs_dim))
        # self.ags = np.empty((self.args.buffer_size, self.args.goal_dim))
        self.actions = np.empty((self.args.buffer_size, self.args.action_dim))
        self.rewards = np.empty((self.args.buffer_size, 1))
        self.next_states = np.empty((self.args.buffer_size, self.args.obs_dim))
        # self.next_ags = np.empty((self.args.buffer_size, self.args.goal_dim))
        # self.goals = np.empty((self.args.buffer_size, self.args.goal_dim))
        self.size = 0
        self.position = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # def get_all_ags(self):
    #     return self.ags[:self.size], self.states[:self.size]

    # we don't do HER, so the distribution of goals and obs won't change
    # def single_sample_her_transitions(self, mb_obs, mb_ag_next, mb_g, num_transitions):
    #     return mb_obs, mb_g

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, batch_size)
        states = self.states[indices]
        next_states = self.next_states[indices]
        # ags = self.ags[indices]
        # next_ags = self.next_ags[indices]
        actions = self.actions[indices]
        # goals = self.goals[indices]
        rewards = self.rewards[indices]
        not_dones = np.ones((batch_size, 1))

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        not_dones = torch.FloatTensor(not_dones).to(self.device)
        return states, actions, next_states, rewards, not_dones
        # return states, next_states, ags, next_ags, actions, goals, rewards

    # def sample_uniform_batches(self, batch_size):
    #     indices = np.random.randint(0, self.size, batch_size)
    #     states = self.states[indices]
    #     actions = self.actions[indices]
    #     # next_states = self.next_states[indices]
    #     # goals = self.goals[indices]
    #     # next_ags = self.next_ags[indices]
    #     # return states, actions, next_states, goals, next_ags
    #     return states, actions

    # def start_new_episode(self):
    #     pass

    # def store(self, state, next_state, achieved_goal, next_achieved_goal, goal, action, reward):
    def store(self, state, next_state, action, reward):
        self.states[self.position] = state
        self.next_states[self.position] = next_state
        # self.ags[self.position] = achieved_goal
        # self.next_ags[self.position] = next_achieved_goal
        # self.goals[self.position] = goal
        self.actions[self.position] = action
        self.rewards[self.position] = reward

        self.size = min(self.size + 1, self.args.buffer_size)
        self.position = (self.position + 1) % self.args.buffer_size

    def __len__(self):
        return self.size
