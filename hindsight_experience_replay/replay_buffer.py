import threading
import numpy as np
import torch
from sklearn.neighbors import KernelDensity


class replay_buffer:
    def __init__(self, args, reward_func):
        self.args = args
        # self.distance_threshold = distance_threshold
        self.reward_func = reward_func
        self.future_p = 1 - (1. / (1 + args.replay_k))
        self.size = args.buffer_size
        self.n_transitions_stored = 0
        self.current_size = 0
        
        self.obs = np.empty([self.size, args.obs_dim])
        self.ag = np.empty([self.size, args.goal_dim])
        self.g = np.empty([self.size, args.goal_dim])
        self.actions = np.empty([self.size, args.action_dim])
        self.obs_next = np.empty([self.size, args.obs_dim])
        self.ag_next = np.empty([self.size, args.goal_dim])
        self.starts = np.empty([self.size], dtype=int)  # for any step in self.obs, where did that traj start?
        self.ends = np.empty([self.size], dtype=int)  # for any step in self.obs, where did that traj end?
        self.idx_to_episode_idx = np.empty([self.size], dtype=int)  # for any step, which episode was it in? For PER.
        self.episode_starts = np.empty([self.size], dtype=int)  # for a particular ep, when did it start
        self.episode_lengths = np.empty([self.size], dtype=int)  # for a particular ep, how long was it
        self.pointer = 0
        self.num_episodes = 0
        self.bit_mask = np.ones(self.size, dtype=int)
        self.valid_episode_idxs = list()
        self.start_to_episode_idx = dict()

        self.current_episode_start = 0  # only for REDQ, for storing 1 step at a time
        
        if self.args.bootstrapped:
            self.bootstrap_mask = np.zeros([self.size, self.args.n_internal_critics])
            self.bootstrap_pointer = 0

        # goal KDE stuff
        # self.goal_kde = None
        # self.new_goal_added = False
        # self.unique_goals = np.empty([self.size, args.goal_dim])
        # self.unique_goals_pointer = 0

    # def get_goal_kde(self):
    #     if self.goal_kde is not None:
    #         return self.goal_kde

    #     goals = self.unique_goals[:self.unique_goals_pointer]
    #     kde = KernelDensity(kernel='gaussian', bandwidth=self.args.goal_biasing_kde_bandwidth).fit(goals)
    #     if self.unique_goals_pointer >= self.args.goal_biasing_num_goals_fitted:
    #         self.goal_kde = kde
    #     return kde

    # For REDQ only
    def start_new_episode(self):
        self.current_episode_start = self.pointer
        self.episode_starts[self.num_episodes] = self.pointer
        self.valid_episode_idxs.append(self.num_episodes)
        self.num_episodes += 1

        # self.new_goal_added = False

    # At each timestep, assume the episode is ending and update everything.
    # This keeps everything up-to-date so you can sample the latest data
    def store_step(self, obs, obs_next, ag, ag_next, g, action):
        idxs, new_pointer = self.get_storage_idxs(1)
        self.obs[idxs] = obs
        self.obs_next[idxs] = obs_next
        self.ag[idxs] = ag
        self.ag_next[idxs] = ag_next
        self.g[idxs] = g
        self.actions[idxs] = action

        self.starts[idxs] = self.current_episode_start
        self.ends[self.current_episode_start : self.pointer + 1] = self.pointer  # keep up-to-date
        self.start_to_episode_idx[self.pointer] = self.num_episodes - 1  # num_episodes is always 1 ahead
        # num_episodes is always 1 ahead
        self.episode_lengths[self.num_episodes - 1] = (self.pointer + 1) - self.current_episode_start  # must add +1 to account for the pointer moving up

        self.idx_to_episode_idx[self.pointer] = self.num_episodes - 1
        
        self.n_transitions_stored += 1
        self.pointer = new_pointer
        self.current_size = min(self.current_size + 1, self.size)

        # if not self.new_goal_added:
        #     self.unique_goals[self.unique_goals_pointer] = g
        #     self.unique_goals_pointer += 1
        #     self.new_goal_added = True
        return idxs

    # if bootstrapping, update the binary bootstrap mask at the end of an episode
    def end_episode(self):
        num_samples = self.pointer - self.bootstrap_pointer
        size = [num_samples, self.args.n_internal_critics]
        self.bootstrap_mask[self.bootstrap_pointer:self.pointer] = np.random.binomial(1, 0.5, size)
        self.bootstrap_pointer = self.pointer

    def store_episode(self, mb_obs, mb_ag, mb_g, mb_actions):
        assert False, 'This should be unused'
        num_steps = mb_actions.shape[0]
        idxs, new_pointer = self.get_storage_idxs(num_steps)
        self.obs[idxs] = mb_obs[:-1]
        self.obs_next[idxs] = mb_obs[1:]
        self.ag[idxs] = mb_ag[:-1]
        self.ag_next[idxs] = mb_ag[1:]
        self.g[idxs] = mb_g
        self.actions[idxs] = mb_actions
        self.starts[idxs] = self.pointer
        self.ends[idxs] = self.pointer + num_steps - 1
        self.start_to_episode_idx[self.pointer] = self.num_episodes

        self.episode_starts[self.num_episodes] = self.pointer
        self.episode_lengths[self.num_episodes] = num_steps
        self.n_transitions_stored += num_steps
        self.pointer = new_pointer
        self.current_size = min(self.current_size + num_steps, self.size)
        self.valid_episode_idxs.append(self.num_episodes)
        self.num_episodes += 1
        
        
    def get_all_ags(self):
        return self.ag[:self.current_size], self.obs[:self.current_size]

    def get_traj(self, idx):
        start = self.starts[idx]
        return self.obs[start:idx+1], self.g[start:idx+1], self.actions[start:idx+1]
    
    def sample(self, batch_size, uniform=False):
        return self.sample_her_transitions(batch_size)

    def single_sample_her_transitions(self, obs, ag_next, g, batch_size):
        episode_length = g.shape[0] # (1, 50, 2)
        timestep_idxs = np.random.randint(episode_length, size=batch_size)  # which timestep in the rollout  # (50,)

        selected_obs = obs[timestep_idxs]  # obs is (1, 51, 2)
        selected_g = g[timestep_idxs]  # g is (1, 50, 2)

        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)  
        
        future_offset = (np.random.uniform(size=batch_size) * (episode_length - timestep_idxs)).astype(int)
        future_t = (timestep_idxs + future_offset)[her_indexes]

        future_ag = ag_next[future_t]
        selected_g[her_indexes] = future_ag
        return selected_obs, selected_g

    def sample_uniform_batches(self, batch_size):
        episode_idxs = np.random.choice(self.valid_episode_idxs, batch_size)
        episode_lengths = self.episode_lengths[episode_idxs]
        episode_starts = self.episode_starts[episode_idxs]
        selected_idxs = episode_starts + np.random.randint(0, high=episode_lengths)
        states = self.obs[selected_idxs]
        actions = self.actions[selected_idxs]
        next_states = self.obs_next[selected_idxs]
        goals = self.g[selected_idxs]
        ag_next = self.ag_next[selected_idxs]

        if self.args.bootstrapped:
            bootstrap_mask = self.bootstrap_mask[selected_idxs]
            return states, actions, next_states, goals, ag_next, bootstrap_mask

        # I also want the next achieved goal
        return states, actions, next_states, goals, ag_next
        
    
    def sample_her_transitions(self, batch_size, indices=None):
        if indices is None:
            # episode_idxs = np.random.randint(0, self.num_episodes, batch_size)  # which rollout to sample from
            probs = self.episode_lengths[:self.num_episodes] / self.episode_lengths[:self.num_episodes].sum()
            episode_idxs = np.random.choice(self.valid_episode_idxs, batch_size, p=probs)

            episode_lengths = self.episode_lengths[episode_idxs]
            episode_starts = self.episode_starts[episode_idxs]
            t_samples = np.random.randint(episode_lengths, size=batch_size)
            timestep_idxs = episode_starts + t_samples
        else:
            timestep_idxs = indices
            episode_idxs = self.idx_to_episode_idx[timestep_idxs]
            episode_lengths = self.episode_lengths[episode_idxs]
            episode_starts = self.episode_starts[episode_idxs]
            t_samples = timestep_idxs - episode_starts
            

        selected_obs = self.obs[timestep_idxs]
        selected_ag = self.ag[timestep_idxs]
        selected_actions = self.actions[timestep_idxs]
        selected_obs_next = self.obs_next[timestep_idxs]
        selected_ag_next = self.ag_next[timestep_idxs]
        selected_g = self.g[timestep_idxs]

        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        
        future_offset = (np.random.uniform(size=batch_size) * (episode_lengths - t_samples)).astype(int)
        future_t = (t_samples + future_offset)[her_indexes]

        future_idxs = episode_starts[her_indexes] + future_t
        future_ag = self.ag_next[future_idxs]
        selected_g[her_indexes] = future_ag

        rewards = self.reward_func(selected_ag_next, selected_g, None)[:, None]

        if self.args.bootstrapped:
            bootstrap_mask = self.bootstrap_mask[timestep_idxs]
            return selected_obs, selected_obs_next, selected_ag, selected_ag_next, selected_actions, selected_g, rewards, bootstrap_mask
        
        return selected_obs, selected_obs_next, selected_ag, selected_ag_next, selected_actions, selected_g, rewards

    def get_storage_idxs(self, num_steps):
        if self.pointer + num_steps <= self.size:
            idxs = np.arange(self.pointer, self.pointer + num_steps)
            new_pointer = self.pointer + num_steps
        else:
            overflow = self.pointer + num_steps - self.size
            idx_a = np.arange(self.pointer, self.size)
            idx_b = np.arange(0, overflow)
            idxs = np.concatenate([idx_a, idx_b], axis=0)
            new_pointer = overflow
        return idxs, new_pointer
