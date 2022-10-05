import sys
import os
import torch
import numpy as np
import torch.nn.functional as F

from models import Actor, GaussianEnsembleCritic
from utils import soft_update


# IMPLEMENTATION OF DEEP ENSEMBLES
class GaussianAgent(object):
    def __init__(self, args, replay_buffer, preproc_inputs, writer, device, reward_func, o_norm, g_norm, env):
        self.args = args
        self.device = device
        self.replay_buffer = replay_buffer
        self.preproc_inputs = preproc_inputs
        self.writer = writer
        self.reward_func = reward_func
        self.o_norm = o_norm
        self.g_norm = g_norm
        self.env = env

        # networks
        self.actor_network = Actor(self.args).to(self.device)
        self.critic_network = GaussianEnsembleCritic(self.args).to(self.device)
        self.actor_target_network = Actor(self.args).to(self.device)
        self.critic_target_network = GaussianEnsembleCritic(self.args).to(self.device)
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.copy_weights(self.critic_network)

        # optimizers
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor, weight_decay=self.args.actor_l2_norm)
        self.critic_optim = self.critic_network.optimizer
        self.forward_critic_optim = self.critic_network.forward_optimizer
        self.reward_critic_optim = self.critic_network.reward_optimizer

    def get_critic_loss(self, num_steps):
        if self.args.use_per:
            beta = self.args.initial_beta + num_steps / self.args.beta_steps
            beta = min(beta, 1.)
            (states_batch, next_states_batch, ag_batch, next_ag_batch, actions_batch, goals_batch, rewards_batch), is_weights, sampled_idxs = self.replay_buffer.sample(self.args.batch_size, beta)
        else:
            states_batch, next_states_batch, ag_batch, next_ag_batch, actions_batch, goals_batch, rewards_batch = self.replay_buffer.sample(self.args.batch_size)
            
        inputs_tensor = self.preproc_inputs(states_batch, goals_batch)
        next_inputs_tensor = self.preproc_inputs(next_states_batch, goals_batch)

        actions_tensor = torch.FloatTensor(actions_batch).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards_batch).to(self.device)
        
        with torch.no_grad():
            noise = (torch.randn_like(actions_tensor) * self.args.policy_noise).clamp(-self.args.noise_clip, self.args.noise_clip)
            actions_next = self.actor_target_network(next_inputs_tensor)

            actions_next = (actions_next + noise).clamp(-self.args.action_max, self.args.action_max)

            q_next_values, q_next_variances = self.critic_target_network(next_inputs_tensor, actions_next)

            if self.args.gaussian_pessimistic_q:
                q_idxs = np.random.choice(np.arange(self.args.n_internal_critics), size=self.args.m_target_critics, replace=False)
                q_next_values = q_next_values[q_idxs].min(axis=0)[0]
                
                
            q_next_values = q_next_values.detach()
            target_q_values = rewards_tensor + self.args.gamma * q_next_values
            target_q_values = target_q_values.detach()
            target_q_value = torch.clamp(target_q_values, -self.args.clip_return, 0)

        q_values, q_variances = self.critic_network(inputs_tensor, actions_tensor)
        # if self.args.gaussian_pessimistic_q:
        #     error = target_q_values[None, ...] - q_values
        # else:
        #     error = target_q_values[None, ...] - q_values
        # critic_loss = error.pow(2) / q_variances + torch.log(q_variances)
        for idx, q_value in enumerate(q_values):
            if idx == 0:
                critic_loss = F.mse_loss(target_q_value, q_value, reduction='none')
            else:
                critic_loss += F.mse_loss(target_q_value, q_value, reduction='none')
        
        if self.args.use_per:
            critic_loss *=  torch.FloatTensor(is_weights).to(self.device)[None, :, None]
        critic_loss = critic_loss.mean()
        
        if self.args.use_per:
            with torch.no_grad():
                if self.args.use_td_error:
                    preds = self.critic_network(inputs_tensor, actions_tensor)
                # use forward predictive error
                else:
                    preds = self.critic_network.predict_forward(inputs_tensor, actions_tensor)
                priorities = preds.std(axis=0).mean(axis=1)
                priorities = np.array(priorities.cpu())
                self.replay_buffer.update_priorities(sampled_idxs, priorities)
        return critic_loss

    def get_forward_loss(self, num_steps):
        states_batch, actions_batch, next_states_batch, goals_batch, next_ag_batch = self.replay_buffer.sample_uniform_batches(self.args.forward_batch_size)
        
        actions_tensor = torch.FloatTensor(actions_batch).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states_batch).to(self.device)
        next_ags_tensor = torch.FloatTensor(next_ag_batch).to(self.device)

        inputs_tensor = self.preproc_inputs(states_batch, goals_batch)

        preds, pred_variances = self.critic_network.predict_forward(inputs_tensor, actions_tensor)
        forward_loss = (preds - next_ags_tensor[None, ...]).norm(dim=2).mean()
        # forward_loss = ((preds - next_ags_tensor).pow(2) / pred_variances + torch.log(pred_variances)).mean()
        return forward_loss

    
    def get_reward_loss(self, num_steps):
        states_batch, actions_batch, next_states_batch, goals_batch, next_ag_batch = self.replay_buffer.sample_uniform_batches(self.args.forward_batch_size)
        rewards_batch = self.reward_func(next_ag_batch, goals_batch, None)
        rewards_tensor = torch.FloatTensor(rewards_batch).to(self.device)
        
        actions_tensor = torch.FloatTensor(actions_batch).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states_batch).to(self.device)
        next_ag_tensor = torch.FloatTensor(next_ag_batch).to(self.device)

        inputs_tensor = self.preproc_inputs(states_batch, goals_batch)

        preds, pred_variances = self.critic_network.predict_reward(input_tensor, actions_tensor)
        reward_loss = ((preds - rewards_tensor).pow(2) / pred_variances + torch.log(pred_variances)).mean()
        return reward_loss


    #### COMPUTE UNCERTAINTY ####

    def q_uncertainty(self, obs, selected_goals):
        input_tensor = self.preproc_inputs(obs, selected_goals)
        actions = self.actor_network(input_tensor)

        values, variances = self.critic_network(input_tensor, actions, combine=True)
        return variances

    def forward_uncertainty(self, selected_obs, selected_goals):
        final_goals = selected_goals.copy()
        input_tensor = self.preproc_inputs(selected_obs, final_goals)

        with torch.no_grad():
            all_variances = list()
            for action_idx in range(self.args.n_curiosity_samples):
                action = np.repeat(self.env.action_space.sample()[None, :], input_tensor.shape[0], axis=0)
                action = torch.FloatTensor(action).to(self.device)
                state_preds, state_pred_variances = self.critic_network.predict_forward(input_tensor, action, combine=True)
                all_variances.append(state_pred_variances)
        mean_variances = torch.stack(all_variances, axis=0).mean(axis=0)
        return mean_variances

    def decompositional_difference_uncertainty(self, selected_obs, selected_goals):
        final_goals = selected_goals.copy()

        input_tensor = self.preproc_inputs(selected_obs, final_goals)
        all_uncertainties = list()
        for action_idx in range(self.args.n_curiosity_samples):
            actions = np.repeat(self.env.action_space.sample()[None, :], input_tensor.shape[0], axis=0)
            actions = torch.FloatTensor(actions).to(self.device)

            next_obs_pred, _ = self.critic_network.predict_forward(input_tensor, actions, combine=True)
            reward_pred, reward_vars = self.critic_network.predict_reward(input_tensor, actions, combine=True)
            q_values, q_vars = self.critic_network(input_tensor, actions, combine=True)
            next_input_tensor = self.preproc_inputs(np.array(next_obs_pred.cpu())[:, None], final_goals)
            next_actions = self.actor_network(next_input_tensor)
            next_q_values, next_q_value_vars = self.critic_network(next_input_tensor, next_actions, combine=True)
            corrected_q_values = self.args.gamma * next_q_values + reward_pred
            corrected_q_vars = next_q_value_vars + reward_vars
            errors = q_values - corrected_q_values
            kl_divergence = torch.log(corrected_q_vars/q_vars) + (q_vars.pow(2)+errors.pow(2))/(2*next_q_value_vars.pow(2))
            all_uncertainties.append(kl_divergence)
        all_uncertainties = torch.stack(all_uncertainties, axis=0)
        return all_uncertainties.mean(axis=0)

    def decompositional_uncertainty(self, selected_obs, selected_goals):
        final_goals = selected_goals.copy()

        input_tensor = self.preproc_inputs(selected_obs, final_goals)
        all_uncertainties = list()
        for action_idx in range(self.args.n_curiosity_samples):
            actions = np.repeat(self.env.action_space.sample()[None, :], input_tensor.shape[0], axis=0)
            actions = torch.FloatTensor(actions).to(self.device)

            next_obs_pred = self.critic_network.predict_forward(input_tensor, actions, combine=True)[0]
            reward_pred, reward_vars = self.critic_network.predict_reward(input_tensor, actions, combine=True)
            next_input_tensor = self.preproc_inputs(np.array(next_obs_pred.cpu()[:, None]), final_goals)
            next_actions = self.actor_network(next_input_tensor)
            
            next_q_values, next_q_value_variances = self.critic_network(next_input_tensor, next_actions, combine=True)
            corrected_q_variances = next_q_value_variances + reward_vars
            all_uncertainties.append(corrected_q_variances)
        all_uncertainties = torch.stack(all_uncertainties, axis=0).mean(axis=0)
        return all_uncertainties

    def get_value_distance(self, obs, goal):
        input_tensor = self.preproc_inputs(obs, goal)
        actions = self.actor_network(input_tensor)
        values, variances = self.critic_network(input_tensor, actions)
        return values.mean(axis=0)[:, 0]

    def isolate_forward_head(self):
        self.critic_network.isolate_forward_head()
        
    def isolate_reward_head(self):
        self.critic_network.isolate_reward_head()

    def unfreeze_network(self):
        self.critic_network.unfreeze_network()
    
