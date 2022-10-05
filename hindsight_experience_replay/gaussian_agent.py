import sys
import os
import torch
import numpy as np
import torch.nn.functional as F

from models import Actor, GaussianEnsembleCritic
from utils import soft_update


# Final  Deep Ensembles Implementation that I'm using
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
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # optimizers
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor, weight_decay=self.args.actor_l2_norm)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic, weight_decay=self.args.critic_l2_norm)
        self.forward_critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.forward_lr, weight_decay=self.args.forward_l2_norm)
        self.reward_critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.reward_lr)

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

            if self.args.gaussian_pessimistic_q:
                q_idxs = np.random.choice(np.arange(self.args.n_internal_critics), size=self.args.m_target_critics, replace=False)
                q_next_value = self.critic_target_network.Q_idxs(next_inputs_tensor, actions_next, q_idxs).min(axis=0)[0]
            else:
                if self.args.use_gaussian_q:
                    q_next_value, q_next_variances = self.critic_target_network(next_inputs_tensor, actions_next)
                else:
                    q_next_value = self.critic_target_network(next_inputs_tensor, actions_next)
                
                
            q_next_value = q_next_value.detach()
            target_q_value = rewards_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            target_q_value = torch.clamp(target_q_value, -self.args.clip_return, 0)

        if self.args.use_gaussian_q:
            q_values, q_variances = self.critic_network(inputs_tensor, actions_tensor)
            critic_loss = (target_q_value - q_values).pow(2) / q_variances + torch.log(q_variances)
            critic_loss = critic_loss.mean(axis=0)
        else:
            q_values = self.critic_network(inputs_tensor, actions_tensor)
            critic_loss = F.mse_loss(target_q_value[None, ...], q_values, reduction='none').sum(axis=0)

        if self.args.use_per:
            critic_loss *= torch.FloatTensor(is_weights).to(self.device)[:, None]
        critic_loss = critic_loss.mean()
        if self.args.use_per:
            with torch.no_grad():
                if self.args.use_td_error:
                    if self.args.use_gaussian_q:
                        preds, variances = self.critic_network(inputs_tensor, actions_tensor)
                    else:
                        preds = self.critic_network(inputs_tensor, actions_tensor)
                        priorities = np.array(preds.std(axis=0).mean(axis=1).cpu())
                        self.replay_buffer.update_priorities(sampled_idxs, priorities)
                        return critic_loss
                # use forward predictive error
                else:
                    if self.args.use_gaussian_f:
                        preds, variances = self.critic_network.predict_forward(inputs_tensor, actions_tensor)
                    else:
                        preds = self.critic_network.predict_forward(inputs_tensor, actions_tensor)
                        priorities = np.array(preds.std(axis=0).mean(axis=1).cpu())
                        self.replay_buffer.update_priorities(sampled_idxs, priorities)
                        return critic_loss
                # this has bugs right now
                priorities = variances.mean(axis=0).squeeze()
                self.replay_buffer.update_priorities(sampled_idxs, priorities)
        return critic_loss

    
    def get_forward_loss(self, num_steps):
        states_batch, actions_batch, next_states_batch, goals_batch, next_ag_batch = self.replay_buffer.sample_uniform_batches(self.args.forward_batch_size)
        
        actions_tensor = torch.FloatTensor(actions_batch).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states_batch).to(self.device)
        next_ags_tensor = torch.FloatTensor(next_ag_batch).to(self.device)

        inputs_tensor = self.preproc_inputs(states_batch, goals_batch)

        if self.args.use_gaussian_f:
            pred_means, pred_variances = self.critic_network.predict_forward(inputs_tensor, actions_tensor)
            forward_loss = ((pred_means - next_ags_tensor[None, ...]).pow(2) / pred_variances + torch.log(pred_variances)).mean()
        else:
            preds = self.critic_network.predict_forward(inputs_tensor, actions_tensor)
            forward_loss = (preds - next_ags_tensor[None, ...]).norm(dim=2).mean()
        return forward_loss

    
    def get_reward_loss(self, num_steps):
        states_batch, actions_batch, next_states_batch, goals_batch, next_ag_batch = self.replay_buffer.sample_uniform_batches(self.args.forward_batch_size)
        rewards_batch = self.reward_func(next_ag_batch, goals_batch, None)
        rewards_tensor = torch.FloatTensor(rewards_batch).to(self.device)
        
        actions_tensor = torch.FloatTensor(actions_batch).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states_batch).to(self.device)
        next_ag_tensor = torch.FloatTensor(next_ag_batch).to(self.device)

        inputs_tensor = self.preproc_inputs(states_batch, goals_batch)

        # pred_means, pred_variances = self.critic_network.predict_reward(inputs_tensor, actions_tensor)
        preds = self.critic_network.predict_reward(inputs_tensor, actions_tensor)
        # reward_loss = ((pred_means - rewards_tensor[None, ...]).pow(2) / pred_variances + torch.log(pred_variances)).mean()
        
        # mean-squared error
        reward_loss = preds[:, :, 0] - rewards_tensor[None, :]
        reward_loss = torch.square(reward_loss).mean()
        return reward_loss

    #### COMPUTE UNCERTAINTY ####
    
    def q_uncertainty(self, obs, selected_goals):
        input_tensor = self.preproc_inputs(obs, selected_goals)
        actions = self.actor_network(input_tensor)

        if self.args.use_gaussian_q:
            values, variances = self.critic_network(input_tensor, actions)
            return variances.mean(axis=0).squeeze()
            
        values = self.critic_network(input_tensor, actions)

        uncertainty = values.std(axis=0).squeeze()
        return uncertainty

    def forward_uncertainty(self, selected_obs, selected_goals):
        final_goals = selected_goals.copy()
        input_tensor = self.preproc_inputs(selected_obs, final_goals)

        with torch.no_grad():
            # all_stds = list()
            all_variances = list()
            for action_idx in range(self.args.n_curiosity_samples):
                action = np.repeat(self.env.action_space.sample()[None, :], input_tensor.shape[0], axis=0)
                action = torch.FloatTensor(action).to(self.device)

                if self.args.use_gaussian_f:
                    state_preds, pred_variances = self.critic_network.predict_forward(input_tensor, action)
                    all_variances.append(pred_variances)
                else:
                    state_preds = self.critic_network.predict_forward(input_tensor, action)
                    stds = state_preds.std(axis=0).mean(axis=1)
                    all_variances.append(stds)

        if self.args.use_gaussian_f:
            mean_variances = torch.stack(all_variances).mean(axis=0).mean(axis=0).squeeze()
        else:
            mean_variances = torch.stack(all_variances, axis=0).mean(axis=0)
        return mean_variances

    def decompositional_uncertainty(self, selected_obs, selected_goals):
        return self.q_uncertainty(selected_obs, selected_goals)
        # final_goals = selected_goals.copy()
        # input_tensor = self.preproc_inputs(selected_obs, final_goals)

        # with torch.no_grad():
        #     all_uncertainties = list()
        #     for action_idx in range(self.args.n_curiosity_samples):
        #         actions = np.repeat(self.env.action_space.sample()[None, :], input_tensor.shape[0], axis=0)
        #         actions = torch.FloatTensor(actions).to(self.device)

        #         predicted_next_obs = self.critic_network.predict_forward(input_tensor, actions).mean(axis=0)
        #         predicted_reward = self.critic_network.predict_reward(input_tensor, actions).mean(axis=0)
        #         q_values = self.critic_network(input_tensor, actions)
        #         next_input_tensor = self.preproc_inputs(np.array(predicted_next_obs.cpu()), final_goals)
        #         next_actions = self.actor_network(next_input_tensor)
        #         next_q_values = self.critic_network(next_input_tensor, next_actions)
        #         corrected_q_values = self.args.gamma * next_q_values + predicted_reward[None, :, :]
        #         errors = abs(q_values - corrected_q_values)
        #         uncertainties = errors.std(axis=0)[:, 0]
        #         all_uncertainties.append(uncertainties)
        # all_uncertainties = torch.stack(all_uncertainties, axis=0)
        # return all_uncertainties.mean(axis=0)

    def decompositional_difference_uncertainty(self, selected_obs, selected_goals):
        final_goals = selected_goals.copy()

        input_tensor = self.preproc_inputs(selected_obs, final_goals)
        all_q_distributions, all_uncertainties = list(), list()
        with torch.no_grad():
            for action_idx in range(self.args.n_curiosity_samples):
                actions = np.repeat(self.env.action_space.sample()[None, :], input_tensor.shape[0], axis=0)
                actions = torch.FloatTensor(actions).to(self.device)

                q_values, q_variances = self.critic_network(input_tensor, actions)
                predicted_next_obs = self.critic_network.predict_forward(input_tensor, actions).mean(axis=0)
                predicted_reward = self.critic_network.predict_reward(input_tensor, actions).mean(axis=0)
                next_input_tensor = self.preproc_inputs(np.array(predicted_next_obs.cpu()), final_goals)
                next_actions = self.actor_network(next_input_tensor)
                next_q_values, next_q_variances = self.critic_network(next_input_tensor, next_actions)
                corrected_q_values = self.args.gamma * next_q_values + predicted_reward

                for first_idx in range(self.args.n_internal_critics):
                    for second_idx in range(first_idx+1, self.args.n_internal_critics):
                        js_divergence = self.js_divergence(q_values[first_idx], q_variances[first_idx], next_q_values[second_idx], next_q_variances[second_idx])
                        all_uncertainties.append(js_divergence)
                        
        return torch.stack(all_uncertainties).mean(axis=0).squeeze()

    # jenson-shannon divergence
    def js_divergence(self, first_values, first_variances, second_values, second_variances):
        first_term = torch.log(second_variances.pow(0.5) / first_variances.pow(0.5))
        second_term = (first_variances + (first_values - second_values).pow(2)) / (2*second_variances)
        divergences = first_term + second_term - 0.5

        first_term = torch.log(first_variances.pow(0.5) / second_variances.pow(0.5))
        second_term = (second_variances + (second_values - first_values).pow(2)) / (2*first_variances)
        divergences += first_term + second_term - 0.5
        return divergences

    def get_value_distance(self, obs, goal):
        input_tensor = self.preproc_inputs(obs, goal)
        actions = self.actor_network(input_tensor)
        if self.args.use_gaussian_q:
            values = self.critic_network(input_tensor, actions)[0].mean(axis=0)[:, 0]
        else:
            values = self.critic_network(input_tensor, actions).mean(axis=0)[:, 0]
        return values

    def log_overall_q_errors(self):
        assert False, 'this is buggy and not ready to be used'
        observation = np.repeat(self.env.start_location[None, :], self.sampled_loc.shape[0], axis=0)
        goals = self.sampled_loc
        input_tensor = self.preproc_inputs(observation, goals)
        with torch.no_grad():
            actions = self.actor_network(input_tensor)
            predicted_next_obs, _ = self.critic_network.predict_forward(input_tensor, actions).mean(axis=0)
            predicted_reward = self.critic_network.predict_reward(input_tensor, actions).mean(axis=0)
            q_values = self.critic_network(input_tensor, actions).cpu()
            next_input_tensor = self.preproc_inputs(np.array(predicted_next_obs.cpu()), goals)
            next_actions = self.actor_network(next_input_tensor)
            next_q_values = self.critic_network(next_input_tensor, next_actions)
            corrected_q_values = self.args.gamma * next_q_values + predicted_reward[None, :, :]
            corrected_q_values = corrected_q_values.cpu()

        q_difference = q_values - self.optimal_values
        q_error = abs(q_difference).sum()
        optimistic_q_error = abs(q_difference[q_difference > 0]).sum()
        pessimistic_q_error = abs(q_difference[q_difference < 0]).sum()

        corrected_q_difference = corrected_q_values - self.optimal_values
        corrected_q_error = abs(corrected_q_difference).sum()
        optimistic_corrected_q_error = abs(corrected_q_difference[corrected_q_difference > 0]).sum()
        pessimistic_corrected_q_error = abs(corrected_q_difference[corrected_q_difference < 0]).sum()

        decomp_difference = q_values - corrected_q_values
        decomp_error = abs(decomp_difference).sum()
        optimistic_decomp_error = abs(decomp_difference[decomp_difference > 0]).sum()
        pessimistic_decomp_error = abs(decomp_difference[decomp_difference < 0]).sum()

        self.writer.add_scalar('q/q_diff', q_error, self.num_steps)
        self.writer.add_scalar('q/optimistic_q_diff', optimistic_q_error, self.num_steps)
        self.writer.add_scalar('q/pessimistic_q_diff', pessimistic_q_error, self.num_steps)

        self.writer.add_scalar('q/corrected_q_diff', corrected_q_error, self.num_steps)
        self.writer.add_scalar('q/optimistic_corrected_q_diff', optimistic_corrected_q_error, self.num_steps)
        self.writer.add_scalar('q/pessimistic_corrected_q_diff', pessimistic_corrected_q_error, self.num_steps)

        self.writer.add_scalar('q/decomp_diff', decomp_error, self.num_steps)
        self.writer.add_scalar('q/optimistic_decomp_diff', optimistic_decomp_error, self.num_steps)
        self.writer.add_scalar('q/pessimistic_decomp_diff', pessimistic_decomp_error, self.num_steps)

    
