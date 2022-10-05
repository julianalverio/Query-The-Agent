import sys
import os
import torch
import numpy as np
import torch.nn.functional as F

from models import Actor, GPEnsembleCritic
from utils import soft_update

import gpytorch
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal

        
class GPEnsembleAgent(object):
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
        self.critic_network = GPEnsembleCritic(self.args).to(self.device)
        self.actor_target_network = Actor(self.args).to(self.device)
        self.critic_target_network = GPEnsembleCritic(self.args).to(self.device)
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # optimizers
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor, weight_decay=self.args.actor_l2_norm)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic, weight_decay=self.args.critic_l2_norm)
        # self.forward_critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.forward_lr, weight_decay=self.args.forward_l2_norm)
        # self.reward_critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.reward_lr)

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


            if self.args.n_internal_critics == 1:
                q_idxs = np.array([0, 0])

            else:
                q_idxs = np.random.choice(np.arange(self.args.n_internal_critics), size=self.args.m_target_critics, replace=False)
            q_next_value = self.critic_target_network.Q_idxs(next_inputs_tensor, actions_next, q_idxs).min(axis=0)[0]

            q_next_value = q_next_value.detach()
            target_q_value = rewards_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            target_q_value = torch.clamp(target_q_value, -self.args.clip_return, 0)

        q_values = self.critic_network(inputs_tensor, actions_tensor)
        critic_loss = F.mse_loss(target_q_value[None, ...], q_values, reduction='none').sum(axis=0)
        if self.args.use_per:
            critic_loss *= torch.FloatTensor(is_weights).to(self.device)[:, None]
        critic_loss = critic_loss.mean()
        if self.args.use_per:
            breakpoint()
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
        if not self.critic_network.gp_initialized:
            self.initialize_gp()
        states_batch, actions_batch, next_states_batch, goals_batch, next_ag_batch = self.replay_buffer.sample_uniform_batches(self.args.forward_batch_size)
        
        actions_tensor = torch.FloatTensor(actions_batch).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states_batch).to(self.device)
        next_ag_tensor = torch.FloatTensor(next_ag_batch).to(self.device)

        inputs_tensor = self.preproc_inputs(states_batch, goals_batch)

        x_preds, x_variances, x_distributions, y_preds, y_variances, y_distributions = self.critic_network.predict_forward(inputs_tensor, actions_tensor, distributions=True)
        x_forward_loss = None
        y_forward_loss = None
        for critic_idx, (x_distribution, y_distribution) in enumerate(zip(x_distributions, y_distributions)):
            elbo_fns = self.elbo_fns[critic_idx]
            if x_forward_loss is None:
                x_forward_loss = -elbo_fns[0](x_distribution, next_ag_tensor[:, 0])
                y_forward_loss = -elbo_fns[1](y_distribution, next_ag_tensor[:, 1])
            else:
                x_forward_loss += -elbo_fns[0](x_distribution, next_ag_tensor[:, 0])
                y_forward_loss += -elbo_fns[1](y_distribution, next_ag_tensor[:, 1])
        return x_forward_loss + y_forward_loss
        
        # for critic_idx, (pred, variance, distribution, elbo_fn)  in enumerate(zip(preds, variances, distributions, self.elbo_fns)):
        #     # # transformation!
        #     # next_ag_tensor = next_ag_tensor.T
            
        #     if forward_loss is None:
        #         forward_loss = -elbo_fn(distribution, next_ag_tensor)
        #     else:
        #         forward_loss += -elbo_fn(distribution, next_ag_tensor)
        # return forward_loss

    
    def get_reward_loss(self, num_steps):
        states_batch, actions_batch, next_states_batch, goals_batch, next_ag_batch = self.replay_buffer.sample_uniform_batches(self.args.forward_batch_size)
        rewards_batch = self.reward_func(next_ag_batch, goals_batch, None)
        rewards_tensor = torch.FloatTensor(rewards_batch).to(self.device)
        
        actions_tensor = torch.FloatTensor(actions_batch).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states_batch).to(self.device)
        next_ag_tensor = torch.FloatTensor(next_ag_batch).to(self.device)

        inputs_tensor = self.preproc_inputs(states_batch, goals_batch)

        preds = self.critic_network.predict_reward(inputs_tensor, actions_tensor)

        # mean-squared error
        reward_loss = preds[:, :, 0] - rewards_tensor[None, :]
        reward_loss = torch.square(reward_loss).mean()
        return reward_loss

    #### COMPUTE UNCERTAINTY ####
    
    def q_uncertainty(self, obs, selected_goals):
        input_tensor = self.preproc_inputs(obs, selected_goals)
        actions = self.actor_network(input_tensor)

        values = self.critic_network(input_tensor, actions)

        uncertainty = values.std(axis=0).squeeze()
        return uncertainty

    def initialize_gp(self):
        states_batch = self.replay_buffer.obs[:self.replay_buffer.current_size]
        goals_batch = self.replay_buffer.g[:self.replay_buffer.current_size]
        inputs_tensor = self.preproc_inputs(states_batch, goals_batch)
        actions_batch = self.replay_buffer.actions[:self.replay_buffer.current_size]
        actions_tensor = torch.FloatTensor(actions_batch).to(self.device)
        self.critic_network.initialize_gp(inputs_tensor, actions_tensor)

        # for GP loss
        self.likelihood = GaussianLikelihood().to(self.device)
        self.likelihood.train()
        self.elbo_fns = list()
        forward_parameters = list(self.critic_network.parameters())
        forward_parameters.extend(self.likelihood.parameters())
        for critic_idx in range(self.args.n_internal_critics):
            critic_elbo_fns = list()
            gp_x = getattr(self.critic_network, f'q{critic_idx}_forward_gp_x')
            gp_y = getattr(self.critic_network, f'q{critic_idx}_forward_gp_y')
            elbo_x = VariationalELBO(self.likelihood, gp_x, num_data=self.args.forward_batch_size)
            elbo_y = VariationalELBO(self.likelihood, gp_y, num_data=self.args.forward_batch_size)
            self.elbo_fns.append([elbo_x, elbo_y])
            forward_parameters.extend(gp_x.parameters())
            forward_parameters.extend(gp_y.parameters())
            
        self.forward_critic_optim = torch.optim.Adam(forward_parameters, lr=self.args.forward_lr, weight_decay=self.args.forward_l2_norm)
        # self.reward_critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.reward_lr)

    def forward_uncertainty(self, selected_obs, selected_goals):
        if not self.critic_network.gp_initialized:
            self.initialize_gp()

        final_goals = selected_goals.copy()
        input_tensor = self.preproc_inputs(selected_obs, final_goals)

        with torch.no_grad():
            # all_stds = list()
            all_variances = list()
            for action_idx in range(self.args.n_curiosity_samples):
                action = np.repeat(self.env.action_space.sample()[None, :], input_tensor.shape[0], axis=0)
                action = torch.FloatTensor(action).to(self.device)
                x_preds, x_variances, y_preds, y_variances = self.critic_network.predict_forward(input_tensor, action)
                variances = (x_variances.mean(axis=0) + y_variances.mean(axis=0)) / 2
                all_variances.append(variances)
                # stds = state_preds.std(axis=0).mean(axis=1)
                # all_stds.append(stds)
        mean_variances = torch.stack(all_variances).mean(axis=0)
        # mean_stds = torch.stack(all_stds, axis=0).mean(axis=0)
        return mean_variances

    def decompositional_difference_uncertainty(self, selected_obs, selected_goals):
        final_goals = selected_goals.copy()
        input_tensor = self.preproc_inputs(selected_obs, final_goals)

        with torch.no_grad():
            all_uncertainties = list()
            for action_idx in range(self.args.n_curiosity_samples):
                actions = np.repeat(self.env.action_space.sample()[None, :], input_tensor.shape[0], axis=0)
                actions = torch.FloatTensor(actions).to(self.device)

                breakpoint()
                predicted_next_obs = self.critic_network.predict_forward(input_tensor, actions).mean(axis=0)
                predicted_reward = self.critic_network.predict_reward(input_tensor, actions).mean(axis=0)
                q_values = self.critic_network(input_tensor, actions)
                next_input_tensor = self.preproc_inputs(np.array(predicted_next_obs.cpu()), final_goals)
                next_actions = self.actor_network(next_input_tensor)
                next_q_values = self.critic_network(next_input_tensor, next_actions)
                corrected_q_values = self.args.gamma * next_q_values + predicted_reward[None, :, :]
                errors = abs(q_values - corrected_q_values)
                uncertainties = errors.std(axis=0)[:, 0]
                all_uncertainties.append(uncertainties)
        all_uncertainties = torch.stack(all_uncertainties, axis=0)
        return all_uncertainties.mean(axis=0)

    
    def decompositional_uncertainty(self, selected_obs, selected_goals):
        final_goals = selected_goals.copy()

        input_tensor = self.preproc_inputs(selected_obs, final_goals)
        all_uncertainties = list()
        with torch.no_grad():
            for action_idx in range(self.args.n_curiosity_samples):
                actions = np.repeat(self.env.action_space.sample()[None, :], input_tensor.shape[0], axis=0)
                actions = torch.FloatTensor(actions).to(self.device)

                breakpoint()
                predicted_next_obs = self.critic_network.predict_forward(input_tensor, actions).mean(axis=0)
                predicted_reward = self.critic_network.predict_reward(input_tensor, actions).mean(axis=0)
                next_input_tensor = self.preproc_inputs(np.array(predicted_next_obs.cpu()), final_goals)
                next_actions = self.actor_network(next_input_tensor)
                next_q_values = self.critic_network(next_input_tensor, next_actions)
                corrected_q_values = self.args.gamma * next_q_values + predicted_reward[None, :, :]
                uncertainties = corrected_q_values.std(axis=0)[:, 0]
                all_uncertainties.append(uncertainties)
        all_uncertainties = torch.stack(all_uncertainties, axis=0)
        return all_uncertainties.mean(axis=0)

    def get_value_distance(self, obs, goal):
        input_tensor = self.preproc_inputs(obs, goal)
        actions = self.actor_network(input_tensor)
        values = self.critic_network(input_tensor, actions).mean(axis=0)[:, 0]
        return values

    def log_overall_q_errors(self):
        observation = np.repeat(self.env.start_location[None, :], self.sampled_loc.shape[0], axis=0)
        goals = self.sampled_loc
        input_tensor = self.preproc_inputs(observation, goals)
        with torch.no_grad():
            actions = self.actor_network(input_tensor)
            breakpoint()
            predicted_next_obs = self.critic_network.predict_forward(input_tensor, actions).mean(axis=0)
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

    
