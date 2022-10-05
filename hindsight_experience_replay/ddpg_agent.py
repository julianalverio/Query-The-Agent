import mujoco_py
print('successfully imported mujoco')
import os
import torch
import math
import copy

from datetime import datetime
import numpy as np
from replay_buffer import replay_buffer
from models import Actor, Critic, EnsembleCritic
from normalizer import normalizer
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import time
import pickle

import argparse
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.neighbors import KernelDensity

import pathlib
import sys
current = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, os.path.join(current, '../clean_baselines/gym/'))
import gym
import os, sys
import random
import torch
import yaml
import json
import re
from scipy.stats import norm

import sys

import pathlib
current = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, os.path.join(current, '../clean_baselines/gym/gym/envs/maze_tests'))
from gym.envs.maze_test.maze_manager import generate_maze

np.seterr(all='raise')  # for debugging

sys.path.insert(0, os.path.join(current, '..'))
from utils import ArgsHolder, soft_update

from collections import deque
from per import PERReplayBuffer
from ensemble_agent import EnsembleAgent
from bootstrapped_agent import BootstrappedAgent
from gaussian_agent import GaussianAgent
# from gp_ensemble_agent import GPEnsembleAgent


class DDPGAgent(object):
    def __init__(self):
        self.setup_args_and_env()
        self.device = torch.device('cuda') if self.args.cuda else torch.device('cpu')

        # replay buffer and normalize
        if self.args.use_per:
            self.buffer = PERReplayBuffer(self.args, self.reward_func)
        else:
            self.buffer = replay_buffer(self.args, self.reward_func)
        self.o_norm = normalizer(size=self.args.obs_dim, default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=self.args.goal_dim, default_clip_range=self.args.clip_range)

        # logging
        writer_path = self.get_writer_path()
        self.writer = SummaryWriter(writer_path)
        with open(os.path.join(writer_path, 'configs.json'), 'w+') as f:
            json.dump(self.args.to_json(), f)

        # agent
        if self.args.simple_ensemble:
            agent_class = EnsembleAgent
        elif self.args.bootstrapped:
            agent_class = BootstrappedAgent
        elif self.args.use_gaussian:
            agent_class = GaussianAgent
        elif self.args.use_gp:
            agent_class = GPEnsembleAgent

        self.agent = agent_class(self.args, self.buffer, self.preproc_inputs, self.writer, self.device, self.reward_func, self.o_norm, self.g_norm, self.env)

        # misc
        self.num_steps = 0
        self.intrinsic_performance_deque = deque(maxlen=10)

        if not self.args._3D and not self.args.robot:
            optimal_values, _, _, sampled_loc, _, optimal_hops = self.env.get_optimal_value_function()
            self.optimal_values = optimal_values
            self.sampled_loc = sampled_loc
            optimal_hops = optimal_hops.max() - optimal_hops + 1
            self.optimal_hops = optimal_hops

            maze_length = optimal_hops.max()
            self.cutoff_lower_bound = -sum([self.args.gamma**n for n in range(1, int(maze_length)+1)])

    def get_writer_path(self):
        current = os.path.dirname(os.path.realpath(__file__))
        log_root = os.path.join(current, 'vds_logs')
        if not os.path.exists(log_root): os.makedirs(log_root)

        writer_dir = f'{self.args.exp_name}_seed{self.args.seed}'
        writer_path = os.path.join(log_root, writer_dir)
        # assert re.match(REGEX, writer_dir)
        return writer_path

    def setup_args_and_env(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int)
        parser.add_argument('--env', type=str)
        parser.add_argument('-config', type=str, default='configs.yml')
        parsed_args = parser.parse_args()

        # args
        configs = yaml.safe_load(open(parsed_args.config, 'r'))
        print('Loaded configs!')
        args = ArgsHolder(configs)
        if hasattr(parsed_args, 'seed') and parsed_args.seed is not None:
            args.seed = parsed_args.seed
        if parsed_args.env:
            args.env_name = parsed_args.env

        args.maze_w_length_lower = args.maze_w_length_upper
        args.maze_hallways_width = 1
        args.maze_free_blocks = 1
        args.maze_density = 10
        args.maze_bridge_length_upper = 1
        args.maze_bridge_length_lower = 1
        args.maze_uniform_goals = False

        # env
        # we will only consider 2D mazes
        args.env_name = 'maze'
        args.maze_dimensionality = 2

        # handle 3D config setup
        if args._3D:
            args.replay_k = 0.  # don't use HER
            args.maze_dimensionality = 3
            args.agent_type = 'ant'
            args.dense_reward = True

        if args.maze_type not in ('w', 'e'):
           env =  gym.make(args.maze_type)
        else:
            env = generate_maze(args)
        self.reward_func = env.compute_reward

        self.args = args
        self.args.buffer_size = self.args.total_num_steps

        # env params
        observation = env.reset()
        self.args.obs_dim = observation['observation'].shape[0]
        self.args.goal_dim = observation['desired_goal'].shape[0]
        ag = observation['achieved_goal']
        self.args.goal_dim = ag.shape[0]
        self.args.action_dim = env.action_space.shape[0]
        self.args.action_max = env.action_space.high[0]
        self.args.episode_length = env._max_episode_steps
        
        # seeds
        env.seed(self.args.seed)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.backends.cudnn.benchmark = True

        self.env = env
        self.args = args

        if self.args.test:
            self.args.init_random_steps = 50
            self.args.eval_freq = 100
            self.args.exp_name = 'test'
            self.args.logging_freq = 100

        if (self.args.use_decompositional or self.args.use_decompositional_difference) and not self.args.num_reward_batches:
            print('WARNING: REWARD BATCHES WAS 0 AND CHANGED TO 1')
            self.args.num_reward_batches = 1

        if 'fetch' in self.args.maze_type.lower():
            self.args.robot = True
        else:
            self.args.robot = False

        ##### CHANGE ARGS FOR GAUSSIAN AGENT
        self.args.use_gaussian_q = False
        self.args.use_gaussian_f = False
        
        if self.args.use_gaussian:
            if self.args.use_forward_uncertainty:
                self.args.use_gaussian_f = True
                self.args.use_gaussian_q = False
                self.args.gaussian_pessimistic_q = False
            elif self.args.use_uncertainty:
                self.args.use_gaussian_q = True
                self.args.use_gaussian_f = False
                self.args.gaussian_pessimistic_q = True
            elif self.args.use_decompositional:
                self.args.use_gaussian_q = True
                self.args.use_gaussian_f = False
                self.args.gaussian_pessimistic_q = True
            elif self.args.use_decompositional_difference:
                self.args.use_gaussian_q = True
                self.args.use_gaussian_f = False
                self.args.use_pessimistic_q = True
            else:
                print('\n\n \033[93m WARNING! YOU HAVE TO SET UP AUTO CONFIGS \033[0m \n\n')
                assert False
            

    # clip, normalize, concatenate
    def preproc_inputs(self, obs, g):
        if len(obs.shape) == len(g.shape) == 1:
            obs = obs[None, :]
            g = g[None, :]
            
        obs, g = self.clip_og(obs, g)
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        return torch.FloatTensor(np.concatenate([obs_norm, g_norm], axis=1)).to(self.device)

    def sample_truncated_gaussian(self):
        mean = 0
        sigma = self.args.gaussian_sigma
        sample = abs(np.random.normal(loc=0, scale=sigma))
        while sample >= 1:
            sample = abs(np.random.normal(loc=0, scale=sigma))
        return 1 - sample  # bias toward difficulty is 1

    # downsample any number of objects
    def downsample(self, inputs_list, output_size=None):
        if inputs_list[0].shape[0] <= output_size:
            return inputs_list
        idxs = np.random.choice(np.arange(inputs_list[0].shape[0]), size=output_size)
        return [input_obj[idxs] for input_obj in inputs_list]
        
    # only supports linear difficulty and sigma difficulty
    def select_goal_from_uncertainty_vector(self, uncertainty_vector, selected_goals, observation, num_goals=1):
        if self.args.use_linear_f:
            normalized_uncertainty = uncertainty_vector - uncertainty_vector.min()
            normalized_uncertainty /= normalized_uncertainty.max()
            likelihoods = self.args.linear_f_intercept + self.args.linear_f_slope*normalized_uncertainty
            likelihoods = likelihoods.clamp(min=0)
            likelihoods = np.array(likelihoods.detach().cpu())
            # normalize distribution + numerical stability
            likelihoods[likelihoods < 0.0001] = 0
            with np.errstate(under='ignore'):
                likelihoods /= likelihoods.sum()
            if np.isnan(likelihoods).any():
                self.writer.add_scalar('train/uniform_likelihoods', 1, self.num_steps)
                print('LIKELIHOODS WERE UNIFORM!!')
                likelihoods = np.full_like(likelihoods, 1./likelihoods.shape[0])
            else:
                self.writer.add_scalar('train/uniform_likelihoods', 0, self.num_steps)
            best_idxs = np.random.choice(np.arange(len(uncertainty_vector)), size=num_goals, p=likelihoods)
            goals = selected_goals[best_idxs]
            if num_goals == 1:
                return goals[0]
            return goals

        # sigma difficulty sampling
        else:
            difficulty = self.sample_truncated_gaussian()
            target_difficulty = (uncertainty_vector.max() - uncertainty_vector.min()) * difficulty + uncertainty_vector.min()
            error = abs(uncertainty_vector - target_difficulty)

            min_error_idxs = (error * -1).topk(num_goals)[1].cpu()
            goals = selected_goals[min_error_idxs]
            return goals
        
    def apply_q_cutoff(self, goal_candidates, obs):
        with torch.no_grad():
            values = self.agent.get_value_distance(obs, goal_candidates).cpu()
        # apply the q filter
        filtered_candidates = goal_candidates[values >= self.args.hard_cutoff]
        filtered_obs = obs[values >= self.args.hard_cutoff]
        if filtered_candidates.shape[0] == 0:
            return goal_candidates, obs
        return filtered_candidates, filtered_obs

    def select_random_goal(self):
        all_ags, _ = self.buffer.get_all_ags()
        idx = np.random.choice(np.arange(all_ags.shape[0]))
        return all_ags[idx]

    def select_gaussian_3d_goal(self):
        scale = np.array([self.args.initial_gaussian_sigma, self.args.initial_gaussian_sigma])
        xy_goal = np.random.normal(loc=self.env.goal[:2], scale=scale)
        qpos_scale = np.array([0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3])
        qpos_goal = np.random.normal(loc=np.zeros(13), scale=qpos_scale)
        return np.concatenate([xy_goal, qpos_goal])
        
    def select_goal(self, observation):
        if self.args.use_her:
            return observation['desired_goal']
        if self.num_steps == 0:  # first time
            return observation['achieved_goal']
        elif self.num_steps <= self.args.init_random_steps:  # during random walk
            if not self.args._3D:
                return self.select_random_goal()
            else:
                return self.select_gaussian_3d_goal()

        selected_goals, selected_obs = self.buffer.get_all_ags()  # return all ags and states (identical for 2D maze)
        selected_goals, selected_obs = self.downsample([selected_goals, selected_obs], output_size=400000)
        obs = np.repeat(observation['observation'][None, :], selected_goals.shape[0], axis=0)
        
        if self.args.hard_cutoff and self.first_goal:
            selected_goals, selected_obs = self.apply_q_cutoff(selected_goals, obs)
        
        if self.args.log_q_histogram:
            with torch.no_grad():
                values = np.array(self.agent.get_value_distance(selected_obs, selected_goals).cpu())
            values = self.downsample([values], output_size=1000)
            values = np.concatenate(values)
            self.writer.add_histogram('Q values', values, self.num_steps)
        # # prevent an unknown bug that popped up
        # if selected_goals.shape[0] != obs.shape[0]:  # TODO: remove this
        #     obs = np.repeat(obs[0][None, :], selected_goals.shape[0], axis=0)
        original_obs = np.repeat(observation['observation'][None, :], selected_goals.shape[0], axis=0)

        with torch.no_grad():
            if self.args.use_uncertainty:
                uncertainty = self.agent.q_uncertainty(selected_goals, obs)
            elif self.args.use_forward_uncertainty:
                uncertainty = self.agent.forward_uncertainty(selected_obs, selected_goals)
            elif self.args.use_decompositional:
                uncertainty = self.agent.decompositional_uncertainty(selected_obs, selected_goals)
            elif self.args.use_decompositional_difference:
                uncertainty = self.agent.decompositional_difference_uncertainty(selected_obs, selected_goals)
            if self.args.uncertainty_vector_sigma:  # ablation
                uncertainty += torch.FloatTensor(np.random.normal(loc=0, scale=self.args.uncertainty_vector_sigma, size=uncertainty.shape[0])).to(self.device)
            goal_candidates = self.select_goal_from_uncertainty_vector(uncertainty, selected_goals, num_goals=self.args.goal_candidates, observation=observation)
            
            if self.args.log_uncertainty_histogram:
                self.writer.add_histogram('uncertainties', uncertainty, self.num_steps)
                normalized_uncertainty = uncertainty - uncertainty.min()
                normalized_uncertainty /= normalized_uncertainty.max()
                self.writer.add_histogram('normalized uncertainties', normalized_uncertainty, self.num_steps)
            if len(goal_candidates.shape) == 1:
                goal_value_distance = self.agent.get_value_distance(observation['observation'], goal_candidates)
                self.writer.add_scalar('train/Goal value distance', goal_value_distance, self.num_steps)
                return goal_candidates
            if self.first_goal:
                goal = goal_candidates[0]
            elif self.args.goal_candidates == 1:
                goal = goal_candidates
            else:
                obs_repeated = np.repeat(obs[0][None, :], goal_candidates.shape[0], axis=0)
                best_candidate_idx = self.agent.get_value_distance(obs_repeated, goal_candidates).argmin().item()
                goal = goal_candidates[best_candidate_idx]

            goal_value_distance = self.agent.get_value_distance(observation['observation'], goal)
            self.writer.add_scalar('train/Goal value distance', goal_value_distance, self.num_steps)
            return goal
    
    def handle_reset(self):
        self.first_goal = True
        observation = self.env.reset()
        obs = observation['observation']
        ag = observation['achieved_goal']
        g = self.select_goal(observation)

        if not self.args._3D and not self.args.robot:
            if self.num_steps % self.args.logging_freq == 0:
                hops_goal_to_final = self.num_hops_from_goal(g)
                self.writer.add_scalar('Q/hops goal to final goal', hops_goal_to_final, self.num_steps)
                hops_start_to_sampled_goal = self.env.get_hops_source_to_target(obs, g)
                self.writer.add_scalar('Q/hops start to sampled goal', hops_start_to_sampled_goal, self.num_steps)
        return observation, obs, ag, g


    def train(self):
        print('Training...')
        while self.num_steps < self.args.total_num_steps:
            print('num steps:', self.num_steps)
            mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
            observation, obs, ag, g = self.handle_reset()
            self.buffer.start_new_episode()
            any_success = False
            self.original_observation = observation.copy()
            logged_free_time = False
            num_goals_reached = 0
            for t in range(self.args.episode_length):
                with torch.no_grad():
                    input_tensor = self.preproc_inputs(obs, g) # normalize, concatenate obs+goal
                    action = self.agent.actor_network(input_tensor).cpu().numpy().squeeze()
                    action = self.add_noise(action)

                observation_new, reward, done, info = self.env.step(action)
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                self.num_steps += 1

                self.buffer.store_step(obs, obs_new, ag, ag_new, g, action)

                mb_obs.append(obs)
                mb_ag.append(ag)
                mb_g.append(g)
                mb_actions.append(action)

                observation = observation_new
                obs = obs_new
                ag = ag_new

                if self.args.save and self.num_steps % self.args.save_freq == 0:
                    self.save()

                # working
                # todo
                # if self.num_steps % self.args.logging_freq == 0:
                #     self.log_overall_q_errors()

                success = np.linalg.norm(obs[:2] - g[:2]) < self.env.distance_threshold
                if success and self.args.new_goal_walk:
                    num_goals_reached += 1
                # change randomness after first goal
                if success and self.first_goal:
                    self.args.noise_eps = self.args.late_noise_eps
                    self.args.random_eps = self.args.late_random_eps
                # first free time of the episode
                if success and not any_success:
                    # Important note: This will not log if you fail to reach the first goal!
                    free_time = self.args.episode_length - t
                    self.writer.add_scalar('train/Free time', free_time, self.num_steps)
                    self.writer.add_scalar('train/Time to first goal', t, self.num_steps)
                    if not self.args._3D and not self.args.robot:
                        hops_to_first_goal = self.env.get_hops_source_to_target(self.original_observation['observation'], obs)
                        self.writer.add_scalar('train/Optimal time to first goal', hops_to_first_goal, self.num_steps)
                        self.writer.add_scalar('train/Steps wasted to first goal', t - hops_to_first_goal, self.num_steps)
                    logged_free_time = True
                any_success = success or any_success

                if self.num_steps > self.args.num_noupdate_steps:
                    self.update_networks(self.num_steps)
                
                if self.num_steps % self.args.eval_freq == 0:
                    success_rate, avg_closest, early_rate, hops_to_end_rate = self.eval_agent()
                    self.writer.add_scalar('eval/Eval Success Rate', success_rate, self.num_steps)
                    if not self.args._3D:
                        self.writer.add_scalar('eval/Hops to end rate', hops_to_end_rate, self.num_steps)
                    print(f'step {self.num_steps} success rate {round(success_rate, 3)}')

                elif success and self.args.new_goal_walk:
                    self.first_goal = False
                    g = self.select_goal(observation)
                    success = False

            if self.args.bootstrapped:
                if not self.args.use_per:
                    self.buffer.end_episode()
                else:
                    self.buffer.replay_buffer.end_episode()

            if self.args.new_goal_walk:
                self.writer.add_scalar('train/num_goals_reached', num_goals_reached, self.num_steps)
            self.intrinsic_performance_deque.append(int(any_success))
            self.writer.add_scalar('Q/Intrinsic success rate', np.mean(self.intrinsic_performance_deque), self.num_steps)

            if not self.args._3D and not self.args.robot:
                hops_to_goal = self.num_hops_from_goal(obs)
                self.writer.add_scalar('train/post-episode hops to goal', hops_to_goal, self.num_steps)
                hops_from_start = self.env.get_hops_source_to_target(self.original_observation['observation'], obs)
                self.writer.add_scalar('train/post-episode hops from start', hops_from_start, self.num_steps)
            elif not self.args.robot:
                adjusted_obs = obs[:2] / self.env.size_scaling
                adjusted_goal = self.env.goal / self.env.size_scaling
                adjusted_start = self.original_observation['observation'][:2] / self.env.size_scaling
                hops_to_goal = self.env.maze_2d.get_hops_source_to_target(adjusted_obs, adjusted_goal)
                hops_from_start = self.env.maze_2d.get_hops_source_to_target(adjusted_obs, adjusted_start)
                self.writer.add_scalar('train/post-episode hops to goal', hops_to_goal, self.num_steps)
                self.writer.add_scalar('train/post-episode hops from start', hops_from_start, self.num_steps)
                
            if not logged_free_time:
                self.writer.add_scalar('train/Free time', 0, self.num_steps)
                
            mb_obs.append(obs)
            mb_ag.append(ag)
            
            mb_obs = np.array(mb_obs)
            mb_ag = np.array(mb_ag)
            mb_g = np.array(mb_g)
            mb_actions = np.array(mb_actions)

            self.update_normalizer(mb_obs, mb_ag, mb_g, mb_actions)

            position = observation['achieved_goal']
            # self.writer.add_scalar('train/hops from goal', self.num_hops_from_goal(position), self.num_steps)

    def num_hops_from_goal(self, position):
        distances = np.linalg.norm(position[None, :] - self.sampled_loc, axis=1)
        min_dist_idx = np.argmin(distances)
        return self.optimal_hops[min_dist_idx]

    def add_noise(self, action):
        action += self.args.noise_eps * self.args.action_max * np.random.randn(*action.shape)
        action = np.clip(action, -self.args.action_max, self.args.action_max)

        random_actions = np.random.uniform(low=-self.args.action_max, high=self.args.action_max, \
                                            size=self.args.action_dim)
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    def update_networks(self, num_steps):
        num_batches = self.args.num_critic_batches
        critic_loss, policy_loss, forward_loss = None, None, None
        forward_loss = torch.FloatTensor(np.array([0.]))
        num_batches = max(self.args.num_critic_batches, self.args.num_reward_batches, self.args.num_forward_batches)
        for update_idx in range(num_batches):
            if update_idx < self.args.num_critic_batches:
                critic_loss = self.agent.get_critic_loss(num_steps)
                self.agent.critic_optim.zero_grad()
                critic_loss.backward()
                self.agent.critic_optim.step()

                # target networks
                soft_update(self.agent.critic_target_network, self.agent.critic_network, self.args.polyak)
                if self.num_steps % self.args.logging_freq == 0:
                    self.writer.add_scalar('train/critic loss', critic_loss.item(), self.num_steps)

            # forward head
            if update_idx < self.args.num_forward_batches:
                forward_loss = self.agent.get_forward_loss(num_steps)
                self.agent.forward_critic_optim.zero_grad()
                forward_loss.backward()
                self.agent.forward_critic_optim.step()
                if self.num_steps % self.args.logging_freq == 0:
                    self.writer.add_scalar('train/forward loss', forward_loss.item(), self.num_steps)


            # reward head
            if update_idx < self.args.num_reward_batches:
                reward_loss = self.agent.get_reward_loss(num_steps)
                self.agent.reward_critic_optim.zero_grad()
                reward_loss.backward()
                self.agent.reward_critic_optim.step()
                if self.num_steps % self.args.logging_freq == 0:
                    self.writer.add_scalar('train/reward loss', reward_loss.item(), self.num_steps)

        policy_loss = self.get_policy_loss()
        self.agent.actor_optim.zero_grad()
        policy_loss.backward()
        self.agent.actor_optim.step()
        soft_update(self.agent.actor_target_network, self.agent.actor_network, self.args.polyak)

        return critic_loss.item(), forward_loss.item(), policy_loss.item()
                
    def save(self):
        actor_save_path = os.path.join(self.writer.log_dir, f'actor{self.num_steps}.torch')
        torch.save(copy.deepcopy(self.actor_network).cpu(), actor_save_path)
        critic_save_path = os.path.join(self.writer.log_dir, f'critic{self.num_steps}.torch')
        torch.save(copy.deepcopy(self.critic_network).cpu(), critic_save_path)
        with open(os.path.join(self.writer.log_dir, f'o_norm{self.num_steps}.pkl'), 'wb') as f:
            pickle.dump(self.o_norm, f)
        with open(os.path.join(self.writer.log_dir, f'g_norm{self.num_steps}.pkl'), 'wb') as f:
            pickle.dump(self.g_norm, f)
        buffer_save_path = os.path.join(self.writer.log_dir, f'buffer{self.num_steps}.npy')
        np.save(buffer_save_path, self.buffer.obs[:self.buffer.current_size])

    def update_normalizer(self, mb_obs, mb_ag, mb_g, mb_actions):
        mb_obs_next = mb_obs[1:, :]
        mb_ag_next = mb_ag[1:, :]
        num_transitions = mb_actions.shape[0]
        obs, g = self.buffer.single_sample_her_transitions(mb_obs, mb_ag_next, mb_g, num_transitions)
        obs, g = self.clip_og(obs, g)

        with np.errstate(under='print'):
            self.o_norm.update(obs)
            self.g_norm.update(g)
            self.o_norm.recompute_stats()
            self.g_norm.recompute_stats()

    # in 2D maze environment clipping will never matter
    def clip_og(self, o, g):
        # o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        # g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    def get_policy_loss(self):
        if self.args.use_per:
            obs, obs_next, ag, ag_next, actions, g, rewards = self.buffer.replay_buffer.sample(self.args.batch_size)[:7]
        else:
            obs, obs_next, ag, ag_next, actions, g, rewards = self.buffer.sample(self.args.batch_size)[:7]

        inputs_norm_tensor = self.preproc_inputs(obs, g)

        actions_real = self.agent.actor_network(inputs_norm_tensor)
        # if self.args.use_gaussian:
        #     actor_loss = -self.agent.critic_network(inputs_norm_tensor, actions_real, combine=True)[0]
        # else:
        #     actor_loss = -self.agent.critic_network(inputs_norm_tensor, actions_real)
        if self.args.use_gaussian_q:
            actor_loss = -self.agent.critic_network(inputs_norm_tensor, actions_real)[0]
        else:
            actor_loss = -self.agent.critic_network(inputs_norm_tensor, actions_real)
        actor_loss = actor_loss.mean()
        
        actor_loss += (actions_real / self.args.action_max).pow(2).mean()
        return actor_loss

    def eval_agent(self):
        successes, all_closest, early_successes = list(), list(), list()
        for _ in range(self.args.n_test_rollouts):
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            closest = np.inf
            early_success = False
            hops_to_end = list()
            for _ in range(self.args.episode_length):
                with torch.no_grad():
                    input_tensor = self.preproc_inputs(obs, g)
                    action = self.agent.actor_network(input_tensor).cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(action)
                obs = observation_new['observation']
                ag = observation_new['achieved_goal']
                distance = np.linalg.norm(g[:2] - ag[:2])
                
                if distance < closest:
                    closest = distance
                success = distance <= self.env.distance_threshold
                early_success = early_success or success
            if not self.args._3D and not self.args.robot:
                hops_to_end.append(self.num_hops_from_goal(obs))
            successes.append(success)
            all_closest.append(closest)
            early_successes.append(early_success)
        success_rate = np.array(successes).mean()
        if success_rate == 1.0 and self.args.use_convergence_l2:
            self.agent.actor_optim.param_groups[0]['weight_decay'] = self.args.convergence_actor_l2
            self.agent.critic_optim.param_groups[0]['weight_decay'] = self.args.convergence_critic_l2
            self.agent.forward_critic_optim.param_groups[0]['weight_decay'] = self.args.convergence_forward_l2
        
        avg_closest = np.array(all_closest).mean()
        early_success_rate = np.array(early_successes).mean()
        if not self.args._3D and not self.args.robot:
            hops_to_end_rate = np.array(hops_to_end).mean()
        else:
            hops_to_end_rate = 0
        return success_rate, avg_closest, early_success_rate, hops_to_end_rate

if __name__ == '__main__':
    DDPGAgent().train()
