import sys
import os
import yaml
import re
CURRENT = os.path.dirname(os.path.realpath(__file__))
ROOT = '/'.join(CURRENT.split('/')[:-1])
GYM_ROOT = os.path.join(ROOT, 'gym')
sys.path.insert(0, GYM_ROOT)
import gym  # must import custom gym without pip-installing

sys.path.insert(0, ROOT)
sys.path.insert(0, CURRENT)
import random
import datetime
import numpy as np
import itertools
import argparse
import json
import copy

from sklearn.neighbors import KernelDensity
from collections import deque
import time
from sac.sac import SAC
from td3.TD3 import TD3
from sac.replay_memory import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam


class ArgsHolder(object):
    def __init__(self, args):
        for key, value in dict(args).items():
            setattr(self, key, value)

    def merge(self, args):
        for key, value in dict(args).items():
            if not hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f'Config attribute {key} already exists')
    
class Handler(object):
    def __init__(self):
        self.setup_config()
        self.setup_tb_writer()
        self.setup_os_env()
        self.get_envs()
        self.seed()

        if self.configs.cuda == 1:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device('cpu')

        if self.configs.agent_type == 'sac':
            self.agent = SAC(
                state_dim=self.env.observation_space.shape[0],
                action_space=self.env.action_space,
                configs=self.configs,
                device=self.device)
        elif self.configs.agent_type == 'td3':
            self.agent = TD3(
                state_dim=self.env.observation_space.shape[0],
                action_dim=self.env.action_space.shape[0],
                max_action=float(self.env.action_space.high[0]),
                configs=self.configs,
                device=self.device)

        self.replay_buffer = ReplayBuffer(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0],
            capacity=self.configs.replay_size,
            device=self.device)

    def setup_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='simple.yml')
        parser.add_argument('--seed', type=int)
        parser.add_argument('--log_dir', type=str)
        parsed = parser.parse_args()

        config = yaml.safe_load(open(parsed.config, 'r'))
        config = ArgsHolder(config)
        # robot-specific config
        robot_config_path = os.path.join(CURRENT, config.agent_type, 'configs', f'{config.robot}.yml')
        config.merge(yaml.safe_load(open(robot_config_path)))
        if parsed.seed is not None:
            config.seed = parsed.seed
        self.configs = config

    def setup_tb_writer(self):
        log_root = os.path.join(CURRENT, 'simple_logs')
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        writer_path = os.path.join(log_root, f'{self.configs.agent_type}_{self.configs.robot}_seed{self.configs.seed}')
        self.writer = SummaryWriter(writer_path)

        # save configs
        config_save_path = os.path.join(writer_path, 'configs.json')
        with open(config_save_path, 'w+') as f:
            json.dump(self.configs.__dict__, f)

    def setup_os_env(self):
        cores = str(self.configs.cores)
        os.environ["OMP_NUM_THREADS"] = cores # export OMP_NUM_THREADS=4
        os.environ["OPENBLAS_NUM_THREADS"] = cores # export OPENBLAS_NUM_THREADS=4 
        os.environ["MKL_NUM_THREADS"] = cores # export MKL_NUM_THREADS=6
        os.environ["VECLIB_MAXIMUM_THREADS"] = cores # export VECLIB_MAXIMUM_THREADS=4
        os.environ["NUMEXPR_NUM_THREADS"] = cores # export NUMEXPR_NUM_THREADS=6

    def get_envs(self):
        if 'ant' in self.configs.robot:
            env_name = 'Ant-v2'
        if 'humanoid' in self.configs.robot:
            env_name = 'Humanoid-v2'
        if 'ball' in self.configs.robot:
            env_name = 'Ball'
        self.env = gym.custom_make(env_name)
        eval_env = gym.custom_make(env_name)
        self.eval_env = eval_env

    def seed(self):
        seed = self.configs.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.env.seed(seed)
        self.env.action_space.np_random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.eval_env.seed(seed+1)
        self.eval_env.action_space.np_random.seed(seed+1)


    def evaluate(self, total_steps):
        avg_reward = 0.
        for _  in range(self.configs.num_eval_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = self.agent.select_action(state[None, :], eval=True)

                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        avg_reward /= self.configs.num_eval_episodes
        self.writer.add_scalar('Average Return', avg_reward, total_steps)

    def save(self, total_steps):
        save_root = self.writer.log_dir
        actor_filename = f'{total_steps}_actor.torchdict'
        critic_filename = f'{total_steps}_critic.torchdict'
        actor_path = os.path.join(save_root, actor_filename)
        critic_path = os.path.join(save_root, critic_filename)
        self.agent.save_model(actor_path, critic_path)


    def train(self):
        total_steps = 0
        while 1:
            episode_steps = 0
            done = False
            state = self.env.reset()
            episode_reward = 0
            while not done:
                if total_steps < self.configs.start_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.select_action(state[None, :])

                if self.replay_buffer.size > self.configs.batch_size:
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.agent.update_parameters(self.replay_buffer, self.configs.batch_size)

                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                replay_done = 0 if episode_steps == self.env._max_episode_steps else float(done)
                self.replay_buffer.add(state, action, next_state, reward, replay_done)
                state = next_state
                if total_steps % 100 == 0:
                    self.writer.add_scalar('train return', episode_reward, total_steps)
                if total_steps % self.configs.eval_freq == 0:
                    self.evaluate(total_steps)
                if total_steps % self.configs.save_freq == 0:
                    self.save(total_steps)

            print('total steps:', total_steps)
            if total_steps >= self.configs.num_steps:
                return

if __name__ == '__main__':
    Handler().train()
