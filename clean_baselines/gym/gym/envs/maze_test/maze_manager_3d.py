import sys
import os
import pathlib
current = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, os.path.join(current, '..'))
import numpy as np
import argparse
import gym
import os
import xml.etree.ElementTree as ET
import tempfile
from gym.envs.maze_test.ant import AntEnv
from gym.envs.maze_test.ball import BallEnv
from gym.envs.maze_test.half_cheetah import HalfCheetahEnv
from gym.envs.maze_test.hopper import HopperEnv
from gym.envs.maze_test.humanoid import HumanoidEnv
from gym.envs.maze_test.swimmer import SwimmerEnv
from gym.envs.maze_test.walker2d import Walker2dEnv

from gym import spaces
from gym.spaces import Box, Dict


# class MazeEnv3D(object):
class MazeEnv3D(gym.GoalEnv):
    def __init__(self, maze, args=None):
        self.args = args
        self.maze_2d = maze
        self.grid = maze.grid
        self.height = 0.5
        self.t = 0
        if hasattr(args, 'maze_size_scaling'):
            self.size_scaling = args.maze_size_scaling
        else:
            self.size_scaling = 8
        self.distance_threshold = 5  # from HIRO paper
        self.position_threshold = 0.393  # 22.5 degrees in radians

        self.setup_env()
        self._max_episode_steps = 1000

    def seed(self, seed):
        self.wrapped_env.seed(seed)

    def set_robot_position(self, position):
        return self.wrapped_env.set_xy(position)

    def get_robot_position(self):
        return self.wrapped_env.get_xy()

    def set_robot_height(self, height):
        self.wrapped_env.set_z(height)  # TEST!

    def setup_env(self):
        current_dir = '/'.join(os.path.realpath(__file__).split('/')[:-1])

        if 'ant' in self.args.agent_type.lower():
            xml_name = 'ant.xml'
            env_class = AntEnv
        elif 'humanoid' in self.args.agent_type.lower():
            xml_name = 'humanoid.xml'
            env_class = HumanoidEnv
        elif 'cheetah' in self.args.agent_type.lower():
            xml_name = 'half_cheetah.xml'
            env_class = HalfCheetahEnv
        elif 'ball' in self.args.agent_type.lower():
            xml_name = 'ball.xml'
            env_class = BallEnv
        elif 'hopper' in self.args.agent_type.lower():
            xml_name = 'hopper.xml'
            env_class = HopperEnv
        elif 'swimmer' in self.args.agent_type.lower():
            xml_name = 'swimmer.xml'
            env_class = SwimmerEnv
        elif 'walker' in self.args.agent_type.lower():
            xml_name = 'walker2d.xml'
            env_class = Walker2dEnv

        xml_path = os.path.join(current_dir, 'assets', xml_name)
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")
        self.worldbody = worldbody

        # set up blocks
        for row_idx in range(self.grid.shape[0]):
          for col_idx in range(self.grid.shape[1]):
            if self.grid[row_idx, col_idx]:
              ET.SubElement(
                  worldbody, "geom",
                  name="block_%d_%d" % (row_idx, col_idx),
                  pos="%f %f %f" % (col_idx * self.size_scaling + self.size_scaling/2.,
                                    row_idx * self.size_scaling + self.size_scaling/2.,
                                    self.height / 2 * self.size_scaling),
                  size="%f %f %f" % (0.5 * self.size_scaling,
                                     0.5 * self.size_scaling,
                                     self.height / 2 * self.size_scaling),
                  type="box",
                  material="",
                  contype="1",
                  conaffinity="1",
                  rgba="0.4 0.4 0.4 1",
              )

        _, file_path = tempfile.mkstemp(text=True, suffix='.xml')
        tree.write(file_path)
        # self.wrapped_env = AntEnv(file_path=file_path)
        self.wrapped_env = env_class(file_path=file_path)
        self.reset()

    def step(self, action):
        self.t += 1
        _, _, done, _ = self.wrapped_env.step(action)
        observation = self._get_obs()
        reward = self.compute_reward(observation['achieved_goal'], observation['desired_goal'], None)
        success = self.compute_success(reward)

        xy_distance = np.linalg.norm(observation['achieved_goal'][:2] - observation['desired_goal'][:2])

        # desired goal should be get_obs with x, y overwritten (similar for achieved goal)
        info = {'is_success': success}
        info['xy-distance'] = xy_distance

        # reward is only for entropy maximization
        return observation, reward, done, info  # middle 2 are ignored (obs, reward, done, info)

    def _get_obs(self):
        obs = self.wrapped_env._get_obs()
        achieved_goal = self.wrapped_env.get_achieved_goal()
        # only shape of desired goal ever matters, except for at evaluation time
        desired_goal = np.zeros_like(achieved_goal)
        desired_goal[:2] = self.goal

        return dict({
            'observation': obs,
            'desired_goal': desired_goal,
            'achieved_goal': achieved_goal,
            'state_observation': obs,
            'state_desired_goal': desired_goal,
            'state_achieved_goal': achieved_goal,
        })

    def reset(self):
        self.maze_2d.reset()

        self.t = 0
        self.wrapped_env.reset()
        # must transpose bc numpy x is vertical and numpy y is horizontal
        start = np.array([self.maze_2d.position[1], self.maze_2d.position[0]]) * self.size_scaling
        self.set_robot_position(start)
        if self.args.agent_type == 'ant':
            height = 0.6
        elif self.args.agent_type == 'humanoid':
            height = 1.3
        elif self.args.agent_type == 'swimmer':
            height = 0
        elif 'cheetah' in self.args.agent_type:
            height = 0.
        elif 'hopper' in self.args.agent_type:
            height = 1.25
        elif 'walker' in self.args.agent_type:
            height = 1.21
        elif self.args.agent_type == 'ball':
            height = 0

        self.set_robot_height(height)
        self.goal = self.maze_2d.goal * self.size_scaling


        ## visualize the region where goals are sampled from
        # goal_region_site_id = self.wrapped_env.sim.model.site_name2id('completion')
        # goal_region_3d = np.array([self.maze_2d.goal_location[1] + 0.5, self.maze_2d.goal_location[0] + 0.5, 0])
        # goal_region_3d *= self.size_scaling
        # self.wrapped_env.sim.model.site_pos[goal_region_site_id] = goal_region_3d

        # visualize the goal
        goal_site_id = self.wrapped_env.sim.model.site_name2id('goal')
        goal_3d = np.array([self.goal[1], self.goal[0], 0.5])
        self.wrapped_env.sim.model.site_pos[goal_site_id] = goal_3d
        self.wrapped_env.sim.forward()
        return self._get_obs()

    @property
    def viewer(self):
        return self.wrapped_env.viewer

    def set_viewer(self, viewer, mode):
        self.wrapped_env.viewer = viewer
        self.wrapped_env._viewers[mode] = viewer

    @property
    def model(self):
        return self.wrapped_env.model

    def render(self, mode='rgb_array', camera_name=None):
        return self.wrapped_env.render(mode=mode, camera_name=camera_name)

    def compute_success(self, reward):
        if self.args.dense_reward:
            return abs(reward) <= self.distance_threshold
        else:
            return bool(reward + 1.)

    def compute_reward(self, achieved_goal, goal, info=None):
        if self.args.dense_reward:
            return self.compute_reward_dense(achieved_goal, goal, info)
        return self.compute_reward_sparse(achieved_goal, goal, info)

    def compute_reward_dense(self, achieved_goal, goal, info=None):
        return -np.linalg.norm(achieved_goal[..., :2] - goal[..., :2], axis=-1)

    def compute_reward_sparse(self, achieved_goal, goal, info=None):
        location_distance = np.linalg.norm(achieved_goal[..., :2] - goal[..., :2], axis=-1)
        location_reward = -(location_distance > self.distance_threshold).astype(np.float32)
        if goal.shape[0] > 2: # reward for additional dims
            qpos_distance = np.linalg.norm(achieved_goal[..., 2:] - goal[..., 2:], axis=-1)
            distance_threshold = self.position_threshold*(1./np.sqrt(goal.shape[-1] - 2))
            qpos_reward = -(qpos_distance > distance_threshold).astype(np.float32)
            if len(goal.shape) == 1:
                reward = (qpos_reward == 0 and location_reward == 0) - 1.  # if both are zero, reward is 0. -1 otherwise
            else:
                reward = np.logical_and(qpos_reward == 0, location_reward == 0) - 1
                reward = reward.astype(float)
        else:
            reward = location_reward
        return reward

    @property
    def observation_space(self):
        goal_samples = self.sample_goals(10000, True)
        goal_maxes = goal_samples.max(axis=0)*self.size_scaling
        if hasattr(self.args, 'alg'):
            if self.args.alg == 'imagined_subgoals':
                goal_maxes = np.pad(goal_maxes, (0, 28), mode='constant')

        self.goal_space = Box(low=np.zeros_like(goal_maxes), high=goal_maxes, dtype=np.float32)

        # self.obs_space = Box(low=np.array([-1.6, -1.6, -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1., -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1., -1.,  -1. ]),
        #                      high=np.array([1.6, 1.6, 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]),
        #                      dtype=np.float32)
        self.obs_space = Box(low=np.full_like(self._get_obs()['observation'],-np.inf), high=np.full_like(self._get_obs()['observation'],np.inf), dtype=np.float32)
        spaces = [
            ('observation', self.obs_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
            ('state_observation', self.obs_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.goal_space),
        ]
        return Dict(spaces)

    @property
    def action_space(self):
        return self.wrapped_env.action_space

    def sample_goals(self, num_samples, uniform=False):
        return self.maze_2d.sample_goals(num_samples, uniform=uniform)

    def _compute_xy_distances(self, obs):
        achieved_goals = obs['achieved_goal'][:]
        desired_goals = obs['desired_goal'][:]
        diff = achieved_goals - desired_goals
        return np.atleast_1d(np.linalg.norm(diff, ord=2))

    def get_xy(self):
        return self.wrapped_env.get_xy()
