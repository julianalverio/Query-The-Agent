"""Random policy on an environment."""

import numpy as np
import argparse

from gym.envs.maze_test.create_maze_env import create_maze_env
from gym.envs.maze_test.maze_env import CustomMazeEnv
from gym.envs.maze_test.maze_generators.obstacle_maze import generate_obstacle_maze
from gym.envs.maze_test.maze_generators.prim_maze import generate_prim_maze
from gym.envs.maze_test.maze_manager_2d import MazeEnv2D
from gym.envs.maze_test.maze_manager_3d import MazeEnv3D


def get_goal_sample_fn(env_name, evaluate):
    if env_name == 'AntMaze':
        # NOTE: When evaluating (i.e. the metrics shown in the paper,
        # we use the commented out goal sampling function.    The uncommented
        # one is only used for training.
        if evaluate:
            return lambda: np.array([0., 16.])
        else:
            return lambda: np.random.uniform((-4, -4), (20, 20))
    elif env_name == 'AntFlat':
        if evaluate:
            return lambda: np.array([0., 16.])
        else:
            return lambda: np.random.uniform((-4, -4), (20, 20))
    elif env_name == 'AntPush':
        return lambda: np.array([0., 19.])
    elif env_name == 'AntFall':
        return lambda: np.array([0., 27., 4.5])
    else:
        assert False, 'Unknown env'


def get_reward_fn(env_name):
    if env_name == 'AntMaze':
        return lambda obs, goal: -np.sum(np.square(obs[0, :2] - goal)) ** 0.5
    elif env_name == 'AntFlat':
        return lambda obs, goal: -np.sum(np.square(obs[0, :2] - goal)) ** 0.5
    elif env_name == 'AntPush':
        return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
    elif env_name == 'AntFall':
        return lambda obs, goal: -np.sum(np.square(obs[:3] - goal)) ** 0.5
    else:
        assert False, 'Unknown env'


def success_fn(last_reward):
    return last_reward > -5.0


class CustomEnvWithGoal(object):
    def __init__(self, maze_type='prim', maze=None, height=10, width=10):
        maze_type = maze_type.lower()
        assert maze_type in ('obstacle', 'prim')
        if maze is None:
            if maze_type == 'obstacle':
                maze = generate_obstacle_maze(height, width)
            elif maze_type == 'prim':
                maze = generate_prim_maze(height, width)

        self.base_env = self.create_custom_maze_env(maze, maze_type)
        self.evaluate = False
        self.reward_fn = lambda obs, goal: -np.sum(np.square(obs[0, :2] - goal)) ** 0.5
        self.goal = None
        self.distance_threshold = 5
        self.count = 0
        self.num_envs = 1  # so we can identify this from ParallelEnv easily

    def create_custom_maze_env(self, maze, maze_type):
        maze_height_dimension, maze_width_dimension = maze.shape
        env = CustomMazeEnv(
            maze_height=0.5,
            maze_type=maze_type,
            maze_size_scaling=8,
            maze=maze,
            maze_height_dimension=maze_height_dimension,
            maze_width_dimension=maze_width_dimension,
        )
        return env

    def goal_sample_fn(self):
        if self.evaluate:
            return lambda: np.array([0., 16.])
        else:
            return lambda: np.random.uniform((-4, -4), (20, 20))

    def activate_evaluation(self):
        self.evaluate = True

    def deactivate_evaluation(self):
        self.evaluate = False

    def get_robot_position(self):
        return self.base_env.wrapped_env.get_xy()[None, :]

    def set_robot_position(robot_position):
        self.base_env.wrapped_env.set_xy(robot_position)

    def seed(self, seed):
        self.base_env.seed(seed)

    def reset(self):
        obs = self.base_env.reset()
        self.count = 0
        # numpy array of shape (2,)
        self.goal = self.goal_sample_fn()

        return obs[None, :], self.goal[None, :]

    def step(self, a):
        obs, _, done, info = self.base_env.step(a)
        obs = obs[None, :]
        done = np.array([done])  # don't return True for done if you time out. We want to use this data in maze env environments.
        reward = self.reward_fn(obs, self.goal)
        self.count += 1
        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        self.base_env.render(mode=mode)

    def get_image(self):
        self.render()
        data = self.base_env.viewer.get_image()

        img_data = data[0]
        width = data[1]
        height = data[2]

        tmp = np.fromstring(img_data, dtype=np.uint8)
        image_obs = np.reshape(tmp, [height, width, 3])
        image_obs = np.flipud(image_obs)

        return image_obs

    @property
    def action_space(self):
        return self.base_env.action_space

    @property
    def observation_space(self):
        return self.base_env.observation_space



class EnvWithGoal(object):
    def __init__(self, env_name='AntMaze', n_bins=0, maze=None):
        self.base_env = create_maze_env(env_name, n_bins=n_bins)
        self.env_name = env_name
        self.evaluate = False
        self.reward_fn = get_reward_fn(env_name)
        self.goal = None
        self.distance_threshold = 5
        self.count = 0
        self.num_envs = 1  # so we can identify this from ParallelEnv easily
        self.state_dim = self.base_env.observation_space.shape[0]
        self.action_dim = self.base_env.action_space.shape[0]

    def activate_evaluation(self):
        self.evaluate = True

    def deactivate_evaluation(self):
        self.evaluate = False

    def get_robot_position(self):
        return self.base_env.wrapped_env.get_xy()[None, :]

    def set_robot_position(robot_position):
        self.base_env.wrapped_env.set_xy(robot_position)

    def seed(self, seed):
        self.base_env.seed(seed)

    def reset(self):
        # self.viewer_setup()
        self.goal_sample_fn = get_goal_sample_fn(self.env_name, self.evaluate)
        obs = self.base_env.reset()
        self.count = 0
        # numpy array of shape (2,)
        self.goal = self.goal_sample_fn()

        return obs[None, :], self.goal[None, :]

        # return {
        #     'observation': obs,
        #     'achieved_goal': obs[:2],
        #     'desired_goal': self.goal,
        # }

    def step(self, a):
        obs, _, done, info = self.base_env.step(a)
        obs = obs[None, :]
        # done = np.array([done or self.count >= 500])  # custimization for batching!
        done = np.array([done])  # don't return True for done if you time out. We want to use this data in maze env environments.
        reward = self.reward_fn(obs, self.goal)
        self.count += 1
        # next_obs = {
        #     'observation': obs,
        #     'achieved_goal': obs[:2],
        #     'desired_goal': self.goal,
        # }
        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        self.base_env.render(mode=mode)

    def get_image(self):
        self.render()
        data = self.base_env.viewer.get_image()

        img_data = data[0]
        width = data[1]
        height = data[2]

        tmp = np.fromstring(img_data, dtype=np.uint8)
        image_obs = np.reshape(tmp, [height, width, 3])
        image_obs = np.flipud(image_obs)

        return image_obs

    @property
    def action_space(self):
        return self.base_env.action_space

    @property
    def observation_space(self):
        return self.base_env.observation_space

def run_environment(env_name, episode_length, num_episodes):
    env = EnvWithGoal(
            create_maze_env(env_name),
            env_name)

    def action_fn(obs):
        action_space = env.action_space
        action_space_mean = (action_space.low + action_space.high) / 2.0
        action_space_magn = (action_space.high - action_space.low) / 2.0
        random_action = (action_space_mean +
            action_space_magn *
            np.random.uniform(low=-1.0, high=1.0,
            size=action_space.shape))

        return random_action

    rewards = []
    successes = []
    for ep in range(num_episodes):
        rewards.append(0.0)
        successes.append(False)
        obs = env.reset()
        for _ in range(episode_length):
            env.render()
            print(env.get_image().shape)
            obs, reward, done, _ = env.step(action_fn(obs))
            rewards[-1] += reward
            successes[-1] = success_fn(reward)
            if done:
                break

        print('Episode {} reward: {}, Success: {}'.format(ep + 1, rewards[-1], successes[-1]))

    print('Average Reward over {} episodes: {}'.format(num_episodes, np.mean(rewards)))
    print('Average Success over {} episodes: {}'.format(num_episodes, np.mean(successes)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="AntEnv", type=str)
    parser.add_argument("--episode_length", default=500, type=int)
    parser.add_argument("--num_episodes", default=100, type=int)

    args = parser.parse_args()
    run_environment(args.env_name, args.episode_length, args.num_episodes)
