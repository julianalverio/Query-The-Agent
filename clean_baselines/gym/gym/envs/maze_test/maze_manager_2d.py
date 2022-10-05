import copy
import pathlib
import os
import pickle

import numpy as np
import sys
# sys.path.append('~/hrl/clean_baselines/imagined_subgoals/gym/')
import gym
from gym import spaces
from gym.utils import seeding
import pathlib
current = pathlib.Path(__file__).parent.resolve()
import sys
# sys.path.insert(0, "/afs/csail.mit.edu/u/r/rsonecha/hrl/hindsight_experience_replay/")
from PIL import Image, ImageDraw
from gym.spaces import Box, Dict
from collections import OrderedDict


class MazeEnv2D(gym.GoalEnv):
    def __init__(self, grid, args=None):
        self.args = args
        # self.force_nonuniform = True

        try:
            self.start_location = np.argwhere(grid == 2)[0]
        except:
            import pdb; pdb.set_trace()
        self.goal_location = np.argwhere(grid == 3)[0]
        assert self.start_location.shape == (2,), 'Maze lacks start location'
        assert self.goal_location.shape == (2,), 'Maze lacks goal location'

        modified_grid = grid.copy()
        modified_grid[self.start_location[0], self.start_location[1]] = 0
        modified_grid[self.goal_location[0], self.goal_location[1]] = 0

        self.grid = modified_grid
        self.free_positions = np.argwhere(self.grid==0)
        self.reset()

        if args is not None:
            np.random.seed(args.seed)

        self.step_size = 0.1
        self.num_collision_steps = 10
        self.distance_threshold = 0.35
        self.max_action = 1
        self.block_size = 70  # for visualization only

        # for backwards compatibility
        self.action_space = Box(-self.max_action, self.max_action, shape=(2,), dtype="float32")
        self.create_observation_space()
        self._max_episode_steps = 50


    def create_observation_space(self):
        self.goal_space = Box(low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf]), dtype=np.float32)

        spaces = [
            ('observation', self.goal_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
            ('state_observation', self.goal_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.goal_space),
        ]
        self.observation_space = Dict(spaces)

    def compute_success(self, reward):
        return bool(reward + 1.)

    # for backwards compatibility for HER buffer
    # used by imagined subgoals
    def compute_rewards(self, achieved_goal, goal):
        # distance = np.linalg.norm(achieved_goal - goal, axis=-1)
        distance = np.linalg.norm(goal['achieved_goal'][:2] - goal['desired_goal'][:2])
        return -distance
        # return -(distance > self.distance_threshold).astype(np.float32)

    # for entropy maximization
    def compute_reward(self, achieved_goal, desired_goal, info=None):
        if len(achieved_goal.shape) == 2:
            ag = achieved_goal[:,:2]
            dg = desired_goal[:,:2]
        else:
            ag = achieved_goal[:2]
            dg = desired_goal[:2]
        d = np.linalg.norm(ag - dg, axis=-1)
        return -(d >= self.distance_threshold).astype(np.float32)

    # TODO: something is weird with this method signature
    def reset(self, *, uniform_goal=False):
        self.position = self.start_location + 0.5 + np.random.uniform(-0.3, 0.3, size=2)
        if hasattr(self.args, 'maze_uniform_goals'):
            if self.args.maze_uniform_goals:
                self.goal = self.sample_goals(1)[0]
            else:
                self.goal = self.goal_location + 0.5 + np.random.uniform(-0.3, 0.3, size=2)
        else:
            self.goal = self.goal_location + 0.5 + np.random.uniform(-0.3, 0.3, size=2)
        return self.get_obs()

    # to be compatible with tensorflow VDS
    def _reset_sim(self):
        self.reset()
        return True

    # Can you start at a location and transition to the other?
    def valid_transition(self, starting_position, ending_position):
        if np.linalg.norm(ending_position - starting_position) > 1.:
            return False
        action = ending_position - starting_position
        current_position = starting_position
        for _ in range(self.num_collision_steps):
            new_position = current_position + action * self.step_size
            new_block = np.floor(new_position).astype(np.uint8)
            if self.grid[new_block[0], new_block[1]]:
                return False
        return True

    def get_nearest_sampled_point(self, original_point, sampled_loc):
        nearest_idx = np.argmin(np.linalg.norm(original_point[None, :] - sampled_loc, axis=1))
        nearest = sampled_loc[nearest_idx]
        return nearest, nearest_idx

    def get_hops_source_to_target(self, original_source, original_target):
        sampled_loc, length = self.get_locations()
        source, source_idx = self.get_nearest_sampled_point(original_source, sampled_loc)
        target, _ = self.get_nearest_sampled_point(original_target, sampled_loc)
        idx_lookup = {tuple(loc):idx for idx, loc in enumerate(sampled_loc)}

        visited = np.zeros(sampled_loc.shape[0])
        visited[source_idx] = 1
        
        first_queue = [source]
        second_queue = list()
        num_hops = 1
        success = False

        while first_queue and not success:
            current = first_queue.pop()
            unvisited = sampled_loc[visited==0]
            candidates = unvisited[np.linalg.norm(current[None, :] - unvisited, axis=1) < 1.]
            
            for candidate in candidates:
                if np.all(candidate == target):
                    success = True
                    break
                if self.valid_transition(current, candidate):
                    second_queue.append(candidate)
                    idx = idx_lookup[tuple(candidate)]
                    visited[idx] = 1
                    
            if not first_queue:
                first_queue = second_queue
                second_queue = list()
                num_hops += 1
        return num_hops
        
    
    def get_optimal_value_function(self, gamma=0.99):
        sampled_loc, length = self.get_locations()
        idx_lookup = {tuple(loc):idx for idx, loc in enumerate(sampled_loc)}

        first_queue = [self.start_location + 0.5]
        second_queue = list()
        optimal_hops = np.full((len(sampled_loc)), np.nan, dtype=np.float32)
        num_hops = 1  # number of hops taken from the first location to get here
        while first_queue:
            current = first_queue.pop()
            unvisited = sampled_loc[np.isnan(optimal_hops)]
            candidates = unvisited[np.linalg.norm(current[None, :] - unvisited, axis=1) <= 1.]
            for candidate in candidates:
                if self.valid_transition(current, candidate):
                    idx = idx_lookup[tuple(candidate)]
                    optimal_hops[idx] = num_hops
                    second_queue.append(candidate)
            if not first_queue:
                first_queue = second_queue
                second_queue = list()
                num_hops += 1

        # optimal_values may contain nan values for unreachable regions
        optimal_values, max_hops = self.hops_to_q_values(optimal_hops, gamma) 
        assert optimal_values.shape[0] == sampled_loc.shape[0]

        normalized_values = optimal_values * -1
        normalized_values -= normalized_values[~np.isnan(normalized_values)].min()
        normalized_values /= normalized_values[~np.isnan(normalized_values)].max()
        return optimal_values, normalized_values, max_hops, sampled_loc, length, optimal_hops


    def hops_to_q_values(self, optimal_hops, gamma=0.99):
        optimal_hops[np.isnan(optimal_hops)] = -1  # areas that can't be reached
        hops_to_values = dict()
        hops_to_values[1] = 0
        hops_to_values[-1] = -1
        max_hops = int(optimal_hops[~np.isnan(optimal_hops)].max())
        current_q_value = 0
        for hop_count in range(max_hops+1):
            if hop_count > 1:
                current_q_value -= gamma ** (hop_count - 1)
                hops_to_values[hop_count] = current_q_value
        optimal_values = list()
        for hop in optimal_hops:
            optimal_values.append(hops_to_values[hop])
        optimal_values = np.array(optimal_values)
        optimal_values[optimal_values == -1] = np.nan  # nan --> -1 --> nan
        return optimal_values, max_hops

    def step(self, action):
        action = np.clip(action, -self.max_action, self.max_action)
        for _ in range(self.num_collision_steps):
            new_position = self.position + action * self.step_size
            new_block = np.floor(new_position).astype(np.uint8)
            if not self.grid[new_block[0], new_block[1]]:
                self.position = new_position
            else:
                break

        obs = self.get_obs()
        done = False
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], None)
        info = {'is_success': self.compute_success(reward),}
        info['xy-distance'] = self._compute_xy_distances(obs)

        return obs, reward, done, info

    def get_obs(self):
        return dict(
            desired_goal=self.goal,
            achieved_goal=self.position,
            observation=self.position,
            state_desired_goal=self.goal,
            state_achieved_goal=self.position,
            state_observation=self.position,
        )

    # to be compatible with tensorflow VDS
    def _get_obs(self):
        return self.get_obs()

    def sample_goals(self, num_samples, uniform=True):
        # if uniform and not self.force_nonuniform:
        if uniform:
            goal_positions = self.free_positions[np.random.choice(len(self.free_positions), num_samples)]
            goal_positions = goal_positions.astype(np.float32)
            goal_positions += 0.5  # center it in the square
            goal_positions += np.random.uniform(low=-0.5, high=0.5, size=(num_samples, 2))
            return goal_positions
        else:
            return self.goal_location[None, :] + np.random.uniform(size=(num_samples, 2))

    def sample_goal_space(self):
        return self.goal_location + 0.5 + np.random.uniform(-0.3, 0.3, size=2)

    # to be compatible with tensorflow VDS
    def _sample_goal(self):
        return self.sample_goals(1)[0]

    def get_locations(self):
        num_blocks = 10
        length = 1./(num_blocks)
        sampled_loc = list()
        for block in self.free_positions:
            for row_idx in range(num_blocks):
                for col_idx in range(num_blocks):
                    location = block + np.array([row_idx, col_idx]) * length + length/2
                    sampled_loc.append(location)
        return np.array(sampled_loc), length

    def render(self, show_agent=True, show_colors=False):
        image_height = self.grid.shape[0] * self.block_size
        image_width = self.grid.shape[1] * self.block_size
        image = Image.new("RGB", (image_width, image_height), "white")
        draw = ImageDraw.Draw(image)
        # note: in PIL the first dim is left to right, the second dim is top to bottom. Origin is top-left corner.
        for row_idx in range(self.grid.shape[0]):
            for col_idx in range(self.grid.shape[1]):
                grid_value = self.grid[row_idx, col_idx]
                if col_idx == self.start_location[1] and row_idx == self.start_location[0]:
                    top_left = (col_idx * self.block_size, row_idx * self.block_size)
                    top_right = (col_idx * self.block_size, (row_idx + 1) * self.block_size)
                    bottom_right = ((col_idx+1) * self.block_size, (row_idx + 1) * self.block_size)
                    bottom_left = ((col_idx+1) * self.block_size, row_idx * self.block_size)
                    if show_colors:
                        draw.polygon((top_left, top_right, bottom_right, bottom_left), fill='blue')
                    else:
                        draw.polygon((top_left, top_right, bottom_right, bottom_left), fill='white')
                elif col_idx == self.goal_location[1] and row_idx == self.goal_location[0]:
                    top_left = (col_idx * self.block_size, row_idx * self.block_size)
                    top_right = (col_idx * self.block_size, (row_idx + 1) * self.block_size)
                    bottom_right = ((col_idx+1) * self.block_size, (row_idx + 1) * self.block_size)
                    bottom_left = ((col_idx+1) * self.block_size, row_idx * self.block_size)
                    if show_colors:
                        draw.polygon((top_left, top_right, bottom_right, bottom_left), fill='green')
                    else:
                        draw.polygon((top_left, top_right, bottom_right, bottom_left), fill='white')
                elif self.grid[row_idx, col_idx]:
                    top_left = (col_idx * self.block_size, row_idx * self.block_size)
                    top_right = (col_idx * self.block_size, (row_idx + 1) * self.block_size)
                    bottom_right = ((col_idx+1) * self.block_size, (row_idx + 1) * self.block_size)
                    bottom_left = ((col_idx+1) * self.block_size, row_idx * self.block_size)
                    draw.polygon((top_left, top_right, bottom_right, bottom_left), fill='gray')

        if show_agent:
            self.agent_size = 0.1 * self.block_size
            agent_location = self.position * self.block_size
            upper_left = tuple(tuple(agent_location - self.agent_size/2)[::-1])
            lower_right = tuple(tuple(agent_location + self.agent_size/2)[::-1])
            draw.ellipse((upper_left, lower_right), fill='green')
        return image
                    
    def draw_heatmap(self, sampled_loc, values, length, image):
        draw = ImageDraw.Draw(image)
        for loc, value in zip(sampled_loc, values):
            top_left = tuple(tuple((loc - length/2.) * self.block_size)[::-1])
            bottom_right = tuple(tuple((loc + length/2.) * self.block_size)[::-1])
            if np.isnan(value):
                continue
            else:
                color_value = round(float(value * 255.))  # higher certainty should be lighter
                color = (color_value, 0, 0)
            draw.rectangle([top_left, bottom_right], fill=color, width=0)
        return image

    def draw_points(self, locations, image, color='black'):
        point_size = 0.075 * self.block_size
        draw = ImageDraw.Draw(image)
        for location in locations:
            scaled_location = np.array(location) * self.block_size
            upper_left = tuple(scaled_location - point_size/2)
            lower_right = tuple(scaled_location + point_size/2)
            draw.ellipse((upper_left, lower_right), fill=color)
        return image

    def _compute_xy_distances(self, obs):
        achieved_goals = obs['achieved_goal'][:]
        desired_goals = obs['desired_goal'][:]
        diff = achieved_goals - desired_goals
        return np.atleast_1d(np.linalg.norm(diff, ord=2))


if __name__ == '__main__':
    class ArgsHolder(object):
        def __init__(self, args):
            self.keys = list()
            for key, value in dict(args).items():
                setattr(self, key, value)
                self.keys.append(key)

        def merge(self, args):
            for key, value in dict(args).items():
                if not hasattr(self, key):
                    setattr(self, key, value)
                    self.keys.append(key)
                else:
                    raise KeyError(f'Config attribute {key} already exists')

        def to_json(self):
            json = dict()
            for key in self.keys:
                json[key] = getattr(self, key)
            return json

    # This reads in visualize_configs.yml, produces a heatmap, and saves it to test.jpg
    
    from maze_generators.w_maze import generate_w_maze
    from maze_generators.e_maze import generate_e_maze
    from maze_generators.block_maze import generate_block_maze
    import yaml

    configs = yaml.safe_load(open('/storage/jalverio/hrl/hindsight_experience_replay/visualization_configs.yml', 'r'))
    args = ArgsHolder(configs)
    maze = generate_w_maze(5, 5, 5, 1, 1)
    maze = generate_e_maze(height=12, width=5, deadends=1, hallway_width=1)
    # maze = generate_block_maze(10, 10)
    env = MazeEnv2D(maze, args=args)
    optimal_values, normalized_values, max_hops, sampled_locations, length, optimal_hops = env.get_optimal_value_function()

    # THIS CODE IS FOR SAVING OPTIMAL Q VALUES
    breakpoint()
    np.save('/storage/jalverio/hrl/hindsight_experience_replay/optimal_q_values/12x5E.npy', optimal_values)

    image = env.render(show_agent=True, show_colors=True)
    drawn_image = image
    drawn_image = env.draw_heatmap(sampled_locations, normalized_values, length, image)
    drawn_image.save('/storage/jalverio/test.jpg')
    
