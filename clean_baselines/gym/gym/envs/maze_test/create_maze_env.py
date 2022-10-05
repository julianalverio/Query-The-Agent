# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from gym.envs.maze_test.ant_maze_env import AntMazeEnv
# from envs.point_maze_env import PointMazeEnv


def create_custom_maze_env(maze_type, maze_arr):
  # class is normally AntMaze, the class of the maze that you're going to instantiate
  # normally define n_bins for range sensor data
  # define env_name (normally Ant)
  # define maze_size_scaling (normally 8)
  # normally manual_collision is False
  # Maze id would be Maze (could also be flat, push, fall, block)
  # observe_blocks is False
  # put_spin_near_agent is False


  # gym_mujoco_kwargs = {
  #     'maze_id': maze_id,  # Maze
  #     'n_bins': n_bins,  # 0
  #     'observe_blocks': observe_blocks,  # False
  #     'put_spin_near_agent': put_spin_near_agent,  # False
  #     'top_down_view': top_down_view, # False
  #     'manual_collision': manual_collision,  # False
  #     'maze_size_scaling': maze_size_scaling  # 8
  # }

  # this would normally call AntMazeEnv with the kwargs. This simply has MODEL_CLASS = AntEnv and inherits from MazeEnv

  gym_env = cls(**gym_mujoco_kwargs)  # cls is AntMazeEnv class (trivial wrapper around MazeEnv class)
  gym_env.reset()
  return gym_env


def create_maze_env(env_name=None, top_down_view=False, n_bins=0):
  manual_collision = False
  if env_name.startswith('Ego'):
    n_bins = 8
    env_name = env_name[3:]
  if env_name.startswith('Ant'):
    cls = AntMazeEnv
    env_name = env_name[3:]
    maze_size_scaling = 8
  elif env_name.startswith('Point'):
    raise 'Point Not Yet Implemented'
    cls = PointMazeEnv
    manual_collision = True
    env_name = env_name[5:]
    maze_size_scaling = 4
  elif env_name.lower() == 'custom':
    pass
  else:
    assert False, 'unknown env %s' % env_name

  maze_id = None
  observe_blocks = False
  put_spin_near_agent = False
  if env_name == 'Maze':
    maze_id = 'Maze'
  elif env_name == 'Flat':
    maze_id = 'Flat'
  elif env_name == 'Push':
    maze_id = 'Push'
  elif env_name == 'Fall':
    maze_id = 'Fall'
  elif env_name == 'Block':
    maze_id = 'Block'
    put_spin_near_agent = True
    observe_blocks = True
  elif env_name == 'BlockMaze':
    maze_id = 'BlockMaze'
    put_spin_near_agent = True
    observe_blocks = True
  else:
    raise ValueError('Unknown maze environment %s' % env_name)

  gym_mujoco_kwargs = {
      'maze_id': maze_id,  # Maze
      'n_bins': n_bins,  # 0
      'observe_blocks': observe_blocks,  # False
      'put_spin_near_agent': put_spin_near_agent,  # False
      'top_down_view': top_down_view, # False
      'manual_collision': manual_collision,  # False
      'maze_size_scaling': maze_size_scaling  # 8
  }
  gym_env = cls(**gym_mujoco_kwargs)  # cls is AntMazeEnv class (trivial wrapper around MazeEnv class)
  gym_env.reset()
  # wrapped_env = gym_wrapper.GymWrapper(gym_env)
  # return wrapped_env
  return gym_env


# class TFPyEnvironment(tf_py_environment.TFPyEnvironment):

#   # import tensorflow as tf
#   # from tf_agents.environments import gym_wrapper
#   # from tf_agents.environments import tf_py_environment

#   def __init__(self, *args, **kwargs):
#     import pdb; pdb.set_trace()
#     super(TFPyEnvironment, self).__init__(*args, **kwargs)

#   def start_collect(self):
#     pass

#   def current_obs(self):
#     time_step = self.current_time_step()
#     return time_step.observation[0]  # For some reason, there is an extra dim.

#   def step(self, actions):
#     actions = tf.expand_dims(actions, 0)
#     next_step = super(TFPyEnvironment, self).step(actions)
#     return next_step.is_last()[0], next_step.reward[0], next_step.discount[0]

#   def reset(self):
#     return super(TFPyEnvironment, self).reset()
