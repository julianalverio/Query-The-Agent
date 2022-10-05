import os
import pathlib
current = pathlib.Path(__file__).parent.resolve()
import sys
sys.path.insert(0, os.path.join(current, '../../gym'))
from gym.envs.registration import register

for grid_name in ['a', 'b', 'c']:
    register(
        id=f'Maze{grid_name.capitalize()}-v0',
        entry_point='baselines.envs.maze.maze:ParticleMazeEnv',
        kwargs={'grid_name': grid_name, 'reward_type': 'sparse'},
        max_episode_steps=50,
    )
